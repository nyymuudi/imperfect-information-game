"""
Python mirror of the corrected C++ NLHEMCCFREngine full-traversal CFR+ path,
for parity testing on a small enumerable subtree.

Mirrors EXACTLY:
  traverse_full        -> _traverse_full (vanilla CFR, expand all, opp-reach weight)
  accumulate_cfrplus   -> keyed on STATE VECTOR (not info_set_key);
                          R[slot] = max(R[slot] + instant, 0); store state once
  emit_cfrplus_targets -> for each accumulated state, emit 4 slots R[slot]/visits
  _collapse_by_state   -> group by exact state vector into an [m,4] target matrix

The result is the [m,4] regret-target matrix the regret network would be fit on.
The C++ parity test compares this matrix against the engine's emitted+collapsed
output for the SAME fixed deal.

Uses PostflopNLHE for game logic and NLHEEncoder for state vectors — the same
classes the live Python pipeline uses; test_parity.py already locks these to the
C++ NLHEGame / NLHEStateEncoder.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.state_encoder import NLHEEncoder

NLHE_NUM_ACTIONS = 4
STATE_SIZE = 124

# Map PostflopNLHE action chars to the C++ NLHEAction enum slots.
#   slot 0 = fold/check ('f' or 'c'), 1 = call ('k'), 2 = raise ('r'), 3 = all-in ('a')
ACTION_SLOT = {"f": 0, "c": 0, "k": 1, "r": 2, "a": 3}


# Dims 120 (preflop equity) and 121 (board_strength) DIVERGE between the C++
# encoder (preflop_equity() formula / score-based) and the Python encoder
# (Monte-Carlo equity_vs_random / evaluate_7card) — a documented residual in
# test_parity.py. They are network INPUT features only and never enter the CFR+
# regret computation, so we exclude them from the parity key. All other dims
# (cards, street, betting, action history, pot-odds, SPR) uniquely identify the
# node and ARE bit-identical across implementations.
_KEY_DIMS = [i for i in range(124) if i not in (120, 121)]


def _state_key(sv):
    """Quantise the parity-relevant dims to a 1e-6 grid; pack as int32 bytes.
    Excludes dims 120-121 (equity/board_strength) — see note above."""
    arr = np.asarray(sv, dtype=np.float64)[_KEY_DIMS]
    q = np.rint(arr * 1e6).astype(np.int32)
    return q.tobytes()


class NLHEFullCfrPlusRef:
    def __init__(self, game: PostflopNLHE, encoder: NLHEEncoder, target="cfrplus"):
        self.game = game
        self.enc = encoder
        self.target = target
        # state_key -> {"R": np.array(4), "state": np.array(124), "visits": int}
        self.acc = {}
        self.instant_rows = []   # for target="instant": (state_key, slot, val)
        self._state_cache = {}   # state_key -> state vector (for emission)

    def _strategy(self, n):
        return np.ones(n) / n

    def _traverse_full(self, h, tp, reach_opp):
        if self.game.is_terminal(h):
            return self.game.terminal_payoffs(h)[tp]
        p = self.game.current_player(h)
        acts = self.game.legal_actions(h)
        n = len(acts)
        strat = self._strategy(n)

        vals = np.zeros(n)
        for a in range(n):
            nh = self.game.apply_action(h, acts[a])
            opp_reach = reach_opp if p == tp else reach_opp * strat[a]
            vals[a] = self._traverse_full(nh, tp, opp_reach)
        nv = float((strat * vals).sum())

        if p == tp:
            sv = self.enc.encode(h, p).astype(np.float32)
            key = _state_key(sv)
            instant = reach_opp * (vals - nv)
            if self.target == "instant":
                for a in range(n):
                    self.instant_rows.append((key, ACTION_SLOT[acts[a]],
                                              float(instant[a]), sv))
            else:
                e = self.acc.get(key)
                if e is None:
                    e = {"R": np.zeros(NLHE_NUM_ACTIONS),
                         "state": sv.copy(), "visits": 0}
                    self.acc[key] = e
                for a in range(n):
                    slot = ACTION_SLOT[acts[a]]
                    e["R"][slot] = max(e["R"][slot] + instant[a], 0.0)
                e["visits"] += 1
        return nv

    def run_deal(self, deal):
        for tp in range(2):
            self._traverse_full(deal, tp, 1.0)

    def emit_targets(self):
        """Return (states [m,124], targets [m,4]) — the collapsed regret matrix."""
        rows = []          # (state_vec, slot, value)
        if self.target == "instant":
            for key, slot, val, sv in self.instant_rows:
                rows.append((sv, slot, val))
        else:
            for key, e in self.acc.items():
                inv = 1.0 / e["visits"]
                for slot in range(NLHE_NUM_ACTIONS):
                    rows.append((e["state"], slot, e["R"][slot] * inv))
        return self._collapse(rows)

    def _collapse(self, rows):
        """Mirror DeepCFRSolver._collapse_by_state: group by exact state vector,
        scatter values into [m,4]."""
        if not rows:
            return np.zeros((0, STATE_SIZE), np.float32), np.zeros((0, 4), np.float32)
        X = np.stack([r[0] for r in rows]).astype(np.float32)
        a = np.array([r[1] for r in rows], dtype=np.int64)
        v = np.array([r[2] for r in rows], dtype=np.float32)
        uniq, inverse = np.unique(X, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        targets = np.zeros((uniq.shape[0], NLHE_NUM_ACTIONS), dtype=np.float32)
        np.add.at(targets, (inverse, a), v)
        return uniq.astype(np.float32), targets


def make_ref(stack=10.0, max_raises=1, equity_sims=100):
    game = PostflopNLHE(starting_stack=stack, max_raises_per_street=max_raises,
                        raise_fractions=(0.75,))
    NLHEEncoder._shared_equity_cache = None
    enc = NLHEEncoder(starting_stack=stack, equity_sims=equity_sims)
    return game, enc


if __name__ == "__main__":
    from src.abstraction.equity import str_to_card as C
    game, enc = make_ref()
    # Fixed deal: AhKh vs QdJd, fixed board (low stack keeps the tree small).
    deal = ((C("Ah"), C("Kh")), (C("Qd"), C("Jd")),
            (C("7c"), C("8s"), C("9c"), C("2h"), C("3d")))
    ref = NLHEFullCfrPlusRef(game, enc, target="cfrplus")
    ref.run_deal(deal)
    states, targets = ref.emit_targets()
    print(f"unique states: {states.shape[0]}")
    print(f"target matrix: {targets.shape}")
    print(f"nonzero target rows: {int((np.abs(targets).sum(1) > 1e-9).sum())}")
    print(f"target range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"all targets >= 0: {(targets >= -1e-9).all()}")