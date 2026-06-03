"""
Python mirror of the corrected C++ MCCFREngine full-traversal CFR+ path.

Mirrors EXACTLY:
  traverse_full      -> _traverse_full   (vanilla CFR, expand all, opp-reach weight)
  accumulate_cfrplus -> R[a] = max(R[a] + instant[a], 0); visits += 1
  emit_cfrplus_targets -> target[a] = R[a] / visits   (once per infoset per iter)

Then regret-matches on R (drives current strategy, same as the live C++ engine),
accumulates Linear-CFR average strategy, and measures exact exploitability vs
the tabular CFR+ ground truth (~0.13 region). This is the reference the C++
parity test will compare against bit-for-bit on a single deterministic iteration.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver, InfoSetData

MAX_ACTIONS = 4


class FullCfrPlusRef:
    def __init__(self, game, target="cfrplus", seed=42):
        self.game = game
        self.target = target
        self.R = {}              # iset -> np.array (clipped cumulative, MAX_ACTIONS)
        self.visits = {}         # iset -> int
        self.nacts = {}          # iset -> int
        self.acts = {}           # iset -> list[action]
        self.cum_strategy = {}   # iset -> np.array  (Linear CFR avg)
        self.iterations = 0
        # Emitted targets, mirrors the regret buffer rows from emit_cfrplus_targets.
        self.emitted = []        # list of (iset, action_idx, value, iteration)

    def _ensure(self, key, acts):
        if key not in self.R:
            n = len(acts)
            self.R[key] = np.zeros(MAX_ACTIONS)
            self.visits[key] = 0
            self.nacts[key] = n
            self.acts[key] = list(acts)
            self.cum_strategy[key] = np.zeros(n)

    def _strategy(self, key, n):
        if key not in self.R:
            return np.ones(n) / n
        pos = np.maximum(self.R[key][:n], 0.0)
        s = pos.sum()
        return pos / s if s > 0 else np.ones(n) / n

    def _traverse_full(self, h, tp, reach_opp):
        if self.game.is_terminal(h):
            return self.game.terminal_payoffs(h)[tp]
        p = self.game.current_player(h)
        acts = self.game.legal_actions(h)
        n = len(acts)
        key = self.game.info_set_key(h, p)
        self._ensure(key, acts)
        strat = self._strategy(key, n)

        vals = np.zeros(n)
        for a in range(n):
            nh = self.game.apply_action(h, acts[a])
            opp_reach = reach_opp if p == tp else reach_opp * strat[a]
            vals[a] = self._traverse_full(nh, tp, opp_reach)
        nv = float((strat * vals).sum())

        if p == tp:
            # Linear-CFR strategy accumulation (weight by opp reach, like CFR).
            self.cum_strategy[key] += (self.iterations + 1) * reach_opp * strat
            instant = reach_opp * (vals - nv)
            if self.target == "instant":
                for a in range(n):
                    self.emitted.append((key, a, float(instant[a]), self.iterations))
            else:
                R = self.R[key]
                for a in range(n):
                    R[a] = max(R[a] + instant[a], 0.0)
                self.visits[key] += 1
        return nv

    def emit_cfrplus(self):
        for key, R in self.R.items():
            v = self.visits[key]
            if v <= 0:
                continue
            n = self.nacts[key]
            inv = 1.0 / v
            for a in range(n):
                self.emitted.append((key, a, float(R[a] * inv), self.iterations))

    def run(self, iters):
        deals = self.game.initial_histories()
        for _ in range(iters):
            for tp in range(2):
                for h, prob in deals:
                    self._traverse_full(h, tp, prob)
            if self.target == "cfrplus":
                self.emit_cfrplus()
            self.iterations += 1

    def avg_strategy(self):
        out = {}
        for key, cs in self.cum_strategy.items():
            s = cs.sum()
            out[key] = (cs / s if s > 0 else np.ones(len(cs)) / len(cs), self.acts[key])
        return out


def exploitability_of(game, avg):
    ref = CFRSolver(game=game, linear_averaging=True)
    for key, (strat, acts) in avg.items():
        ref.info_sets[key] = InfoSetData(
            actions=list(acts),
            cumulative_regret=np.zeros(len(strat)),
            cumulative_strategy=np.asarray(strat, dtype=np.float64).copy())
    return ref.exploitability()


if __name__ == "__main__":
    game = LeducHoldem()
    for ITERS in (50, 200, 500):
        print(f"--- full-traversal CFR+, iters={ITERS} ---")
        for mode in ["instant", "cfrplus"]:
            soln = FullCfrPlusRef(game, target=mode)
            soln.run(ITERS)
            expl = exploitability_of(game, soln.avg_strategy())
            print(f"  {mode:8s}  exploitability={expl:.4f}  emitted_rows={len(soln.emitted)}")
        ref = CFRSolver(game=game, linear_averaging=True, cfr_plus=True)
        ref.solve(iterations=ITERS)
        print(f"  {'tabular':8s}  exploitability={ref.exploitability():.4f}")