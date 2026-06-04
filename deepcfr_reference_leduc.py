#!/usr/bin/env python3
"""
deepcfr_reference_leduc.py

A clean, self-contained Deep CFR reference implementation for Leduc Hold'em,
written directly from Brown et al. (2019), "Deep Counterfactual Regret
Minimization", Algorithm 1. No C++, no project buffer/solver code — only the
LeducHoldem game (known-good: tabular CFR converges on it) and PyTorch.

Purpose: establish a KNOWN-GOOD baseline. If this converges toward the tabular
reference exploitability (~0.138 at 1000 tabular iters; Deep CFR should trend
clearly downward), we then diff it against the project's deep_cfr_solver.py to
locate the discrepancy deterministically.

Faithful to the paper:
  - External-sampling MCCFR traversal. Traverser explores ALL actions; the
    opponent samples ONE action from the current strategy. (Paper §2.2.)
  - Strategy at each infoset = regret matching on the advantage network's
    output. When all predicted regrets <= 0, play the SINGLE highest-regret
    action with probability 1 (NOT uniform). (Paper §2.1 + Fig. 4.)
  - One sample of this iteration's INSTANTANEOUS regret vector per traverser
    infoset visited, added to a per-player reservoir advantage memory. (§4.)
  - Advantage network RE-TRAINED FROM SCRATCH each iteration from a random
    init, fitting the reservoir with MSE over the FULL action vector. (§4 +
    Fig. 4: fine-tuning instead of from-scratch raises exploitability ~50%.)
  - Strategy memory collects (infoset, iteration-weighted strategy) samples;
    the average-strategy network is trained once at the end. Here, because
    Leduc is tiny, we AVERAGE strategies tabularly for the exploitability
    check (exact), sidestepping a second network — this isolates the value
    network, which is the part under test.

Key correctness points that the project code got wrong (for later diffing):
  - The reservoir stores per-(infoset, iteration) INSTANTANEOUS regret VECTORS.
    It does NOT pre-sum within an iteration, and it does NOT collapse multiple
    visits into one averaged row. Each traverser-infoset visit in one traversal
    contributes one full-action regret vector; the network fits the mean over
    the reservoir, which (with from-scratch training + alternating updates)
    approximates the average regret that drives regret matching.
  - Regrets are NOT iteration-weighted going into the value memory.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver


# ── Infoset encoding (self-contained; does not use project encoder) ───────────
# Leduc infoset key from the game is like "J|Q|crc" (rank | community | actions).
# We encode it as a fixed-length float vector. This is deliberately simple and
# fully under our control so the encoder cannot be the hidden bug.

RANKS = {"J": 0, "Q": 1, "K": 2, "": 3}   # "" = community not yet revealed
ACTIONS = ["c", "r", "f", "k"]            # check, raise, fold, call
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MAX_ACT_HIST = 8
STATE_SIZE = 4 + 4 + MAX_ACT_HIST * 4     # priv rank(4) + comm rank(4) + hist


def encode_infoset(key: str) -> np.ndarray:
    """Encode a Leduc infoset key 'priv|comm|actions' into a float vector."""
    parts = key.split("|")
    priv = parts[0] if len(parts) > 0 else ""
    comm = parts[1] if len(parts) > 1 else ""
    hist = parts[2] if len(parts) > 2 else ""

    v = np.zeros(STATE_SIZE, dtype=np.float32)
    v[RANKS.get(priv, 3)] = 1.0
    v[4 + RANKS.get(comm, 3)] = 1.0
    for i, a in enumerate(hist[:MAX_ACT_HIST]):
        if a in ACTION_IDX:
            v[8 + i * 4 + ACTION_IDX[a]] = 1.0
    return v


# ── Advantage (regret) network ────────────────────────────────────────────────

class AdvantageNet(nn.Module):
    def __init__(self, state_size: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


N_ACTIONS = 4   # Leduc: c, r, f, k  (we map legal actions into these slots)

# Diagnostic log of raw instantaneous regrets for the K|| infoset.
_KREG_LOG = []


def regret_matching(pred: np.ndarray, legal_idx: list[int]) -> np.ndarray:
    """
    Regret matching over the LEGAL action subset.
    pred: length-N_ACTIONS predicted regrets.
    legal_idx: indices (into the 0..N_ACTIONS-1 slots) that are legal here.
    Returns a distribution over legal_idx (same order).
    """
    r = np.array([pred[i] for i in legal_idx], dtype=np.float64)
    pos = np.maximum(r, 0.0)
    s = pos.sum()
    if s > 0:
        return pos / s
    # All non-positive: fall back to UNIFORM over legal actions.
    #
    # Brown et al. play the single highest-regret action w.p. 1 here, which is
    # correct when the advantage net is well-trained: at equilibrium some action
    # genuinely dominates and deterministic play is right. But with a noisy net
    # in a small game it is actively harmful. When predicted regrets hover near
    # zero (as K|| does -- true cumulative [0.77, 1.81] is a shallow 30/70 mix),
    # noise routinely pushes BOTH actions slightly negative; argmax then collapses
    # the mixed equilibrium strategy to a pure one, which in external sampling
    # stops exploring the unplayed action, distorts its regret estimate, and
    # feeds a self-reinforcing divergence (observed: K|| drifting -0.04 -> -0.75
    # and locking to [0,1] despite the accumulator being correct in isolation).
    # Uniform preserves exploration and breaks the feedback loop. This is the
    # original Hart & Mas-Colell (2000) regret matching.
    n = len(legal_idx)
    return np.ones(n, dtype=np.float64) / n


# ── Action <-> slot mapping ───────────────────────────────────────────────────
# The game returns legal actions as strings like ['c','r'] or ['f','k','r'].
# We map each to a fixed slot so the network output is consistent.

def legal_slots(game, history):
    acts = game.legal_actions(history)
    return acts, [ACTION_IDX[a] for a in acts]


# ── External-sampling MCCFR traversal ─────────────────────────────────────────

def traverse(game, history, traverser, net, value_mem, strat_mem, cum_plus, t, rng):
    """
    External sampling. Returns the traverser's counterfactual value at `history`.

    Records, per traverser infoset, a regret target into value_mem. The target
    form is selected by DCFR_TARGET: 'cumplus' (CFR+-clipped cumulative regret,
    the quantity tabular CFR+ uses; default) or 'instant' (raw instantaneous,
    previous form whose mean had the wrong sign). cum_plus is the per-player
    CumulativePlus table maintaining the running clipped cumulative regret.
    """
    if game.is_terminal(history):
        return game.terminal_payoffs(history)[traverser]

    player = game.current_player(history)
    acts, slots = legal_slots(game, history)
    key = game.info_set_key(history, player)

    # Current strategy from the advantage net (regret matching).
    state = encode_infoset(key)
    with torch.no_grad():
        pred = net(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
    strat = regret_matching(pred, slots)   # over legal acts, in `acts` order

    if player == traverser:
        # Explore ALL actions; compute action values and the node value.
        action_values = np.zeros(len(acts), dtype=np.float64)
        for i, a in enumerate(acts):
            action_values[i] = traverse(
                game, game.apply_action(history, a),
                traverser, net, value_mem, strat_mem, cum_plus, t, rng
            )
        node_value = float(np.dot(strat, action_values))

        # Instantaneous regret r^t(I,a) = v(I,a) - v(I), over full slots.
        regret_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        for i, slot in enumerate(slots):
            regret_vec[slot] = action_values[i] - node_value

        # Lightweight instrumentation: record only the K|| instantaneous regret
        # (first two slots) so the measurement loop can report reg_SE -- the
        # standing check that 1000 traversals keeps external-sampling variance
        # resolvable. The earlier heavy per-visit logging (strategy used, action
        # values, node value) served the sign diagnosis, which is now resolved.
        if key == "K||":
            _KREG_LOG.append((float(regret_vec[0]), float(regret_vec[1])))

        # Target form selected by DCFR_TARGET (see module note).
        #   cumplus: CFR+-clipped cumulative regret -- the quantity tabular CFR+
        #            regret-matches on. Preserves cumulative sign (fixes the
        #            measured sign bug: K mean [-1.1,+0.05] but tabular cumulative
        #            [+10.8,+60.1]) and CFR+ clipping keeps it bounded without
        #            normalisation. The reservoir stores SNAPSHOTS of this running
        #            cumulative; from-scratch fit converges to its mean, which
        #            tracks the cumulative since later (larger) snapshots dominate.
        #   instant: raw instantaneous regret (previous form, wrong-sign mean).
        if _DCFR_TARGET_MODE == "instant":
            target = regret_vec
        else:
            # CFR+ cumulative, normalised by iteration t to bound the scale.
            # Measured: tabular cum_regret/iterations = [0.011,0.060] for K, a
            # network-fittable O(0.01-0.1) scale. Division by t>0 preserves sign
            # and ratios (regret matching is scale-invariant), so the strategy is
            # unchanged while the regression target stops diverging (was +4466).
            target = (cum_plus.add(key, regret_vec) / float(t)).astype(np.float32)
        value_mem.add(state, target, float(t))

        # Strategy memory at the traverser's nodes, Linear-weighted by t.
        strat_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        for i, slot in enumerate(slots):
            strat_vec[slot] = strat[i]
        strat_mem.add_strategy(key, strat_vec, float(t))

        return node_value
    else:
        # Opponent: SAMPLE one action from the current strategy and recurse.
        idx = rng.choice(len(acts), p=strat)
        return traverse(
            game, game.apply_action(history, acts[idx]),
            traverser, net, value_mem, strat_mem, cum_plus, t, rng
        )


# NOTE: The CumulativeRegret table and DCFR alpha/beta discounting were REMOVED.
# They were a departure from Brown et al. (2019): the paper stores raw
# instantaneous regrets in the reservoir and weights the LOSS by t', it does not
# maintain an accumulated (and therefore unbounded, diverging) per-infoset value.
# The accumulated target diverged (D_cum for J reached -50397) and per-infoset
# normalisation D/||D|| only converted that into a moving unit-vector target the
# net could not fit. The advantage D_T is now produced IMPLICITLY by the
# t'-weighted from-scratch fit over bounded instantaneous samples.


# ── Regret target form (DCFR_TARGET) ──────────────────────────────────────────
# 'cumplus' : reservoir stores the CFR+-clipped CUMULATIVE regret (per infoset,
#             max(R,0) each update). This is the quantity tabular CFR+ uses for
#             regret matching; it preserves the cumulative SIGN (the bug fix) and
#             CFR+ clipping keeps it BOUNDED without normalisation (~60 for K in
#             Leduc at 1000 iters). This is the default.
# 'instant' : reservoir stores the raw instantaneous regret r^t (previous form).
#             Kept for A/B comparison. Its t'-weighted MEAN has the wrong sign
#             relative to the cumulative sum (measured: K target [-1.1,+0.05] vs
#             tabular cum_regret [+10.8,+60.1]).
_DCFR_TARGET_MODE = os.environ.get("DCFR_TARGET", "cumplus").strip().lower()


class CumulativePlus:
    """Per-infoset CFR+-clipped cumulative regret: R <- max(R + r^t, 0).

    Mirrors the tabular MCCFRSolver (cfr_plus=True) that CONVERGES on Leduc.
    Clipping prevents negative accumulation toward -inf (no divergence) and
    yields the correct sign for regret matching. No DCFR discounting, no
    normalisation -- CFR+ alone keeps the scale network-fittable.
    """
    def __init__(self):
        self.table = {}   # key -> cumulative regret vector (N_ACTIONS,)

    def add(self, key, regret_vec):
        R = self.table.get(key)
        if R is None:
            R = np.zeros(N_ACTIONS, dtype=np.float64)
        R = np.maximum(R + regret_vec, 0.0)   # CFR+ clip
        self.table[key] = R
        return R.astype(np.float32)


# ── Reservoir advantage memory (Vitter) ───────────────────────────────────────

class Reservoir:
    """Vitter (1985) Algorithm R. Stores (state, instantaneous_regret, t').

    Per Brown et al. (2019) Algorithm 1, the reservoir holds raw INSTANTANEOUS
    regret vectors r~_t' together with the iteration number t' at which they
    were sampled. There is NO accumulation and NO normalisation: Lemma 1 proves
    the sampled instantaneous regret is an unbiased estimator of the advantage,
    so it is stored as-is. The iteration weight t' is consumed by the loss
    (later iterations weighted more), so the from-scratch fit converges to the
    t'-weighted mean of instantaneous regrets -- which equals D_T, the Linear-CFR
    advantage. Crucially this target is BOUNDED (instantaneous regrets are
    bounded) even though their cumulative sum is not -- the divergence that the
    CumulativeRegret table produced.
    """
    def __init__(self, capacity, state_size, n_actions, seed=0):
        self.cap = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.targets = np.zeros((capacity, n_actions), dtype=np.float32)
        self.weights = np.zeros(capacity, dtype=np.float32)   # iteration t'
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, target, weight):
        if self.size < self.cap:
            i = self.size
            self.states[i] = state
            self.targets[i] = target
            self.weights[i] = weight
            self.size += 1
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < self.cap:
                self.states[j] = state
                self.targets[j] = target
                self.weights[j] = weight
        self.seen += 1

    def sample(self, n):
        n = min(n, self.size)
        idx = self.rng.integers(0, self.size, size=n)
        return self.states[idx], self.targets[idx], self.weights[idx]


# ── Tabular strategy memory (exact average for Leduc exploitability) ──────────
# Brown 2019 trains a separate strategy network; for Leduc we can average
# strategies tabularly and feed them to CFRSolver's exact best-response. This
# isolates the ADVANTAGE network as the unit under test.

class StrategyMemory:
    def __init__(self):
        # key -> [weighted strategy sum (N_ACTIONS), weight sum]
        self.acc = {}

    def add_strategy(self, key, strat_vec, weight):
        if key not in self.acc:
            self.acc[key] = [np.zeros(N_ACTIONS, dtype=np.float64), 0.0]
        self.acc[key][0] += weight * strat_vec
        self.acc[key][1] += weight

    def average(self, key, legal_slots_list):
        if key not in self.acc or self.acc[key][1] <= 0:
            n = len(legal_slots_list)
            return np.ones(n) / n
        full = self.acc[key][0] / self.acc[key][1]
        sub = np.array([full[s] for s in legal_slots_list], dtype=np.float64)
        tot = sub.sum()
        if tot > 0:
            return sub / tot
        n = len(legal_slots_list)
        return np.ones(n) / n


# ── Train advantage net from scratch on the reservoir ─────────────────────────

# Loss-muoto valitaan ymparistomuuttujalla DCFR_LOSS (diagnostiikkaa varten):
#   'batchnorm' (oletus): (Sum_i t'_i * MSE_i) / Sum_i t'_i   -- numeerinen turva
#   'raw'                : (t'_i * MSE_i).mean()              -- paperin raaka paino
# raw testaa hypoteesia etta batch-normalisointi laimentaa korkean t':n tuoreet
# nayttet ja estaa D_T-approksimaation. gradient-clip pitaa raw:n skaalan kurissa.
_DCFR_LOSS_MODE = os.environ.get("DCFR_LOSS", "batchnorm").strip().lower()


def train_from_scratch(reservoir, hidden, sgd_steps, batch, lr, device):
    """Train the advantage net FROM SCRATCH (Brown et al. 2019 Algorithm 1).

    Loss: L(theta) = E_(I,t',r~)~MV [ t' * Sum_a (r~(a) - V(I,a))^2 ].

    Returns (net, final_loss). final_loss is the last SGD step's loss, used to
    distinguish H1 (net cannot fit reservoir -> high loss -> training-capacity
    problem) from H2 (net fits well -> low loss -> the TARGET itself is wrong).

    DCFR_LOSS env var selects the weighting form (see module note above).
    """
    net = AdvantageNet(STATE_SIZE, N_ACTIONS, hidden).to(device)
    if reservoir.size < 2:
        return net, float("nan")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    final_loss = float("nan")
    for _ in range(sgd_steps):
        s, tgt, w = reservoir.sample(batch)
        s = torch.from_numpy(s).to(device)
        tgt = torch.from_numpy(tgt).to(device)
        w = torch.from_numpy(w).to(device)               # iteration t', shape [B]
        pred = net(s)
        per_sample_mse = ((pred - tgt) ** 2).mean(dim=1)  # MSE over actions, [B]
        # cumplus target is ALREADY time-weighted (it is an accumulation), so
        # t'-weighting the loss would double-count. Use plain MSE there.
        # instant target keeps the previous t'-weighted behaviour for A/B.
        if _DCFR_TARGET_MODE != "instant":
            loss = per_sample_mse.mean()                  # plain MSE (cumplus)
        elif _DCFR_LOSS_MODE == "raw":
            loss = (w * per_sample_mse).mean()            # paper's raw t' weight
        else:
            loss = (w * per_sample_mse).sum() / (w.sum() + 1e-8)  # batch-normalised
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())
    net.eval()
    return net, final_loss


# ── Exact exploitability of the averaged strategy ─────────────────────────────

def exact_exploitability(game, strat_mem):
    """Build a CFRSolver carrying the averaged strategy; use its exact BR."""
    ref = CFRSolver(game=game, linear_averaging=True)

    def walk(history):
        if game.is_terminal(history):
            return
        player = game.current_player(history)
        acts, slots = legal_slots(game, history)
        key = game.info_set_key(history, player)
        if key not in ref.info_sets:
            data = ref._get_or_create_info_set(key, acts)
            avg = strat_mem.average(key, slots)
            data.cumulative_strategy = np.asarray(avg, dtype=np.float64).copy()
        for a in acts:
            walk(game.apply_action(history, a))

    for init_h, _ in game.initial_histories():
        walk(init_h)
    return ref.exploitability()


def current_strategy_exploitability(game, nets):
    """
    Exact exploitability of the CURRENT regret-matching strategy (read straight
    off the advantage nets), as opposed to the time-average. If this converges
    but the average does not, the bug is in StrategyMemory; if neither
    converges, the bug is in the advantage nets / regret target.
    """
    ref = CFRSolver(game=game, linear_averaging=True)

    def walk(history):
        if game.is_terminal(history):
            return
        player = game.current_player(history)
        acts, slots = legal_slots(game, history)
        key = game.info_set_key(history, player)
        if key not in ref.info_sets:
            state = encode_infoset(key)
            with torch.no_grad():
                pred = nets[player](
                    torch.from_numpy(state).unsqueeze(0)
                ).squeeze(0).numpy()
            strat = regret_matching(pred, slots)
            data = ref._get_or_create_info_set(key, acts)
            data.cumulative_strategy = np.asarray(strat, dtype=np.float64).copy()
        for a in acts:
            walk(game.apply_action(history, a))

    for init_h, _ in game.initial_histories():
        walk(init_h)
    return ref.exploitability()


# ── Main Deep CFR loop ────────────────────────────────────────────────────────

def main():
    device = "cpu"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    game = LeducHoldem()

    # Reference
    tab = CFRSolver(game=game, linear_averaging=True)
    tab.solve(iterations=1000)
    ref = tab.exploitability()
    print(f"[reference] tabular CFR exploitability = {ref:.5f}\n")

    # Convergence run: 1000 iterations, no average-strategy reset. The reset is
    # a degree of freedom that would HIDE the early transient rather than
    # measure it; run clean first. If avg_expl reaches the function-approx floor
    # (~0.3-0.8) by 1000 without it, the transient is harmless. Only if the
    # descent stalls clearly above the floor do we add a post-transient reset
    # and measure the difference -- one variable at a time.
    ITERS         = 1000
    TRAVERSALS    = 1000
    HIDDEN        = 128
    SGD_STEPS     = 1000
    BATCH         = 1024
    LR            = 1e-3
    RES_CAP       = 1_000_000
    MEASURE_EVERY = 25

    value_mem = {0: Reservoir(RES_CAP, STATE_SIZE, N_ACTIONS, seed=1),
                 1: Reservoir(RES_CAP, STATE_SIZE, N_ACTIONS, seed=2)}
    strat_mem = StrategyMemory()
    # Per-player CFR+ cumulative regret tables (used when DCFR_TARGET=cumplus).
    cum_plus = {0: CumulativePlus(), 1: CumulativePlus()}

    # Advantage nets start returning ~0 (random init is fine; paper inits to 0,
    # but a small random net is standard and works). One per player.
    nets = {0: AdvantageNet(STATE_SIZE, N_ACTIONS, HIDDEN).to(device),
            1: AdvantageNet(STATE_SIZE, N_ACTIONS, HIDDEN).to(device)}
    _last_train_loss = {0: float("nan"), 1: float("nan")}

    print(f"Deep CFR reference | Leduc | hidden={HIDDEN} "
          f"traversals={TRAVERSALS} iters={ITERS} target={_DCFR_TARGET_MODE} "
          f"(Brown et al. 2019)")
    print(f"{'iter':>6} {'avg_expl':>9} {'cur_expl':>9} {'ref':>7} {'ratio':>6} "
          f"{'t(s)':>7}  reg_SE  mean_regret per preflop infoset (J / Q / K)")

    t0 = time.time()
    for t in range(1, ITERS + 1):
        # Alternating updates: one traverser per iteration (paper uses
        # alternating; we alternate by parity).
        traverser = (t % 2)
        for _ in range(TRAVERSALS):
            init_states = game.initial_histories()
            idx = rng.integers(0, len(init_states))
            init_h = init_states[idx][0]
            traverse(game, init_h, traverser,
                     nets[traverser], value_mem[traverser], strat_mem,
                     cum_plus[traverser], t, rng)

        # Retrain THIS traverser's advantage net from scratch on its reservoir.
        nets[traverser], _last_loss = train_from_scratch(
            value_mem[traverser], HIDDEN, SGD_STEPS, BATCH, LR, device
        )
        _last_train_loss[traverser] = _last_loss

        if t % MEASURE_EVERY == 0:
            # avg and cur on the SAME cadence: the "avg falls / cur oscillates"
            # split is the DCFR signature used to localise convergence, and it
            # only reads if both share a timeline.
            avg_expl = exact_exploitability(game, strat_mem)
            cur_expl = current_strategy_exploitability(game, nets)
            ratio = avg_expl / ref if ref > 0 else float("nan")

            # t'-weighted mean instantaneous regret for three canonical preflop
            # infosets (weak / medium / strong hand). Tabular truths:
            #   J||: a marginal hand; Q||: medium; K||: strong, mix [0.30,0.70].
            # This is the BOUNDED target the net fits (replacing the removed,
            # diverging cumulative table); sign/ratio settling = convergence.
            def dstr(key):
                # t'-weighted mean instantaneous regret for `key`, computed over
                # whichever player's reservoir contains it. This is the target
                # the net fits; its sign/ratio settling is the convergence
                # signal. Bounded, unlike the removed cumulative table.
                st = encode_infoset(key)
                for p in (0, 1):
                    r = value_mem[p]
                    if r.size == 0:
                        continue
                    rows = np.where(
                        (r.states[:r.size] == st).all(axis=1))[0]
                    if len(rows) == 0:
                        continue
                    w = r.weights[rows]
                    tgt = r.targets[rows]
                    wmean = (w[:, None] * tgt).sum(axis=0) / (w.sum() + 1e-8)
                    return f"[{wmean[0]:+7.3f},{wmean[1]:+7.3f}]"
                return "[ --  , --  ]"

            # reg_SE: cheap insurance that the variance fix (1000 traversals)
            # holds across the whole run. Standard error of the K|| v0-v1 signal.
            if _KREG_LOG:
                diff = np.array([e[0] - e[1] for e in _KREG_LOG], dtype=np.float64)
                reg_se = diff.std() / np.sqrt(max(len(diff), 1))
                _KREG_LOG.clear()
            else:
                reg_se = float("nan")

            # DIAGNOSTIIKKA (H1 vs H2): fittaako verkko reservoiria, ja onko
            # kohde oikea. Verrataan K||-ennustetta sen reservoir-kohteiden
            # t'-painotettuun keskiarvoon (= dstr('K||') yllÃ¤). Jos final_loss
            # on matala MUTTA ennuste poikkeaa kohteesta -> kohde on epÃ¤vakaa
            # (H2). Jos final_loss on korkea -> verkko ei fittaa (H1).
            st_k = encode_infoset("K||")
            with torch.no_grad():
                pk = nets[traverser](
                    torch.from_numpy(st_k).unsqueeze(0)).squeeze(0).numpy()
            print(f"{t:>6} {avg_expl:>9.4f} {cur_expl:>9.4f} {ref:>7.4f} "
                  f"{ratio:>6.2f} {time.time()-t0:>7.1f}  reg_SE={reg_se:.3f}  "
                  f"J{dstr('J||')} Q{dstr('Q||')} K{dstr('K||')}")
            print(f"       [diag] loss(P{traverser})={_last_train_loss[traverser]:.4f} "
                  f"mode={_DCFR_LOSS_MODE} "
                  f"K||_net_pred=[{pk[0]:+.3f},{pk[1]:+.3f}] "
                  f"res_size={value_mem[traverser].size}")

    print("\nExpected: avg_expl trends DOWN toward the function-approx floor")
    print("(~0.3-0.8) while cur_expl oscillates -- expected and never converges")
    print("(only the average policy converges in CFR). mean_regret signs should")
    print("settle (J/Q/K) and stay BOUNDED, and reg_SE ~0.1 throughout. If avg")
    print("reaches the floor, this is the validated baseline for the project diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())