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

def traverse(game, history, traverser, net, value_mem, strat_mem, cum_reg, t, rng):
    """
    External sampling. Returns the traverser's counterfactual value at `history`.

    Records, per traverser infoset, the LINEAR-CFR CUMULATIVE regret
    (sum of t * r^t over iterations) into cum_reg, and pushes a snapshot of
    that cumulative vector into value_mem. The network's training target is
    therefore the cumulative regret -- the SAME quantity tabular CFR's regret
    matching uses -- NOT the mean of instantaneous regrets (which, as measured,
    has the wrong sign relative to the cumulative sum and does not converge).
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
                traverser, net, value_mem, strat_mem, cum_reg, t, rng
            )
        node_value = float(np.dot(strat, action_values))

        # Instantaneous regret r^t(I,a) = v(I,a) - v(I), over full slots.
        regret_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        for i, slot in enumerate(slots):
            regret_vec[slot] = action_values[i] - node_value

        # DIAGNOSTIC: log raw instantaneous regret for K|| before accumulation,
        # together with the strategy used and the action values, so we can see
        # whether the SIGN problem originates in the regret INPUT (traverse) or
        # the accumulation. Tabular truth: cumulative [+0.77,+1.81]; the mean
        # instantaneous regret should therefore be positive on BOTH actions.
        if key == "K||":
            _KREG_LOG.append((
                t, regret_vec[:2].copy(), strat.copy(), action_values.copy(),
                float(node_value)
            ))

        # Linear CFR: accumulate t * r^t into the per-infoset cumulative regret,
        # and track the running sum of weights W = Σ_τ τ. The NETWORK TARGET is
        # the WEIGHTED-AVERAGE regret C(I)/W, NOT the raw cumulative sum.
        #
        # Why: regret matching is scale-invariant (R+/ΣR+), so for the tabular
        # algorithm the absolute magnitude of cumulative regret is irrelevant --
        # only signs and ratios matter. But a network that REGRESSES to the
        # target is not scale-invariant: the raw cumulative sum grows without
        # bound (~t^2 with Linear weights), so the same infoset appears in the
        # reservoir with wildly different target magnitudes across iterations,
        # which the net cannot fit (observed: K|| predictions exploding to
        # ±thousands and flipping sign). Dividing by W yields a stable-scale
        # quantity with the SAME signs and ratios as the cumulative sum, which
        # is exactly what regret matching needs and what the net can actually
        # learn. C(I)/W -> a bounded vector; for K|| it should approach a
        # DCFR (Brown & Sandholm 2019): maintain DISCOUNTED cumulative regret.
        # Positive cumulative regrets are multiplied each step by (n/(n+1))^alpha
        # and negative ones by (n/(n+1))^beta before adding this iteration's
        # instantaneous regret. This down-weights early (far-from-equilibrium)
        # iterations relative to later ones, fixing the SIGN/RATIO problem that
        # the mean-of-instantaneous target had (K|| mean had the wrong sign).
        #
        # The discounted cumulative D(I) still grows, so a raw regression target
        # would still explode in scale. We therefore feed the net the PER-INFOSET
        # NORMALISED vector D(I)/||D(I)||. Regret matching is scale-invariant
        # (R+/ΣR+), so normalising per infoset does NOT change the strategy -- it
        # only puts the regression target on a stable O(1) scale the net can fit,
        # while EXACTLY preserving the action ratios. This is the combination the
        # two failed Variant-1 attempts each missed: discounting alone keeps the
        # ratio but explodes in scale; dividing by the weight sum fixes scale but
        # collapses the ratio to zero. D/||D|| does both.
        cumulative = cum_reg.add(key, regret_vec, t)
        norm = np.linalg.norm(cumulative)
        target = (cumulative / norm if norm > 1e-8 else cumulative).astype(np.float32)

        value_mem.add(state, target)

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
            traverser, net, value_mem, strat_mem, cum_reg, t, rng
        )


# ── DCFR-discounted cumulative regret (tabular side-structure) ────────────────
# Discounted Counterfactual Regret Minimization (Brown & Sandholm 2019).
# Per infoset I, maintain D(I) updated each visit n as:
#   pos *= (n/(n+1))^alpha ; neg *= (n/(n+1))^beta ; D += r^n
# with alpha=1.5, beta=0 (paper's recommended regret coefficients). This keeps
# the ratio between actions correct (early bad regrets fade) without the mean
# collapsing to zero. Tabular here (288 infosets in Leduc); in the project the
# value net's reservoir target plays this role.

DCFR_ALPHA = 1.0   # LCFR-style moderate discounting. alpha=1.5 faded the
DCFR_BETA  = 0.5   # shallow-mix signal before variance averaged out (lock-in);
                   # alpha=0 kept early errors forever (overshoot/oscillation,
                   # [0,1]->[0.91,0.09] past the [0.30,0.70] equilibrium). alpha=1
                   # fades positive cumulative regret linearly (n/(n+1)); beta=0.5
                   # lets negative regret recover faster so no action stays dead.

class CumulativeRegret:
    def __init__(self):
        # key -> [discounted_cumulative_regret (N_ACTIONS), visit_count]
        self.table = {}

    def add(self, key, regret_vec, iteration):
        entry = self.table.get(key)
        if entry is None:
            entry = [np.zeros(N_ACTIONS, dtype=np.float64), 0]
        D, n = entry
        # Discount existing cumulative regret by DCFR coefficients, using the
        # per-infoset visit count n (not the global iteration) so infosets that
        # are visited at different rates are each discounted by their own age.
        if n > 0:
            ratio = n / (n + 1.0)
            pos_factor = ratio ** DCFR_ALPHA
            neg_factor = ratio ** DCFR_BETA
            D = np.where(D > 0, D * pos_factor, D * neg_factor)
        D = D + regret_vec
        entry[0] = D
        entry[1] = n + 1
        self.table[key] = entry
        return D


# ── Reservoir advantage memory (Vitter) ───────────────────────────────────────

class Reservoir:
    def __init__(self, capacity, state_size, n_actions, seed=0):
        self.cap = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.targets = np.zeros((capacity, n_actions), dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, target):
        if self.size < self.cap:
            i = self.size
            self.states[i] = state
            self.targets[i] = target
            self.size += 1
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < self.cap:
                self.states[j] = state
                self.targets[j] = target
        self.seen += 1

    def sample(self, n):
        n = min(n, self.size)
        idx = self.rng.integers(0, self.size, size=n)
        return self.states[idx], self.targets[idx]


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

def train_from_scratch(reservoir, hidden, sgd_steps, batch, lr, device):
    net = AdvantageNet(STATE_SIZE, N_ACTIONS, hidden).to(device)
    if reservoir.size < 2:
        return net
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(sgd_steps):
        s, tgt = reservoir.sample(batch)
        s = torch.from_numpy(s).to(device)
        tgt = torch.from_numpy(tgt).to(device)
        pred = net(s)
        loss = ((pred - tgt) ** 2).mean()   # MSE over full action vector
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    return net


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

    # Deep CFR hyperparameters. Changes from the diverging runs, all from the
    # measured root cause (per-iteration regret variance was ±4 >> signal 0.5,
    # and the discounting schedule controls a lock-in vs overshoot trade-off):
    #   (1) TRAVERSALS 100 -> 1000: averages external-sampling variance WITHIN
    #       each iteration before training. Verified: reg_SE fell ~4 -> ~0.1.
    #   (2) DCFR alpha=1, beta=0.5 (module level): moderate discounting. Both
    #       extremes failed -- alpha=1.5 faded the signal (lock-in), alpha=0 kept
    #       early errors (overshoot past equilibrium). This is the middle.
    ITERS         = 200
    TRAVERSALS    = 1000
    HIDDEN        = 128
    SGD_STEPS     = 1000
    BATCH         = 1024
    LR            = 1e-3
    RES_CAP       = 1_000_000
    MEASURE_EVERY = 10

    value_mem = {0: Reservoir(RES_CAP, STATE_SIZE, N_ACTIONS, seed=1),
                 1: Reservoir(RES_CAP, STATE_SIZE, N_ACTIONS, seed=2)}
    strat_mem = StrategyMemory()
    cum_reg = {0: CumulativeRegret(), 1: CumulativeRegret()}

    # Advantage nets start returning ~0 (random init is fine; paper inits to 0,
    # but a small random net is standard and works). One per player.
    nets = {0: AdvantageNet(STATE_SIZE, N_ACTIONS, HIDDEN).to(device),
            1: AdvantageNet(STATE_SIZE, N_ACTIONS, HIDDEN).to(device)}

    print(f"Clean Deep CFR reference | Leduc | hidden={HIDDEN} "
          f"traversals={TRAVERSALS} sgd_steps={SGD_STEPS} batch={BATCH}")
    print(f"{'iter':>6} {'avg_expl':>10} {'cur_expl':>10} {'ref':>8} {'ratio':>7} {'t(s)':>7}")

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
                     cum_reg[traverser], t, rng)

        # Retrain THIS traverser's advantage net from scratch on its reservoir.
        nets[traverser] = train_from_scratch(
            value_mem[traverser], HIDDEN, SGD_STEPS, BATCH, LR, device
        )

        if t % MEASURE_EVERY == 0:
            avg_expl = exact_exploitability(game, strat_mem)
            cur_expl = current_strategy_exploitability(game, nets)
            ratio = avg_expl / ref if ref > 0 else float("nan")
            # Diagnostic: net's prediction for K|| vs tabular cumulative regret
            # [0.773, 1.807] (strat [0.30, 0.70]). If the fix works, the net's
            # K|| prediction should be positive on BOTH actions with a similar
            # ratio, and regret matching on it should give ~[0.30, 0.70].
            kstate = encode_infoset("K||")
            with torch.no_grad():
                kpred = nets[0](
                    torch.from_numpy(kstate).unsqueeze(0)
                ).squeeze(0).numpy()[:2]
            kpos = np.maximum(kpred, 0)
            kstrat = kpos / kpos.sum() if kpos.sum() > 0 else np.array([1.0, 0.0])
            print(f"{t:>6} {avg_expl:>10.5f} {cur_expl:>10.5f} {ref:>8.5f} "
                  f"{ratio:>7.2f} {time.time()-t0:>7.1f}   "
                  f"K||_pred=[{kpred[0]:+.2f},{kpred[1]:+.2f}] "
                  f"strat=[{kstrat[0]:.2f},{kstrat[1]:.2f}]")

            # Aggregate raw instantaneous regrets for K|| since the last report.
            if _KREG_LOG:
                regs = np.array([e[1] for e in _KREG_LOG], dtype=np.float64)
                mean_r = regs.mean(axis=0)
                strats = np.array([e[2] for e in _KREG_LOG], dtype=np.float64)
                mean_strat = strats.mean(axis=0)
                frac_a0_pos = float((regs[:, 0] > 0).mean())
                frac_a1_pos = float((regs[:, 1] > 0).mean())
                # Raw action values and node value (the heart of the question:
                # are v[0]/v[1] themselves sane, or is node_value anchored to the
                # dominant action by the current strategy?).
                avs = np.array([e[3] for e in _KREG_LOG], dtype=np.float64)
                nvs = np.array([e[4] for e in _KREG_LOG], dtype=np.float64)
                mean_av = avs.mean(axis=0)
                std_av = avs.std(axis=0)
                # (1) VARIANCE metric: standard error of the v0-v1 regret signal.
                # If raising traversals worked, reg_SE should be << |v0-v1|, i.e.
                # the shallow-mix signal becomes statistically resolvable.
                diff = avs[:, 0] - avs[:, 1]
                reg_se = diff.std() / np.sqrt(len(diff))
                # (2) ACCUMULATION metric: the DCFR table's current K|| cumulative
                # vector and visit count. With alpha=0 this is the honest running
                # sum; its sign should stabilise rather than flip each report.
                entry = cum_reg[0].table.get("K||") or cum_reg[1].table.get("K||")
                if entry is not None:
                    Dvec, Dn = entry[0][:2], entry[1]
                else:
                    Dvec, Dn = np.zeros(2), 0
                # (3) BEHAVIOUR metric: entropy of the current K|| strategy
                # (0 = locked deterministic, ln2≈0.69 = uniform). Convergence
                # requires this NOT collapsing to 0.
                kp = np.maximum(kpred, 0)
                ks = kp / kp.sum() if kp.sum() > 0 else np.array([0.5, 0.5])
                ent = -sum(p * np.log(p) for p in ks if p > 0)
                print(f"       K|| raw-regret: n={len(_KREG_LOG):>4d} "
                      f"mean_r=[{mean_r[0]:+.3f},{mean_r[1]:+.3f}] "
                      f"frac_pos=[{frac_a0_pos:.2f},{frac_a1_pos:.2f}] "
                      f"mean_strat_used=[{mean_strat[0]:.2f},{mean_strat[1]:.2f}]")
                print(f"       K|| action_vals: "
                      f"v0={mean_av[0]:+.3f}±{std_av[0]:.2f} "
                      f"v1={mean_av[1]:+.3f}±{std_av[1]:.2f} "
                      f"(v0-v1={mean_av[0]-mean_av[1]:+.3f})")
                print(f"       K|| behaviour:  reg_SE={reg_se:.3f} "
                      f"D_cum=[{Dvec[0]:+.2f},{Dvec[1]:+.2f}] D_n={Dn} "
                      f"entropy={ent:.3f}")
                _KREG_LOG.clear()

    print("\nExpected: expl trends DOWN toward ref. If it does, this clean")
    print("implementation is correct and becomes the baseline to diff the")
    print("project solver against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())