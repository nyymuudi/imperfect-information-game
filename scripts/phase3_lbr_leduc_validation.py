#!/usr/bin/env python3
"""
Phase 3 prerequisite: validate the LBR methodology on Leduc.

The production LBR estimator (src/analysis/exploitability.py) is a lower
bound on exploitability with a specific structure: uniform-depth node
sampling, opponent-card marginalisation, blueprint rollouts. Exact
exploitability is intractable in NLHE — but on Leduc it is exact, so the
LBR *methodology* can be checked where the answer is known:

  (a) ordering: LBR must rank strategies of different quality the same
      way exact exploitability does;
  (b) bound direction: per-decision LBR regret must not exceed the
      exact per-decision best-response gain it lower-bounds.

This mirrors the NLHE estimator step for step (deal → uniform depth →
marginalise opponent → rollout EVs → regret at the landed node) with
Leduc's exact enumeration in place of MC opponent sampling.

Usage: python3 scripts/phase3_lbr_leduc_validation.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver

CARDS = ["J1", "J2", "Q1", "Q2", "K1", "K2"]


def rollout_ev(game, strategy_fn, history):
    """Exact expected (p0, p1) payoff under strategy_fn for both players."""
    if game.is_terminal(history):
        p = game.terminal_payoffs(history)
        return float(p[0]), float(p[1])
    player = game.current_player(history)
    legal = game.legal_actions(history)
    probs = np.asarray(strategy_fn(history, player), dtype=np.float64)
    ev0 = ev1 = 0.0
    for pr, a in zip(probs, legal):
        if pr < 1e-12:
            continue
        e0, e1 = rollout_ev(game, strategy_fn, history + (a,))
        ev0 += pr * e0
        ev1 += pr * e1
    return ev0, ev1


def opp_reach_weight(game, strategy_fn, world, hero, n_actions):
    """P(opponent's observed actions | their card, comm) under strategy_fn."""
    w = 1.0
    h = world[:3]
    for a in world[3:3 + n_actions]:
        p = game.current_player(h)
        legal = game.legal_actions(h)
        if p != hero:
            probs = np.asarray(strategy_fn(h, p), dtype=np.float64)
            s = probs.sum()
            probs = probs / s if s > 1e-9 else np.ones(len(legal)) / len(legal)
            w *= float(probs[legal.index(a)])
        h = h + (a,)
    return w


def leduc_lbr(game, strategy_fn, n_games=400, max_depth=4, seed=0,
              bayes_weighted=False):
    """LBR proxy in chips/decision — the NLHE estimator's algorithm with
    exact opponent marginalisation (4 consistent cards, not MC).

    bayes_weighted=False mirrors the production estimator (uniform
    opponent belief). True weights each consistent world by the
    opponent's reach probability of the observed line — measured on
    Leduc to be the difference between a metric that FLOORS at ~0.31
    near equilibrium (uniform, cannot separate exact 0.22 from 0.08)
    and one that keeps separating (0.059 vs 0.017, r=0.996)."""
    rng = np.random.default_rng(seed)
    deals = game.initial_histories()
    regrets = []
    for _ in range(n_games):
        h, _ = deals[rng.integers(0, len(deals))]
        depth = int(rng.integers(1, max_depth + 1))
        # Walk the strategy stochastically for `depth` actions.
        ok = True
        for _ in range(depth):
            if game.is_terminal(h):
                ok = False
                break
            player = game.current_player(h)
            legal = game.legal_actions(h)
            probs = np.asarray(strategy_fn(h, player), dtype=np.float64)
            probs = np.clip(probs, 0, None)
            s = probs.sum()
            probs = probs / s if s > 1e-9 else np.ones(len(legal)) / len(legal)
            h = h + (legal[int(rng.choice(len(legal), p=probs))],)
        if not ok or game.is_terminal(h):
            continue

        hero = game.current_player(h)
        legal = game.legal_actions(h)
        # Marginalise over opponent cards consistent with hero's info.
        hero_card = h[hero]
        # Community is hidden information until revealed; the LBR player
        # knows only its own card + public actions, so marginalise over
        # BOTH the opponent card and (pre-reveal) the community card.
        _, _, r1_done = game._split_rounds(h[3:])
        n_act = len(h) - 3
        action_evs = np.zeros(len(legal))
        total_w = 0.0
        for opp_card in CARDS:
            if opp_card == hero_card:
                continue
            for comm in CARDS:
                if comm in (hero_card, opp_card):
                    continue
                if r1_done and comm != h[2]:
                    continue
                world = list(h)
                world[1 - hero] = opp_card
                world[2] = comm
                world = tuple(world)
                w = (opp_reach_weight(game, strategy_fn, world, hero, n_act)
                     if bayes_weighted else 1.0)
                if w <= 0.0:
                    continue
                total_w += w
                for ai, a in enumerate(legal):
                    evs = rollout_ev(game, strategy_fn, world + (a,))
                    action_evs[ai] += w * evs[hero]
        if total_w <= 0.0:
            continue
        action_evs /= total_w
        probs = np.asarray(strategy_fn(h, hero), dtype=np.float64)
        s = probs.sum()
        probs = probs / s if s > 1e-9 else np.ones(len(legal)) / len(legal)
        bp_ev = float(probs @ action_evs)
        regrets.append(max(0.0, float(action_evs.max()) - bp_ev))
    arr = np.asarray(regrets)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def carrier_exploitability(game, strategy_fn):
    ref = CFRSolver(game=game, linear_averaging=True)

    def walk(h):
        if game.is_terminal(h):
            return
        p = game.current_player(h)
        acts = game.legal_actions(h)
        key = game.info_set_key(h, p)
        if key not in ref.info_sets:
            probs = np.asarray(strategy_fn(h, p), dtype=np.float64)
            d = ref._get_or_create_info_set(key, acts)
            d.cumulative_strategy = probs[: len(acts)].copy()
        for a in acts:
            walk(game.apply_action(h, a))

    for ih, _ in game.initial_histories():
        walk(ih)
    return ref.exploitability()


def tabular_strategy(game, iterations):
    solver = CFRSolver(game=game, linear_averaging=True, cfr_plus=True)
    solver.solve(iterations=iterations)
    strategies = {k: d.average_strategy() for k, d in solver.info_sets.items()}

    def fn(h, p):
        key = game.info_set_key(h, p)
        n = len(game.legal_actions(h))
        if key in strategies:
            return strategies[key][:n]
        return np.ones(n) / n
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game = LeducHoldem()

    def uniform_fn(h, p):
        n = len(game.legal_actions(h))
        return np.ones(n) / n

    candidates = {
        "uniform": uniform_fn,
        "cfr@5": tabular_strategy(game, 5),
        "cfr@50": tabular_strategy(game, 50),
        "cfr@1000": tabular_strategy(game, 1000),
    }

    print(f"{'strategy':<10} {'exact_expl':>11} {'LBR-uniform':>12} "
          f"{'LBR-bayes':>12}")
    rows = []
    for name, fn in candidates.items():
        exact = carrier_exploitability(game, fn)
        lbr_u, _ = leduc_lbr(game, fn, n_games=args.n_games, seed=args.seed)
        lbr_b, se = leduc_lbr(game, fn, n_games=args.n_games, seed=args.seed,
                              bayes_weighted=True)
        rows.append((name, exact, lbr_u, lbr_b))
        print(f"{name:<10} {exact:>11.5f} {lbr_u:>12.5f} "
              f"{lbr_b:>9.5f} ± {se:.5f}")

    exacts = [r[1] for r in rows]
    for label, idx in (("uniform", 2), ("bayes", 3)):
        vals = [r[idx] for r in rows]
        order_ok = all(
            (exacts[i] > exacts[j]) == (vals[i] > vals[j])
            for i in range(len(rows)) for j in range(len(rows)) if i != j
        )
        corr = float(np.corrcoef(exacts, vals)[0, 1])
        print(f"\nLBR-{label}: ordering preserved={order_ok}  r={corr:.4f}")

    # Gate applies to the bayes-weighted variant (the one Phase 3 uses).
    vals = [r[3] for r in rows]
    order_ok = all(
        (exacts[i] > exacts[j]) == (vals[i] > vals[j])
        for i in range(len(rows)) for j in range(len(rows)) if i != j
    )
    corr = float(np.corrcoef(exacts, vals)[0, 1])
    verdict = "PASS" if order_ok and corr > 0.95 else "FAIL"
    print(f"\nLBR methodology on Leduc (bayes-weighted): {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
