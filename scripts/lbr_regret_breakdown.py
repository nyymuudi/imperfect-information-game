#!/usr/bin/env python3
"""LBR regret breakdown by player / street / action category.

Diagnoses WHERE the exploitability comes from in a blueprint, so we know
what to target for the next training run. Produces three views:

  1. Mean regret per (player, street) — surfaces positional asymmetry and
     the street at which a blueprint leaks most.
  2. Mean regret per (player, best_action_category) — what *should* the
     blueprint be doing more of? Categorised as fold/check/call/raise/allin.
  3. Top-N leakiest individual decision nodes with full info-set context.

Usage:
    python3 scripts/lbr_regret_breakdown.py blueprints/50bb_v11_optuna_v2 \
        --n-games 600 --n-opp-samples 8 --max-depth 8
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.abstraction.equity import card_to_str, canonical_hand_class
from src.analysis.exploitability import (
    _lbr_action_evs,
    _live_cards,
    _sample_deal,
    _substitute_opp_cards,
    _walk_to_depth,
)
from src.analysis.head_to_head import ev_adjusted_payoffs
from src.deep_cfr.blueprint import Blueprint
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE


ACTION_NAMES = ["fold_check", "call", "raise", "allin"]
STREET_NAMES = ["preflop", "flop", "turn", "river"]


def _street_idx(game, history) -> int:
    st = game._parse_state(history)
    return int(st["street_idx"])


def collect_regret_samples(bp, game, encoder, n_games: int,
                           n_opp_samples: int, max_depth: int,
                           seed: int = 42) -> list[dict]:
    """Sample n_games LBR decision nodes and return per-node regret records."""
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    bb_chips = float(getattr(game, "bb", 2.0))

    for i in range(n_games):
        deal = _sample_deal(rng)
        target = int(rng.integers(1, max_depth + 1))
        node = _walk_to_depth(bp, encoder, game, deal, target, rng)
        if node is None or game.is_terminal(node):
            continue

        hero = game.current_player(node)
        opp  = 1 - hero
        actions = game.legal_actions(node)
        n = len(actions)

        action_evs = _lbr_action_evs(
            bp, encoder, game, node, hero, opp, rng,
            n_opp_samples=n_opp_samples,
            payoff_fn=ev_adjusted_payoffs,
        )
        state = encoder.encode(node, hero)
        strategy = np.asarray(bp.query(state, n), dtype=np.float64)
        s = strategy.sum()
        strategy = strategy / s if s > 1e-9 else np.ones(n) / n

        bp_ev = float(strategy @ action_evs)
        best_idx = int(np.argmax(action_evs))
        best_ev = float(action_evs[best_idx])
        regret_chips = max(0.0, best_ev - bp_ev)

        hero_cards = node[hero]
        hand_class = canonical_hand_class(hero_cards[0], hero_cards[1])
        board = node[2]
        st_idx = _street_idx(game, node)
        n_visible = [0, 3, 4, 5][min(st_idx, 3)]
        board_str = " ".join(card_to_str(c) for c in board[:n_visible])

        samples.append({
            "player":         hero,
            "street_idx":     st_idx,
            "n_actions":      n,
            "best_action":    actions[best_idx],
            "regret_chips":   regret_chips,
            "regret_mbb":     (regret_chips / bb_chips) * 1000.0,
            "bp_ev_chips":    bp_ev,
            "best_ev_chips":  best_ev,
            "strategy":       strategy.tolist(),
            "action_evs":     action_evs.tolist(),
            "hand_class":     hand_class,
            "board_str":      board_str,
            "depth":          target,
            "actions":        list(actions),
        })

        if (i + 1) % max(1, n_games // 10) == 0:
            print(f"  [{i+1:>4d}/{n_games}]  collected={len(samples)}")

    return samples


def _categorise_action(action, n_actions: int, is_facing_bet: bool) -> str:
    """Map action index/symbol to a category that's stable across contexts."""
    # PostflopNLHE actions are positional: [0]=check/fold, [1]=call (only when
    # facing bet), [2]=raise, [3]=allin. Strings 'c','k','r','a','f' also seen.
    if isinstance(action, str):
        if action in ("f",): return "fold_check"
        if action in ("c", "k"): return "fold_check" if action == "k" else "call"
        if action == "r": return "raise"
        if action == "a": return "allin"
    # Fall back to index → name table.
    return ACTION_NAMES[min(int(action), len(ACTION_NAMES) - 1)] \
        if isinstance(action, int) else "unknown"


def report(samples: list[dict]) -> None:
    if not samples:
        print("No samples collected.")
        return
    print(f"\n{'='*70}\nCollected {len(samples)} LBR decision nodes\n{'='*70}")

    by_player_street = defaultdict(list)
    by_player_action = defaultdict(list)
    for s in samples:
        by_player_street[(s["player"], s["street_idx"])].append(s["regret_mbb"])
        cat = _categorise_action(s["best_action"], s["n_actions"],
                                 is_facing_bet=False)
        by_player_action[(s["player"], cat)].append(s["regret_mbb"])

    print("\n— Mean regret per (player × street), mbb/dec —")
    print(f"  {'street':<10} {'P0 (SB)':>14} {'P1 (BB)':>14}")
    for st_idx, st_name in enumerate(STREET_NAMES):
        p0 = by_player_street.get((0, st_idx), [])
        p1 = by_player_street.get((1, st_idx), [])
        p0_str = f"{np.mean(p0):7.1f} (n={len(p0):>3})" if p0 else "         —    "
        p1_str = f"{np.mean(p1):7.1f} (n={len(p1):>3})" if p1 else "         —    "
        print(f"  {st_name:<10} {p0_str:>14} {p1_str:>14}")

    print("\n— Mean regret per (player × best_action), mbb/dec —")
    print(f"  {'best_action':<12} {'P0 (SB)':>14} {'P1 (BB)':>14}")
    cats = sorted({c for (_, c) in by_player_action.keys()})
    for cat in cats:
        p0 = by_player_action.get((0, cat), [])
        p1 = by_player_action.get((1, cat), [])
        p0_str = f"{np.mean(p0):7.1f} (n={len(p0):>3})" if p0 else "         —    "
        p1_str = f"{np.mean(p1):7.1f} (n={len(p1):>3})" if p1 else "         —    "
        print(f"  {cat:<12} {p0_str:>14} {p1_str:>14}")

    print("\n— Top-10 leakiest individual decisions —")
    ranked = sorted(samples, key=lambda r: r["regret_mbb"], reverse=True)
    for i, s in enumerate(ranked[:10]):
        strat_str = " ".join(f"{p*100:>3.0f}%" for p in s["strategy"])
        evs_str = " ".join(f"{e:+5.2f}" for e in s["action_evs"])
        seat = "SB" if s["player"] == 0 else "BB"
        print(f"  #{i+1:>2}  regret={s['regret_mbb']:6.0f} mbb  "
              f"{seat} {STREET_NAMES[s['street_idx']]:<7} "
              f"{s['hand_class']:>4}  board=[{s['board_str']:<14}]")
        print(f"       strategy = [{strat_str}]  ev = [{evs_str}]  "
              f"→ best={s['best_action']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("blueprint", help="Path to blueprint directory")
    p.add_argument("--n-games",       type=int, default=600)
    p.add_argument("--n-opp-samples", type=int, default=8)
    p.add_argument("--max-depth",     type=int, default=8)
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    bp = Blueprint.load(args.blueprint, device="cpu")
    game = PostflopNLHE(starting_stack=bp.metadata.starting_stack,
                        max_raises_per_street=bp.metadata.max_raises,
                        raise_fractions=(bp.metadata.raise_fraction,))
    enc = NLHEEncoder(starting_stack=bp.metadata.starting_stack)

    print(f"Blueprint: {args.blueprint}")
    print(f"Sampling {args.n_games} games × {args.n_opp_samples} opp draws, "
          f"max_depth={args.max_depth} ...")
    samples = collect_regret_samples(
        bp, game, enc,
        n_games=args.n_games, n_opp_samples=args.n_opp_samples,
        max_depth=args.max_depth, seed=args.seed,
    )
    report(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
