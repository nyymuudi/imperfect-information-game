#!/usr/bin/env python3
"""
Diagnose why exploitability and h2h disagree for two blueprints.

Computes for both blueprints over a shared sample of game trajectories:
    1. Strategy entropy per decision node (mean and distribution)
    2. TV distance between strategies (and per-state argmax disagreement rate)
    3. Per-preflop-class breakdown of paired-Δ payoff under CRN+EV
    4. Action distribution per representative scenario
    5. Best-response-style "where the regret comes from" per blueprint

The output is verbose by design — diagnostic, not a quick score.

Usage:
    python3 scripts/diagnose_expl_vs_h2h.py --a v3_path --b v11_path
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.blueprint import Blueprint
from src.abstraction.equity import canonical_hand_class
from src.analysis.head_to_head import play_hand


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    n = min(len(p), len(q))
    return float(0.5 * np.abs(p[:n] - q[:n]).sum())


def sample_decision_states(game, encoder, bp, n_games: int,
                            rng: np.random.Generator,
                            max_depth: int = 30):
    """Play n_games as self-play with bp; return list of (history, player, probs)."""
    samples = []
    for _ in range(n_games):
        deal = game.sample_deal(rng)
        history = deal
        depth = 0
        while not game.is_terminal(history) and depth < max_depth:
            player = game.current_player(history)
            actions = game.legal_actions(history)
            state = encoder.encode(history, player)
            probs = bp.query(state, len(actions))
            probs = np.asarray(probs, dtype=np.float64)
            s = probs.sum()
            if s <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs /= s
            samples.append((history, player, probs.copy()))
            idx = int(rng.choice(len(actions), p=probs))
            history = game.apply_action(history, actions[idx])
            depth += 1
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a",        required=True, help="Blueprint A path")
    p.add_argument("--b",        required=True, help="Blueprint B path")
    p.add_argument("--n-games",  type=int, default=500,
                   help="Number of self-play trajectories per blueprint.")
    p.add_argument("--n-hands",  type=int, default=2000,
                   help="Hands for per-preflop-class CRN+EV breakdown.")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    bp_a = Blueprint.load(args.a, device="cpu")
    bp_b = Blueprint.load(args.b, device="cpu")
    game = PostflopNLHE(
        starting_stack=bp_a.metadata.starting_stack,
        max_raises_per_street=bp_a.metadata.max_raises,
        raise_fractions=(bp_a.metadata.raise_fraction,),
    )
    encoder = NLHEEncoder(starting_stack=bp_a.metadata.starting_stack)

    print(f"A: {args.a}")
    print(f"B: {args.b}\n")

    # ── 1. Strategy entropy per blueprint ────────────────────────────────────
    print("=" * 60)
    print("1. Strategy entropy distribution (self-play trajectories)")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)
    samples_a = sample_decision_states(game, encoder, bp_a, args.n_games, rng)
    rng = np.random.default_rng(args.seed)
    samples_b = sample_decision_states(game, encoder, bp_b, args.n_games, rng)

    h_a = np.array([entropy(p) for _, _, p in samples_a])
    h_b = np.array([entropy(p) for _, _, p in samples_b])
    print(f"A entropy: mean={h_a.mean():.3f}  median={np.median(h_a):.3f}  "
          f"std={h_a.std():.3f}  n={len(h_a)}")
    print(f"B entropy: mean={h_b.mean():.3f}  median={np.median(h_b):.3f}  "
          f"std={h_b.std():.3f}  n={len(h_b)}")
    print(f"  → {'A more deterministic' if h_a.mean() < h_b.mean() else 'B more deterministic'}")
    print(f"  → Δ entropy = {h_b.mean() - h_a.mean():+.3f}  "
          f"(positive = B more random)")
    print()

    # ── 2. TV distance + per-state disagreement on SHARED states ─────────────
    print("=" * 60)
    print("2. Strategy distance on shared sampled states")
    print("=" * 60)

    rng = np.random.default_rng(args.seed + 17)
    shared = sample_decision_states(game, encoder, bp_a, args.n_games // 2, rng)

    tvs = []
    argmax_dis = 0
    for hist, player, _ in shared:
        actions = game.legal_actions(hist)
        state = encoder.encode(hist, player)
        pa = np.asarray(bp_a.query(state, len(actions)), dtype=np.float64)
        pb = np.asarray(bp_b.query(state, len(actions)), dtype=np.float64)
        pa /= max(pa.sum(), 1e-9)
        pb /= max(pb.sum(), 1e-9)
        tvs.append(tv_distance(pa, pb))
        if np.argmax(pa) != np.argmax(pb):
            argmax_dis += 1

    tvs = np.array(tvs)
    print(f"TV distance: mean={tvs.mean():.3f}  median={np.median(tvs):.3f}  "
          f"max={tvs.max():.3f}  n={len(tvs)}")
    print(f"argmax disagreement: {argmax_dis}/{len(tvs)} "
          f"({100*argmax_dis/len(tvs):.1f}%)")
    print()

    # ── 3. Per-preflop-class CRN+EV breakdown ────────────────────────────────
    print("=" * 60)
    print("3. Per-preflop-class paired-Δ (CRN+EV, A-Δ-B, hero seat alternated)")
    print("=" * 60)

    # Group hands by canonical hand class (e.g. AA, AKs, 72o, ...)
    # and compute average paired diff per class.
    by_class_diffs = defaultdict(list)

    master = np.random.default_rng(args.seed + 31)
    pair_seeds = master.integers(1, 2**63 - 1, size=args.n_hands, dtype=np.int64)

    for i in range(args.n_hands):
        pseed = int(pair_seeds[i])
        deal_rng = np.random.default_rng(pseed)
        deal = game.sample_deal(deal_rng)
        hero_pos = i % 2
        action_seed = pseed ^ 0xDEADBEEF

        # Identify hero's hand class (the player at hero_pos in this match)
        hero_cards = deal[hero_pos]
        cls = canonical_hand_class(hero_cards[0], hero_cards[1])

        rng1 = np.random.default_rng(action_seed)
        payoff_a = play_hand(bp_a, bp_b, game, encoder, encoder, deal,
                             hero=hero_pos, rng=rng1, ev_adjusted=True)
        rng2 = np.random.default_rng(action_seed)
        payoff_b = play_hand(bp_b, bp_a, game, encoder, encoder, deal,
                             hero=hero_pos, rng=rng2, ev_adjusted=True)
        diff = payoff_a - payoff_b
        by_class_diffs[cls].append(diff)

    bb = float(getattr(game, "bb", 2.0))
    rows = []
    for cls, diffs in by_class_diffs.items():
        d = np.array(diffs)
        mbb = (d.mean() / bb) * 1000.0
        rows.append((cls, mbb, len(diffs)))
    # Top 5 best classes for A, top 5 worst (= best for B)
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nA's BEST hand classes (largest paired Δ, A wins most):")
    for cls, mbb, n in rows[:7]:
        print(f"  {cls:>4s}: {mbb:+8.1f} mbb/hand  (n={n})")
    print(f"\nA's WORST hand classes (most negative Δ, B wins most):")
    for cls, mbb, n in rows[-7:][::-1]:
        print(f"  {cls:>4s}: {mbb:+8.1f} mbb/hand  (n={n})")
    print()

    # ── 4. Action distribution for representative scenarios ─────────────────
    print("=" * 60)
    print("4. Action distributions for representative spots")
    print("=" * 60)

    def show_scenario(label, hole, board, player, actions_taken):
        # Build a placeholder history with full board.
        opp_cards = (24, 1) if hole != (24, 1) else (48, 49)  # avoid clash
        used = set(hole) | set(opp_cards) | set(board)
        rest = [c for c in range(52) if c not in used]
        full_board = tuple(board) + tuple(rest[: 5 - len(board)])
        h = (hole, opp_cards, full_board) + actions_taken
        state = encoder.encode(h, player)
        actions = game.legal_actions(h)
        pa = np.asarray(bp_a.query(state, len(actions)), dtype=np.float64)
        pb = np.asarray(bp_b.query(state, len(actions)), dtype=np.float64)
        pa /= max(pa.sum(), 1e-9)
        pb /= max(pb.sum(), 1e-9)
        names = ["fold/k", "call", "raise", "all-in"][: len(actions)]
        print(f"\n  {label}")
        print(f"    A: " + "  ".join(f"{n}={pa[i]*100:.0f}%"
                                       for i, n in enumerate(names)))
        print(f"    B: " + "  ".join(f"{n}={pb[i]*100:.0f}%"
                                       for i, n in enumerate(names)))

    show_scenario("AA preflop, hero",   (48, 49), (),                  0, ())
    show_scenario("72o preflop, hero",  (24, 1),  (),                  0, ())
    show_scenario("KK flop top, hero",  (44, 45), (0, 5, 10, 15, 20), 0, ("k","c"))
    show_scenario("Q hi bluff, hero",   (11, 22), (0, 5, 10, 15, 20), 0, ("k","c"))
    show_scenario("AA vs raise, hero",  (48, 49), (),                  1, ("r",))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
