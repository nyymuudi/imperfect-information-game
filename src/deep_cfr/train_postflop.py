#!/usr/bin/env python3
"""
Train Deep CFR on Postflop Heads-Up NLHE.

Usage:
    # Quick test (~150s)
    python3 -m src.deep_cfr.train_postflop \\
        --iterations 100 --traversals 500 --hidden 256

    # Full training run with blueprint save
    python3 -m src.deep_cfr.train_postflop \\
        --iterations 1000 --traversals 500 --hidden 256 \\
        --buffer 10000 --epochs 20 \\
        --save-blueprint blueprints/200bb_75pot_1000iter

    # Resume evaluation from saved blueprint
    python3 -m src.deep_cfr.train_postflop \\
        --load-blueprint blueprints/200bb_75pot_1000iter \\
        --eval-only

Notes on buffer sizing:
    The regret (value) buffer is a RESERVOIR over many iterations, following
    Deep CFR (Brown et al. 2019) and Single Deep CFR (Steinberger 2019). The
    value network is re-fitted each iteration on samples drawn from ALL past
    iterations; that is how it comes to approximate CUMULATIVE counterfactual
    regret without explicit summation. It must therefore be LARGE — default
    1_000_000. A small FIFO window (e.g. ~10×traversals) is WRONG: the network
    would fit only the latest iteration's instantaneous regrets, which is not
    CFR and does not converge (verified empirically on Leduc). The strategy
    buffer is likewise a large reservoir (time-average strategy).
"""

import argparse
import time
import torch
import numpy as np

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.cpp_backend import export_for_libtorch
from src.deep_cfr.blueprint import Blueprint
from src.analysis.exploitability import estimate_exploitability


# ── Strategy evaluation helpers ───────────────────────────────────────────────

ACTION_LABELS = {
    "preflop": ["fold/check", "call", "raise", "all-in"],
    "postflop": ["check/fold", "call", "raise", "all-in"],
}

SAMPLE_HANDS = [
    # (description, hole_cards_p0, board, player, label)
    ("AA preflop",   (48, 49), (),                     0, "value"),
    ("72o preflop",  (24, 1),  (),                     0, "trash"),
    ("KK flop top",  (44, 45), (0, 5, 10, 15, 20),    0, "value"),
    ("Q hi bluff",   (11, 22), (0, 5, 10, 15, 20),    0, "bluff"),
]


def evaluate_blueprint(bp: Blueprint, encoder: NLHEEncoder) -> None:
    """Print strategy snapshots for representative hands."""
    print("\n" + "=" * 62)
    print("BLUEPRINT STRATEGY SNAPSHOTS")
    print("=" * 62)

    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=(bp.metadata.raise_fraction,),
    )
    rng = np.random.default_rng(0)

    for desc, hole, board_cards, player, _ in SAMPLE_HANDS:
        # Build a minimal deal tuple
        remaining = [c for c in range(52) if c not in hole and c not in board_cards]
        rng.shuffle(remaining)
        opp_cards = tuple(remaining[:2])
        full_board = tuple(board_cards) + tuple(remaining[2:2 + (5 - len(board_cards))])
        deal = (hole, opp_cards, full_board)

        state_vec = encoder.encode(deal, player)
        num_actions = len(game.legal_actions(deal))
        probs = bp.query(state_vec, num_actions)

        labels = ACTION_LABELS["preflop" if not board_cards else "postflop"]
        parts = "  ".join(
            f"{labels[i]}={p:.0%}" for i, p in enumerate(probs)
        )
        print(f"  {desc:<20s}: {parts}")

    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep CFR on Postflop NLHE")

    # Training
    p.add_argument("--iterations",      "-n", type=int,   default=50)
    p.add_argument("--traversals",      "-t", type=int,   default=200)
    p.add_argument("--hidden",               type=int,   default=128)
    p.add_argument("--buffer",               type=int,   default=0,
                   help="Regret reservoir capacity. 0 = auto (1_000_000). "
                        "Must be large: it holds value samples across many "
                        "iterations. See module docstring.")
    p.add_argument("--strategy-buffer",      type=int,   default=0,
                   help="Strategy buffer capacity (0 = 1_000_000). Should be large.")
    p.add_argument("--epochs",               type=int,   default=20)

    # Game config
    p.add_argument("--stack",                type=float, default=200.0)
    p.add_argument("--raise-fraction",       type=float, default=0.75,
                   help="Raise size as fraction of pot (default: 0.75)")

    # Convergence measurement
    p.add_argument("--expl-games",           type=int,   default=50,
                   help="Games per exploitability estimate in callback "
                        "(0 disables mid-training measurement).")

    # Blueprint I/O
    p.add_argument("--save-blueprint",       type=str,   default=None,
                   metavar="PATH",
                   help="Save trained blueprint to this directory")
    p.add_argument("--load-blueprint",       type=str,   default=None,
                   metavar="PATH",
                   help="Load existing blueprint (skips training)")
    p.add_argument("--resume-from",          type=str,   default=None,
                   metavar="CHECKPOINT_PATH",
                   help="Resume training from a saved checkpoint blueprint")
    p.add_argument("--eval-only",            action="store_true",
                   help="Only run strategy evaluation (requires --load-blueprint)")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    encoder = NLHEEncoder(starting_stack=args.stack)

    # ── Load-only path ────────────────────────────────────────────────────────
    if args.load_blueprint:
        bp = Blueprint.load(args.load_blueprint, device=device)
        evaluate_blueprint(bp, encoder)
        return 0

    if args.eval_only and not args.load_blueprint:
        print("Error: --eval-only requires --load-blueprint")
        return 1

    # ── Resume-from checkpoint ────────────────────────────────────────────────
    resume_iter = 0
    resume_strategy_state = None
    if args.resume_from:
        try:
            resume_bp   = Blueprint.load(args.resume_from, device=device)
            resume_iter = resume_bp.metadata.iterations
            resume_strategy_state = resume_bp._net.state_dict()
            print(f"Resuming from checkpoint: iter={resume_iter}")
            remaining = args.iterations - resume_iter
            if remaining <= 0:
                print("Checkpoint is already at target iterations — nothing to do.")
                evaluate_blueprint(resume_bp, NLHEEncoder(starting_stack=args.stack))
                return 0
        except Exception as e:
            print(f"[warn] Could not load checkpoint: {e}. Starting fresh.")

    # ── Derive buffer sizes ───────────────────────────────────────────────────
    # Regret (value) buffer is a RESERVOIR over many iterations (Deep CFR /
    # SD-CFR). It must be LARGE so the network fits cumulative regret across
    # iterations — NOT a small ~10×traversals window (that only captures the
    # latest iteration's instantaneous regrets and does not converge).
    regret_buf = args.buffer if args.buffer > 0 else 1_000_000
    strat_buf  = args.strategy_buffer if args.strategy_buffer > 0 else 1_000_000

    if args.buffer > 0 and args.buffer < 50_000:
        print(
            f"[warn] --buffer {args.buffer:,} is small for a reservoir. The "
            f"regret network approximates CUMULATIVE regret via a reservoir "
            f"over many iterations; too small a capacity reintroduces the "
            f"window pathology (fits only recent iterations). Prefer >= 1e6."
        )

    # ── Training path ─────────────────────────────────────────────────────────
    game = PostflopNLHE(
        starting_stack=args.stack,
        max_raises_per_street=2,
        raise_fractions=(args.raise_fraction,),
    )

    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,
        buffer_capacity=regret_buf,
        strategy_buffer_capacity=strat_buf,
        hidden_size=args.hidden,
        train_epochs=args.epochs,
        train_batch=256,
        traversals_per_iter=args.traversals,
        use_cpp_engine=True,
        device=device,
        lr=1e-3,
    )

    print(f"Deep CFR — HU Postflop NLHE")
    print(f"  Stack: {args.stack:.0f}BB | "
          f"Raise: {args.raise_fraction:.0%} pot | "
          f"Iterations: {args.iterations}")
    print(f"  Hidden: {args.hidden} | "
          f"Regret buf: {regret_buf:,} | "
          f"Strategy buf: {strat_buf:,}")
    print(f"  Device: {device}\n")

    t0 = time.time()

    def callback(s, i):
        elapsed = time.time() - t0
        reg_loss = getattr(s, "_last_regret_loss", 0.0)

        # Exploitability of the CURRENT regret-matching strategy.
        # This is the strategy CFR actually iterates and it updates every
        # iteration — unlike strategy_net, which is trained only at the end.
        # Measuring it here gives a true convergence signal mid-training.
        expl_str = "  expl=  n/a "
        if args.expl_games > 0:
            try:
                cur = s.current_strategy_blueprint()
                expl = estimate_exploitability(
                    cur, game, encoder, n_games=args.expl_games, seed=0
                )
                expl_str = f"  expl={expl:6.1f} mbb/h"
            except Exception as e:
                expl_str = f"  expl=ERR ({type(e).__name__})"

        print(
            f"  iter {i:>4d}: "
            f"regret_buf={len(s.regret_buffer):>7,} | "
            f"strat_buf={len(s.strategy_buffer):>7,} | "
            f"loss={reg_loss:.4f}"
            f"{expl_str} | "
            f"t={elapsed:.1f}s"
        )
        # Checkpoint every 1000 iterations
        if args.save_blueprint and i % 1000 == 0:
            ckpt_path = args.save_blueprint + f"_ckpt{i}"
            Blueprint.from_solver(s, device="cpu").save(ckpt_path)
            print(f"  [checkpoint saved → {ckpt_path}]")

    # Jos resumataan, lataa strategy_net-painot ja aseta iteraatiolaskuri
    if resume_strategy_state is not None:
        solver.strategy_net.load_state_dict(resume_strategy_state)
        solver.iterations = resume_iter

    remaining_iters = args.iterations - resume_iter
    solver.solve(
        iterations=remaining_iters,
        callback=callback,
        callback_freq=max(1, args.iterations // 10),
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ── Export TorchScript for C++ ────────────────────────────────────────────
    try:
        export_for_libtorch(solver.regret_net)
        print("Regret network exported for LibTorch.")
    except Exception as e:
        print(f"[warn] LibTorch export failed: {e}")

    # ── Blueprint save ────────────────────────────────────────────────────────
    bp = Blueprint.from_solver(solver, device=device)

    if args.save_blueprint:
        bp.save(args.save_blueprint)
    else:
        print("\n[tip] Pass --save-blueprint PATH to persist this run.")

    # ── Final convergence check on the TRAINED strategy network ───────────────
    # The blueprint wraps strategy_net (now trained). This is the deployable
    # artefact; its exploitability is the number that matters.
    try:
        final_expl = estimate_exploitability(
            bp, game, encoder, n_games=max(args.expl_games, 200), seed=0
        )
        print(f"\nFinal blueprint exploitability: {final_expl:.1f} mbb/hand "
              f"(untrained ≈ 200–500; target < 100)")
    except Exception as e:
        print(f"[warn] Final exploitability measurement failed: {e}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluate_blueprint(bp, encoder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())