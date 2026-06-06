#!/usr/bin/env python3
"""
Train Deep CFR on Postflop Heads-Up NLHE.

Usage:
    # Quick test (~60s, 50BB)
    python3 -m src.deep_cfr.train_postflop \\
        --iterations 100 --traversals 1000 --hidden 256

    # Full training run with blueprint save (50BB, 1 raise/street)
    python3 -m src.deep_cfr.train_postflop \\
        --iterations 500 --traversals 1000 --hidden 256 \\
        --buffer 1000000 --epochs 50 \\
        --save-blueprint blueprints/50bb_75pot_500iter

    # 200BB (vaatii huomattavasti enemmän traversaaleja — ei suositella ilman
    # card abstraktiota, puu on liian suuri Deep CFR:lle tällä budjetilla)
    python3 -m src.deep_cfr.train_postflop \\
        --stack 200 --max-raises 2 \\
        --iterations 1000 --traversals 5000 --hidden 512 \\
        --buffer 5000000 --epochs 50 \\
        --save-blueprint blueprints/200bb_75pot_1000iter

    # Resume evaluation from saved blueprint
    python3 -m src.deep_cfr.train_postflop \\
        --load-blueprint blueprints/50bb_75pot_500iter \\
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

Exploitability units:
    estimate_exploitability returns a PER-DECISION proxy in mbb/decision (milli-
    big-blinds per decision node), NOT per hand. The callback and final report
    below print mbb/decision accordingly.
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

    p.add_argument("--iterations",      "-n", type=int,   default=50)
    p.add_argument("--traversals",      "-t", type=int,   default=200)
    p.add_argument("--hidden",               type=int,   default=128)
    p.add_argument("--buffer",               type=int,   default=0,
                   help="Regret reservoir capacity. 0 = auto (1_000_000). "
                        "Must be large: it holds value samples across many "
                        "iterations. See module docstring.")
    p.add_argument("--strategy-buffer",      type=int,   default=0,
                   help="Strategy buffer capacity (0 = 1_000_000). Should be large.")
    p.add_argument("--epochs",               type=int,   default=50)
    p.add_argument("--no-warm-start",        action="store_true",
                   help="Alusta regret-verkko nollista joka iteraatiolla "
                        "(Brown et al. 2019 cold-start). Oletus: warm-start.")
    p.add_argument("--dcfr-gamma",           type=float, default=2.0,
                   help="DCFR temporaalipainotuksen eksponentti γ. "
                        "0 = uniform (vanilla Deep CFR), 2 = DCFR (oletus). "
                        "Näytteet painotetaan t^γ näytteistysvaiheessa.")
    p.add_argument("--dcfr-alpha",           type=float, default=1.5,
                   help="DCFR regret-diskontaus α (oletus 1.5). "
                        "Näytepaino = t^α/(t^α+1). 0 = ei diskontausta.")
    p.add_argument("--regret-target",        type=str,   default="instant",
                   choices=["instant", "cfrplus"],
                   help="Regret-kohde C++-traversaalille. "
                        "'instant' = hetkellinen regret per solmu (Brown et al. "
                        "2019 Algorithm 1, oikea valinta jatkuvalle tilavektorille). "
                        "'cfrplus' = CFR+/visits (toimii vain diskreetin infoset-avaimen kanssa).")

    p.add_argument("--stack",                type=float, default=50.0,
                   help="Effective stack in BB (default: 50). 200BB puu on "
                        "liian suuri Deep CFR:lle ilman vahvaa abstraktiota — "
                        "käytä 50BB tai 100BB tuotantoajossa.")
    p.add_argument("--max-raises",          type=int,   default=1,
                   help="Max raises per street (default: 1). 2 kasvattaa "
                        "puun koon ~4x per street.")
    p.add_argument("--raise-fraction",       type=float, default=0.75,
                   help="Raise size as fraction of pot (default: 0.75)")

    p.add_argument("--expl-games",           type=int,   default=50,
                   help="Games per exploitability estimate in callback "
                        "(0 disables mid-training measurement).")

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
        max_raises_per_street=args.max_raises,
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
        warm_start=not args.no_warm_start,
        dcfr_gamma=args.dcfr_gamma,
        dcfr_alpha=args.dcfr_alpha,
        regret_target=args.regret_target,
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

        # Exploitability of the CURRENT regret-matching strategy (updates every
        # iteration, unlike strategy_net which is trained only at the end).
        # Reported in mbb/decision — the unit estimate_exploitability returns.
        expl_str = "  expl=  n/a "
        if args.expl_games > 0:
            try:
                cur = s.current_strategy_blueprint()
                expl = estimate_exploitability(
                    cur, game, encoder, n_games=args.expl_games, seed=0
                )
                expl_str = f"  expl={expl:6.1f} mbb/decision"
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
        if args.save_blueprint and i % 1000 == 0:
            ckpt_path = args.save_blueprint + f"_ckpt{i}"
            Blueprint.from_solver(s, device="cpu").save(ckpt_path)
            print(f"  [checkpoint saved → {ckpt_path}]")

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
    try:
        final_expl = estimate_exploitability(
            bp, game, encoder, n_games=max(args.expl_games, 200), seed=0
        )
        print(f"\nFinal blueprint exploitability: {final_expl:.1f} mbb/decision "
              f"(untrained ≈ order 100s; lower is better)")
    except Exception as e:
        print(f"[warn] Final exploitability measurement failed: {e}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluate_blueprint(bp, encoder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())