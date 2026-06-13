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

Validated 50BB / 1-raise baseline (2026-06-08, blueprint v2_subcat):
    --iterations 500 --traversals 1000 --hidden 256 --buffer 1000000 --epochs 50
    (and let --lr-decay-factor / --finetune-epochs at their defaults).
    Final exploitability: ~3397 mbb/decision — beats the previous monotone-proxy
    baseline (~3585) by ~190. State encoder uses subcategory board bucketing
    (cat + top-rank / pair-rank sub-bin) — see state_encoder.py:_board_bucket.

    Ablations that REGRESSED the baseline (do not turn on without re-validating):
      * --lr-decay-factor 0.97 from iter 100  → +850 mbb/decision (strategy
        collapses to ~uniform; decay outpaces buffer warm-up).
      * --finetune-epochs 100 on top of the above → neutral on top of the LR
        regression, gave no signal alone — kept available but off by default.
"""

import argparse
import sys
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

    p.add_argument("--seed",                 type=int,   default=42,
                   help="RNG seed for C++ MCCFR (default 42). Eri seedit "
                        "antavat eri näytteistyspolut → eri blueprint "
                        "samoilla hyperparametreilla. Käytetään multi-run "
                        "baseline -ajamisessa.")
    p.add_argument("--warm-start-lr-factor", type=float, default=5.0,
                   help="Warm-start training käyttää lr / warm_start_lr_factor "
                        "(default 5.0). Iso arvo = pienempi LR warm-startissa.")
    p.add_argument("--lr",                   type=float, default=1e-3,
                   help="Regret-network Adam learning rate (default 1e-3). "
                        "Optuna trial 17 paras: 1.756e-3.")
    p.add_argument("--train-batch",          type=int,   default=256,
                   help="Mini-batch koko regret- ja strategy-koulutuksessa "
                        "(default 256).")
    p.add_argument("--lr-decay-start",       type=int,   default=100,
                   help="Iteraatio jonka jälkeen regret-LR aletaan decay:taa "
                        "lr_decay_factor:lla per iteraatio (default: 100).")
    p.add_argument("--lr-decay-factor",      type=float, default=1.0,
                   help="LR-decay per iteraatio lr_decay_start:n jälkeen. "
                        "1.0 = EI decay:tä (default, validoitu). 0.97 regressoi "
                        "+850 mbb/dec — älä laita päälle ilman tuoretta "
                        "validointia. Mahdollinen kokeilu: 0.995, start 200.")
    p.add_argument("--finetune-epochs",      type=int,   default=0,
                   help="Strategy-verkon fine-tune-epochit pääajon jälkeen "
                        "pienemmällä LR:llä (default: 0 = pois). Ablation "
                        "B vs alkuperäinen baseline: ei signaalia.")
    p.add_argument("--finetune-lr",          type=float, default=1e-4,
                   help="Fine-tune-vaiheen LR (default: 1e-4).")

    p.add_argument("--expl-games",           type=int,   default=500,
                   help="Games per exploitability estimate in callback "
                        "(0 disables mid-training measurement). Default 500: "
                        "50 produced ~20%% stddev/mean sampling noise which "
                        "masked the actual convergence trajectory in earlier "
                        "ablations. 500 brings relative noise to ~7%%.")
    p.add_argument("--bucket-scheme",        type=str,   default="flat",
                   choices=["flat", "tree", "super", "tree42"],
                   help="Board-bucket encoding scheme. 'flat'=v3 K=8 one-hot "
                        "(default, production, todennettu). Muut: 'tree'=4×4, "
                        "'super'=K=4, 'tree42'=4×2 — kaikki kokeellisia jotka "
                        "eivät vakaasti voita v3:a. tree42-tuotantokoulutus "
                        "(1500 iter, Optuna trial #4 -params) collapsoi "
                        "strategian uniformiin → revertattu defaultiin flat.")
    p.add_argument("--ev-adjusted-expl",     action="store_true",
                   help="Käytä EV-adjusted payoffeja mid-training "
                        "exploitability-estimaatissa. Korvaa all-in-runoutien "
                        "realisoidun lopputuloksen equity:llä → ~2-5x vähemmän "
                        "kohinaa per arvio (eli sama signaali ~4-25x harvemmalla "
                        "callback game count:lla).")
    p.add_argument("--ev-adjusted-mccfr",    action="store_true",
                   help="Aja C++ MCCFR EV-adjusted terminaaleilla: all-in-"
                        "showdownin payoff = equity yli jäljellä olevien board-"
                        "korttien, ei realisoitunutta runoutia. Tiukempi regret-"
                        "kohde → nopeampi konvergenssi. Lisää overheadia "
                        "(equity-cache:llä mitigoitavissa).")
    p.add_argument("--diagnostics",          action="store_true",
                   help="Print buffer diagnostics every callback: unique-state "
                        "fraction, iter-histogram of samples, target |mag|, "
                        "and 3-window expl stddev. Use for debugging "
                        "convergence stagnation/oscillation.")

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

    p.add_argument("--mlflow",               action="store_true",
                   help="Log run to MLflow (./mlruns file backend). Params: "
                        "all CLI args. Metrics: per-callback loss/expl/buffer "
                        "sizes/elapsed; final exploitability. Artifacts: log + "
                        "blueprint manifest. View with: mlflow ui --backend-store-uri "
                        "file://./mlruns")
    p.add_argument("--mlflow-run-name",      type=str, default=None,
                   help="Custom MLflow run name (default: derived from --save-blueprint "
                        "or 'unnamed').")
    p.add_argument("--mlflow-experiment",    type=str, default="deep-cfr-nlhe",
                   help="MLflow experiment name (default: deep-cfr-nlhe).")

    return p


def _setup_mlflow(args):
    """Start an MLflow run + log all params. Returns mlflow module or None.
    Quiet failure if MLflow is missing — never blocks training."""
    if not args.mlflow:
        return None
    try:
        import mlflow as _ml
    except ImportError:
        print("[warn] --mlflow set but mlflow not installed. Skipping.")
        return None

    # SQLite backend (MLflow 3.x deprecated the file store). Artifacts still
    # land in ./mlruns. Single sqlite file per repo → easy to commit/share if
    # desired; no daemon required.
    from pathlib import Path
    tracking_db = (Path.cwd() / "mlflow.db").resolve()
    artifacts   = (Path.cwd() / "mlruns").resolve()
    _ml.set_tracking_uri(f"sqlite:///{tracking_db}")
    # Create/select experiment with artifact location pointed at ./mlruns.
    try:
        exp = _ml.get_experiment_by_name(args.mlflow_experiment)
        if exp is None:
            _ml.create_experiment(args.mlflow_experiment,
                                   artifact_location=f"file://{artifacts}")
        _ml.set_experiment(args.mlflow_experiment)
    except Exception as e:
        print(f"[warn] MLflow experiment setup failed: {e}. Disabling MLflow.")
        return None

    # Derive run name from --save-blueprint basename if not given.
    name = args.mlflow_run_name
    if name is None and args.save_blueprint:
        name = args.save_blueprint.split("/")[-1]
    if name is None:
        name = "unnamed"

    run = _ml.start_run(run_name=name)
    print(f"MLflow run: {name}  (id={run.info.run_id})")
    print(f"  view: mlflow ui --backend-store-uri sqlite:///{tracking_db}")

    # Log every CLI arg as a param. Skip None / private fields.
    for k, v in sorted(vars(args).items()):
        if v is None or k.startswith("_"):
            continue
        # Coerce to MLflow-compatible types (strings for paths, ints/floats for nums).
        _ml.log_param(k, v if isinstance(v, (int, float, bool)) else str(v))
    return _ml


def main() -> int:
    # Force line-buffered stdout so `tee` / pipe redirects show progress live,
    # not just at the very end. Without this Python uses block-buffered stdout
    # when output isn't a tty, and a 76-min training run produces zero visible
    # output until completion.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    args = build_arg_parser().parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Tree scheme: switch C++ encoder BEFORE constructing solver so MCCFR
    # samples are produced with tree layout. Default 0 (flat) is harmless.
    if args.ev_adjusted_mccfr:
        try:
            import cfr_engine
            cfr_engine.NLHEMCCFREngine.set_ev_adjusted_terminals(True)
            print("MCCFR terminals: EV-adjusted (equity over remaining runout)")
        except (ImportError, AttributeError):
            print("[warn] cfr_engine.set_ev_adjusted_terminals not available — "
                  "rebuild .so to enable EV-adjusted MCCFR.")

    if args.bucket_scheme in ("tree", "super", "tree42"):
        try:
            import cfr_engine
            cpp_scheme = {"tree": 1, "super": 2, "tree42": 3}[args.bucket_scheme]
            cfr_engine.NLHEStateEncoder.set_scheme(cpp_scheme)
            label = {
                1: "TREE (super 4-hot + fine 4-hot)",
                2: "SUPER (K=4 one-hot, [12:16] zero)",
                3: "TREE42 (super 4-hot + fine 2-hot low/high)",
            }[cpp_scheme]
            print(f"Encoder scheme: {label}")
        except (ImportError, AttributeError):
            print(f"[warn] cfr_engine.set_scheme not available — running flat-only "
                  f"(rebuild .so for {args.bucket_scheme} scheme).")

    encoder = NLHEEncoder(starting_stack=args.stack,
                          bucket_scheme=args.bucket_scheme)

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
        train_batch=args.train_batch,
        traversals_per_iter=args.traversals,
        use_cpp_engine=True,
        device=device,
        lr=args.lr,
        warm_start=not args.no_warm_start,
        warm_start_lr_factor=args.warm_start_lr_factor,
        seed=args.seed,
        dcfr_gamma=args.dcfr_gamma,
        dcfr_alpha=args.dcfr_alpha,
        regret_target=args.regret_target,
        lr_decay_start=args.lr_decay_start,
        lr_decay_factor=args.lr_decay_factor,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
    )

    print(f"Deep CFR — HU Postflop NLHE")
    print(f"  Stack: {args.stack:.0f}BB | "
          f"Raise: {args.raise_fraction:.0%} pot | "
          f"Iterations: {args.iterations}")
    print(f"  Hidden: {args.hidden} | "
          f"Regret buf: {regret_buf:,} | "
          f"Strategy buf: {strat_buf:,}")
    print(f"  Device: {device}\n")

    # MLflow setup (no-op if --mlflow not given). Runs at this point so the
    # encoder/scheme/MCCFR config from above is already locked in but
    # solver.solve() hasn't started — params are logged before any metric.
    mlf = _setup_mlflow(args)

    t0 = time.time()
    # Rolling window of recent exploitability estimates for stddev diagnostic.
    expl_history: list[float] = []

    def _print_diagnostics(s) -> None:
        """Buffer-level convergence diagnostics — see --diagnostics."""
        buf = s.regret_buffer
        n   = buf.size
        if n == 0:
            print("  [diag] buffer empty")
            return
        # Unique state fraction (sampled to keep cost bounded).
        idx = np.random.default_rng(0).choice(n, size=min(n, 50_000),
                                              replace=False)
        states_sample = buf.states[idx]
        uniq = np.unique(states_sample, axis=0).shape[0]
        uniq_frac = uniq / len(idx)
        # Iteration histogram (5 bins).
        iters = buf.iters[:n]
        if iters.max() > 0:
            bins = np.linspace(1, max(iters.max(), 2), 6)
            hist, _ = np.histogram(iters, bins=bins)
            hist_str = "/".join(f"{c:>5d}" for c in hist)
        else:
            hist_str = "n/a"
        # Target magnitude distribution.
        tgt_mag = np.abs(buf.targets[:n]).mean()
        tgt_max = np.abs(buf.targets[:n]).max()
        # Exploitability rolling stddev (last 3 measurements).
        if len(expl_history) >= 2:
            recent = expl_history[-min(3, len(expl_history)):]
            expl_std = float(np.std(recent))
            expl_std_str = f"{expl_std:6.0f}"
        else:
            expl_std_str = "  n/a"
        print(f"  [diag] uniq={uniq_frac:.3f} of {len(idx):,} | "
              f"iter_hist={hist_str} | "
              f"|tgt| mean={tgt_mag:.4f} max={tgt_max:.4f} | "
              f"expl_std3={expl_std_str}")

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
                    cur, game, encoder, n_games=args.expl_games, seed=0,
                    ev_adjusted=args.ev_adjusted_expl,
                )
                expl_str = f"  expl={expl:6.1f} mbb/decision"
                expl_history.append(float(expl))
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
        # MLflow per-iter metrics. Logged at the iteration step the trainer
        # actually finished (i is 1-indexed). Skip expl if it errored.
        if mlf is not None:
            mlf.log_metric("regret_loss",      reg_loss,                step=i)
            mlf.log_metric("regret_buf_size",  len(s.regret_buffer),    step=i)
            mlf.log_metric("strategy_buf_size",len(s.strategy_buffer),  step=i)
            mlf.log_metric("elapsed_sec",      elapsed,                 step=i)
            if expl_history:
                mlf.log_metric("expl_current_strategy",
                               expl_history[-1], step=i)
        if args.diagnostics:
            _print_diagnostics(s)
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
    final_expl = None
    try:
        final_expl = estimate_exploitability(
            bp, game, encoder, n_games=max(args.expl_games, 200), seed=0,
            ev_adjusted=args.ev_adjusted_expl,
        )
        print(f"\nFinal blueprint exploitability: {final_expl:.1f} mbb/decision "
              f"(untrained ≈ order 100s; lower is better)")
    except Exception as e:
        print(f"[warn] Final exploitability measurement failed: {e}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluate_blueprint(bp, encoder)

    # ── MLflow finalization ───────────────────────────────────────────────────
    if mlf is not None:
        try:
            if final_expl is not None:
                mlf.log_metric("final_exploitability_mbb_per_decision",
                               float(final_expl))
            mlf.log_metric("total_training_seconds", elapsed)
            # Blueprint metadata as artifact (manifest hash + state size etc).
            if args.save_blueprint:
                from pathlib import Path
                manifest = Path(args.save_blueprint) / "model_manifest.json"
                if manifest.exists():
                    mlf.log_artifact(str(manifest))
        finally:
            mlf.end_run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())