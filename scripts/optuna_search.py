#!/usr/bin/env python3
"""
Optuna hyperparameter search for Deep CFR on Postflop NLHE.

v3: FULL-budget trials (500 iter × 1000 traversals, ~21 min each) +
    CRN+EV h2h paired-Δ vs v3_coarse blueprint as the OBJECTIVE
    (NOT exploitability). Each trial total: ~24 min (21 train + 3 h2h).

Why h2h objective: rounds v1+v2 showed exploitability and CRN+EV h2h disagree
systematically. v11_optuna_v2 reached expl=1764→1999 mbb/dec internally but
LOST +494 mbb/h to v3 in h2h (z=3.60, statistically significant). Diagnostic
(scripts/diagnose_expl_vs_h2h.py) attributed this to entropy + loose-aggressive
play: Optuna's expl optimum is closer to GTO in the "worst-case best-response"
sense but loses to a tighter opponent (v3) in practice. Switching the objective
to h2h directly optimises production-relevant quality.

Trial measurement: 1000 CRN+EV pairs ≈ 3 min on MPS, per-X stderr ~140 mbb/h.
TPE handles noisy measurements at this scale. Pruning still uses mid-training
exploitability (cheap signal during training that correlates with regret-loss
collapse — if expl goes to 10000 mbb/dec mid-training, h2h won't recover).

Each trial is logged as an MLflow run under experiment "deep-cfr-nlhe-optuna"
with params + per-iter expl + final expl. The Optuna study itself is persisted
to optuna.db (sqlite) so you can resume / inspect.

Usage:
    python3 scripts/optuna_search.py                # 20 trials, ~7h
    python3 scripts/optuna_search.py --best         # print best trial only
    python3 scripts/optuna_search.py --fresh        # delete old study, restart
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import optuna
import mlflow

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.analysis.exploitability import estimate_exploitability


# ── Fixed config (NOT searched — keeps trials comparable) ─────────────────────
#
# v2 budget: 500 iter × 1000 traversals — same as production v3 training. Why
# this matters: the v1 short-budget search (100 iter × 500 traversals) found a
# "Trial 17" optimum that scored 1538 mbb/dec internally but COLLAPSED to a
# near-uniform strategy when extrapolated to 500 iter (v10 = 2551 mbb/dec
# exploitability AND h2h v3 +1338 mbb/h with z=8.81 in v3's favour). The
# 100-iter optimum favoured aggressive LR + early decay (gets fast initial
# gains then deteriorates); at 500 iter that schedule oscillates badly.
#
# Trade-off: each trial now takes ~21 min instead of ~3 min, so we drop
# n_trials default 50 → 20. Pruner stays aggressive, MedianPruner kicks in
# after 8 warm-up trials.
FIXED = {
    "stack":          50.0,
    "max_raises":     1,
    "raise_fraction": 0.75,
    "buffer":         1_000_000,  # production-sized buffer
    "strategy_buf":   1_000_000,
    "hidden":         256,
    "iterations":     500,        # full production budget
    "traversals":     1000,       # full production budget
    "expl_games":     200,        # tighter than 500 default to keep callback fast
    "expl_freq":      50,         # 10 callbacks per trial — 10 pruning points
    "bucket_scheme":  "flat",     # v3 winner
    # ev_adjusted_mccfr collapsed v9 training (z=8.62 worse than v3). Disabled
    # here until the scale interaction is understood.
    "ev_adjusted_mccfr":   False,
    "ev_adjusted_expl":    True,  # callback signal benefits from variance reduction
    "regret_target":  "instant",
    # CRN+EV paired comparisons per trial. 1000 pairs ≈ 3 min on MPS; gives
    # per-X stderr ~140 mbb/h based on prior measurements. With 20 trials we
    # don't need exact precision per trial — TPE works fine with noisy
    # measurements at this scale.
    "h2h_pairs": 1000,
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Single training run (lives outside train_postflop.py's argparse) ──────────
# Reference opponent for h2h objective. Loaded lazily once per process.
REFERENCE_BLUEPRINT_PATH = "blueprints/50bb_validation_v3_coarse"
_reference_bp = None


def _get_reference_blueprint():
    global _reference_bp
    if _reference_bp is None:
        from src.deep_cfr.blueprint import Blueprint
        _reference_bp = Blueprint.load(REFERENCE_BLUEPRINT_PATH, device="cpu")
    return _reference_bp


def train_one_run(cfg: dict, trial: optuna.Trial | None = None):
    """Train Deep CFR with cfg and return (bp, game, encoder).

    Final exploitability is logged to MLflow but no longer used as objective —
    caller computes CRN+EV h2h vs the reference blueprint to score the trial.

    cfg keys: lr, dcfr_gamma, dcfr_alpha, warm_start_lr_factor, epochs,
              train_batch, lr_decay_factor, lr_decay_start, finetune_epochs,
              finetune_lr (and FIXED keys).
    """
    device = _device()
    encoder = NLHEEncoder(starting_stack=cfg["stack"],
                          bucket_scheme=cfg["bucket_scheme"])

    if cfg.get("ev_adjusted_mccfr"):
        try:
            import cfr_engine
            cfr_engine.NLHEMCCFREngine.set_ev_adjusted_terminals(True)
        except Exception:
            pass

    game = PostflopNLHE(
        starting_stack=cfg["stack"],
        max_raises_per_street=cfg["max_raises"],
        raise_fractions=(cfg["raise_fraction"],),
    )
    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,
        buffer_capacity=cfg["buffer"],
        strategy_buffer_capacity=cfg["strategy_buf"],
        hidden_size=cfg["hidden"],
        train_epochs=cfg["epochs"],
        train_batch=cfg["train_batch"],
        traversals_per_iter=cfg["traversals"],
        use_cpp_engine=True,
        device=device,
        lr=cfg["lr"],
        warm_start=True,
        warm_start_lr_factor=cfg["warm_start_lr_factor"],
        dcfr_gamma=cfg["dcfr_gamma"],
        dcfr_alpha=cfg["dcfr_alpha"],
        regret_target=cfg["regret_target"],
        lr_decay_start=cfg["lr_decay_start"],
        lr_decay_factor=cfg["lr_decay_factor"],
        finetune_epochs=cfg["finetune_epochs"],
        finetune_lr=cfg["finetune_lr"],
    )

    t0 = time.time()

    def cb(s, i):
        # Mid-training expl for pruning + MLflow logging.
        try:
            cur_bp = s.current_strategy_blueprint()
            expl = estimate_exploitability(
                cur_bp, game, encoder,
                n_games=cfg["expl_games"], seed=0,
                ev_adjusted=cfg.get("ev_adjusted_expl", True),
            )
            mlflow.log_metric("expl_current_strategy", float(expl), step=i)
            mlflow.log_metric("regret_loss",
                              float(getattr(s, "_last_regret_loss", 0.0)),
                              step=i)
            mlflow.log_metric("elapsed_sec", time.time() - t0, step=i)
            if trial is not None:
                trial.report(float(expl), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  [warn cb] {type(e).__name__}: {e}")

    solver.solve(iterations=cfg["iterations"],
                 callback=cb,
                 callback_freq=cfg["expl_freq"])

    # Build the trained blueprint and report its final exploitability for
    # logging (NOT the objective in v3). The actual objective is computed by
    # the caller via CRN+EV h2h vs the v3 reference blueprint.
    from src.deep_cfr.blueprint import Blueprint
    bp = Blueprint.from_solver(solver, device=device)
    try:
        final_expl = estimate_exploitability(
            bp, game, encoder, n_games=max(cfg["expl_games"], 200), seed=0,
            ev_adjusted=cfg.get("ev_adjusted_expl", True),
        )
        mlflow.log_metric("final_exploitability_mbb_per_decision",
                          float(final_expl))
    except Exception as e:
        print(f"  [warn final expl] {type(e).__name__}: {e}")
    return bp, game, encoder


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    # Search space refined after v1 (100-iter) round + v10 full-budget failure.
    #
    # NARROWED:
    #   * lr: 1e-4 to 1e-3 (v1 found 1.7e-3 best at 100 iter; 500 iter
    #     oscillated at that LR. Cap at 1e-3 to stay in stable territory.)
    #   * dcfr_alpha: 1.5 to 3.0 (v1 best > 2.0 consistently; bottom raised
    #     since alpha < 1.5 was never competitive.)
    #   * warm_start_lr_factor: 5.0 to 10.0 (winners clustered at the top.)
    #   * train_batch: 256 or 512 (128 never won.)
    #   * lr_decay_start: 100 to 400 (CRITICAL — at 500 iter, decay starting
    #     at 91 means 409 iter of decay → over-decays. Push later.)
    #   * finetune_epochs: 0 to 50 (winners always < 20; top capped at 50.)
    #
    # KEPT WIDE:
    #   * dcfr_gamma: 0.5 to 4.0 (no clear pattern yet.)
    #   * lr_decay_factor: 0.95 to 1.0 (consistent narrow range, keep as is.)
    #   * epochs: 30 to 80 (mixed signal, keep wide.)
    #   * finetune_lr: 1e-5 to 5e-4 log (keep wide.)
    cfg = dict(FIXED)
    cfg["lr"]               = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    cfg["dcfr_gamma"]       = trial.suggest_float("dcfr_gamma", 0.5, 4.0)
    # v2 boundary hits: dcfr_alpha 1.79 (lower bound was 1.5),
    # warm_start_lr_factor 5.23 (lower bound was 5.0). Widened here so optimum
    # can live below the previous floors.
    cfg["dcfr_alpha"]       = trial.suggest_float("dcfr_alpha", 0.5, 3.0)
    cfg["warm_start_lr_factor"] = trial.suggest_float("warm_start_lr_factor",
                                                       2.0, 10.0)
    cfg["epochs"]           = trial.suggest_int("epochs", 30, 80)
    cfg["train_batch"]      = trial.suggest_categorical("train_batch",
                                                         [256, 512])
    cfg["lr_decay_factor"]  = trial.suggest_float("lr_decay_factor", 0.95, 1.0)
    cfg["lr_decay_start"]   = trial.suggest_int("lr_decay_start", 100, 400)
    cfg["finetune_epochs"]  = trial.suggest_int("finetune_epochs", 0, 50)
    cfg["finetune_lr"]      = trial.suggest_float("finetune_lr", 1e-5, 5e-4,
                                                   log=True)

    run_name = f"optuna_trial_{trial.number:04d}"
    with mlflow.start_run(run_name=run_name):
        for k, v in cfg.items():
            mlflow.log_param(k, v if isinstance(v, (int, float, bool)) else str(v))
        mlflow.set_tag("optuna_trial", str(trial.number))
        try:
            bp, game, encoder = train_one_run(cfg, trial=trial)
        except optuna.TrialPruned:
            mlflow.set_tag("status", "pruned")
            raise
        except Exception as e:
            mlflow.set_tag("status", f"error:{type(e).__name__}")
            raise

        # CRN+EV h2h vs v3. Returns v3's paired-Δ in mbb/hand:
        #   positive = v3 wins this much per hand (trial is worse than v3)
        #   negative = trial wins this much per hand (trial beats v3)
        # Optuna minimises → smaller paired-Δ → trial wants to beat v3.
        from src.analysis.head_to_head import match_crn
        ref_bp = _get_reference_blueprint()
        h2h = match_crn(ref_bp, bp, game, encoder, encoder,
                        n_pairs=cfg["h2h_pairs"],
                        seed=trial.number * 17 + 3,
                        ev_adjusted=True)
        paired_diff = float(h2h["win_rate_mbb"])
        stderr      = float(h2h["stderr_mbb"])
        mlflow.log_metric("h2h_v3_paired_diff_mbb_per_hand", paired_diff)
        mlflow.log_metric("h2h_v3_stderr_mbb_per_hand",      stderr)
        mlflow.log_metric("h2h_z_score",
                          paired_diff / stderr if stderr > 0 else 0.0)
        mlflow.set_tag("status", "completed")
        return paired_diff


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of Optuna trials (default: 20). Each trial is "
                        "a full-budget 500-iter × 1000-traversals training run "
                        "(~21 min on MPS), so 20 trials ≈ 7h. Pruner kills "
                        "trials with bad mid-training expl, typically cutting "
                        "30-50% of compute on losing regions.")
    p.add_argument("--study-name", default="deep-cfr-postflop-50bb-h2h",
                   help="Optuna study name (used for resume).")
    p.add_argument("--storage",   default="sqlite:///optuna.db",
                   help="Optuna storage (default: sqlite:///optuna.db).")
    p.add_argument("--fresh",     action="store_true",
                   help="Force a fresh study even if --study-name already exists "
                        "(default: resume any existing study with that name).")
    p.add_argument("--resume",    action="store_true",
                   help="Backwards-compat no-op (resume is the default).")
    p.add_argument("--best",      action="store_true",
                   help="Print best trial from existing study and exit.")
    p.add_argument("--timeout",   type=int, default=None,
                   help="Wall-clock seconds to spend (overrides --n-trials).")
    args = p.parse_args()

    tracking_db = (Path.cwd() / "mlflow.db").resolve()
    artifacts   = (Path.cwd() / "mlruns").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    try:
        exp = mlflow.get_experiment_by_name("deep-cfr-nlhe-optuna")
        if exp is None:
            mlflow.create_experiment("deep-cfr-nlhe-optuna",
                                      artifact_location=f"file://{artifacts}")
    except Exception:
        pass
    mlflow.set_experiment("deep-cfr-nlhe-optuna")

    # Load/create study. MedianPruner skips trials worse than median at each
    # report step after the first n_startup_trials have completed.
    # Default behaviour: resume any existing study (idempotent). --fresh forces
    # a delete first so re-running with same name starts over.
    if args.fresh:
        try:
            optuna.delete_study(study_name=args.study_name, storage=args.storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                            n_warmup_steps=2),
    )

    if args.best:
        try:
            best = study.best_trial
        except ValueError:
            print("No completed trials yet."); return 0
        print(f"Best trial #{best.number}: expl={best.value:.1f} mbb/decision")
        for k, v in best.params.items():
            print(f"  {k:25s} = {v}")
        return 0

    print(f"Optuna study '{args.study_name}' → {args.storage}")
    print(f"  trials so far: {len(study.trials)}")
    print(f"  fixed budget per trial: {FIXED['iterations']} iter × "
          f"{FIXED['traversals']} traversals  (~3 min @ MPS)")
    if args.timeout:
        print(f"  timeout: {args.timeout}s")
    else:
        print(f"  n_trials: {args.n_trials}")
    print()

    study.optimize(objective,
                   n_trials=args.n_trials,
                   timeout=args.timeout,
                   show_progress_bar=False)

    print("\nSearch done.")
    try:
        best = study.best_trial
        print(f"Best trial #{best.number}: expl={best.value:.1f} mbb/decision")
        for k, v in best.params.items():
            print(f"  {k:25s} = {v}")
    except ValueError:
        print("No completed trials.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
