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
    estimate_exploitability returns an LBR (Local Best Response) proxy in
    mbb/decision (milli-big-blinds per decision node), NOT per hand. It
    marginalises opponent hands at each evaluated node and integrates all-in
    runouts by default. The returned ExplResult is a float subclass — it
    also carries .stderr_mbb, used below to log noise alongside the mean.

    Pre-2026-06-14 baselines reported under the deal-specific clairvoyant
    estimator (no opp-card marginalisation, ev_adjusted=False default) are
    NOT comparable: that version overstated exploitability ~2x via runout
    variance and ~2-5x further via clairvoyant best-response. Re-baseline
    before comparing against historical numbers.

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


def _action_labels_for_game(game, deal, prefix: str) -> list[str]:
    """Build snapshot labels that match the actual legal_actions list.

    Labels are produced by mapping each action SYMBOL to a human label.
    Previously this took ``ACTION_LABELS[prefix][:N]`` which assumed slot
    1 was always "call" — false on postflop no-bet states (legal_actions
    returns ['c','r','a']: slot 1 is "raise", slot 2 is "all-in"). The
    symbol-based map is correct for any subset of legal actions and for
    both single-raise and multi-raise puu.
    """
    legal = game.legal_actions(deal)
    base  = "fold/check" if prefix == "preflop" else "check/fold"
    label_map = {"f": base, "c": base, "k": "call", "a": "all-in"}
    n_raises = len(game.raise_fractions)
    if n_raises <= 1:
        # Single-raise legacy: action symbol is plain 'r'.
        label_map["r"] = "raise"
    else:
        # Multi-raise puu: 'r0','r1','r2' indexed by raise_fractions.
        for i, frac in enumerate(game.raise_fractions):
            label_map[f"r{i}"] = f"r{int(round(frac * 100))}%"
    return [label_map.get(str(a), str(a)) for a in legal]

SAMPLE_HANDS = [
    # (description, hole_p0, board, hero, prefix_actions, label)
    #
    # Prefix actions: applied verbatim before encoding so the snapshot
    # reflects the intended street/role. Pre-2026-06-14 entries omitted
    # prefix_actions, which made every "flop" snapshot collapse to a
    # preflop state — see project_evaluate_snapshot_fix.
    #
    # ('k','c') sequence after deal: SB calls preflop → BB checks first
    # on flop → hand reaches SB's flop decision with to_call=0.
    ("AA preflop SB",   (48, 49), (),                    0, (),          "value"),
    ("72o preflop SB",  (24, 1),  (),                    0, (),          "trash"),
    ("KK flop top SB",  (44, 45), (0, 5, 10, 15, 20),    0, ("k", "c"), "value"),
    ("Q-hi flop SB",    (11, 22), (0, 5, 10, 15, 20),    0, ("k", "c"), "bluff"),
]


def evaluate_blueprint(bp: Blueprint, encoder: NLHEEncoder) -> None:
    """Print strategy snapshots for representative hands."""
    print("\n" + "=" * 62)
    print("BLUEPRINT STRATEGY SNAPSHOTS")
    print("=" * 62)

    # Restore multi-raise puu from metadata if present.
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    rng = np.random.default_rng(0)

    for desc, hole, board_cards, player, prefix_actions, _ in SAMPLE_HANDS:
        remaining = [c for c in range(52) if c not in hole and c not in board_cards]
        rng.shuffle(remaining)
        opp_cards = tuple(remaining[:2])
        full_board = tuple(board_cards) + tuple(remaining[2:2 + (5 - len(board_cards))])
        deal = (hole, opp_cards, full_board)

        # Apply prefix actions so the encoded state matches the intended
        # street and role. Without this, board-only scenarios collapse to
        # a preflop encoding (see history-only-determines-street invariant).
        history = deal + tuple(prefix_actions)
        if game.is_terminal(history):
            print(f"  {desc:<22s}: SKIPPED (terminal after prefix {prefix_actions})")
            continue

        actual_player = game.current_player(history)
        if actual_player != player:
            print(f"  {desc:<22s}: SKIPPED (expected hero={player}, "
                  f"got current_player={actual_player} after prefix)")
            continue

        state_vec   = encoder.encode(history, player)
        legal       = game.legal_actions(history)
        num_actions = len(legal)
        # Slot-indexed query so postflop no-bet states (legal=['c','r','a'])
        # show actual check/raise/all-in probabilities — the legacy
        # bp.query(state, num_actions) returned slots [0..n-1] contiguously
        # which misaligned the snapshot at any non-contiguous legal set.
        from src.deep_cfr.action_slots import legal_actions_to_slots
        slots = legal_actions_to_slots(legal, bp.metadata.action_size)
        probs = bp.query_by_slots(state_vec, slots)

        prefix = "preflop" if not prefix_actions else "postflop"
        labels = _action_labels_for_game(game, history, prefix)
        parts = "  ".join(
            f"{labels[i] if i < len(labels) else f'slot{i}'}={p:.0%}"
            for i, p in enumerate(probs)
        )
        print(f"  {desc:<22s}: {parts}")

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
    p.add_argument("--dcfr-alpha",           type=float, default=1.0,
                   help="DCFR regret-diskontaus α (oletus 1.0 = Linear CFR). "
                        "Näytepaino = t^α/(t^α+1). 0 = ei diskontausta, "
                        "1.0 = puhdas Linear CFR (Brown & Sandholm 2019), "
                        "1.5 = aggressiivisempi DCFR-diskontaus.")
    p.add_argument("--linear-cfr-iters",    type=int, default=0,
                   help="Pluribus-tyylinen Linear CFR -varhaisbias: kun N > 0, "
                        "ensimmäisillä N iteraatiolla käytetään α=1.0 (linear), "
                        "ja N:n jälkeen vaihdetaan --dcfr-alpha:han. "
                        "Pluribus: linear vain alkuun koska myöhemmin hyöty "
                        "katoaa. Suositus: N ≈ iteraatiomäärä/3.")
    p.add_argument("--prune-threshold",     type=float, default=0.0,
                   help="Pluribus-tyylinen dynaaminen pruning C++ MCCFR:lle. "
                        "Actionit, joiden blueprint-prob < threshold, jätetään "
                        "traversoimatta. Säästää compute → enemmän iteraatioita "
                        "samalla budjetilla. 0 = pois. Suositus: 0.001…0.01.")
    p.add_argument("--prune-after-iter",    type=int, default=100,
                   help="Iteraatio, jonka jälkeen pruning aktivoituu. "
                        "Alkuiteraatioilla blueprint on liian uniformi → kaikki "
                        "actionit nikkaroituisi pois. Suositus: ≥100.")
    p.add_argument("--no-position-bit",     action="store_true",
                   help="Encoderin position-bit pois käytöstä (v15 ablation). "
                        "Kirjoittaa vakio 0.0 viimeiseen dimensioon SB/BB-signaalin "
                        "sijaan. State_size pysyy 37:nä → yhteensopiva ilman C++ "
                        "rebuildiä. Testaa onko position-bitti aiheuttanut v14:n "
                        "regression v11-baseliniin verrattuna.")
    p.add_argument("--save-best-checkpoint", action="store_true",
                   help="Tallentaa parhaan LBR-iteraation snapshotin erikseen "
                        "polkuun {--save-blueprint}_best. Vaatii --save-blueprint:n "
                        "ja --expl-games>0:n. Treenaa snapshot-strategy-netin "
                        "current strategy-bufferista (--best-snapshot-epochs).")
    p.add_argument("--best-snapshot-epochs", type=int, default=50,
                   help="Best-checkpoint snapshot-treenin epoch-määrä per "
                        "tallennus. Pienempi = nopeampi, suurempi = polishoidumpi "
                        "snapshot. Suositus 50 (vs Final 300).")
    p.add_argument("--best-margin-stderr",   type=float, default=0.25,
                   help="Margin uudelle 'best':lle: uusi expl < best_expl - "
                        "margin * combined_stderr. Estää tallentamasta marginaalisia "
                        "sample-kohina-parannuksia. Suositus 0.25-1.0.")
    p.add_argument("--cfr-cache",            type=str, default="",
                   help="Polku CFR advisor cache:hen (rakennettu komennolla "
                        "build_cfr_cache.py). Kun asetettu, encoder appendaa 12 "
                        "advisor-dimiä (6 action-probs + 6 EVs) state-vektoriin. "
                        "State_size 37 → 49 → verkko on suurempi mutta saa "
                        "korkealaatuista ohjausta jokaisessa decisionissa. Pakottaa "
                        "Python-solverin (C++ engine ei vielä tunne cache:ta).")
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
    p.add_argument("--raise-fraction",       type=float, default=0.50,
                   help="Raise size as fraction of pot (default: 0.50). "
                        "Vaihdettu 0.75:stä 2026-06-14 — LBR-diagnostiikka "
                        "näytti että 75%% raise oli liian iso useille spotille; "
                        "50%% pot on tyypillisempi GTO-koko ja antaa verkolle "
                        "paremman value/bluff-balanssin 4-action-puussa. "
                        "Ohitetaan jos --raise-fractions on annettu.")
    p.add_argument("--raise-fractions",      type=str, default="",
                   help="Pluribus-tyylinen multi-raise puu: pilkulla erotettu "
                        "lista raise-kokoja pottin osuutena, esim. "
                        "'0.33,0.66,1.0'. 1-3 kokoa tuettu (C++ enum-kapasiteetti). "
                        "Tyhjä = yksittäisraise --raise-fraction:lla. "
                        "Multi-raise muuttaa action-spacen 4 → 4+(N-1) ja "
                        "vaatii saman --raise-fractions:n h2h-arvioinnissa.")

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
    p.add_argument("--aux-ev-weight",        type=float, default=0.0,
                   help="Painokerroin regret-verkon auxiliary EV-prediction "
                        "headille. Vaatii --cfr-cache (state_size=49). Verkko "
                        "saa pienen sivuhead:n joka oppii ennustamaan advisor-"
                        "EV:t [slots 43:49] siten että jaettu trunk pakottuu "
                        "internalisoimaan cache-signaalia. 0.0 = pois päältä "
                        "(default; v4-käytös). Suositus aloitukseen: 0.1.")
    p.add_argument("--exploit-gap-map",      type=str, default="",
                   help="Polku scripts/mine_exploit_gaps.py:n tuottamaan .npz-"
                        "tiedostoon. Aktivoi BRD (best-response-defense) -"
                        "painotuksen: regret-buffer-näytteet, joiden "
                        "abstraktio-avain mappaa korkeaan exploit-gap:iin, "
                        "saavat suuremman loss-painon. Tarvitsee --exploit-gap-"
                        "lambda > 0 jotta vaikutus on >0.")
    p.add_argument("--exploit-gap-lambda",   type=float, default=0.0,
                   help="BRD-painotuksen voimakkuus. Sample-paino = base * "
                        "(1 + lambda * gap/median_gap). 0 = ei BRD-painotusta, "
                        "5-10 voimakas painotus. Suositus aloitukseen: 5.0.")
    p.add_argument("--teacher-blueprint",    type=str, default="",
                   help="Polku olemassa olevaan blueprint-hakemistoon joka "
                        "toimii teacher-mallina strategy-network "
                        "distillationiin. Vaatii --teacher-kl-weight > 0. "
                        "Teacher-mallin metadata.state_size + action_size + "
                        "raise-fractions tulee vastata treenattavaa konfiguraatiota.")
    p.add_argument("--teacher-kl-weight",    type=float, default=0.0,
                   help="KL(teacher || student) -painokerroin strategy-"
                        "network-koulutuksen loss-funktioon. 0 = ei "
                        "distillationia (default). Suositus aloitukseen: 0.5.")
    p.add_argument("--aug-bucket-prob",      type=float, default=0.0,
                   help="Counterfactual hand-bucket -augmentaation "
                        "todennäköisyys per regret-näyte. Siirtää "
                        "hand-bucket-one-hotin naapuriin (±radius) ja "
                        "päivittää cache-advisor-dimit uudella bucket:lla. "
                        "Regret-target säilyy → pakottaa verkon olemaan "
                        "sileä equity-abstraktion sisällä. 0 = pois (default). "
                        "Suositus aloitukseen: 0.3.")
    p.add_argument("--aug-bucket-radius",    type=int, default=1,
                   help="Maksimi siirtyma hand-bucket-augmentaatiossa "
                        "(±radius). Default 1 = vain välittömät naapurit.")
    p.add_argument("--predictive-alpha",     type=float, default=0.0,
                   help="Predictive CFR+ (Brown 2020) momentum-kerroin. "
                        "Lisää R_t -päivitykseen termin alpha*(r_t - r_{t-1}) "
                        "joka ennakoi seuraavan iteraation regretia. "
                        "Empirically 3-5x nopeampi konvergenssi vs vanilla "
                        "CFR+. 0 = pois (default). Suositus aloitukseen: 1.0.")
    p.add_argument("--value-head-weight",    type=float, default=0.0,
                   help="V(s) scalar value head -painokerroin. Verkko saa "
                        "rinnakkaisen value-head:n joka oppii ennustamaan "
                        "Σ probs * EVs (cache:n implikoiman state-arvon). "
                        "DeepStack/ReBeL-tyylinen tukisignaali joka pakottaa "
                        "trunkin enkoodamaan state-tason expected valuen. "
                        "Vaatii --cfr-cache. 0 = pois (default). Suositus: 0.1.")

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

    # Parse multi-raise fractions early so the encoder gets the same list
    # the game uses (encoder._parse_state needs all 'rN' actions resolvable).
    if args.raise_fractions.strip():
        _early_raise_fracs = tuple(
            float(x) for x in args.raise_fractions.split(",") if x.strip()
        )
    else:
        _early_raise_fracs = (args.raise_fraction,)

    # Optional CFR advisor cache (C2 design): adds 12 dims to state_size
    # via cache lookup + live MC EV fallback. State_size becomes 49 when
    # set, forces Python solver path (use_cpp_engine=False) until C++
    # cache loader lands.
    cfr_cache = None
    if args.cfr_cache:
        from src.deep_cfr.cfr_cache import CFRCache
        cfr_cache = CFRCache.load(args.cfr_cache)
        print(f"Loaded CFR cache: {len(cfr_cache)} entries from {args.cfr_cache}")

    encoder = NLHEEncoder(starting_stack=args.stack,
                          bucket_scheme=args.bucket_scheme,
                          raise_fractions=_early_raise_fracs,
                          include_position_bit=not args.no_position_bit,
                          cfr_cache=cfr_cache)

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
    # Parse --raise-fractions (multi-raise puu) tai fall back to single
    # --raise-fraction (legacy 4-action puu).
    if args.raise_fractions.strip():
        raise_fracs = tuple(
            float(x) for x in args.raise_fractions.split(",") if x.strip()
        )
        if not raise_fracs:
            raise ValueError("--raise-fractions parsed to empty list")
        if len(raise_fracs) > 3:
            raise ValueError(
                "--raise-fractions max 3 entries (C++ enum NLHE_RAISE_0..2)"
            )
    else:
        raise_fracs = (args.raise_fraction,)

    # Action space size: fold/check + (optional call) + N raises + allin.
    # Capacity (network output dim, regret sample slots): 3 + N raises.
    # Single raise → 4 actions (legacy). Multi raise N=2 → 5, N=3 → 6.
    n_actions_capacity = 3 + len(raise_fracs)

    game = PostflopNLHE(
        starting_stack=args.stack,
        max_raises_per_street=args.max_raises,
        raise_fractions=raise_fracs,
    )

    # C++ engine emits 49-dim state vectors (advisor slots at [37:49] = 0).
    # When a CFR cache is attached, cpp_backend.to_tensors backfills those
    # slots via key_from_state_vector lookup before samples reach the regret
    # buffer (see DeepCFRSolver._run_cpp_iteration → set_cache_context wiring).
    # Cache build, Python lookup, and C++ backfill all go through the same
    # key_from_state_vector path → keys agree by construction.
    _use_cpp = bool(args.use_cpp) if hasattr(args, "use_cpp") else True
    if cfr_cache is not None:
        print(f"[note] CFR cache attached → C++ path with state-vector "
              f"backfill (use_cpp={_use_cpp}).")

    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=n_actions_capacity,
        buffer_capacity=regret_buf,
        strategy_buffer_capacity=strat_buf,
        hidden_size=args.hidden,
        train_epochs=args.epochs,
        train_batch=args.train_batch,
        traversals_per_iter=args.traversals,
        use_cpp_engine=_use_cpp,
        device=device,
        lr=args.lr,
        warm_start=not args.no_warm_start,
        warm_start_lr_factor=args.warm_start_lr_factor,
        seed=args.seed,
        dcfr_gamma=args.dcfr_gamma,
        dcfr_alpha=args.dcfr_alpha,
        linear_cfr_iters=args.linear_cfr_iters,
        prune_threshold=args.prune_threshold,
        prune_after_iter=args.prune_after_iter,
        include_position_bit=not args.no_position_bit,
        regret_target=args.regret_target,
        lr_decay_start=args.lr_decay_start,
        lr_decay_factor=args.lr_decay_factor,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        aux_ev_weight=args.aux_ev_weight,
        aug_bucket_prob=args.aug_bucket_prob,
        aug_bucket_radius=args.aug_bucket_radius,
        predictive_alpha=args.predictive_alpha,
        value_head_weight=args.value_head_weight,
    )

    # Teacher distillation: load teacher blueprint's strategy net and pin
    # it to the solver so train_strategy_network can add the KL term.
    if args.teacher_blueprint and args.teacher_kl_weight > 0:
        from pathlib import Path as _P
        if not _P(args.teacher_blueprint).exists():
            sys.exit(f"[error] teacher blueprint not found: {args.teacher_blueprint}")
        teacher_bp = Blueprint.load(args.teacher_blueprint, device=device)
        if teacher_bp.metadata.state_size != solver.encoder.state_size():
            sys.exit(
                f"[error] teacher state_size {teacher_bp.metadata.state_size} "
                f"≠ student encoder state_size {solver.encoder.state_size()}. "
                f"Match --cfr-cache / --raise-fractions to teacher's config."
            )
        if teacher_bp.metadata.action_size != solver.max_actions:
            sys.exit(
                f"[error] teacher action_size {teacher_bp.metadata.action_size} "
                f"≠ student action_size {solver.max_actions}."
            )
        solver.teacher_net = teacher_bp._net          # frozen, just for forward
        solver.teacher_kl_weight = float(args.teacher_kl_weight)
        # Ensure no gradient tracking on teacher.
        for p_ in solver.teacher_net.parameters():
            p_.requires_grad = False
        print(f"[teacher] loaded {args.teacher_blueprint}, "
              f"KL weight={args.teacher_kl_weight}")

    # BRD exploit-gap map (loaded once before solver fully constructed).
    if args.exploit_gap_map and args.exploit_gap_lambda > 0:
        from pathlib import Path as _P
        if not _P(args.exploit_gap_map).exists():
            sys.exit(f"[error] exploit gap map not found: {args.exploit_gap_map}")
        d = np.load(args.exploit_gap_map, allow_pickle=True)
        keys = np.asarray(d["keys"], dtype=np.uint64)
        gaps = np.asarray(d["gaps"], dtype=np.float32)
        gap_map = {int(k): float(g) for k, g in zip(keys, gaps)}
        solver.exploit_gap_map    = gap_map
        solver.exploit_gap_lambda = float(args.exploit_gap_lambda)
        print(f"[BRD] loaded {len(gap_map)} exploit-gap entries from "
              f"{args.exploit_gap_map}, lambda={args.exploit_gap_lambda}")

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
    # Best-LBR-iteration checkpoint tracking. None until --save-best-checkpoint
    # observes the first LBR measurement. best_stderr is tracked so the margin
    # rule (--best-margin-stderr) can demand a statistically meaningful drop
    # before incurring snapshot-train cost.
    best_state = {"expl": None, "stderr": None, "iter": None, "saved_to": None}

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
                expl_str = (f"  expl={expl:6.1f} ± {expl.stderr_mbb:5.1f}"
                            f" mbb/decision")
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
                # Log stderr alongside so we can tell sample noise from
                # actual strategy quality changes between iterations.
                if hasattr(expl, "stderr_mbb"):
                    mlf.log_metric("expl_current_strategy_stderr",
                                   float(expl.stderr_mbb), step=i)
        if args.diagnostics:
            _print_diagnostics(s)
        if args.save_blueprint and i % 1000 == 0:
            ckpt_path = args.save_blueprint + f"_ckpt{i}"
            Blueprint.from_solver(s, device="cpu").save(ckpt_path)
            print(f"  [checkpoint saved → {ckpt_path}]")

        # ── Best-LBR-iteration checkpoint (--save-best-checkpoint) ────────────
        if (args.save_best_checkpoint and args.save_blueprint
                and args.expl_games > 0 and expl_history):
            cur_expl   = float(expl_history[-1])
            cur_stderr = float(expl.stderr_mbb) if hasattr(expl, "stderr_mbb") else 0.0
            be = best_state["expl"]
            if be is None:
                _new_best = True
            else:
                # Margin rule: insist current is sig. lower than best by
                # `margin` × combined stderr — avoids snapshot-thrashing on
                # noise. Larger margin = stricter criterion.
                combined = (cur_stderr**2 + (best_state["stderr"] or 0.0)**2) ** 0.5
                _new_best = (cur_expl + args.best_margin_stderr * combined) < be
            if _new_best:
                from src.deep_cfr.networks import (
                    StrategyNetwork, train_strategy_network,
                )
                from src.deep_cfr.blueprint import (
                    BlueprintMetadata, ScriptableStrategyNet,
                )
                t_snap = time.time()
                # Train a fresh strategy-net from the current buffer. Fewer
                # epochs than the final pass — fast enough to run multiple
                # times during training.
                snap_net = StrategyNetwork(
                    s.encoder.state_size(),
                    solver.max_actions,
                    s.hidden_size,
                ).to(device)
                train_strategy_network(
                    snap_net, s.strategy_buffer,
                    epochs=args.best_snapshot_epochs,
                    batch_size=args.train_batch,
                    lr=args.lr,
                )
                wrapper = ScriptableStrategyNet.from_strategy_network(snap_net)
                _rfs = solver.game.raise_fractions
                meta = BlueprintMetadata(
                    state_size=s.encoder.state_size(),
                    action_size=solver.max_actions,
                    hidden_size=s.hidden_size,
                    starting_stack=solver.game.starting_stack,
                    raise_fraction=float(_rfs[0]),
                    raise_fractions=([float(x) for x in _rfs]
                                     if len(_rfs) > 1 else []),
                    max_raises=solver.game.max_raises_per_street,
                    iterations=i,
                    traversals_per_iter=solver.traversals_per_iter,
                    strategy_samples=len(solver.strategy_buffer),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                best_path = args.save_blueprint + "_best"
                Blueprint(wrapper, meta, device="cpu").save(best_path)
                print(f"  [best snapshot saved → {best_path}  "
                      f"iter={i}, expl={cur_expl:.1f} mbb/dec, "
                      f"snap_time={time.time() - t_snap:.1f}s]")
                best_state["expl"]     = cur_expl
                best_state["stderr"]   = cur_stderr
                best_state["iter"]     = i
                best_state["saved_to"] = best_path

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
        print(f"\nFinal blueprint LBR exploitability: "
              f"{final_expl:.1f} ± {final_expl.stderr_mbb:.1f} mbb/decision "
              f"(lower is better; large stderr → metric is noise-dominated)")
        # Best-LBR-iteration summary (if --save-best-checkpoint was on).
        if best_state["saved_to"] is not None:
            print(f"Best LBR snapshot     : {best_state['saved_to']}")
            print(f"  iter={best_state['iter']}, "
                  f"during-train LBR={best_state['expl']:.1f} mbb/dec")
            if float(best_state["expl"]) < float(final_expl):
                gap = float(final_expl) - float(best_state["expl"])
                print(f"  → Best snapshot has {gap:.0f} mbb/dec LOWER mid-training "
                      f"LBR than final. Measure both with compare_ablations.py "
                      f"to confirm whether the snapshot is actually better.")
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