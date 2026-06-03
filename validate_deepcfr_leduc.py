#!/usr/bin/env python3
"""
validate_deepcfr_leduc.py

Ajaa PROJEKTIN oman DeepCFRSolverin Leduc Hold'emilla ja mittaa konvergenssin
tabulaarista totuutta (~0.138) vasten. Tama mittaus on PUUTTUNUT projektista:
DeepCFRSolveria on ajettu vain NLHE:lla (train_postflop.py), jossa ei ole
tunnettua vastausta -- joten sen konvergenssia ei ole koskaan validoitu pelilla
jolla on tabulaarinen totuus. Tama on todennakoisesti suurin syy siihen miksi
solverin viat ovat jaaneet piiloon: niita ei voi mitata siella missa solveria
ajetaan.

Kaikki palat ovat jo projektissa:
  - LeducHoldem            (src/games/leduc.py)
  - LeducEncoder           (src/deep_cfr/state_encoder.py, 20-dim, valmis)
  - DeepCFRSolver          (src/deep_cfr/deep_cfr_solver.py, use_cpp_engine=False)
  - CFRSolver              (src/solvers/cfr.py, tabulaarinen totuus + exploitability)

Skripti rakentaa Leduc-yhteensopivan Blueprintin solverin verkoista, syottaa sen
keskistrategian CFRSolverin EKSAKTIIN best-response-exploitabilityyn (sama metodi
jota referenssi kaytti), ja tulostaa konvergenssikayran.

Kaytto:
  python3 validate_deepcfr_leduc.py
  python3 validate_deepcfr_leduc.py --iterations 500 --traversals 1000

Tulkinta:
  - avg_expl laskee kohti tabulaarisen MCCFR:n kayraa (1000 iter -> ~1.26,
    10000 -> ~0.37) tai paremmin -> solver on terve, konvergoi.
  - avg_expl jaa ~5:een tai nousee -> solverissa on vika, ja nyt se on
    MITATTAVISSA pelilla jolla on tunnettu vastaus. Vertaa referenssin
    loyytamiin vikoihin: regret-kohteen muoto, CFR+-klippaus, buffer-moodi.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.deep_cfr.state_encoder import LeducEncoder
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver


def deepcfr_strategy_exploitability(game, solver, encoder):
    """
    Eksakti exploitability solverin NYKYISESTA regret-matching-strategiasta.

    Rakentaa CFRSolverin joka kantaa solverin regret-verkon tuottaman strategian
    jokaisessa infosetissa, ja kayttaa sen eksaktia best-responsea. Sama metodi
    kuin referenssin current_strategy_exploitability -- isoloi sen mita verkko
    on oppinut.
    """
    ref = CFRSolver(game=game, linear_averaging=True)

    def walk(history):
        if game.is_terminal(history):
            return
        player = game.current_player(history)
        acts = game.legal_actions(history)
        key = game.info_set_key(history, player)
        if key not in ref.info_sets:
            state = encoder.encode(history, player)
            strat = solver._get_regret_strategy(state, len(acts))
            data = ref._get_or_create_info_set(key, acts)
            data.cumulative_strategy = np.asarray(strat, dtype=np.float64).copy()
        for a in acts:
            walk(game.apply_action(history, a))

    for init_h, _ in game.initial_histories():
        walk(init_h)
    return ref.exploitability()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--traversals", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--measure-every", type=int, default=25)
    ap.add_argument("--buffer", type=int, default=500_000,
                    help="regret/strategy buffer capacity (test: 5_000_000 = ei tayty)")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    game = LeducHoldem()
    encoder = LeducEncoder()

    # Tabulaarinen totuus vertailukohdaksi.
    tab = CFRSolver(game=game, linear_averaging=True)
    tab.solve(iterations=1000)
    ref = tab.exploitability()
    print(f"[reference] tabular CFR exploitability = {ref:.5f}")
    print(f"[reference] tabular MCCFR @1000 ~1.26, @10000 ~0.37 (Deep CFR target band)\n")

    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,                # Leduc: c, r, f, k
        buffer_capacity=args.buffer,
        hidden_size=args.hidden,
        train_epochs=args.epochs,
        traversals_per_iter=args.traversals,
        use_cpp_engine=False,         # puhdas Python -- ei C++-riippuvuutta
        device="cpu",
        lr=1e-3,
    )

    print(f"PROJECT DeepCFRSolver | Leduc | hidden={args.hidden} "
          f"traversals={args.traversals} epochs={args.epochs} buffer={args.buffer:,}")
    print(f"{'iter':>6} {'cur_expl':>10} {'ref':>8} {'ratio':>7} {'reg_buf':>9} "
          f"{'loss':>10} {'t(s)':>7}")

    t0 = time.time()

    def callback(s, i):
        cur = deepcfr_strategy_exploitability(game, s, encoder)
        ratio = cur / ref if ref > 0 else float("nan")
        loss = getattr(s, "_last_regret_loss", float("nan"))
        print(f"{i:>6} {cur:>10.5f} {ref:>8.5f} {ratio:>7.2f} "
              f"{len(s.regret_buffer):>9,} {loss:>10.4f} {time.time()-t0:>7.1f}")

    solver.solve(
        iterations=args.iterations,
        callback=callback,
        callback_freq=args.measure_every,
    )

    print(f"\nValmis. Jos cur_expl laskee kohti tabulaarista kayraa, projektin")
    print(f"DeepCFRSolver konvergoi Leducilla -- ja tama skripti on pysyva")
    print(f"regressiotesti. Jos se jaa korkealle, vika on nyt mitattavissa")
    print(f"tunnettua vastausta vasten (vrt. regret-kohde / CFR+ / buffer-moodi).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
