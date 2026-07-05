"""
Test suite for Kuhn Poker + CFR solver.

Tests are structured in three tiers:
    1. Game mechanics — verify Kuhn Poker rules and payoffs
    2. CFR correctness — verify solver converges and produces valid strategies
    3. Nash verification — verify solution matches known analytical equilibrium
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.games.kuhn import KuhnPoker
from src.solvers.cfr import CFRSolver
from src.analysis.convergence import ConvergenceTracker, verify_kuhn_nash


# ── Tier 1: Game Mechanics ──────────────────────────────────────────

class TestKuhnGameMechanics:
    """Verify Kuhn Poker rules are correctly implemented."""

    def setup_method(self):
        self.game = KuhnPoker()

    def test_num_players(self):
        assert self.game.num_players() == 2

    def test_initial_histories(self):
        histories = self.game.initial_histories()
        assert len(histories) == 6  # P(3,2) = 6 dealings
        total_prob = sum(p for _, p in histories)
        assert abs(total_prob - 1.0) < 1e-10

    def test_initial_histories_cover_all_dealings(self):
        dealings = {h for h, _ in self.game.initial_histories()}
        expected = {
            ("J", "Q"), ("J", "K"), ("Q", "J"),
            ("Q", "K"), ("K", "J"), ("K", "Q"),
        }
        assert dealings == expected

    def test_not_terminal_at_start(self):
        for h, _ in self.game.initial_histories():
            assert not self.game.is_terminal(h)

    def test_check_check_is_terminal(self):
        assert self.game.is_terminal(("K", "J", "c", "c"))

    def test_bet_fold_is_terminal(self):
        assert self.game.is_terminal(("K", "J", "b", "f"))

    def test_bet_call_is_terminal(self):
        assert self.game.is_terminal(("K", "J", "b", "k"))

    def test_check_bet_fold_is_terminal(self):
        assert self.game.is_terminal(("K", "J", "c", "b", "f"))

    def test_check_bet_call_is_terminal(self):
        assert self.game.is_terminal(("K", "J", "c", "b", "k"))

    def test_single_action_not_terminal(self):
        assert not self.game.is_terminal(("K", "J", "c"))
        assert not self.game.is_terminal(("K", "J", "b"))

    def test_check_bet_not_terminal(self):
        assert not self.game.is_terminal(("K", "J", "c", "b"))

    # ── Payoff tests ──

    def test_payoff_check_check_higher_wins(self):
        assert self.game.terminal_payoffs(("K", "J", "c", "c")) == (1.0, -1.0)

    def test_payoff_check_check_lower_loses(self):
        assert self.game.terminal_payoffs(("J", "K", "c", "c")) == (-1.0, 1.0)

    def test_payoff_bet_fold(self):
        assert self.game.terminal_payoffs(("J", "K", "b", "f")) == (1.0, -1.0)

    def test_payoff_bet_call_higher_wins(self):
        assert self.game.terminal_payoffs(("K", "J", "b", "k")) == (2.0, -2.0)

    def test_payoff_bet_call_lower_loses(self):
        assert self.game.terminal_payoffs(("J", "K", "b", "k")) == (-2.0, 2.0)

    def test_payoff_check_bet_fold(self):
        assert self.game.terminal_payoffs(("K", "J", "c", "b", "f")) == (-1.0, 1.0)

    def test_payoff_check_bet_call(self):
        assert self.game.terminal_payoffs(("K", "J", "c", "b", "k")) == (2.0, -2.0)

    def test_payoff_zero_sum(self):
        terminals = [
            ("K", "J", "c", "c"),
            ("K", "J", "b", "f"),
            ("K", "J", "b", "k"),
            ("K", "J", "c", "b", "f"),
            ("K", "J", "c", "b", "k"),
            ("J", "K", "c", "c"),
            ("Q", "K", "b", "k"),
        ]
        for t in terminals:
            payoffs = self.game.terminal_payoffs(t)
            assert abs(sum(payoffs)) < 1e-10, f"Non-zero-sum at {t}: {payoffs}"

    def test_player_0_acts_first(self):
        assert self.game.current_player(("K", "J")) == 0

    def test_player_1_acts_second(self):
        assert self.game.current_player(("K", "J", "c")) == 1
        assert self.game.current_player(("K", "J", "b")) == 1

    def test_player_0_acts_on_check_bet(self):
        assert self.game.current_player(("K", "J", "c", "b")) == 0

    def test_info_set_hides_opponent_card(self):
        key1 = self.game.info_set_key(("K", "J"), 0)
        key2 = self.game.info_set_key(("K", "Q"), 0)
        assert key1 == key2 == "K:"

    def test_info_set_includes_actions(self):
        key = self.game.info_set_key(("K", "J", "c", "b"), 0)
        assert key == "K:cb"

    def test_info_sets_distinct_by_card(self):
        key_k = self.game.info_set_key(("K", "J"), 0)
        key_j = self.game.info_set_key(("J", "K"), 0)
        assert key_k != key_j

    def test_legal_actions_at_root(self):
        assert self.game.legal_actions(("K", "J")) == ["c", "b"]

    def test_legal_actions_after_check(self):
        assert self.game.legal_actions(("K", "J", "c")) == ["c", "b"]

    def test_legal_actions_after_bet(self):
        assert self.game.legal_actions(("K", "J", "b")) == ["f", "k"]


# ── Tier 2: CFR Solver Correctness ──────────────────────────────────

class TestCFRSolver:
    @pytest.fixture(autouse=True)
    def setup_and_run(self):
        self.game = KuhnPoker()
        self.solver = CFRSolver(game=self.game, linear_averaging=True)
        self.tracker = ConvergenceTracker()
        # solve palauttaa nyt listan [p0_strat, p1_strat]
        self.strategy = self.solver.solve(
            iterations=5000,
            callback=self.tracker.record,
            callback_freq=500,
        )

    def test_all_info_sets_discovered(self):
        # Kuhn poker has 12 information sets
        assert len(self.strategy) == 12

    def test_strategies_are_valid_distributions(self):
        for key, strat in self.strategy.items():
            assert all(s >= -1e-10 for s in strat), f"Negative prob at {key}"
            assert abs(strat.sum() - 1.0) < 1e-6, f"Doesn't sum to 1 at {key}"

    def test_exploitability_decreases(self):
        exploits = self.tracker.exploitabilities()
        assert exploits[-1] < exploits[0]

    def test_exploitability_below_threshold(self):
        assert self.solver.exploitability() < 0.05

    def test_convergence_rate(self):
        exploits = self.tracker.exploitabilities()
        ratio = exploits[-1] / exploits[0]
        assert ratio < 0.5


# ── Tier 3: Nash Equilibrium Verification ────────────────────────────

class TestNashVerification:
    @pytest.fixture(autouse=True)
    def setup_deep(self):
        self.game = KuhnPoker()
        self.solver = CFRSolver(game=self.game, linear_averaging=True)
        self.strategy = self.solver.solve(iterations=10000)

    def test_exploitability_near_zero(self):
        exploit = self.solver.exploitability()
        assert exploit < 0.01, f"Exploitability too high: {exploit}"

    def test_nash_structural_properties(self):
        results = verify_kuhn_nash(self.strategy, tolerance=0.05)
        failures = {k: v for k, v in results.items() if not v["match"]}
        if failures:
            msg = "\n".join(
                f"  {k}: {v['detail']} (err={v['error']:.4f})"
                for k, v in failures.items()
            )
            pytest.fail(f"Nash property violations:\n{msg}")

    def test_beta_three_alpha(self):
        j_bet = self.strategy["J:"][1]
        k_bet = self.strategy["K:"][1]
        if j_bet > 0.01:
            assert abs(k_bet - 3 * j_bet) < 0.1

    def test_queen_always_checks(self):
        assert self.strategy["Q:"][0] > 0.95

    def test_king_always_calls(self):
        assert self.strategy["K:b"][1] > 0.95

    def test_jack_always_folds_to_bet(self):
        assert self.strategy["J:b"][0] > 0.95

    def test_queen_calls_one_third(self):
        call_prob = self.strategy["Q:b"][1]
        assert abs(call_prob - 1/3) < 0.05

    def test_alpha_in_valid_range(self):
        j_bet = self.strategy["J:"][1]
        assert -0.01 < j_bet < 1/3 + 0.05

    def test_game_value_converges_to_analytical(self):
        """
        Kuhn Pokerin analyyttinen peliarvo on -1/18 ≈ -0.05556 (P0:n EV).
        Tämä on vahvin yksittäinen korrektiivisuustarkistus koko solverille:
        tarttuu sekä peli- että solverilogiikan virheisiin.
        Viite: Kuhn (1950), s. 99.

        PELIARVO vs BEST-RESPONSE:
        Peliarvo on P0:n EV kun MOLEMMAT pelaavat tasapainostrategiaa — ei
        best-response-arvo. Se lasketaan suoraan strategiaprofiilin odotusarvona
        traversoimalla peli niin että kumpikin pelaaja noudattaa
        keskiarvostrategiaansa (chance-painotettuna). Tämä on info-set-turvallinen
        (toisin kuin per-historia-maksimoiva best response, joka antaisi
        pelaajalle selvänäkijän kyvyn imperfect-information-pelissä).
        """
        game     = KuhnPoker()
        solver   = CFRSolver(game=game, linear_averaging=True)
        strategy = solver.solve(iterations=20000)

        ev_p0 = _profile_value_p0(game, strategy)
        analytical = KuhnPoker.known_game_value()  # -1/18

        assert abs(ev_p0 - analytical) < 0.005, (
            f"P0 EV={ev_p0:.6f}, odotettu {analytical:.6f} "
            f"(virhe {abs(ev_p0-analytical):.6f} > 0.005)"
        )


def _profile_value_p0(game, strategy) -> float:
    """
    P0:n odotusarvo kun MOLEMMAT pelaavat annettua keskiarvostrategiaa.

    Suora strategiaprofiilin EV — ei best response. Jokaisessa solmussa
    toimiva pelaaja noudattaa info-set-avaimen mukaista strategiaansa, ja
    payoffit painotetaan strategian todennäköisyyksillä ja alkujaon
    chance-todennäköisyydellä. Info-set-turvallinen Kuhnin kaltaisille
    imperfect-information-peleille.
    """
    def ev(history):
        if game.is_terminal(history):
            return game.terminal_payoffs(history)[0]
        player  = game.current_player(history)
        actions = game.legal_actions(history)
        key     = game.info_set_key(history, player)
        strat   = strategy.get(key)
        if strat is None:
            strat = np.ones(len(actions)) / len(actions)
        return sum(
            strat[i] * ev(game.apply_action(history, a))
            for i, a in enumerate(actions)
        )

    return sum(cp * ev(h) for h, cp in game.initial_histories())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])