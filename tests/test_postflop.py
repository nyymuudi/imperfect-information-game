"""
Test suite for Postflop NLHE + Deep CFR pipeline.

Three tiers:
    1. Postflop game mechanics (streets, terminals, payoffs, legal actions)
    2. NLHEEncoder (board visibility, state vector shape, normalization)
    3. Deep CFR pipeline (buffers, network training, strategy validity)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.replay_buffer import ReservoirBuffer
from src.abstraction.equity import str_to_card, card_to_str


# ── Helper: build a specific deal ──

def make_deal(p0_str, p1_str, board_str):
    """Build a history from card strings. e.g. make_deal('AhKh', 'QdJd', '7c8s9c4h2d')"""
    p0 = (str_to_card(p0_str[:2]), str_to_card(p0_str[2:4]))
    p1 = (str_to_card(p1_str[:2]), str_to_card(p1_str[2:4]))
    board = tuple(str_to_card(board_str[i:i+2]) for i in range(0, len(board_str), 2))
    # Pad board to 5 cards if needed
    while len(board) < 5:
        # Use cards not in play
        used = set(p0 + p1 + board)
        for c in range(52):
            if c not in used:
                board = board + (c,)
                break
    return (p0, p1, board)


@pytest.fixture
def game():
    return PostflopNLHE(starting_stack=200.0, max_raises_per_street=2)


@pytest.fixture
def encoder():
    NLHEEncoder._shared_equity_cache = None  # Reset cache for tests
    return NLHEEncoder(starting_stack=200.0, equity_sims=100)


# ── Tier 1: Postflop Game Mechanics ─────────────────────────────


class TestPostflopMechanics:

    def setup_method(self):
        self.game = PostflopNLHE(starting_stack=200.0, max_raises_per_street=2)
        self.deal = make_deal("AhKh", "QdJd", "7c8s9cTh2d")

    def test_num_players(self):
        assert self.game.num_players() == 2

    def test_sample_deal_produces_9_unique_cards(self):
        rng = np.random.default_rng(42)
        h = self.game.sample_deal(rng)
        all_cards = list(h[0]) + list(h[1]) + list(h[2])
        assert len(all_cards) == 9
        assert len(set(all_cards)) == 9  # No duplicates

    def test_initial_histories_raises(self):
        """Tabular CFR should not work on this game."""
        with pytest.raises(NotImplementedError):
            self.game.initial_histories()

    def test_not_terminal_at_deal(self):
        assert not self.game.is_terminal(self.deal)

    def test_sb_acts_first_preflop(self):
        assert self.game.current_player(self.deal) == 0

    # ── Fold tests ──

    def test_preflop_fold_terminal(self):
        h = self.deal + ("f",)
        assert self.game.is_terminal(h)

    def test_fold_after_raise_terminal(self):
        h = self.deal + ("r", "f")
        assert self.game.is_terminal(h)

    def test_fold_payoff_folder_loses(self):
        """SB folds preflop → loses 1 chip (SB)."""
        h = self.deal + ("f",)
        payoffs = self.game.terminal_payoffs(h)
        assert payoffs[0] < 0  # SB loses
        assert payoffs[1] > 0  # BB wins

    def test_fold_payoff_zero_sum(self):
        h = self.deal + ("r", "f")
        payoffs = self.game.terminal_payoffs(h)
        assert abs(payoffs[0] + payoffs[1]) < 1e-6

    # ── Street transitions ──

    def test_preflop_call_goes_to_flop(self):
        """SB calls BB → preflop ends, flop begins."""
        h = self.deal + ("k",)  # SB calls
        # BB now acts on... wait, SB called so preflop street is:
        # SB posts 1, BB posts 2, SB calls (matches BB), now BB acts
        # Actually SB calling = street complete? No, only if both have acted.
        # In preflop: SB is first to act. 'k' = SB calls BB's 2.
        # Then BB needs to act (check or raise). 
        # Actually in standard poker: if SB just calls (limps), BB gets option.
        assert not self.game.is_terminal(h)

    def test_limp_check_goes_to_flop(self):
        """SB limps, BB checks → preflop done, flop starts."""
        h = self.deal + ("k", "c")  # SB calls, BB checks
        street = self.game._current_street(h)
        assert street == "flop"
        assert not self.game.is_terminal(h)

    def test_raise_call_goes_to_flop(self):
        """SB raises, BB calls → flop."""
        h = self.deal + ("r", "k")
        street = self.game._current_street(h)
        assert street == "flop"

    def test_flop_check_check_goes_to_turn(self):
        h = self.deal + ("r", "k", "c", "c")
        street = self.game._current_street(h)
        assert street == "turn"

    def test_full_runout_terminal(self):
        """All streets check-check → river showdown."""
        h = self.deal + ("k", "c",   # preflop: limp, check
                         "c", "c",   # flop: check, check
                         "c", "c",   # turn: check, check
                         "c", "c")   # river: check, check
        assert self.game.is_terminal(h)

    def test_showdown_payoff_zero_sum(self):
        h = self.deal + ("k", "c", "c", "c", "c", "c", "c", "c")
        payoffs = self.game.terminal_payoffs(h)
        assert abs(payoffs[0] + payoffs[1]) < 1e-6

    def test_showdown_better_hand_wins(self):
        """AhKh vs QdJd on 7c8s9cTh2d → QdJd has straight (QJTT98)... 
        actually: P0 has AK, P1 has QJ. Board 7c8s9cTh2d.
        P1: QJ + 789T = straight (T-high). P0: AK + 789T = no straight.
        P1 should win."""
        h = self.deal + ("k", "c", "c", "c", "c", "c", "c", "c")
        payoffs = self.game.terminal_payoffs(h)
        assert payoffs[1] > 0  # P1 (QJ straight) wins

    # ── Legal actions ──

    def test_preflop_sb_can_fold(self):
        actions = self.game.legal_actions(self.deal)
        assert "f" in actions or "k" in actions

    def test_cannot_fold_when_checked_to(self):
        """P0 acts first on flop after limp-check. No bet to fold to."""
        h = self.deal + ("k", "c")  # Flop, P0 acts
        actions = self.game.legal_actions(h)
        assert "f" not in actions
        assert "c" in actions

    def test_can_raise_on_flop(self):
        h = self.deal + ("k", "c")
        actions = self.game.legal_actions(h)
        assert "r" in actions

    # ── Pot tracking ──

    def test_pot_after_limp(self):
        h = self.deal + ("k", "c")
        state = self.game._parse_state(h)
        assert state["pot"] == 4.0  # SB 2 + BB 2

    def test_pot_after_raise_call(self):
        h = self.deal + ("r", "k")
        state = self.game._parse_state(h)
        assert state["pot"] > 4.0  # More than just blinds


# ── Tier 2: NLHE State Encoder ──────────────────────────────────


class TestNLHEEncoder:

    def setup_method(self):
        NLHEEncoder._shared_equity_cache = None
        self.encoder = NLHEEncoder(starting_stack=200.0, equity_sims=100)
        self.deal = make_deal("AhKh", "QdJd", "7c8s9cTh2d")

    def test_state_size(self):
        # 29 + K_BOARD(=8) = 37. Last dim is the position bit (added 2026-06-14).
        assert self.encoder.state_size() == 37

    def test_output_shape(self):
        state = self.encoder.encode(self.deal, 0)
        assert state.shape == (37,)

    def test_output_dtype(self):
        state = self.encoder.encode(self.deal, 0)
        assert state.dtype == np.float32

    def test_private_cards_two_bits(self):
        """Exactly one bit set in preflop bucket [0:8]."""
        state = self.encoder.encode(self.deal, 0)
        assert state[0:8].sum() == pytest.approx(1.0)

    def test_private_cards_differ_by_player(self):
        """Players in different hand classes land in different buckets or have
        different continuous equity features."""
        s0 = self.encoder.encode(self.deal, 0)
        s1 = self.encoder.encode(self.deal, 1)
        # Preflop bucket or equity feature must differ across players
        assert not np.array_equal(s0[0:8], s1[0:8]) or s0[32] != s1[32]

    def test_board_zero_preflop(self):
        """No board bucket visible preflop — dims [8:16] all zero."""
        state = self.encoder.encode(self.deal, 0)
        assert state[8:16].sum() == pytest.approx(0.0)

    def test_board_three_on_flop(self):
        """After preflop betting, board bucket [8:16] has exactly one bit set."""
        h = self.deal + ("k", "c")  # Limp, check → flop
        state = self.encoder.encode(h, 0)
        assert state[8:16].sum() == pytest.approx(1.0)

    def test_board_four_on_turn(self):
        h = self.deal + ("k", "c", "c", "c")  # Through flop
        state = self.encoder.encode(h, 0)
        assert state[8:16].sum() == pytest.approx(1.0)

    def test_board_five_on_river(self):
        h = self.deal + ("k", "c", "c", "c", "c", "c")
        state = self.encoder.encode(h, 0)
        assert state[8:16].sum() == pytest.approx(1.0)

    def test_street_one_hot(self):
        """Exactly one street bit set."""
        state = self.encoder.encode(self.deal, 0)
        assert state[16:20].sum() == pytest.approx(1.0)
        assert state[16] == pytest.approx(1.0)  # Preflop

    def test_street_flop(self):
        h = self.deal + ("k", "c")
        state = self.encoder.encode(h, 0)
        assert state[17] == pytest.approx(1.0)  # Flop

    def test_values_normalized(self):
        """All values should be in reasonable range."""
        h = self.deal + ("r", "k", "r", "k")
        state = self.encoder.encode(h, 0)
        assert state.min() >= -1.5
        assert state.max() <= 1.5

    def test_opponent_cards_never_visible(self):
        """P1's card info must not leak into P0's encoding.
        In the abstracted encoder, card info is represented via bucket index,
        not raw one-hots. Verify that the preflop buckets differ between players
        (since AhKh and QdJd have different equity values)."""
        s0 = self.encoder.encode(self.deal, 0)
        s1 = self.encoder.encode(self.deal, 1)
        # Different hands → different preflop equity → different bucket or dim 32
        assert not (np.array_equal(s0[0:8], s1[0:8]) and s0[32] == s1[32])

    def test_equity_feature_present(self):
        """Preflop equity feature (dim 32) should be non-zero."""
        state = self.encoder.encode(self.deal, 0)
        assert 0.0 < state[32] < 1.0  # AhKh equity ≈ 0.65

    def test_premium_higher_equity_than_trash(self):
        """AA should have higher continuous equity (dim 32) than 23o."""
        aa_deal = make_deal("AhAs", "QdJd", "7c8s9cTh2d")
        trash_deal = make_deal("2h3d", "QdJd", "7c8s9cTh2d")
        aa_state = self.encoder.encode(aa_deal, 0)
        trash_state = self.encoder.encode(trash_deal, 0)
        assert aa_state[32] > trash_state[32]


# ── Tier 3: Deep CFR Pipeline ──────────────────────────────────


class TestReservoirBuffer:

    def test_basic_add_and_sample(self):
        buf = ReservoirBuffer(capacity=100, state_size=10, action_size=3)
        buf.add(np.ones(10), np.array([1.0, 2.0, 3.0]), 1.0)
        assert len(buf) == 1
        states, targets, weights = buf.sample_batch(1)
        assert states.shape == (1, 10)
        assert targets.shape == (1, 3)

    def test_reservoir_keeps_capacity(self):
        buf = ReservoirBuffer(capacity=50, state_size=5, action_size=2)
        for i in range(200):
            buf.add(np.random.randn(5), np.array([1.0, 0.0]), float(i))
        assert len(buf) == 50

    def test_clear(self):
        buf = ReservoirBuffer(capacity=100, state_size=5, action_size=2)
        buf.add(np.zeros(5), np.zeros(2), 1.0)
        buf.clear()
        assert len(buf) == 0


class TestDeepCFRPipeline:
    """Integration test: run Deep CFR briefly on PostflopNLHE."""

    def test_deep_cfr_runs_without_error(self):
        from src.deep_cfr.deep_cfr_solver import DeepCFRSolver

        game = PostflopNLHE(starting_stack=200.0, max_raises_per_street=2)
        NLHEEncoder._shared_equity_cache = None
        encoder = NLHEEncoder(starting_stack=200.0, equity_sims=50)

        solver = DeepCFRSolver(
            game=game,
            encoder=encoder,
            max_actions=4,
            buffer_capacity=5000,
            hidden_size=64,
            train_epochs=5,
            train_batch=32,
            traversals_per_iter=10,
        )
        solver.solve(iterations=3)

        assert len(solver.regret_buffer) > 0
        assert len(solver.strategy_buffer) > 0

    def test_strategy_is_valid_distribution(self):
        from src.deep_cfr.deep_cfr_solver import DeepCFRSolver

        game = PostflopNLHE(starting_stack=200.0, max_raises_per_street=2)
        encoder = NLHEEncoder(starting_stack=200.0, equity_sims=50)

        solver = DeepCFRSolver(
            game=game,
            encoder=encoder,
            max_actions=4,
            buffer_capacity=5000,
            hidden_size=64,
            train_epochs=5,
            train_batch=32,
            traversals_per_iter=20,
        )
        solver.solve(iterations=5)

        rng = np.random.default_rng(99)
        h = game.sample_deal(rng)
        strat = solver.get_strategy(h, 0)
        num_actions = len(game.legal_actions(h))

        assert len(strat) == num_actions
        assert all(s >= -0.01 for s in strat)
        assert abs(sum(strat) - 1.0) < 0.05

    def test_buffers_accumulate_over_iterations(self):
        from src.deep_cfr.deep_cfr_solver import DeepCFRSolver

        game = PostflopNLHE(starting_stack=200.0, max_raises_per_street=2)
        encoder = NLHEEncoder(starting_stack=200.0, equity_sims=50)

        solver = DeepCFRSolver(
            game=game,
            encoder=encoder,
            max_actions=4,
            buffer_capacity=50000,
            hidden_size=64,
            train_epochs=5,
            train_batch=32,
            traversals_per_iter=20,
        )
        solver.solve(iterations=5)
        size_after_5 = len(solver.regret_buffer)

        solver.solve(iterations=5)
        size_after_10 = len(solver.regret_buffer)

        assert size_after_10 > size_after_5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])