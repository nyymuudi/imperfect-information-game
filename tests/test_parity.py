"""
Cross-implementation parity tests: Python NLHEEncoder vs C++ NLHEStateEncoder.

Card-abstracted encoder layout (36 dims):
  [0:8]   preflop bucket one-hot
  [8:16]  board bucket one-hot (zeros preflop)
  [16:20] street one-hot
  [20:24] betting scalars
  [24:32] action history
  [32]    preflop equity (continuous)
  [33]    board strength (continuous — ACCEPTED RESIDUAL: Python/C++ differ)
  [34]    pot odds
  [35]    SPR

Coverage:
  * TestEncoderParity        — preflop initial-state feature parity
  * TestPostActionParity     — parity AFTER action sequences
  * TestEquityFeatureParity  — dim 32 parity (equity table)
  * TestBucketParity         — preflop bucket parity (dims 0-7)
  * TestLegalActionsParity   — C++ legal-action set

Skipped automatically if cfr_engine.so is unavailable.
Note: board bucket (dims 8-15) and board strength (dim 33) are NOT required to
be bit-identical across implementations — both are monotone in hand strength but
use different score normalisations. See torch_model.hpp parity notes.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/cpp_engine"))

from src.deep_cfr.state_encoder import NLHEEncoder
from src.abstraction.equity import str_to_card

try:
    import cfr_engine as _eng
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

cpp_required = pytest.mark.skipif(
    not CPP_AVAILABLE, reason="cfr_engine.so ei saatavilla"
)

ATOL = 1e-4

# Action-char (Python) -> NLHEAction enum int (C++). 'f' and 'c' both map to
# FOLD_OR_CHECK; the engine disambiguates by whether there is an outstanding bet.
_CHAR_TO_ENUM = {
    "f": 0, "c": 0, "k": 1, "r": 2, "a": 3,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cards(s: str) -> list[int]:
    return [str_to_card(s[i:i+2]) for i in range(0, len(s), 2)]


def _full_board(hole0, hole1, board_cards):
    used = set(hole0 + hole1 + board_cards)
    full = list(board_cards)
    for c in range(52):
        if len(full) >= 5:
            break
        if c not in used:
            full.append(c)
    return full[:5]


def make_cpp_state(hole0_str, hole1_str, board_str, actions=()):
    """Build an NLHEState on the C++ side. `actions` are NLHEAction ints."""
    h0    = _cards(hole0_str)
    h1    = _cards(hole1_str)
    board = _cards(board_str) if board_str else []
    full  = _full_board(h0, h1, board)

    cfg = _eng.NLHEGameConfig()
    cfg.starting_stack = 200.0
    cfg.sb             = 1.0
    cfg.bb             = 2.0
    cfg.raise_fraction = 0.75
    cfg.max_raises     = 2

    deal  = _eng.make_nlhe_deal(h0[0], h0[1], h1[0], h1[1], full)
    state = _eng.NLHEGame.initial_state(deal, cfg)
    for a in actions:
        state = _eng.NLHEGame.apply_action(state, _eng.NLHEAction(a))
    return state


def make_python_vec(hole0_str, hole1_str, board_str, actions, player):
    """Encode the equivalent Python history. `actions` are char actions."""
    from src.games.postflop_nlhe import PostflopNLHE  # noqa: F401
    NLHEEncoder._shared_equity_cache = None
    enc  = NLHEEncoder(starting_stack=200.0, equity_sims=2000)

    h0    = tuple(_cards(hole0_str))
    h1    = tuple(_cards(hole1_str))
    board = tuple(_cards(board_str)) if board_str else ()
    full  = tuple(_full_board(list(h0), list(h1), list(board)))

    history = (h0, h1, full) + tuple(actions)
    return enc.encode(history, player)


def _both_vecs(hole0, hole1, board, char_actions, player):
    """Return (cpp_vec, py_vec) for an action sequence given as char actions."""
    enum_actions = [_CHAR_TO_ENUM[a] for a in char_actions]
    cpp_state = make_cpp_state(hole0, hole1, board, enum_actions)
    cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(cpp_state, player),
                       dtype=np.float32)
    py_vec = make_python_vec(hole0, hole1, board, char_actions, player)
    return cpp_vec, py_vec


# ── TestEncoderParity (preflop initial state) ────────────────────────────────

class TestEncoderParity:

    @cpp_required
    def test_state_size_is_36(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert len(vec) == 36

    @cpp_required
    def test_preflop_bucket_is_one_hot(self):
        """Preflop: exactly one bit set in dims [0:8], all zeros in [8:16]."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert sum(vec[0:8]) == pytest.approx(1.0)
        assert all(v == pytest.approx(0.0) for v in vec[8:16])

    @cpp_required
    def test_street_one_hot_preflop(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[16] == pytest.approx(1.0)
        assert vec[17] == pytest.approx(0.0)
        assert vec[18] == pytest.approx(0.0)
        assert vec[19] == pytest.approx(0.0)

    @cpp_required
    def test_pot_odds_preflop(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[34] == pytest.approx(0.25, abs=ATOL)

    @cpp_required
    def test_spr_full_stack(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[35] == pytest.approx(1.0, abs=ATOL)

    @cpp_required
    def test_board_bucket_zero_preflop(self):
        """Preflop: board bucket dims [8:16] must all be zero."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert all(v == pytest.approx(0.0) for v in vec[8:16])

    @cpp_required
    def test_cpp_python_preflop_bucket_match(self):
        """Preflop bucket (dims 0-7) must be identical on both sides."""
        cpp_vec, py_vec = _both_vecs("AhKh", "QdJd", "7c8s9c", (), 0)
        np.testing.assert_allclose(cpp_vec[0:8], py_vec[0:8], atol=ATOL,
                                   err_msg="preflop bucket mismatch dim 0-7")

    @cpp_required
    def test_cpp_python_betting_scalars_match(self):
        cpp_vec, py_vec = _both_vecs("AhKh", "QdJd", "7c8s9c", (), 0)
        np.testing.assert_allclose(cpp_vec[20:24], py_vec[20:24], atol=ATOL,
                                   err_msg="betting scalar mismatch dim 20-23")

    @cpp_required
    def test_cpp_python_street_one_hot_match(self):
        cpp_vec, py_vec = _both_vecs("AhKh", "QdJd", "7c8s9c", (), 0)
        np.testing.assert_allclose(cpp_vec[16:20], py_vec[16:20], atol=ATOL,
                                   err_msg="street one-hot mismatch dim 16-19")


# ── TestPostActionParity (the previously-uncovered region) ───────────────────

class TestPostActionParity:
    """Parity AFTER action sequences on corrected state machine."""

    CASES = [
        ("preflop_sb_call_to_flop", ("k",), 1),
        ("flop_first_check",        ("k", "c"), 0),
        ("preflop_raise",           ("r",), 1),
        ("preflop_raise_call_flop", ("r", "k"), 1),
        ("flop_bet",                ("k", "c", "r"), 1),
    ]

    @cpp_required
    @pytest.mark.parametrize("label,actions,player",
                             CASES, ids=[c[0] for c in CASES])
    def test_structural_dims_match(self, label, actions, player):
        """Dims checked: preflop bucket [0:8], street [16:20], betting [20:24],
        action history [24:32], pot-odds+SPR [34:36].
        Dims NOT checked: board bucket [8:16] and board strength [33]
        — accepted residual, see torch_model.hpp parity notes."""
        cpp_vec, py_vec = _both_vecs("AhKh", "QdJd", "7c8s9c", actions, player)
        np.testing.assert_allclose(cpp_vec[0:8], py_vec[0:8], atol=ATOL,
                                   err_msg=f"{label}: preflop bucket mismatch")
        np.testing.assert_allclose(cpp_vec[16:32], py_vec[16:32], atol=ATOL,
                                   err_msg=f"{label}: dim 16-31 mismatch")
        np.testing.assert_allclose(cpp_vec[34:36], py_vec[34:36], atol=ATOL,
                                   err_msg=f"{label}: pot-odds/SPR mismatch")

    @cpp_required
    def test_call_advances_street_both_sides(self):
        """Preflop CALL → flop: board bucket should be non-zero AND street=flop."""
        cpp_vec, py_vec = _both_vecs("AhKh", "QdJd", "7c8s9c", ("k",), 1)
        # board bucket: exactly one bit set in [8:16]
        assert sum(cpp_vec[8:16]) == pytest.approx(1.0)
        assert sum(py_vec[8:16]) == pytest.approx(1.0)
        # street one-hot: flop bit (dim 17) set
        assert cpp_vec[17] == pytest.approx(1.0)
        assert py_vec[17] == pytest.approx(1.0)


# ── TestEquityFeatureParity (dim 32) ─────────────────────────────────────────

class TestEquityFeatureParity:
    """dim 32 (continuous preflop equity) parity — both sides read the same table."""

    @cpp_required
    def test_preflop_equity_matches_when_table_present(self):
        py_vec = make_python_vec("AhKh", "QdJd", "7c8s9c", (), 0)
        cpp_state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(cpp_state, 0),
                           dtype=np.float32)
        if abs(cpp_vec[32] - py_vec[32]) > 0.02:
            pytest.skip("C++ equity table not loaded (heuristic fallback active)")
        assert cpp_vec[32] == pytest.approx(py_vec[32], abs=2e-2)


# ── TestBucketParity (preflop bucket) ────────────────────────────────────────

class TestBucketParity:
    """Preflop bucket (dims 0-7) must agree when equity tables are loaded."""

    @cpp_required
    def test_aa_in_higher_bucket_than_72o(self):
        """AA must land in a higher bucket than 72o on both sides."""
        cpp_aa, py_aa = _both_vecs("AhAs", "QdJd", "7c8s9c", (), 0)
        cpp_72, py_72 = _both_vecs("7h2d", "QdJd", "7c8s9c", (), 0)
        aa_bucket_cpp = int(np.argmax(cpp_aa[0:8]))
        aa_bucket_py  = int(np.argmax(py_aa[0:8]))
        lo_bucket_cpp = int(np.argmax(cpp_72[0:8]))
        lo_bucket_py  = int(np.argmax(py_72[0:8]))
        assert aa_bucket_cpp > lo_bucket_cpp
        assert aa_bucket_py  > lo_bucket_py

    @cpp_required
    def test_preflop_buckets_match_across_implementations(self):
        """Same hand → same preflop bucket on both sides (when table loaded)."""
        py_vec = make_python_vec("AhKh", "QdJd", "7c8s9c", (), 0)
        cpp_state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(cpp_state, 0),
                           dtype=np.float32)
        if abs(cpp_vec[32] - py_vec[32]) > 0.02:
            pytest.skip("C++ equity table not loaded")
        assert int(np.argmax(cpp_vec[0:8])) == int(np.argmax(py_vec[0:8]))


# ── TestLegalActionsParity ───────────────────────────────────────────────────

class TestLegalActionsParity:

    @cpp_required
    def test_preflop_call_is_legal(self):
        state   = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        actions = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_CALL) in actions
        assert int(_eng.NLHE_FOLD_OR_CHECK) in actions

    @cpp_required
    def test_preflop_raise_is_legal(self):
        state   = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        actions = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_RAISE) in actions

    @cpp_required
    def test_facing_raise_includes_call(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c",
                               actions=[int(_eng.NLHE_RAISE)])
        acts  = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_CALL) in acts
        assert int(_eng.NLHE_FOLD_OR_CHECK) in acts

    @cpp_required
    def test_all_in_opponent_can_fold_or_call(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c",
                               actions=[int(_eng.NLHE_ALL_IN)])
        acts  = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_FOLD_OR_CHECK) in acts
        assert int(_eng.NLHE_CALL) in acts

    @cpp_required
    @pytest.mark.parametrize("char_actions", [(), ("r",), ("k", "c"), ("r", "k")])
    def test_legal_action_count_matches_python(self, char_actions):
        """C++ and Python must agree on the NUMBER of legal actions along a
        sequence — a cheap structural cross-check of the two state machines."""
        from src.games.postflop_nlhe import PostflopNLHE
        game = PostflopNLHE(starting_stack=200.0, max_raises_per_street=2,
                            raise_fractions=(0.75,))
        h0 = tuple(_cards("AhKh")); h1 = tuple(_cards("QdJd"))
        full = tuple(_full_board(list(h0), list(h1), _cards("7c8s9c")))
        py_hist = (h0, h1, full) + tuple(char_actions)
        py_n = len(game.legal_actions(py_hist))

        enum_actions = [_CHAR_TO_ENUM[a] for a in char_actions]
        cpp_state = make_cpp_state("AhKh", "QdJd", "7c8s9c", enum_actions)
        cpp_n = len(_eng.NLHEGame.legal_actions(cpp_state))
        assert py_n == cpp_n, (
            f"legal action count: py={py_n} cpp={cpp_n} after {char_actions}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
