"""
Cross-implementation parity tests: Python NLHEEncoder vs C++ NLHEStateEncoder.

Käyttää cfr_engine.make_nlhe_deal() + NLHEGame.initial_state() + NLHEStateEncoder.encode_vec()
— kaikki eksportattu bindings.cpp:stä.

Hypätään yli jos cfr_engine.so ei ole saatavilla.
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


# ── Apufunktiot ──────────────────────────────────────────────────────────────

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
    """
    Rakenna NLHEState C++-puolella.
    actions: tuple/list of int — muunnetaan NLHEAction-enumeiksi automaattisesti.
    """
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
        # pybind11-enum ei automuunnu int:stä — eksplisiittinen cast pakollinen
        state = _eng.NLHEGame.apply_action(state, _eng.NLHEAction(a))
    return state


def make_python_vec(hole0_str, hole1_str, board_str, actions, player):
    from src.games.postflop_nlhe import PostflopNLHE
    NLHEEncoder._shared_equity_cache = None
    enc  = NLHEEncoder(starting_stack=200.0, equity_sims=200)

    h0    = tuple(_cards(hole0_str))
    h1    = tuple(_cards(hole1_str))
    board = tuple(_cards(board_str)) if board_str else ()
    full  = tuple(_full_board(list(h0), list(h1), list(board)))

    history = (h0, h1, full) + tuple(actions)
    return enc.encode(history, player)


# ── TestEncoderParity ────────────────────────────────────────────────────────

class TestEncoderParity:

    @cpp_required
    def test_state_size_is_124(self):
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert len(vec) == 124

    @cpp_required
    def test_hole_card_bits_player0(self):
        """Dim 0-51: pelaajan 0 kortit merkitty 1.0:ksi, vastustajan 0.0."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        ah, kh = str_to_card("Ah"), str_to_card("Kh")
        qd, jd = str_to_card("Qd"), str_to_card("Jd")
        assert vec[ah] == pytest.approx(1.0)
        assert vec[kh] == pytest.approx(1.0)
        assert vec[qd] == pytest.approx(0.0)
        assert vec[jd] == pytest.approx(0.0)

    @cpp_required
    def test_street_one_hot_preflop(self):
        """Dim 104-107: preflop → [1,0,0,0]."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[104] == pytest.approx(1.0)
        assert vec[105] == pytest.approx(0.0)
        assert vec[106] == pytest.approx(0.0)
        assert vec[107] == pytest.approx(0.0)

    @cpp_required
    def test_pot_odds_preflop(self):
        """Dim 122: pot odds preflopilla.
        SB=1 on jo postattu, BB=2 on jo postattu → pot=3, to_call=1.
        pot_odds = to_call / (pot + to_call) = 1/4 = 0.25
        """
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[122] == pytest.approx(0.25, abs=ATOL)

    @cpp_required
    def test_spr_full_stack(self):
        """Dim 123: SPR täydellä stackilla → yli 10 → normalisoitu 1.0."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert vec[123] == pytest.approx(1.0, abs=ATOL)

    @cpp_required
    def test_board_bits_zero_preflop(self):
        """Dim 52-103: preflopilla ei board-kortteja → kaikki 0."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        vec   = _eng.NLHEStateEncoder.encode_vec(state, 0)
        assert all(v == pytest.approx(0.0) for v in vec[52:104])

    @cpp_required
    def test_cpp_python_card_bits_match(self):
        """Kriittisin testi: Python ja C++ tuottavat identtiset korttibitit (dim 0-103)."""
        state   = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(state, 0), dtype=np.float32)
        py_vec  = make_python_vec("AhKh", "QdJd", "7c8s9c", (), player=0)
        np.testing.assert_allclose(
            cpp_vec[:104], py_vec[:104], atol=ATOL,
            err_msg="Korttibitti-mismatch dim 0-103"
        )

    @cpp_required
    def test_cpp_python_betting_scalars_match(self):
        """Dim 108-111: pot, to_call, my_stack, opp_stack."""
        state   = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(state, 0), dtype=np.float32)
        py_vec  = make_python_vec("AhKh", "QdJd", "7c8s9c", (), player=0)
        np.testing.assert_allclose(
            cpp_vec[108:112], py_vec[108:112], atol=ATOL,
            err_msg="Betting scalar mismatch dim 108-111"
        )

    @cpp_required
    def test_cpp_python_street_one_hot_match(self):
        """Dim 104-107: street one-hot."""
        state   = make_cpp_state("AhKh", "QdJd", "7c8s9c")
        cpp_vec = np.array(_eng.NLHEStateEncoder.encode_vec(state, 0), dtype=np.float32)
        py_vec  = make_python_vec("AhKh", "QdJd", "7c8s9c", (), player=0)
        np.testing.assert_allclose(
            cpp_vec[104:108], py_vec[104:108], atol=ATOL,
            err_msg="Street one-hot mismatch dim 104-107"
        )


# ── TestLegalActionsParity ───────────────────────────────────────────────────

class TestLegalActionsParity:

    @cpp_required
    def test_preflop_call_is_legal(self):
        """Preflopilla CALL on heti laillinen: BB toimii bettinä (SB limp/call/raise)."""
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
        """Raisen jälkeen: CALL on sallittu."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c",
                               actions=[int(_eng.NLHE_RAISE)])
        acts  = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_CALL) in acts
        assert int(_eng.NLHE_FOLD_OR_CHECK) in acts

    @cpp_required
    def test_all_in_opponent_can_fold_or_call(self):
        """All-inin jälkeen: vastustajalla fold tai call."""
        state = make_cpp_state("AhKh", "QdJd", "7c8s9c",
                               actions=[int(_eng.NLHE_ALL_IN)])
        acts  = [int(a) for a in _eng.NLHEGame.legal_actions(state)]
        assert int(_eng.NLHE_FOLD_OR_CHECK) in acts
        assert int(_eng.NLHE_CALL) in acts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
