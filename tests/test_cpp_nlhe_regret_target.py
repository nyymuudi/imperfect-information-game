"""
Parity test: C++ NLHEMCCFREngine CFR+ target == Python NLHEFullCfrPlusRef.

Validates the NLHE port of the CFR+ regret target on a SMALL ENUMERABLE subtree
(low stack, max_raises=1, one fixed deal) — NLHE's full tree is not enumerable,
so we follow the project's inductive method: prove correctness on a tiny exact
instance before the large acceptance run.

Both sides:
  - deterministic full (vanilla) traversal, uniform strategy, one fixed deal
  - CFR+ accumulation keyed on the EXACT state vector (encoder output)
  - emit R[slot]/visits for all 4 action slots per state
  - collapse by exact state vector into an [m, 4] regret-target matrix

We compare the collapsed [m,4] matrices. Because both group by the bit-identical
encoder output, rows are matched by state vector (not by key string), so the
comparison is implementation-independent w.r.t. info_set_key formatting.

Skipped automatically if cfr_engine.so is unavailable. Run after rebuilding:

    cd src/cpp_engine && bash scripts/build.sh && cd ../.. \
        && pytest tests/test_cpp_nlhe_regret_target.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/cpp_engine"))

from src.abstraction.equity import str_to_card as C
# nlhe_full_cfrplus_ref.py lives at the repo root; the parent dir is already on
# sys.path (inserted above), so this import resolves regardless of CWD.
from nlhe_full_cfrplus_ref import NLHEFullCfrPlusRef, make_ref, _state_key  # noqa: E402

try:
    import cfr_engine as _eng
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

cpp_required = pytest.mark.skipif(
    not CPP_AVAILABLE, reason="cfr_engine.so ei saatavilla"
)

ATOL = 1e-3   # looser than Leduc: NLHE encoder has equity MC + larger payoffs
STACK = 10.0
MAX_RAISES = 1

# Fixed deal: AhKh vs QdJd on a fixed 5-card board.
DEAL_P0 = ("Ah", "Kh")
DEAL_P1 = ("Qd", "Jd")
DEAL_BOARD = ("7c", "8s", "9c", "2h", "3d")


def _deal_ints():
    p0 = tuple(C(s) for s in DEAL_P0)
    p1 = tuple(C(s) for s in DEAL_P1)
    board = tuple(C(s) for s in DEAL_BOARD)
    return p0, p1, board


def _python_matrix():
    game, enc = make_ref(stack=STACK, max_raises=MAX_RAISES, equity_sims=200)
    p0, p1, board = _deal_ints()
    deal = (p0, p1, board)
    ref = NLHEFullCfrPlusRef(game, enc, target="cfrplus")
    ref.run_deal(deal)
    states, targets = ref.emit_targets()
    return {_state_key(states[i]): targets[i] for i in range(len(states))}


def _cpp_matrix():
    cfg = _eng.NLHETraversalConfig()
    cfg.target = _eng.RegretTarget.CFRPLUS
    cfg.max_actions = 4
    gc = _eng.NLHEGameConfig()
    gc.starting_stack = STACK
    gc.sb = 1.0; gc.bb = 2.0
    gc.raise_fraction = 0.75
    gc.max_raises = MAX_RAISES
    cfg.game_cfg = gc

    engine = _eng.NLHEMCCFREngine(cfg)
    engine.reset_cfrplus()
    engine.clear_buffers()

    p0, p1, board = _deal_ints()
    deal = _eng.make_nlhe_deal(p0[0], p0[1], p1[0], p1[1], list(board))
    engine.set_iteration(0)
    engine.run_full_traversal_deal_uniform(0, deal)
    engine.run_full_traversal_deal_uniform(1, deal)
    engine.emit_cfrplus_targets()

    exp = engine.export_regret_buffer()
    n = exp.n_samples
    states = np.array(exp.states, dtype=np.float32).reshape(n, 124)
    actions = np.array(list(exp.actions), dtype=np.int64)
    values = np.array(list(exp.values), dtype=np.float32)
    # collapse to {state_key: [4]}
    out = {}
    for i in range(n):
        k = _state_key(states[i])
        if k not in out:
            out[k] = np.zeros(4, dtype=np.float32)
        out[k][actions[i]] += values[i]
    return out


class TestCppNLHERegretTargetParity:

    @cpp_required
    def test_same_state_set(self):
        py = _python_matrix()
        cpp = _cpp_matrix()
        missing_cpp = set(py) - set(cpp)
        missing_py = set(cpp) - set(py)
        assert not missing_cpp, f"{len(missing_cpp)} states in py missing from C++"
        assert not missing_py, f"{len(missing_py)} states in C++ missing from py"

    @cpp_required
    def test_target_matrices_match(self):
        py = _python_matrix()
        cpp = _cpp_matrix()
        common = set(py) & set(cpp)
        assert len(common) > 0, "no common states"
        max_diff = 0.0
        for k in common:
            d = float(np.max(np.abs(py[k] - cpp[k])))
            max_diff = max(max_diff, d)
        assert max_diff < ATOL, f"target matrix max_diff={max_diff:.2e} (>= {ATOL})"

    @cpp_required
    def test_targets_nonnegative(self):
        cpp = _cpp_matrix()
        for v in cpp.values():
            assert (v >= -1e-9).all()

    @cpp_required
    def test_row_count_matches(self):
        py = _python_matrix()
        cpp = _cpp_matrix()
        assert len(py) == len(cpp), f"state count: py={len(py)} cpp={len(cpp)}"


if __name__ == "__main__":
    if not CPP_AVAILABLE:
        print("cfr_engine.so not available — build first, then run via pytest.")
    else:
        py = _python_matrix()
        cpp = _cpp_matrix()
        common = set(py) & set(cpp)
        md = max((float(np.max(np.abs(py[k] - cpp[k]))) for k in common), default=float("nan"))
        print(f"py states={len(py)}  cpp states={len(cpp)}  common={len(common)}  max_diff={md:.2e}")