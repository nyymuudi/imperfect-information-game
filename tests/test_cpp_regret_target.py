"""
Parity test: C++ MCCFREngine CFR+ target  ==  Python full_cfrplus_ref.

WHY KEY STRINGS ARE NOT COMPARED DIRECTLY
-----------------------------------------
The C++ LeducGame::info_set_key serialises as e.g. "r0b1a1a1" (rank index,
community index, action enum ints), while the Python LeducHoldem.info_set_key
serialises as "J|J|cc" (rank letters, community letter, action letters). These
are different *formats* for the same information partition -- they never match
as raw strings, but that is irrelevant: the Leduc key is only an internal
grouping tag, and on the NLHE side samples carry the 124-d state vector, not the
key.

What MUST match is the COMPUTED TARGET, independent of key format. We compare
implementation-independent invariants of the emitted CFR+ targets:

  1. count of emitted (infoset, action) rows
  2. infoset action-arity histogram (how many 2- and 3-action infosets)
  3. the sorted multiset of target VALUES (R/visits across all rows)

If all three match for several iteration counts, the CFR+ accumulation logic
(reach weighting, node value, clip, /visits, emit) is identical across
implementations regardless of key serialisation.

Strategy isolation: deterministic full (vanilla) traversal with a UNIFORM
strategy on both sides removes the strategy-feedback loop, so the emitted
targets are a pure deterministic function of the game tree.

Skipped automatically if cfr_engine.so is unavailable. Run after rebuilding:

    cd src/cpp_engine && bash scripts/build.sh && cd ../.. \
        && pytest tests/test_cpp_regret_target.py -v
"""
import sys
import os
import numpy as np
import pytest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/cpp_engine"))

from src.games.leduc import LeducHoldem
from full_cfrplus_ref import FullCfrPlusRef  # noqa: E402

try:
    import cfr_engine as _eng
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

cpp_required = pytest.mark.skipif(
    not CPP_AVAILABLE, reason="cfr_engine.so ei saatavilla"
)

ATOL = 1e-4
ROUND = 6


def _python_rows(iters):
    game = LeducHoldem()
    ref = FullCfrPlusRef(game, target="cfrplus")
    ref._strategy = lambda key, n: np.ones(n) / n  # type: ignore
    ref.run(iters)
    last = iters - 1
    vals = sorted(round(v, ROUND) for _, _, v, it in ref.emitted if it == last)
    arity = Counter(ref.nacts[k] for k in ref.nacts)
    return vals, dict(arity), len(vals)


def _cpp_rows(iters):
    cfg = _eng.TraversalConfig()
    cfg.target = _eng.RegretTarget.CFRPLUS
    engine = _eng.MCCFREngine(cfg)
    engine.reset_cfrplus()
    for it in range(iters):
        engine.set_iteration(it)
        engine.run_full_traversal_uniform(0)
        engine.run_full_traversal_uniform(1)
    engine.clear_buffers()
    engine.set_iteration(iters - 1)
    engine.emit_cfrplus_targets()
    exp = engine.export_regret_buffer()
    vals = sorted(round(float(v), ROUND) for v in exp.values)
    per_iset = Counter()
    for iset in exp.info_sets:
        per_iset[iset] += 1
    arity = Counter(per_iset.values())
    return vals, dict(arity), len(vals)


class TestCppRegretTargetParity:

    @cpp_required
    @pytest.mark.parametrize("iters", [1, 3, 10])
    def test_row_count_matches(self, iters):
        _, _, n_py = _python_rows(iters)
        _, _, n_cpp = _cpp_rows(iters)
        assert n_py == n_cpp, f"row count: py={n_py} cpp={n_cpp}"

    @cpp_required
    @pytest.mark.parametrize("iters", [1, 3, 10])
    def test_arity_histogram_matches(self, iters):
        _, ar_py, _ = _python_rows(iters)
        _, ar_cpp, _ = _cpp_rows(iters)
        assert ar_py == ar_cpp, f"arity hist: py={ar_py} cpp={ar_cpp}"

    @cpp_required
    @pytest.mark.parametrize("iters", [1, 3, 10])
    def test_target_value_multiset_matches(self, iters):
        v_py, _, _ = _python_rows(iters)
        v_cpp, _, _ = _cpp_rows(iters)
        assert len(v_py) == len(v_cpp), (
            f"length mismatch: py={len(v_py)} cpp={len(v_cpp)}"
        )
        a = np.array(v_py); b = np.array(v_cpp)
        max_diff = float(np.max(np.abs(a - b)))
        assert max_diff < ATOL, (
            f"target value multiset mismatch (max_diff={max_diff:.2e}). "
            f"py distinct={len(set(v_py))} cpp distinct={len(set(v_cpp))}"
        )

    @cpp_required
    def test_cfrplus_targets_nonnegative(self):
        v_cpp, _, _ = _cpp_rows(5)
        assert all(v >= -1e-9 for v in v_cpp)

    @cpp_required
    def test_target_scale_bounded(self):
        v_cpp, _, _ = _cpp_rows(10)
        assert max(v_cpp) < 15.0, "target scale divergence?"


if __name__ == "__main__":
    if not CPP_AVAILABLE:
        print("cfr_engine.so not available -- build first, then run via pytest.")
    else:
        for it in (1, 3, 10):
            v_py, ar_py, n_py = _python_rows(it)
            v_cpp, ar_cpp, n_cpp = _cpp_rows(it)
            md = float(np.max(np.abs(np.array(v_py) - np.array(v_cpp)))) \
                if len(v_py) == len(v_cpp) else float("nan")
            print(f"iters={it:2d}  n: py={n_py} cpp={n_cpp}  "
                  f"arity py={ar_py} cpp={ar_cpp}  max_diff={md:.2e}")