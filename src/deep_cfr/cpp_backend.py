"""
src/deep_cfr/cpp_backend.py

C++ MCCFR backend for DeepCFRSolver.
Buffer samples carry full 37-dim state vectors — training features
are identical to LibTorch inference features. (Was 36-dim before
2026-06-14; position bit added at last index.)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from typing import Optional
import torch
import numpy as np


# ── Locate cfr_engine.so ─────────────────────────────────────────────────────

def _find_so() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for d in [
        here.parent / "cpp_engine",
        here.parent / "cpp_engine" / "build",
        here.parent.parent / "cpp_engine",
        here.parent.parent / "cpp_engine" / "build",
    ]:
        hits = list(d.glob("cfr_engine*.so"))
        if hits:
            return hits[0].parent
    return None

_ENGINE_AVAILABLE = False
_eng = None
_so_dir = _find_so()
if _so_dir is not None:
    sys.path.insert(0, str(_so_dir))
    try:
        import cfr_engine as _eng
        _ENGINE_AVAILABLE = True
    except ImportError:
        pass

def engine_available() -> bool:
    return _ENGINE_AVAILABLE


# ── TorchScript export for regret network ────────────────────────────────────
#
# The regret network is exported as TorchScript every iteration so LibTorch
# can load it for zero-callback C++ traversal.
#
# torch.jit.script is deprecated in PyTorch >= 2.5, but LibTorch's
# torch::jit::load() is unaffected — the format itself is unchanged.
# The deprecation is on the Python saving API only.
#
# We suppress the warning here because:
#   (a) TorchScript is the ONLY format LibTorch can load
#   (b) torch.compile / torch.export do NOT produce LibTorch-loadable files
#   (c) This is a training-internal artefact, re-exported every iteration
#   (d) The blueprint STRATEGY network uses ONNX (blueprint.py) — no issue there
#
# CRITICAL FIX (device safety):
#   The previous implementation built `nn.Sequential(*list(net.net.children()))`,
#   which REUSES the original layer objects (children() returns references, not
#   copies). Calling load_state_dict(..., assign=True) on that sequential then
#   rebinds those shared layers' parameters to CPU tensors — silently moving the
#   ORIGINAL regret_net to CPU on every iteration. On an MPS/CUDA machine this
#   meant training quietly fell back to CPU and inference hit a device mismatch.
#   We now deep-copy the module first so the export never touches the live net.

def export_for_libtorch(net) -> "torch.jit.ScriptModule":
    """
    Export RegretNetwork as TorchScript for C++ LibTorch inference.
    DeprecationWarning is suppressed intentionally — see module docstring.

    Does NOT mutate `net` or move it across devices: it operates on a
    detached CPU deep-copy.
    """
    import copy
    import torch.nn as nn

    class _ScriptableNet(nn.Module):
        def __init__(self, layers: nn.Sequential):
            super().__init__()
            self.layers = layers

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    # Deep-copy the inner Sequential so we never alias the live network's
    # layer objects, then move the COPY to CPU. The original `net` is untouched.
    cpu_layers = copy.deepcopy(net.net).to("cpu")
    cpu_layers.eval()

    wrapper = _ScriptableNet(cpu_layers)
    wrapper.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.jit.script(wrapper)


# ── Leduc backend ─────────────────────────────────────────────────────────────

_CPP_ACTION_IDX: dict = {}

def _build_leduc_action_map():
    global _CPP_ACTION_IDX
    if not _ENGINE_AVAILABLE:
        return
    _CPP_ACTION_IDX = {
        _eng.Action.FOLD: 0, _eng.Action.CHECK: 1,
        _eng.Action.CALL: 2, _eng.Action.RAISE: 3,
    }

_build_leduc_action_map()


def info_set_to_tensor(key: str) -> torch.Tensor:
    """Leduc info-set key → 20-dim tensor (Leduc encoder layout)."""
    parts  = key.split("|")
    state  = torch.zeros(20)
    card   = parts[0] if parts else ""
    action = parts[1] if len(parts) > 1 else ""
    rank_map = {"J": 0, "Q": 1, "K": 2}
    if card and card[0] in rank_map:
        state[rank_map[card[0]]] = 1.0
    for i, a in enumerate(action[-10:]):
        enc = {"c": 0.0, "r": 0.5, "f": -1.0, "k": 1.0}.get(a, 0.0)
        if i + 10 < 20:
            state[10 + i] = enc
    return state


# ── NLHE C++ backend ──────────────────────────────────────────────────────────

class CppMCCFRBackend:
    """Generic Leduc C++ backend (kept for compatibility)."""

    STATE_SIZE = 20

    def __init__(self, n_traversals=500, regret_capacity=1 << 20,
                 strategy_capacity=1 << 20, device="cpu", seed=42):
        if not _ENGINE_AVAILABLE:
            raise ImportError("cfr_engine.so not found.")
        from cfr_engine import TraversalConfig, MCCFREngine
        cfg = TraversalConfig()
        cfg.n_traversals      = n_traversals
        cfg.regret_capacity   = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.seed              = seed
        self._engine = MCCFREngine(cfg)
        self.device  = device

    def run_iteration(self, iteration: int, regret_net=None):
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()
        if regret_net is not None:
            try:
                path = "/tmp/cfr_regret_net_leduc.pt"
                export_for_libtorch(regret_net).save(path)
            except Exception:
                pass
        self._engine.run_traversals_uniform(0)
        self._engine.run_traversals_uniform(1)
        return (
            self._engine.export_regret_buffer(),
            self._engine.export_strategy_buffer(),
        )

    def to_tensors(self, export) -> tuple:
        n  = export.n_samples
        if n == 0:
            sz = self.STATE_SIZE
            return (torch.zeros(0, sz), torch.zeros(0, dtype=torch.long),
                    torch.zeros(0))
        X = torch.tensor(
            [info_set_to_tensor(s).tolist() for s in export.info_sets],
            dtype=torch.float32,
        )
        a = torch.tensor(list(export.actions), dtype=torch.long)
        v = torch.tensor(list(export.values),  dtype=torch.float32)
        return X, a, v


class NLHECppBackend:
    """
    NLHE C++ backend using state-vector buffers.
    Buffer samples store float[STATE_SIZE] state vectors — no string parsing.

    Buffer insertion is handled by DeepCFRSolver._run_cpp_iteration, which
    routes the exported (state, action, value) triples through
    _collapse_by_state so that one row per unique state carries the FULL
    per-action target vector. There is intentionally no add_batch helper here:
    a per-row scatter (one non-zero slot per row) would reintroduce the
    conflicting-one-hot pathology that _collapse_by_state exists to remove.
    """

    # 2026-06-16: STATE_SIZE bumped 37 → 49 to make room for the 12 CFR
    # advisor dims appended at the tail (6 probs + 6 EVs). C++ writes
    # ZEROS to those slots (no cache lookup in-engine yet); the Python
    # wrapper backfills them in to_tensors() when a cache is attached.
    # When no cache is attached, advisor dims stay at 0.0 and the network
    # treats them as a no-information default — equivalent to the legacy
    # 37-dim layout from a learnability standpoint.
    BASE_STATE_SIZE = 37
    ADVISOR_DIMS    = 12
    STATE_SIZE      = BASE_STATE_SIZE + ADVISOR_DIMS  # 49

    def __init__(self, n_traversals=500, regret_capacity=1 << 20,
                 strategy_capacity=1 << 20, device="cpu", seed=42,
                 starting_stack=200.0, raise_fraction=0.75, max_raises=2,
                 regret_target: str = "instant",
                 prune_threshold: float = 0.0,
                 prune_after_iter: int = 0,
                 raise_fractions: tuple[float, ...] | None = None,
                 max_actions: int = 4,
                 include_position_bit: bool = True,
                 predictive_alpha: float = 0.0):
        if not _ENGINE_AVAILABLE:
            raise ImportError("cfr_engine.so not found.")
        self.device = device
        self._regret_target = regret_target.strip().lower()

        cfg             = _eng.NLHETraversalConfig()
        cfg.n_traversals      = n_traversals
        cfg.regret_capacity   = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.collect_strategy  = True
        cfg.seed              = seed
        cfg.max_actions       = int(max_actions)
        # Pluribus-style dynamic pruning. 0.0 = off (default; backward compat).
        cfg.prune_threshold   = float(prune_threshold)
        cfg.prune_after_iter  = int(prune_after_iter)
        # Predictive CFR+ (Brown 2020): adds momentum term alpha*(r_t-r_{t-1})
        # to each accumulator update. Only active when target=CFRPLUS.
        try:
            cfg.predictive_alpha = float(predictive_alpha)
        except AttributeError:
            if predictive_alpha > 0:
                print("[warn] cfr_engine.so missing predictive_alpha — "
                      "rebuild engine for Predictive CFR+ support.")
        # Position-bit ablation: when False, both Python encoder and C++
        # engine write 0.0 in the slot for both players. State shape stays 37
        # so all buffers and networks remain compatible without a rebuild —
        # but the SB/BB signal is dropped at the source.
        self.include_position_bit = bool(include_position_bit)
        try:
            _eng.NLHEStateEncoder.set_zero_position_bit(
                not self.include_position_bit
            )
        except AttributeError:
            # Old .so without the binding — log and rely on Python encoder
            # alone. C++ traversal will still write the SB/BB signal, which
            # means the v15 ablation is not clean. Rebuild the .so to fix.
            if not self.include_position_bit:
                print("[warn] cfr_engine.so missing set_zero_position_bit — "
                      "C++ traversal will still emit position bit. Rebuild "
                      "src/cpp_engine for a clean ablation.")

        # Regret target: INSTANT = per-traversal instantaneous regret (Brown et al.
        # 2019 Algorithm 1). CFRPLUS = CFR+/visits kumulatiivisena kohteena.
        # INSTANT on oikea valinta jatkuvalle tilavektorille koska visits(I)≈1
        # aina — CFR+/visits redusoituu kohinaiseksi yhden näytteen estimaatiksi.
        if self._regret_target == "instant":
            cfg.target = _eng.RegretTarget.INSTANT
        else:
            cfg.target = _eng.RegretTarget.CFRPLUS

        game_cfg               = _eng.NLHEGameConfig()
        game_cfg.starting_stack = starting_stack
        game_cfg.sb             = 1.0
        game_cfg.bb             = 2.0
        game_cfg.raise_fraction = raise_fraction
        game_cfg.max_raises     = max_raises
        # Pluribus-style multi-raise puu. None tai single-element tuple
        # → legacy single-raise (n_raise_fractions=0). 2..MAX-element tuple
        # → multi-raise: legal_actions tuottaa RAISE_0..RAISE_{N-1}.
        if raise_fractions is not None and len(raise_fractions) > 1:
            game_cfg.set_raise_fractions(list(raise_fractions))
        cfg.game_cfg            = game_cfg

        self._engine    = _eng.NLHEMCCFREngine(cfg)
        self._model_path = "/tmp/cfr_regret_net.pt"

    def run_iteration(self, iteration: int, regret_net=None) -> tuple:
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()
        # NOTE: do NOT reset_cfrplus() here. The CFR+ accumulator is CUMULATIVE
        # across iterations (R <- max(R + r^t, 0) per infoset), exactly like the
        # Python solver's persistent _cfrplus_regret. clear_buffers() empties
        # only the reservoir; the accumulator must persist so emit_cfrplus_targets
        # below exports the up-to-date cumulative R/visits each iteration.

        if regret_net is not None and getattr(_eng, "TORCH_AVAILABLE", False):
            try:
                export_for_libtorch(regret_net).save(self._model_path)
                self._engine.load_model(self._model_path)
            except Exception as e:
                warnings.warn(
                    f"TorchScript export failed (iter {iteration}): {e}. "
                    "Falling back to uniform strategy.",
                    RuntimeWarning,
                )

        if self._engine.model_loaded():
            self._engine.run_traversals_model(0)
            self._engine.run_traversals_model(1)
        else:
            self._engine.run_traversals_uniform(0)
            self._engine.run_traversals_uniform(1)

        # CFRPLUS-modessa: emit akkumuloidut R/visits regret-bufferiin.
        # INSTANT-modessa: traversaali kirjoittaa suoraan regret-bufferiin,
        # emit on turha ja hidastava — ohitetaan.
        if self._regret_target == "cfrplus":
            self._engine.emit_cfrplus_targets()

        return (
            self._engine.export_regret_buffer(),
            self._engine.export_strategy_buffer(),
        )

    def to_tensors(self, export) -> tuple:
        """
        Convert NLHEBufferExport → (X [N, STATE_SIZE], actions [N], values [N]).
        State vectors come directly from C++ (49 dims since 2026-06-16,
        with advisor dims [37:49] left at zero).

        When a CFR cache + encoder is attached (via ``set_cache_context``),
        this method re-derives the abstraction key from each state vector
        and backfills the advisor dims via cache lookup. The result is a
        49-dim state vector matching what the Python encoder would have
        produced — the network trains on consistent advisor signals.
        """
        n = export.n_samples
        if n == 0:
            return (torch.zeros(0, self.STATE_SIZE, dtype=torch.float32),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32))

        states = np.array(export.states, dtype=np.float32).reshape(n, self.STATE_SIZE)

        if not self.include_position_bit:
            # Position bit is at index BASE_STATE_SIZE-1 (= 36) — slot 48
            # belongs to the advisor pad, NOT the position bit.
            states[:, self.BASE_STATE_SIZE - 1] = 0.0

        # Backfill advisor dims from the cache when a context is set.
        cache    = getattr(self, "_cache", None)
        encoder  = getattr(self, "_encoder", None)
        if cache is not None and encoder is not None:
            from .cfr_cache import (
                ADVISOR_DIMS, EV_DIMS, PROB_DIMS,
                key_from_state_vector,
            )
            base = self.BASE_STATE_SIZE
            for i in range(n):
                k = key_from_state_vector(states[i], encoder)
                entry = cache.lookup_by_key(k)
                if entry is not None:
                    probs, evs = entry
                    states[i, base               : base + PROB_DIMS] = probs
                    states[i, base + PROB_DIMS   : base + ADVISOR_DIMS] = evs
                # cache miss → leave 12 zeros (Python's encoder would have
                # used the live MC fallback here, but doing MC inside the
                # buffer-export hot path would add ~10ms per sample. Accept
                # the zeros — training tolerates partial advisor coverage).

        X = torch.from_numpy(states)
        actions = np.array(list(export.actions), dtype=np.int64)
        values  = np.array(list(export.values),  dtype=np.float32)
        return X, torch.from_numpy(actions), torch.from_numpy(values)

    def set_cache_context(self, cfr_cache, encoder) -> None:
        """Wire a CFRCache + encoder so that to_tensors() backfills advisor
        dims into each buffer entry. Called by DeepCFRSolver when the
        encoder has a cache attached and the C++ engine is in use."""
        self._cache   = cfr_cache
        self._encoder = encoder