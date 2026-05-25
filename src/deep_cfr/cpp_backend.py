"""
src/deep_cfr/cpp_backend.py

C++ MCCFR backend for DeepCFRSolver. Drop-in replacement for the
Python external-sampling traversal loop. Requires cfr_engine.so
compiled from cpp_engine/ via:

    cd src/cpp_engine && bash scripts/build.sh

Integrated into DeepCFRSolver via use_cpp_engine=True:

    solver = DeepCFRSolver(game, use_cpp_engine=True, n_traversals=1000)

Falls back to Python traversal if the .so is not found.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np

if TYPE_CHECKING:
    from .networks import RegretNetwork, StrategyNetwork

# ── Locate and import cfr_engine.so ──────────────────────────────────────────

def _find_so() -> Optional[Path]:
    """Search for cfr_engine .so relative to this file."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "cpp_engine",               # src/cpp_engine/
        here.parent / "cpp_engine" / "build",
        here.parent.parent / "cpp_engine",        # fallback: root/cpp_engine/
        here.parent.parent / "cpp_engine" / "build",
        here.parent.parent,
    ]
    for d in candidates:
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
        import cfr_engine as _eng  # type: ignore
        _ENGINE_AVAILABLE = True
    except ImportError:
        pass

def engine_available() -> bool:
    """True if cfr_engine.so was found and loaded."""
    return _ENGINE_AVAILABLE


# ── Action mapping ────────────────────────────────────────────────────────────

_CPP_ACTION_IDX: dict = {}

def _build_action_map():
    global _CPP_ACTION_IDX
    if not _ENGINE_AVAILABLE:
        return
    _CPP_ACTION_IDX = {
        _eng.Action.FOLD:  0,
        _eng.Action.CHECK: 1,
        _eng.Action.CALL:  2,
        _eng.Action.RAISE: 3,
    }

_build_action_map()


# ── Info set encoder ──────────────────────────────────────────────────────────

def info_set_to_tensor(key: str) -> torch.Tensor:
    """
    Parse C++ info set key → 20-dim float tensor.

    Key format: r{rank}[b{comm_rank}][a{action}]*
    Matches LeducEncoder in state_encoder.py:
        [0:3]   private rank one-hot
        [3:6]   community rank one-hot  (zeros if round 0)
        [6]     round == 0
        [7]     round == 1
        [8:18]  action history (up to 5 actions × 4 types)
        [18:20] unused
    """
    vec = torch.zeros(20)
    i, action_slot = 0, 0

    while i < len(key):
        c = key[i]
        if c == 'r':
            vec[int(key[i + 1])] = 1.0
            i += 2
        elif c == 'b':
            vec[3 + int(key[i + 1])] = 1.0
            vec[7] = 1.0
            i += 2
        elif c == 'a':
            slot = 8 + action_slot * 4 + int(key[i + 1])
            if slot < 18:
                vec[slot] = 1.0
            action_slot += 1
            i += 2
        else:
            i += 1

    if vec[7] == 0.0:
        vec[6] = 1.0

    return vec


# ── Strategy callback ─────────────────────────────────────────────────────────

def _make_strategy_fn(regret_net: Optional["RegretNetwork"],
                      device: str = "cpu"):
    if regret_net is None or not _ENGINE_AVAILABLE:
        def uniform(key, actions):
            n = len(actions)
            return [1.0 / n] * n
        return uniform

    regret_net.eval()

    def network_fn(key: str, actions) -> list:
        with torch.no_grad():
            x = info_set_to_tensor(key).unsqueeze(0).to(device)
            raw = regret_net(x).squeeze(0).cpu().numpy()

        indices = [_CPP_ACTION_IDX[a] for a in actions]
        pos = np.array([max(0.0, raw[i]) for i in indices])
        total = pos.sum()
        if total > 1e-7:
            return (pos / total).tolist()
        return [1.0 / len(actions)] * len(actions)

    return network_fn


# ── CppMCCFRBackend ───────────────────────────────────────────────────────────

class CppMCCFRBackend:
    def __init__(
        self,
        n_traversals: int = 1000,
        regret_capacity: int = 1 << 20,
        strategy_capacity: int = 1 << 20,
        device: str = "cpu",
        seed: int = 42,
    ):
        if not _ENGINE_AVAILABLE:
            raise ImportError(
                "cfr_engine.so not found. Build it with:\n"
                "  cd src/cpp_engine && bash scripts/build.sh"
            )

        self.device = device

        cfg = _eng.TraversalConfig()
        cfg.n_traversals      = n_traversals
        cfg.regret_capacity   = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.collect_strategy  = True
        cfg.seed              = seed

        self._engine = _eng.MCCFREngine(cfg)

    def run_iteration(self, iteration: int, regret_net=None) -> tuple:
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()

        fn = _make_strategy_fn(regret_net, self.device)
        self._engine.run_traversals(0, fn)
        self._engine.run_traversals(1, fn)

        return (
            self._engine.export_regret_buffer(),
            self._engine.export_strategy_buffer(),
        )

    def to_tensors(self, export) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(export)
        if n == 0:
            return (
                torch.zeros(0, 20),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0),
            )

        X       = torch.stack([info_set_to_tensor(k) for k in export.info_sets])
        actions = torch.tensor(list(export.actions), dtype=torch.long)
        values  = torch.tensor(list(export.values),  dtype=torch.float32)

        return X.to(self.device), actions.to(self.device), values.to(self.device)

    @property
    def n_regret_samples(self) -> int:
        return self._engine.regret_buffer_size()

    @property
    def n_strategy_samples(self) -> int:
        return self._engine.strategy_buffer_size()


# ═══════════════════════════════════════════════════════════════════════════════
# NLHE EXTENSION
# ═══════════════════════════════════════════════════════════════════════════════

# NLHE action mapping: C++ enum → network index
_NLHE_ACTION_IDX: dict = {}

def _build_nlhe_action_map():
    global _NLHE_ACTION_IDX
    if not _ENGINE_AVAILABLE:
        return
    _NLHE_ACTION_IDX = {
        _eng.NLHE_FOLD:     0,
        _eng.NLHE_CHECK:    1,
        _eng.NLHE_CALL:     2,
        _eng.NLHE_BET_HALF: 3,
        _eng.NLHE_BET_POT:  4,
        _eng.NLHE_ALL_IN:   5,
    }

_build_nlhe_action_map()


def nlhe_info_set_to_tensor(key: str) -> torch.Tensor:
    """
    Parse NLHE info set key → 120-dim tensor.

    Key format: H{c1}{c2}|S{street}|B{board_cards}|P{pot_bucket}|A{actions}

    Tensor layout (matches NLHEEncoder, simplified):
        [0:52]    hole cards one-hot (card index 0-51)
        [52:104]  visible board one-hot
        [104:108] street one-hot
        [108]     pot bucket / 7 (normalised)
        [109:117] action history (last 8 actions / 5)
        [117:120] padding
    """
    vec = torch.zeros(122)

    parts = key.split('|')
    if len(parts) < 5:
        return vec

    # Hole cards: H{xx}{xx} — 2-char hex each
    h_part = parts[0][1:]   # strip 'H'
    if len(h_part) >= 4:
        try:
            c0 = int(h_part[0:2], 16)
            c1 = int(h_part[2:4], 16)
            if 0 <= c0 < 52: vec[c0] = 1.0
            if 0 <= c1 < 52: vec[c1] = 1.0
        except ValueError:
            pass

    # Street: S{0-3}
    try:
        street = int(parts[1][1:])
        if 0 <= street < 4:
            vec[104 + street] = 1.0
    except (ValueError, IndexError):
        pass

    # Board: B{xx}{xx}...
    b_part = parts[2][1:]
    for i in range(0, len(b_part) - 1, 2):
        try:
            card = int(b_part[i:i+2], 16)
            if 0 <= card < 52:
                vec[52 + card] = 1.0
        except ValueError:
            pass

    # Pot bucket: P{0-7}
    try:
        pot_bucket = int(parts[3][1:])
        vec[108] = pot_bucket / 7.0
    except (ValueError, IndexError):
        pass

    # Action history: A{a1}{a2}...
    a_part = parts[4][1:]
    for i, ch in enumerate(a_part[:8]):
        try:
            vec[109 + i] = int(ch) / 5.0
        except ValueError:
            pass

    return vec


def _make_nlhe_strategy_fn(regret_net, device: str = "cpu"):
    """Wrap PyTorch regret network as NLHE strategy callback."""
    if regret_net is None or not _ENGINE_AVAILABLE:
        def uniform(key, actions):
            return [1.0 / len(actions)] * len(actions)
        return uniform

    regret_net.eval()

    def network_fn(key: str, actions) -> list:
        with torch.no_grad():
            x = nlhe_info_set_to_tensor(key).unsqueeze(0).to(device)
            raw = regret_net(x).squeeze(0).cpu().numpy()

        indices = [_NLHE_ACTION_IDX.get(a, int(a)) for a in actions]
        pos = np.array([max(0.0, raw[min(i, len(raw)-1)]) for i in indices])
        total = pos.sum()
        if total > 1e-7:
            return (pos / total).tolist()
        return [1.0 / len(actions)] * len(actions)

    return network_fn


class NLHECppBackend:
    """
    C++ MCCFR backend for full HU NLHE (postflop_nlhe.py).

    Drop-in replacement for Python traversal in DeepCFRSolver when
    game is PostflopNLHE. Use via:

        solver = DeepCFRSolver(game, encoder, use_cpp_engine=True)
        # DeepCFRSolver detects PostflopNLHE and routes to NLHECppBackend.

    Or directly:
        backend = NLHECppBackend(n_traversals=500)
        reg, strat = backend.run_iteration(0)
        X, actions, values = backend.to_tensors(reg)  # [N, 120]
    """

    def __init__(
        self,
        n_traversals: int = 500,
        regret_capacity: int = 1 << 20,
        strategy_capacity: int = 1 << 20,
        device: str = "cpu",
        seed: int = 42,
    ):
        if not _ENGINE_AVAILABLE:
            raise ImportError(
                "cfr_engine.so not found. Build: cd src/cpp_engine && bash scripts/build.sh"
            )

        self.device = device

        cfg = _eng.NLHETraversalConfig()
        cfg.n_traversals      = n_traversals
        cfg.regret_capacity   = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.collect_strategy  = True
        cfg.seed              = seed

        self._engine = _eng.NLHEMCCFREngine(cfg)

    def run_iteration(self, iteration: int, regret_net=None) -> tuple:
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()

        # ── LibTorch path: zero Python callbacks ──────────────────────────────
        if regret_net is not None and getattr(_eng, 'TORCH_AVAILABLE', False):
            model_path = "/tmp/cfr_regret_net.pt"
            import copy; scripted = torch.jit.script(copy.deepcopy(regret_net).to("cpu"))
            scripted.save(model_path)
            self._engine.load_model(model_path)

        if self._engine.model_loaded():
            self._engine.run_traversals_model(0)
            self._engine.run_traversals_model(1)
        else:
            # Fallback: Python callback (first iteration or no LibTorch)
            fn = _make_nlhe_strategy_fn(regret_net, self.device)
            self._engine.run_traversals(0, fn)
            self._engine.run_traversals(1, fn)

        return (
            self._engine.export_regret_buffer(),
            self._engine.export_strategy_buffer(),
        )

    def to_tensors(self, export) -> tuple:
        """Returns (X [N,120], actions [N], values [N])."""
        n = len(export)
        if n == 0:
            return (torch.zeros(0, 122),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0))

        X       = torch.stack([nlhe_info_set_to_tensor(k) for k in export.info_sets])
        actions = torch.tensor(list(export.actions), dtype=torch.long)
        values  = torch.tensor(list(export.values),  dtype=torch.float32)

        return X.to(self.device), actions.to(self.device), values.to(self.device)

    @property
    def n_regret_samples(self) -> int:
        return self._engine.regret_buffer_size()

    @property
    def n_strategy_samples(self) -> int:
        return self._engine.strategy_buffer_size()


    def save_strategy_model(self, strategy_net, path: str = "/tmp/cfr_strategy_net.pt") -> str:
        """
        Save strategy network as TorchScript and load into C++ engine.
        Returns the path.
        """
        import copy
        scripted = torch.jit.script(copy.deepcopy(strategy_net).to("cpu"))
        scripted.save(path)
        self._engine.load_strategy_model(path)
        return path

    def get_strategy_cpp(self, hole0: int, hole1: int,
                         street: int = 0,
                         board: list = None,
                         pot: float = 1.5,
                         to_call: float = 0.5,
                         my_stack: float = 99.5) -> list:
        """
        Query strategy from C++ engine (no Python game needed).
        Returns 4-slot probability vector: [fold/check, call/check, raise, all-in]
        """
        if not self._engine.strategy_model_loaded():
            return [0.25, 0.25, 0.25, 0.25]
        return self._engine.query_strategy(
            hole0, hole1, street,
            board or [], pot, to_call, my_stack
        )

    def get_preflop_strategy_cpp(self, hole0: int, hole1: int) -> list:
        """Query preflop SB strategy directly from C++."""
        if not self._engine.strategy_model_loaded():
            return [0.25, 0.25, 0.25, 0.25]
        return self._engine.query_preflop_strategy(hole0, hole1)