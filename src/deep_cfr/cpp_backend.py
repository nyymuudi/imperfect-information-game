"""
src/deep_cfr/cpp_backend.py

C++ MCCFR backend for DeepCFRSolver.
Requires cfr_engine.so: cd src/cpp_engine && bash scripts/build.sh
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional
import torch
import numpy as np


# ── Locate and import cfr_engine.so ──────────────────────────────────────────

def _find_so() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "cpp_engine",
        here.parent / "cpp_engine" / "build",
        here.parent.parent / "cpp_engine",
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
        import cfr_engine as _eng
        _ENGINE_AVAILABLE = True
    except ImportError:
        pass

def engine_available() -> bool:
    return _ENGINE_AVAILABLE


# ── Leduc info-set → tensor ───────────────────────────────────────────────────

_CPP_ACTION_IDX: dict = {}

def _build_action_map():
    global _CPP_ACTION_IDX
    if not _ENGINE_AVAILABLE: return
    _CPP_ACTION_IDX = {
        _eng.Action.FOLD: 0, _eng.Action.CHECK: 1,
        _eng.Action.CALL: 2, _eng.Action.RAISE: 3,
    }

_build_action_map()


def info_set_to_tensor(key: str) -> torch.Tensor:
    """Leduc info-set key → 20-dim tensor."""
    vec = torch.zeros(20)
    i, aslot = 0, 0
    while i < len(key):
        c = key[i]
        if c == 'r':
            vec[int(key[i+1])] = 1.0; i += 2
        elif c == 'b':
            vec[3 + int(key[i+1])] = 1.0; vec[7] = 1.0; i += 2
        elif c == 'a':
            slot = 8 + aslot*4 + int(key[i+1])
            if slot < 18: vec[slot] = 1.0
            aslot += 1; i += 2
        else:
            i += 1
    if vec[7] == 0.0: vec[6] = 1.0
    return vec


def _make_strategy_fn(regret_net, device: str = "cpu"):
    if regret_net is None or not _ENGINE_AVAILABLE:
        return lambda key, actions: [1.0/len(actions)]*len(actions)
    regret_net.eval()
    def fn(key: str, actions) -> list:
        with torch.no_grad():
            x = info_set_to_tensor(key).unsqueeze(0).to(device)
            raw = regret_net(x).squeeze(0).cpu().numpy()
        indices = [_CPP_ACTION_IDX[a] for a in actions]
        pos = np.maximum([raw[i] for i in indices], 0)
        total = pos.sum()
        return (pos/total).tolist() if total > 1e-7 else [1/len(actions)]*len(actions)
    return fn


class CppMCCFRBackend:
    def __init__(self, n_traversals=1000, regret_capacity=1<<20,
                 strategy_capacity=1<<20, device="cpu", seed=42):
        if not _ENGINE_AVAILABLE:
            raise ImportError("cfr_engine.so not found. Build: cd src/cpp_engine && bash scripts/build.sh")
        self.device = device
        cfg = _eng.TraversalConfig()
        cfg.n_traversals = n_traversals
        cfg.regret_capacity = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.collect_strategy = True
        cfg.seed = seed
        self._engine = _eng.MCCFREngine(cfg)

    def run_iteration(self, iteration, regret_net=None):
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()
        fn = _make_strategy_fn(regret_net, self.device)
        self._engine.run_traversals(0, fn)
        self._engine.run_traversals(1, fn)
        return self._engine.export_regret_buffer(), self._engine.export_strategy_buffer()

    def to_tensors(self, export):
        n = len(export)
        if n == 0:
            return torch.zeros(0,20), torch.zeros(0,dtype=torch.long), torch.zeros(0)
        X       = torch.stack([info_set_to_tensor(k) for k in export.info_sets])
        actions = torch.tensor(list(export.actions), dtype=torch.long)
        values  = torch.tensor(list(export.values),  dtype=torch.float32)
        return X.to(self.device), actions.to(self.device), values.to(self.device)

    @property
    def n_regret_samples(self): return self._engine.regret_buffer_size()
    @property
    def n_strategy_samples(self): return self._engine.strategy_buffer_size()


# ═══════════════════════════════════════════════════════════════════════════════
# NLHE BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

# NLHE 4-action enum ints (matching C++ NLHEAction): FOLD_OR_CHECK=0, CALL=1, RAISE=2, ALL_IN=3
_NLHE_ACTION_IDX: dict = {}

def _build_nlhe_action_map():
    global _NLHE_ACTION_IDX
    if not _ENGINE_AVAILABLE: return
    _NLHE_ACTION_IDX = {
        _eng.NLHE_FOLD_OR_CHECK: 0,
        _eng.NLHE_CALL:          1,
        _eng.NLHE_RAISE:         2,
        _eng.NLHE_ALL_IN:        3,
    }

_build_nlhe_action_map()


def nlhe_info_set_to_tensor(key: str) -> torch.Tensor:
    """
    NLHE info-set key → 122-dim tensor.
    Key format: H{c1}{c2}|S{street}|B{board}|P{pot_bucket}|A{actions}
    Matches NLHEEncoder layout exactly (same as C++ NLHEStateEncoder).
    """
    vec = torch.zeros(122)
    parts = key.split('|')
    if len(parts) < 5: return vec

    # Hole cards
    h = parts[0][1:]
    if len(h) >= 4:
        try:
            c0, c1 = int(h[0:2],16), int(h[2:4],16)
            if 0<=c0<52: vec[c0]=1.0
            if 0<=c1<52: vec[c1]=1.0
        except ValueError: pass

    # Street one-hot
    try:
        s = int(parts[1][1:])
        if 0<=s<4: vec[104+s]=1.0
    except (ValueError,IndexError): pass

    # Board
    b = parts[2][1:]
    for i in range(0, len(b)-1, 2):
        try:
            card = int(b[i:i+2],16)
            if 0<=card<52: vec[52+card]=1.0
        except ValueError: pass

    # Pot bucket [108] — normalised as pot/(2*stack) approximation
    try:
        pb = int(parts[3][1:])
        vec[108] = pb / 7.0
    except (ValueError,IndexError): pass

    # Action history [112:120] — 8 slots
    # Python ACTION_ENC: {"f":-1,"c":0,"k":0.25,"r":0.5,"a":1.0}
    # C++ NLHE_ACTION_ENC: {0:0.0, 1:0.25, 2:0.5, 3:1.0}
    a = parts[4][1:]
    for i, ch in enumerate(a[:8]):
        try: vec[112+i] = int(ch) / 3.0   # normalise to [0,1]
        except ValueError: pass

    return vec


def _make_nlhe_strategy_fn(regret_net, device: str = "cpu"):
    """Wrap PyTorch regret network as NLHE strategy callback."""
    if regret_net is None or not _ENGINE_AVAILABLE:
        return lambda key, actions: [1.0/len(actions)]*len(actions)
    regret_net.eval()
    def fn(key: str, actions) -> list:
        with torch.no_grad():
            x = nlhe_info_set_to_tensor(key).unsqueeze(0).to(device)
            raw = regret_net(x).squeeze(0).cpu().numpy()
        # actions are NLHEAction enum values (ints 0-3), direct index into network output
        indices = [int(a) for a in actions]
        pos = np.maximum([raw[min(i,len(raw)-1)] for i in indices], 0)
        total = pos.sum()
        return (pos/total).tolist() if total > 1e-7 else [1/len(actions)]*len(actions)
    return fn


class NLHECppBackend:
    """
    C++ MCCFR backend for PostflopNLHE.
    4-action space matches Python exactly — no remapping layer.
    """

    def __init__(self, n_traversals=500, regret_capacity=1<<20,
                 strategy_capacity=1<<20, device="cpu", seed=42,
                 starting_stack=200.0, raise_fraction=0.75, max_raises=2):
        if not _ENGINE_AVAILABLE:
            raise ImportError("cfr_engine.so not found. Build: cd src/cpp_engine && bash scripts/build.sh")
        self.device = device

        cfg = _eng.NLHETraversalConfig()
        cfg.n_traversals      = n_traversals
        cfg.regret_capacity   = regret_capacity
        cfg.strategy_capacity = strategy_capacity
        cfg.collect_strategy  = True
        cfg.seed              = seed
        cfg.max_actions       = 4

        # Pass game config matching PostflopNLHE parameters
        game_cfg = _eng.NLHEGameConfig()
        game_cfg.starting_stack = starting_stack
        game_cfg.sb             = 1.0
        game_cfg.bb             = 2.0
        game_cfg.raise_fraction = raise_fraction
        game_cfg.max_raises     = max_raises
        cfg.game_cfg = game_cfg

        self._engine = _eng.NLHEMCCFREngine(cfg)

    def run_iteration(self, iteration: int, regret_net=None) -> tuple:
        self._engine.set_iteration(iteration)
        self._engine.clear_buffers()

        # LibTorch path: export regret net and run without Python callbacks
        if regret_net is not None and getattr(_eng, 'TORCH_AVAILABLE', False):
            model_path = "/tmp/cfr_regret_net.pt"
            try:
                from .train_postflop import _export_for_libtorch
                _export_for_libtorch(regret_net).save(model_path)
                self._engine.load_model(model_path)
            except Exception:
                pass

        if self._engine.model_loaded():
            self._engine.run_traversals_model(0)
            self._engine.run_traversals_model(1)
        else:
            fn = _make_nlhe_strategy_fn(regret_net, self.device)
            self._engine.run_traversals(0, fn)
            self._engine.run_traversals(1, fn)

        return (
            self._engine.export_regret_buffer(),
            self._engine.export_strategy_buffer(),
        )

    def to_tensors(self, export) -> tuple:
        """Returns (X [N,122], actions [N], values [N])."""
        n = len(export)
        if n == 0:
            return (torch.zeros(0,122),
                    torch.zeros(0,dtype=torch.long),
                    torch.zeros(0))
        X       = torch.stack([nlhe_info_set_to_tensor(k) for k in export.info_sets])
        actions = torch.tensor(list(export.actions), dtype=torch.long)
        values  = torch.tensor(list(export.values),  dtype=torch.float32)
        return X.to(self.device), actions.to(self.device), values.to(self.device)

    @property
    def n_regret_samples(self): return self._engine.regret_buffer_size()
    @property
    def n_strategy_samples(self): return self._engine.strategy_buffer_size()

    def save_strategy_model(self, strategy_net, path="/tmp/cfr_strategy_net.pt") -> str:
        from .train_postflop import _export_for_libtorch
        _export_for_libtorch(strategy_net).save(path)
        self._engine.load_strategy_model(path)
        return path

    def get_preflop_strategy_cpp(self, hole0: int, hole1: int) -> list:
        if not self._engine.strategy_model_loaded(): return [0.25]*4
        return self._engine.query_preflop_strategy(hole0, hole1)