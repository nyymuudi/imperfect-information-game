"""
Blueprint — serialisable Deep CFR strategy for HU NLHE.

File layout on disk (directory):
    blueprint/
        strategy_net.onnx     # ONNX graph (C++ ONNX Runtime inference)
        strategy_weights.pt   # state_dict (Python inference / fine-tuning)
        metadata.json         # game config + architecture params

Usage:
    bp = Blueprint.from_solver(solver)
    bp.save("blueprints/200bb_75pot_1000iter")

    bp = Blueprint.load("blueprints/200bb_75pot_1000iter")
    probs = bp.query(state_vec, num_actions=3)
    probs = bp.query_batch(state_matrix, num_actions_per_row)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class BlueprintMetadata:
    state_size:          int   = 124
    action_size:         int   = 4
    hidden_size:         int   = 256
    starting_stack:      float = 200.0
    sb:                  float = 1.0
    bb:                  float = 2.0
    raise_fraction:      float = 0.75
    max_raises:          int   = 2
    iterations:          int   = 0
    traversals_per_iter: int   = 0
    strategy_samples:    int   = 0
    timestamp:           str   = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "BlueprintMetadata":
        return cls(**json.loads(s))


# ── ScriptableStrategyNet ─────────────────────────────────────────────────────

class ScriptableStrategyNet(nn.Module):
    """ONNX-exportable strategy network wrapper (no Python default args)."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.action_size = action_size

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits + (1.0 - action_mask) * (-1e9)
        return F.softmax(logits, dim=-1)

    @classmethod
    def from_strategy_network(cls, net: nn.Module) -> "ScriptableStrategyNet":
        meta = _infer_arch(net)
        wrapper = cls(**meta)
        wrapper.net.load_state_dict(net.net.state_dict())
        return wrapper


def _infer_arch(net: nn.Module) -> dict:
    layers = list(net.net.children())
    linear = [l for l in layers if isinstance(l, nn.Linear)]
    return {
        "state_size":  linear[0].in_features,
        "action_size": linear[-1].out_features,
        "hidden_size": linear[0].out_features,
    }


# ── Blueprint ─────────────────────────────────────────────────────────────────

class Blueprint:
    """Trained HU NLHE strategy for deployment and subgame solving."""

    WEIGHTS_FILE  = "strategy_weights.pt"
    ONNX_FILE     = "strategy_net.onnx"
    METADATA_FILE = "metadata.json"

    def __init__(self, net: ScriptableStrategyNet, metadata: BlueprintMetadata,
                 device: str = "cpu"):
        self._net     = net.to(device)
        self._device  = device
        self.metadata = metadata

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_solver(cls, solver, device: str = "cpu") -> "Blueprint":
        """Construct from a trained DeepCFRSolver. Copies weights, no mutation."""
        arch    = _infer_arch(solver.strategy_net)
        wrapper = ScriptableStrategyNet(**arch)
        wrapper.net.load_state_dict(solver.strategy_net.net.state_dict())
        meta = BlueprintMetadata(
            state_size=arch["state_size"],
            action_size=arch["action_size"],
            hidden_size=arch["hidden_size"],
            starting_stack=solver.game.starting_stack,
            raise_fraction=float(solver.game.raise_fractions[0]),
            max_raises=solver.game.max_raises_per_street,
            iterations=solver.iterations,
            traversals_per_iter=solver.traversals_per_iter,
            strategy_samples=len(solver.strategy_buffer),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        return cls(wrapper, meta, device=device)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist to a directory. Creates it if absent.

        Writes:
          strategy_weights.pt  — state_dict (Python inference)
          strategy_net.onnx    — ONNX graph (C++ ONNX Runtime)
          metadata.json        — game config + provenance

        torch.onnx.export is stable (not deprecated). The ONNX file
        uses dynamic batch axis so C++ can query one or many states.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import copy
        cpu_net = copy.deepcopy(self._net).to("cpu")
        cpu_net.eval()

        # 1. state_dict — always works, device-agnostic
        torch.save(cpu_net.state_dict(), p / self.WEIGHTS_FILE)

        # 2. ONNX — stable API, C++ ONNX Runtime compatible
        try:
            dummy_state = torch.zeros(1, self.metadata.state_size)
            dummy_mask  = torch.ones(1, self.metadata.action_size)
            torch.onnx.export(
                cpu_net,
                (dummy_state, dummy_mask),
                str(p / self.ONNX_FILE),
                input_names=["state", "action_mask"],
                output_names=["probs"],
                dynamic_axes={
                    "state":       {0: "batch"},
                    "action_mask": {0: "batch"},
                    "probs":       {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
                verbose=False,
                dynamo=False
            )
        except Exception as exc:  # pragma: no cover
            print(f"[Blueprint] ONNX export failed: {exc}. "
                  "C++ inference unavailable; Python query still works.")

        # 3. metadata
        (p / self.METADATA_FILE).write_text(self.metadata.to_json())

        print(f"[Blueprint] Saved to {p.resolve()}")
        print(f"  state_size={self.metadata.state_size}, "
              f"hidden={self.metadata.hidden_size}, "
              f"iterations={self.metadata.iterations}, "
              f"strategy_samples={self.metadata.strategy_samples:,}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "Blueprint":
        """Load from a saved directory."""
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Blueprint directory not found: {p}")

        meta = BlueprintMetadata.from_json((p / cls.METADATA_FILE).read_text())
        net  = ScriptableStrategyNet(
            state_size=meta.state_size,
            action_size=meta.action_size,
            hidden_size=meta.hidden_size,
        )
        net.load_state_dict(torch.load(p / cls.WEIGHTS_FILE, map_location="cpu"))

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"[Blueprint] Loaded from {p.resolve()}")
        print(f"  stack={meta.starting_stack}BB, raise={meta.raise_fraction:.0%} pot, "
              f"iterations={meta.iterations}, device={device}")

        return cls(net, meta, device=device)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, state_vec: np.ndarray, num_actions: int) -> np.ndarray:
        """Single-state query. Returns float32 array [num_actions], sums to 1."""
        assert 1 <= num_actions <= self.metadata.action_size, (
            f"num_actions={num_actions} out of range [1, {self.metadata.action_size}]"
        )
        s    = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self._device)
        mask = torch.zeros(1, self.metadata.action_size, device=self._device)
        mask[0, :num_actions] = 1.0
        with torch.no_grad():
            probs = self._net(s, mask).squeeze(0).cpu().numpy()
        return probs[:num_actions]

    def query_batch(self, state_matrix: np.ndarray,
                    num_actions_per_row: np.ndarray | list[int]) -> np.ndarray:
        """Vectorised query. Returns float32 array [batch, action_size]."""
        batch = len(state_matrix)
        s     = torch.tensor(state_matrix, dtype=torch.float32).to(self._device)
        mask  = torch.zeros(batch, self.metadata.action_size, device=self._device)
        for i, n in enumerate(num_actions_per_row):
            mask[i, :n] = 1.0
        with torch.no_grad():
            probs = self._net(s, mask).cpu().numpy()
        for i, n in enumerate(num_actions_per_row):
            probs[i, n:] = 0.0
        return probs

    # ── Utilities ─────────────────────────────────────────────────────────────

    def onnx_path(self, base_path: str | Path) -> Path:
        """Path to ONNX file for C++ load_strategy_model() calls."""
        return Path(base_path) / self.ONNX_FILE

    def __repr__(self) -> str:
        m = self.metadata
        return (f"Blueprint(stack={m.starting_stack}BB, raise={m.raise_fraction:.0%}pot, "
                f"iter={m.iterations}, hidden={m.hidden_size}, state={m.state_size})")