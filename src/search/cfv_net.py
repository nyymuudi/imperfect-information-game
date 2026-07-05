"""
PBS value network for Leduc depth-limited re-solving (ReBeL Alg. 1).

Maps a 16-dim round-2 root PBS encoding (see cfv.encode_round2_pbs) to
12 counterfactual values: v0[6] ++ v1[6], per-holding expected values in
chips under the round-2 equilibrium. Trained with Huber loss on exact
solver targets; re-trained FROM SCRATCH each outer-loop epoch (fresh
Adam state — same cold-start rationale as the Deep CFR regret net).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .cfv import ENCODING_DIMS, encode_round2_pbs


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CFVNet(nn.Module):
    """PBS → per-holding CFV regressor.

    input_dim/output_dim are parameters so the NLHE encoder (Phase 3:
    public features + bucketed ranges, larger dims) plugs into the same
    training loop unchanged.
    """

    def __init__(self, input_dim: int = ENCODING_DIMS,
                 output_dim: int = 12, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_cfv_net(
    buffer,
    hidden: int = 128,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    range_slices: tuple = ((7, 13), (13, 19)),
    mass_floor: float = 0.05,
) -> tuple[CFVNet, float]:
    """From-scratch training on the reservoir. Returns (net, final_loss).

    Per-entry Huber weights = holding mass from the input ranges + floor:
    off-support holdings have NON-UNIQUE equilibrium values (opponent
    off-support play is underdetermined), measured as solver-to-solver
    target disagreement up to ±6 chips at y≈0 while game values agree to
    ~0.005. Mass weighting keeps that ill-defined tail from dominating
    the loss; the floor keeps every entry weakly anchored.
    ``range_slices``: (start, end) input slots of each player's range
    (Leduc layout by default; NLHE passes its own).
    """
    device = device or pick_device()
    net = CFVNet(input_dim=buffer.state_size, output_dim=buffer.action_size,
                 hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    final_loss = float("nan")
    for _ in range(epochs):
        states, targets, _w = buffer.sample_batch(min(batch_size, len(buffer)))
        x = torch.as_tensor(states, dtype=torch.float32, device=device)
        y = torch.as_tensor(targets, dtype=torch.float32, device=device)
        (a0, b0), (a1, b1) = range_slices
        mass = torch.cat([x[:, a0:b0], x[:, a1:b1]], dim=1) + mass_floor
        w = mass / mass.sum(dim=1, keepdim=True)
        opt.zero_grad()
        loss = (loss_fn(net(x), y) * w).sum(dim=1).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    return net, final_loss


def net_leaf_evaluator(net: CFVNet, device: str | None = None):
    """VectorCFR leaf evaluator backed by the value net."""
    device = device or next(net.parameters()).device
    net.eval()

    def evaluator(comm, cont, y0, y1):
        enc = encode_round2_pbs(comm, cont, y0, y1)
        with torch.no_grad():
            out = net(torch.as_tensor(enc, dtype=torch.float32,
                                      device=device).unsqueeze(0))
        v = out.squeeze(0).cpu().numpy().astype(np.float64)
        return v[:6], v[6:]

    return evaluator
