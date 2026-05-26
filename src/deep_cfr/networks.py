"""
Neural networks for Deep CFR.

Two networks with different roles:
    RegretNetwork: Predicts counterfactual regrets for each action.
        - Output: unbounded reals (linear activation)
        - Loss: weighted Huber loss (robust to outliers)
        
    StrategyNetwork: Learns the average strategy (final output).
        - Output: probability distribution (softmax activation)
        - Loss: weighted cross-entropy

Both networks share the same MLP architecture but differ in
output activation and loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegretNetwork(nn.Module):
    """
    Predicts counterfactual regrets for each legal action.
    
    Architecture: MLP with ReLU activations, linear output.
    Uses Huber loss for robustness to large regret values
    (common in no-limit games with exponential pot growth).
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            # No activation — regrets are unbounded
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyNetwork(nn.Module):
    """
    Predicts the average strategy (action probabilities).
    
    Architecture: MLP with ReLU activations, softmax output.
    Supports action masking — illegal actions get -1e9 logits
    before softmax, forcing their probability to ~0.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            # No activation — apply softmax with masking externally
        )

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.net(x)
        if action_mask is not None:
            logits = logits + (1 - action_mask) * (-1e9)
        return F.softmax(logits, dim=-1)


def train_regret_network(
    network: RegretNetwork,
    buffer,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    optimizer=None,  # persistent optimizer — momentum survives across iterations
) -> float:
    """
    Train regret network on replay buffer data.
    
    Uses iteration-weighted Huber loss (Linear CFR principle):
    later iterations get more weight since their regrets
    are computed from better strategies.

    Pass a persistent optimizer to retain Adam momentum across
    iterations — avoids cold-start on every network update.
    
    Returns average loss over final epoch.
    """
    if len(buffer) < batch_size:
        return 0.0

    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    final_loss = 0.0
    for epoch in range(epochs):
        states, targets, weights = buffer.sample_batch(batch_size)
        device = next(network.parameters()).device
        s = torch.tensor(states,  dtype=torch.float32).to(device)
        t = torch.tensor(targets, dtype=torch.float32).to(device)
        w = torch.tensor(weights, dtype=torch.float32).to(device)

        pred = network(s)
        # Weighted Huber loss
        element_loss = F.huber_loss(pred, t, reduction='none')
        weighted_loss = (element_loss.mean(dim=1) * w).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = weighted_loss.item()

    return final_loss


def train_strategy_network(
    network: StrategyNetwork,
    buffer,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    optimizer=None,  # persistent optimizer
) -> float:
    """
    Train strategy network on strategy memory.
    
    Uses iteration-weighted cross-entropy loss.
    This is trained ONCE at the end of all CFR iterations.

    Pass a persistent optimizer to retain Adam momentum across
    iterations — avoids cold-start on every network update.
    
    Returns average loss over final epoch.
    """
    if len(buffer) < batch_size:
        return 0.0

    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    final_loss = 0.0
    for epoch in range(epochs):
        states, targets, weights = buffer.sample_batch(batch_size)
        device = next(network.parameters()).device
        s = torch.tensor(states,  dtype=torch.float32).to(device)
        t = torch.tensor(targets, dtype=torch.float32).to(device)
        w = torch.tensor(weights, dtype=torch.float32).to(device)

        pred = network(s)
        # Weighted cross-entropy: -sum(target * log(pred))
        # Clamp pred to avoid log(0)
        log_pred = torch.log(pred.clamp(min=1e-8))
        element_loss = -(t * log_pred).sum(dim=1)
        weighted_loss = (element_loss * w).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        final_loss = weighted_loss.item()

    return final_loss


class ScriptableNet(torch.nn.Module):
    """Single-argument wrapper for TorchScript/LibTorch export."""
    def __init__(self, net: torch.nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
