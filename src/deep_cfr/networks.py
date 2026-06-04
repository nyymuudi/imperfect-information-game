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

NOTE on the LibTorch-export wrapper:
    The single-argument TorchScript wrapper used to export the regret network
    for C++ inference lives in cpp_backend.export_for_libtorch (which deep-
    copies the live net to avoid device aliasing). There is intentionally no
    duplicate wrapper here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    optimizer=None,
) -> float:
    """
    Train the regret network on the replay buffer.

    Uses a per-sample-weighted Huber loss. The DeepCFRSolver inserts regret
    samples UNWEIGHTED (weight = 1.0), so by default this is an unweighted
    Huber regression onto the CFR+ regret targets; the weight column exists so
    the same trainer can serve the strategy buffer's Linear-CFR weights without
    a second code path. The network is re-initialised from scratch each
    iteration by the solver (Brown et al. 2019, Fig. 4), so no optimizer state
    is carried across iterations unless one is explicitly passed in.

    Returns average loss over the final epoch.
    """
    if len(buffer) < batch_size:
        return 0.0

    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(epochs):
        states, targets, weights = buffer.sample_batch(batch_size)
        device = next(network.parameters()).device
        s = torch.tensor(states,  dtype=torch.float32).to(device)
        t = torch.tensor(targets, dtype=torch.float32).to(device)
        w = torch.tensor(weights, dtype=torch.float32).to(device)

        pred = network(s)
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
    optimizer=None,
) -> float:
    """
    Train the strategy network on the strategy buffer.

    Uses an iteration-weighted cross-entropy loss (Linear CFR — later
    iterations carry more weight because their strategies are closer to
    equilibrium). Trained ONCE at the end of all CFR iterations.

    Returns average loss over the final epoch.
    """
    if len(buffer) < batch_size:
        return 0.0

    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(epochs):
        states, targets, weights = buffer.sample_batch(batch_size)
        device = next(network.parameters()).device
        s = torch.tensor(states,  dtype=torch.float32).to(device)
        t = torch.tensor(targets, dtype=torch.float32).to(device)
        w = torch.tensor(weights, dtype=torch.float32).to(device)

        pred = network(s)
        # Weighted cross-entropy: -sum(target * log(pred)), clamp to avoid log(0)
        log_pred = torch.log(pred.clamp(min=1e-8))
        element_loss = -(t * log_pred).sum(dim=1)
        weighted_loss = (element_loss * w).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        final_loss = weighted_loss.item()

    return final_loss
