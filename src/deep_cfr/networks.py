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

    Architecture: MLP trunk + linear regret head. Optional auxiliary EV
    head taps the post-trunk hidden activation and is trained on the
    CFR cache's per-action EV targets (slots [43:49] of the augmented
    state vector). The aux loss forces the shared trunk to encode
    advisor-derived value information, which the main regret head can
    then leverage — addresses the observation that v15_c2_v4's network
    produced identical output with and without filled advisor dims.

    The ``net`` Sequential alias is preserved for backward compatibility
    with ``cpp_backend.export_for_libtorch`` (which deep-copies ``net``).
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_size: int = 256, ev_head_dim: int = 0):
        super().__init__()
        self.ev_head_dim = ev_head_dim
        # Trunk + regret head as one Sequential so that ``self.net`` keeps
        # the legacy single-output layout for LibTorch export.
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
        if ev_head_dim > 0:
            # EV head taps the last hidden activation (the ReLU output
            # at index 5 in self.net). Forward computes both heads in
            # one pass when ``return_aux=True``.
            self.ev_head = nn.Linear(hidden_size, ev_head_dim)

    def forward(self, x: torch.Tensor,
                return_aux: bool = False) -> torch.Tensor:
        if not return_aux or self.ev_head_dim == 0:
            return self.net(x)
        # Run all trunk layers except the final Linear (= regret head).
        h = x
        layers = list(self.net.children())
        for layer in layers[:-1]:
            h = layer(h)
        regret = layers[-1](h)
        ev = self.ev_head(h)
        return regret, ev


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
    aux_ev_weight: float = 0.0,
    aux_ev_offset: int = 43,
    aux_ev_dim:    int = 6,
    exploit_gap_map: dict | None = None,
    exploit_gap_lambda: float = 0.0,
    exploit_gap_encoder=None,
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

    When ``aux_ev_weight > 0`` AND the network has an ``ev_head``, an
    auxiliary MSE loss is added against the advisor EV slots in the input
    state vector (default slots [43:49] = CFR cache per-action EVs). The
    aux loss forces the shared trunk to encode the cache's EV signal,
    rather than letting the regret head ignore the 12 advisor dims.

    Returns average TOTAL loss over the final epoch.
    """
    if len(buffer) < batch_size:
        return 0.0

    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    use_aux = aux_ev_weight > 0.0 and getattr(network, "ev_head_dim", 0) > 0
    use_bd  = (exploit_gap_map is not None and exploit_gap_lambda > 0
               and exploit_gap_encoder is not None)
    if use_bd:
        # Normalise gaps to median for stable lambda interpretation. Mapping
        # values are in mbb/decision; median is a reasonable per-spot baseline.
        import numpy as _np
        from .cfr_cache import key_from_state_vector as _key_fn
        _gap_values = _np.fromiter(exploit_gap_map.values(),
                                    dtype=_np.float32,
                                    count=len(exploit_gap_map))
        _gap_median = float(_np.median(_gap_values)) if len(_gap_values) else 1.0
        _gap_median = max(_gap_median, 1.0)

    final_loss = 0.0
    for _ in range(epochs):
        states, targets, weights = buffer.sample_batch(batch_size)
        device = next(network.parameters()).device

        if use_bd:
            # Per-sample BRD weight: 1 + λ * min(gap / median_gap, MAX_RATIO).
            # Spots above median exploitation get amplified gradient;
            # spots below median (or unmapped, weight 1.0) stay neutral.
            # Outlier spots (gap >> median) are clipped to MAX_RATIO=10 so
            # a single mega-leak doesn't dominate the batch — without
            # clipping, 24k-mbb spots could get 160× weight on λ=5 and
            # destabilise the Adam optimiser.
            import numpy as _np
            BD_MAX_RATIO = 10.0
            bd_mul = _np.ones(len(states), dtype=_np.float32)
            for i in range(len(states)):
                try:
                    k = _key_fn(states[i], exploit_gap_encoder)
                    g = exploit_gap_map.get(int(k))
                    if g is not None:
                        ratio = min(float(g) / _gap_median, BD_MAX_RATIO)
                        bd_mul[i] = 1.0 + exploit_gap_lambda * ratio
                except Exception:
                    pass
            weights = weights * bd_mul

        s = torch.tensor(states,  dtype=torch.float32).to(device)
        t = torch.tensor(targets, dtype=torch.float32).to(device)
        w = torch.tensor(weights, dtype=torch.float32).to(device)

        if use_aux:
            pred_regret, pred_ev = network(s, return_aux=True)
            ev_target = s[:, aux_ev_offset : aux_ev_offset + aux_ev_dim]
            regret_loss = F.huber_loss(pred_regret, t, reduction='none').mean(dim=1)
            ev_loss     = F.mse_loss(pred_ev, ev_target, reduction='none').mean(dim=1)
            element_loss = regret_loss + aux_ev_weight * ev_loss
            weighted_loss = (element_loss * w).mean()
        else:
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
