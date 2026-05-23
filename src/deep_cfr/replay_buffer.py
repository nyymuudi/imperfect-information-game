"""
Experience replay buffers for Deep CFR.

Two separate buffers:
    MR (Regret Memory): stores (state, regrets, iteration_weight)
    MΠ (Strategy Memory): stores (state, strategy, iteration_weight)

Uses reservoir sampling to maintain a representative sample
when the buffer is full, ensuring uniform coverage over all
iterations without unbounded memory growth.

Reference: Vitter (1985). "Random Sampling with a Reservoir."
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ReservoirBuffer:
    """
    Fixed-capacity replay buffer with reservoir sampling.
    
    When the buffer is full, new samples replace random existing
    samples with decreasing probability, maintaining a uniform
    sample over all inserted data.
    """
    capacity: int
    state_size: int
    action_size: int  # Max number of actions

    def __post_init__(self):
        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.targets = np.zeros((self.capacity, self.action_size), dtype=np.float32)
        self.weights = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.total_added = 0
        self._rng = np.random.default_rng(42)

    def add(self, state: np.ndarray, target: np.ndarray, weight: float):
        """
        Add a sample to the buffer.
        
        Args:
            state: Encoded game state vector
            target: Regret values or strategy probabilities
            weight: Iteration number (for Linear CFR weighting)
        """
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            # Reservoir sampling: replace random element with prob capacity/total
            idx = self._rng.integers(0, self.total_added + 1)
            if idx >= self.capacity:
                self.total_added += 1
                return  # Don't add this sample

        self.states[idx] = state
        # Pad target to action_size
        padded = np.zeros(self.action_size, dtype=np.float32)
        padded[:len(target)] = target
        self.targets[idx] = padded
        self.weights[idx] = weight
        self.total_added += 1

    def sample_batch(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch from the buffer.
        
        Returns:
            (states, targets, weights) — numpy arrays of shape
            (batch_size, ...) ready for neural network training.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = self._rng.integers(0, self.size, size=min(batch_size, self.size))
        return (
            self.states[indices],
            self.targets[indices],
            self.weights[indices],
        )

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        self.total_added = 0