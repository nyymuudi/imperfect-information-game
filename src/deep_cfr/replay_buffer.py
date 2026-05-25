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
        self.states  = np.zeros((self.capacity, self.state_size),  dtype=np.float32)
        self.targets = np.zeros((self.capacity, self.action_size), dtype=np.float32)
        self.weights = np.zeros(self.capacity,                     dtype=np.float32)
        self.size        = 0
        self.total_added = 0
        self._rng = np.random.default_rng(42)

    def add(self, state: np.ndarray, target: np.ndarray, weight: float):
        """Add a single sample (original interface — unchanged)."""
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = self._rng.integers(0, self.total_added + 1)
            if idx >= self.capacity:
                self.total_added += 1
                return

        self.states[idx] = state
        padded = np.zeros(self.action_size, dtype=np.float32)
        padded[:len(target)] = target
        self.targets[idx] = padded
        self.weights[idx] = weight
        self.total_added += 1

    def add_batch(self,
                  states:  np.ndarray,
                  targets: np.ndarray,
                  weights: np.ndarray) -> None:
        """
        Vectorized batch insert — O(1) numpy ops instead of O(N) Python loop.

        Args:
            states:  [N, state_size]  float32
            targets: [N, action_size] float32
            weights: [N]              float32

        Replaces the for-loop in _run_cpp_iteration; critical for C++ backend
        performance where N can be 50k–100k samples per iteration.
        """
        n = len(states)
        if n == 0:
            return

        # ── Phase 1: sequential fill into empty slots ─────────────────────────
        n_fill = min(n, self.capacity - self.size)
        if n_fill > 0:
            slots = np.arange(self.size, self.size + n_fill)
            self.states[slots]  = states[:n_fill]
            self.targets[slots] = targets[:n_fill]
            self.weights[slots] = weights[:n_fill]
            self.size        += n_fill
            self.total_added += n_fill

        # ── Phase 2: reservoir sampling for overflow ──────────────────────────
        n_over = n - n_fill
        if n_over <= 0:
            return

        # For overflow sample i: draw uniform integer in [0, total_added + i).
        # Keep if drawn index < capacity.
        t_base = self.total_added
        # Generate N random values where draw_i ~ Uniform[0, t_base + i)
        # Approximation: draw from [0, t_base + N) and scale — exact enough
        # for large buffers. For exact Vitter, use per-sample bounds below.
        drawn = self._rng.integers(
            0,
            (t_base + np.arange(1, n_over + 1)).astype(np.int64),
        )
        keep = np.where(drawn < self.capacity)[0]  # indices into overflow slice

        if len(keep) > 0:
            write_slots    = drawn[keep]                    # where to write in buffer
            sample_indices = n_fill + keep                  # which incoming samples
            self.states[write_slots]  = states[sample_indices]
            self.targets[write_slots] = targets[sample_indices]
            self.weights[write_slots] = weights[sample_indices]

        self.total_added += n_over

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
        self.size        = 0
        self.total_added = 0