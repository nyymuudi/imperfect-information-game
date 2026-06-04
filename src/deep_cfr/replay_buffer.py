"""
Experience replay buffer for Deep CFR.

ReservoirBuffer maintains a fixed-capacity, uniform sample over ALL inserted
data using Vitter Algorithm R. This is the buffer Deep CFR / Single-Deep-CFR
require for BOTH memories:

    Regret (value) buffer (MR):
        The value network is re-fitted each iteration on a reservoir drawn from
        samples across ALL iterations; that reservoir is how the network comes
        to approximate CUMULATIVE counterfactual regret without explicit
        summation. It must be LARGE.

    Strategy buffer (MΠ):
        Approximates the time-average strategy across all iterations, which is
        the quantity that converges to Nash. Reservoir over the full history.

A 'window' (FIFO) mode is also provided for experimentation, but it is NOT the
right choice for Deep CFR: keeping only the freshest samples makes the network
fit only the latest iteration's instantaneous regrets, which does not converge
(verified: flat exploitability on Leduc). The default is therefore 'reservoir'.

Reference: Vitter (1985). "Random Sampling with a Reservoir."
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ReservoirBuffer:
    """
    Fixed-capacity replay buffer.

    mode='reservoir' (default): uniform sample over all history (Vitter Alg. R)
    mode='window':              circular FIFO — K most recent samples only
    """
    capacity: int
    state_size: int
    action_size: int
    mode: str = 'reservoir'   # 'reservoir' | 'window'

    def __post_init__(self):
        if self.mode not in ('reservoir', 'window'):
            raise ValueError(f"mode must be 'reservoir' or 'window', got {self.mode!r}")
        self.states  = np.zeros((self.capacity, self.state_size),  dtype=np.float32)
        self.targets = np.zeros((self.capacity, self.action_size), dtype=np.float32)
        self.weights = np.zeros(self.capacity,                     dtype=np.float32)
        self.size        = 0
        self.total_added = 0
        self._head       = 0   # next write position (window mode)
        self._rng = np.random.default_rng(42)

    # ── Single insert ─────────────────────────────────────────────────────────

    def add(self, state: np.ndarray, target: np.ndarray, weight: float):
        if self.mode == 'window':
            idx = self._head % self.capacity
            self._head += 1
        else:
            # Reservoir (Vitter Algorithm R)
            if self.size < self.capacity:
                idx = self.size
            else:
                idx = int(self._rng.integers(0, self.total_added + 1))
                if idx >= self.capacity:
                    self.total_added += 1
                    return

        self.states[idx] = state
        padded = np.zeros(self.action_size, dtype=np.float32)
        padded[:len(target)] = target
        self.targets[idx] = padded
        self.weights[idx] = weight
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    # ── Batch insert (vectorised) ─────────────────────────────────────────────

    def add_batch(self,
                  states:  np.ndarray,
                  targets: np.ndarray,
                  weights: np.ndarray) -> None:
        """
        Vectorised batch insert — O(1) numpy ops instead of O(N) Python loop.

        Each row is expected to be one UNIQUE state with its FULL action-target
        vector (see DeepCFRSolver._collapse_by_state). Do not feed per-action
        one-hot rows here: collapse them by state first, or the network is
        trained on conflicting targets for the same input.
        """
        n = len(states)
        if n == 0:
            return

        if self.mode == 'window':
            start = self._head % self.capacity
            end   = start + n
            if end <= self.capacity:
                self.states[start:end]  = states
                self.targets[start:end] = targets
                self.weights[start:end] = weights
            else:
                first = self.capacity - start
                self.states[start:]  = states[:first]
                self.targets[start:] = targets[:first]
                self.weights[start:] = weights[:first]
                rest = n - first
                self.states[:rest]  = states[first:]
                self.targets[:rest] = targets[first:]
                self.weights[:rest] = weights[first:]
            self._head       += n
            self.size         = min(self.size + n, self.capacity)
            self.total_added += n

        else:
            # Reservoir sampling batch insert.
            n_fill = min(n, self.capacity - self.size)
            if n_fill > 0:
                slots = np.arange(self.size, self.size + n_fill)
                self.states[slots]  = states[:n_fill]
                self.targets[slots] = targets[:n_fill]
                self.weights[slots] = weights[:n_fill]
                self.size        += n_fill
                self.total_added += n_fill

            n_over = n - n_fill
            if n_over <= 0:
                return

            t_base = self.total_added
            drawn  = self._rng.integers(
                0, (t_base + np.arange(1, n_over + 1)).astype(np.int64))
            keep   = np.where(drawn < self.capacity)[0]

            if len(keep) > 0:
                write_slots    = drawn[keep]
                sample_indices = n_fill + keep
                self.states[write_slots]  = states[sample_indices]
                self.targets[write_slots] = targets[sample_indices]
                self.weights[write_slots] = weights[sample_indices]

            self.total_added += n_over

    # ── Sample ────────────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        indices = self._rng.integers(0, self.size, size=min(batch_size, self.size))
        return (self.states[indices], self.targets[indices], self.weights[indices])

    def __len__(self):
        return self.size

    def clear(self):
        self.size        = 0
        self.total_added = 0
        self._head       = 0