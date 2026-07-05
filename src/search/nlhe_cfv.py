"""
NLHE river-boundary PBS value-net interface (Stage B2).

The turn depth-limited solver queries river-root PBS values the same way
the Leduc round-1 solver queries reveal-boundary values (Phase 2). The
encoding follows the same findings:

  * Ranges as 50 strength-percentile buckets per player. Buckets are
    BOARD-RELATIVE (percentiles of 7-card strength on this exact board),
    which absorbs most board detail into the range representation — the
    encoding only adds residual texture features that change the
    bucket-vs-bucket dominance structure (flushes, pairs).
  * Values are POT-NORMALISED (v/pot): river pots span 4–100 chips and
    Huber training needs a stable scale. Callers multiply back by pot.
  * Per-bucket targets are own-range-weighted (bucket_values) — the same
    mass-weighting rationale as the Leduc net (off-support values are
    equilibrium-dependent noise).

Encoding (104 dims):
    [0]      pot / (2 · starting_stack)
    [1]      max same-suit count on board / 5   (flush texture)
    [2]      board paired flag
    [3]      top board rank / 12
    [4:54]   P0 bucket range (50)
    [54:104] P1 bucket range (50)
"""

from __future__ import annotations

import numpy as np

from .nlhe_river_vector import (
    RiverVectorCFR, bucket_map, bucket_range, bucket_values,
)

K_BUCKETS = 50
RIVER_ENCODING_DIMS = 4 + 2 * K_BUCKETS


def board_texture(board: tuple) -> tuple[float, float, float]:
    suits = [c % 4 for c in board]
    ranks = [c // 4 for c in board]
    flush = max(suits.count(s) for s in range(4)) / 5.0
    paired = 1.0 if len(set(ranks)) < len(ranks) else 0.0
    top = max(ranks) / 12.0
    return flush, paired, top


def encode_river_pbs(board: tuple, pot: float, starting_stack: float,
                     br0: np.ndarray, br1: np.ndarray) -> np.ndarray:
    enc = np.zeros(RIVER_ENCODING_DIMS, dtype=np.float32)
    enc[0] = pot / (2.0 * starting_stack)
    enc[1], enc[2], enc[3] = board_texture(board)
    enc[4:4 + K_BUCKETS] = br0
    enc[4 + K_BUCKETS:] = br1
    return enc


def river_target(game, node_history, x0: np.ndarray, x1: np.ndarray,
                 solve_iters: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Solve the river subgame exactly; return (encoding[104], target[100]).

    Target = pot-normalised per-bucket CFVs, [v0_buckets ++ v1_buckets].
    """
    board = tuple(node_history[2])
    pot = float(game._parse_state(node_history)["pot"])
    solver = RiverVectorCFR(game, node_history, x0, x1,
                            iterations=solve_iters)
    solver.solve()
    v0, v1 = solver.root_values()

    bm = bucket_map(board, K_BUCKETS)
    enc = encode_river_pbs(
        board, pot, float(game.starting_stack),
        bucket_range(x0, bm, K_BUCKETS),
        bucket_range(x1, bm, K_BUCKETS),
    )
    target = np.concatenate([
        bucket_values(v0, x0, bm, K_BUCKETS),
        bucket_values(v1, x1, bm, K_BUCKETS),
    ]).astype(np.float32) / max(pot, 1e-9)
    return enc, target
