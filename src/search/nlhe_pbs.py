"""
NLHE Public Belief State ranges: Bayes-updated hole-combo distributions.

Phase 3 of the ReBeL extension. The June live-resolve experiments pinned
the hero to its actual hand and gave the opponent a UNIFORM range — the
measured Δ≈0 is consistent with those ranges carrying no information.
This module computes proper DeepStack-style ranges for BOTH players at
any decision point by replaying the public action sequence and applying

    range_p[combo] ∝ range_p[combo] · σ_p(observed action | combo, public)

with the blueprint as σ, plus card-removal for the visible board.

Ranges are dicts {(c1, c2): prob} over sorted hole-card combos — the
exact format UnsafeSubgameSolver.solve() consumes.

Perf note: each action step batches all live combos of the acting player
into ONE network forward (the legal-action slot mask is shared: legal
actions depend only on the public state, never on private cards).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from ..deep_cfr.action_slots import legal_actions_to_slots

_ZERO_MASS_EPS = 1e-12


def _all_combos() -> list[tuple[int, int]]:
    return [tuple(p) for p in combinations(range(52), 2)]


def _substitute(history: tuple, player: int, combo: tuple) -> tuple:
    h = list(history)
    h[player] = combo
    return tuple(h)


def _batched_slot_probs(blueprint, states: np.ndarray,
                        slots: list[int]) -> np.ndarray:
    """[n, len(slots)] probabilities at the given output slots, renormalised
    per row. One forward pass; the slot mask is shared across rows."""
    action_size = blueprint.metadata.action_size
    device = blueprint._device
    s = torch.as_tensor(states, dtype=torch.float32, device=device)
    mask = torch.zeros(len(states), action_size, device=device)
    mask[:, slots] = 1.0
    with torch.no_grad():
        out = blueprint._net(s, mask).cpu().numpy()
    picked = out[:, slots].astype(np.float64)
    row_sum = picked.sum(axis=1, keepdims=True)
    n = len(slots)
    uniform = np.full_like(picked, 1.0 / n)
    return np.where(row_sum > 1e-9, picked / np.where(row_sum > 1e-9,
                                                      row_sum, 1.0), uniform)


def compute_ranges(
    blueprint,
    encoder,
    game,
    history: tuple,
) -> tuple[dict, dict]:
    """Bayes ranges for both players at `history`'s decision point.

    Replays the action prefix from the deal; multiplies the acting
    player's range by the blueprint probability of the observed action
    for every live combo. Combos overlapping the CURRENTLY VISIBLE board
    are excluded at the end (equivalent to progressive removal: per-combo
    updates are independent and normalisation happens once).

    Returns (range_p0, range_p1) as {combo: prob} dicts, each summing
    to 1. A zero-mass line (blueprint prob 0 for every combo) falls back
    to uniform over live combos — Bayes is undefined there.
    """
    combos = _all_combos()
    n = len(combos)
    masses = [np.ones(n), np.ones(n)]

    n_actions = len(history) - 3
    for k in range(n_actions):
        prefix = history[: 3 + k]
        if game.is_terminal(prefix):
            break
        player = game.current_player(prefix)
        legal = game.legal_actions(prefix)
        action = history[3 + k]
        a_idx = legal.index(action)
        slots = legal_actions_to_slots(legal, blueprint.metadata.action_size)

        live = np.where(masses[player] > _ZERO_MASS_EPS)[0]
        if len(live) == 0:
            continue
        states = np.stack([
            encoder.encode(_substitute(prefix, player, combos[i]), player)
            for i in live
        ])
        probs = _batched_slot_probs(blueprint, states, slots)
        masses[player][live] *= probs[:, a_idx]

    # Card removal: visible board at the current street.
    state = game._parse_state(history)
    n_visible = (0, 3, 4, 5)[min(int(state["street_idx"]), 3)]
    board_set = set(history[2][:n_visible])

    out = []
    for p in range(2):
        m = masses[p].copy()
        for i, c in enumerate(combos):
            if board_set & set(c):
                m[i] = 0.0
        total = m.sum()
        if total <= _ZERO_MASS_EPS:
            m = np.asarray([0.0 if (board_set & set(c)) else 1.0
                            for c in combos])
            total = m.sum()
        m /= total
        out.append({c: float(m[i]) for i, c in enumerate(combos)
                    if m[i] > 0.0})
    return out[0], out[1]


def top_k_range(range_dict: dict, k: int,
                must_include: tuple | None = None,
                floor: float = 0.0) -> dict:
    """Truncate a range to its k most probable combos (renormalised).

    The deal cross-product in UnsafeSubgameSolver is O(|r0|·|r1|) in
    Python — full 1326-combo ranges are intractable there. ``must_include``
    forces a combo (the hero's actual hand) into the kept set with at
    least ``floor`` mass so the resolved strategy is defined for it.
    """
    items = sorted(range_dict.items(), key=lambda kv: -kv[1])[:k]
    kept = dict(items)
    if must_include is not None:
        key = tuple(sorted(must_include))
        kept[key] = max(kept.get(key, 0.0), range_dict.get(key, 0.0), floor)
    total = sum(kept.values())
    if total <= 0.0:
        n = len(kept)
        return {c: 1.0 / n for c in kept}
    return {c: p / total for c, p in kept.items()}


def range_sampler(range_dict: dict, exclude: set):
    """Sampler over a range for LBR opponent marginalisation.

    Returns callable(rng, n) -> list of combos drawn from the range
    (restricted to combos disjoint from `exclude`, renormalised).
    """
    items = [(c, p) for c, p in range_dict.items()
             if not (set(c) & exclude) and p > 0.0]
    if not items:
        return None
    combos = [c for c, _ in items]
    probs = np.asarray([p for _, p in items], dtype=np.float64)
    probs /= probs.sum()

    def sample(rng: np.random.Generator, n: int) -> list:
        idx = rng.choice(len(combos), size=n, p=probs)
        return [combos[i] for i in idx]

    return sample
