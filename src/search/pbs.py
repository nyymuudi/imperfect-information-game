"""
Public Belief State (PBS) for Leduc Hold'em.

PBS = (public state, P1 range, P2 range)  — Moravčík et al. 2017 (DeepStack),
Brown et al. 2020 (ReBeL).

Public state in Leduc: the action sequence from the game start plus the
community card once revealed. Ranges are probability vectors over the six
private holdings (J1, J2, Q1, Q2, K1, K2), maintained as per-player
marginals; the joint card-removal constraint (players cannot share a card,
nobody holds the community card) is applied when a subgame enumerates
concrete deals, exactly as DeepStack does.

Bayesian range update after an observed action a by player p:

    range_p[h]  ∝  range_p[h] · σ_p(infoset(h, public), a)

using the strategy σ that player p is modelled to follow (the blueprint for
the opponent; the actually-played re-solved strategy for the hero).
After the community card c is revealed, both ranges zero out c and
renormalise (card removal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..games.leduc import LeducHoldem


LEDUC_CARDS: tuple[str, ...] = ("J1", "J2", "Q1", "Q2", "K1", "K2")
CARD_IDX: dict[str, int] = {c: i for i, c in enumerate(LEDUC_CARDS)}
N_CARDS = len(LEDUC_CARDS)

# Degenerate-range fallback threshold: below this total mass the observed
# line has zero probability under the modelled strategy and Bayes is
# undefined — reset to uniform over consistent cards (standard practice).
_ZERO_MASS_EPS = 1e-12


@dataclass(frozen=True)
class LeducPBS:
    """Immutable PBS node. Updates return new instances."""

    actions: tuple[str, ...]              # public actions from game start
    community: str | None                 # None until revealed
    ranges: tuple[tuple[float, ...], ...]  # (2, 6) row-normalised marginals

    def range_array(self, player: int) -> np.ndarray:
        return np.asarray(self.ranges[player], dtype=np.float64)


def initial_pbs() -> LeducPBS:
    uniform = tuple(1.0 / N_CARDS for _ in range(N_CARDS))
    return LeducPBS(actions=(), community=None, ranges=(uniform, uniform))


def representative_history(
    card: str,
    player: int,
    community: str | None,
    actions: tuple[str, ...],
) -> tuple:
    """Concrete history whose infoset for `player` matches (card, public).

    The opponent's slot and (pre-reveal) the community slot are filled with
    arbitrary distinct placeholder cards: LeducHoldem.info_set_key and
    LeducEncoder.encode read only the acting player's own card, the
    community (post-reveal), and the action string.
    """
    used = {card}
    if community is not None:
        used.add(community)
    fillers = [c for c in LEDUC_CARDS if c not in used]
    opp_card = fillers[0]
    comm = community if community is not None else fillers[1]
    if player == 0:
        return (card, opp_card, comm) + actions
    return (opp_card, card, comm) + actions


def _normalised(vec: np.ndarray, community: str | None) -> tuple[float, ...]:
    """Renormalise; on zero mass fall back to uniform over consistent cards."""
    total = float(vec.sum())
    if total > _ZERO_MASS_EPS:
        return tuple(float(x) for x in vec / total)
    fallback = np.ones(N_CARDS, dtype=np.float64)
    if community is not None:
        fallback[CARD_IDX[community]] = 0.0
    fallback /= fallback.sum()
    return tuple(float(x) for x in fallback)


def update_on_action(
    pbs: LeducPBS,
    game: LeducHoldem,
    player: int,
    action: str,
    strategy_fn,
) -> LeducPBS:
    """Bayes update of `player`'s range after they took `action`.

    strategy_fn(history, player) -> probs aligned with
    game.legal_actions(history); modelled strategy of the acting player.
    """
    new_range = pbs.range_array(player).copy()
    for i, card in enumerate(LEDUC_CARDS):
        if new_range[i] <= 0.0:
            continue
        h = representative_history(card, player, pbs.community, pbs.actions)
        legal = game.legal_actions(h)
        probs = np.asarray(strategy_fn(h, player), dtype=np.float64)
        a_idx = legal.index(action)
        new_range[i] *= max(0.0, float(probs[a_idx]))

    ranges = list(pbs.ranges)
    ranges[player] = _normalised(new_range, pbs.community)
    return LeducPBS(
        actions=pbs.actions + (action,),
        community=pbs.community,
        ranges=tuple(ranges),
    )


def update_on_community(pbs: LeducPBS, card: str) -> LeducPBS:
    """Card-removal update when the community card is revealed."""
    ranges = []
    for p in range(2):
        vec = pbs.range_array(p)
        vec[CARD_IDX[card]] = 0.0
        ranges.append(_normalised(vec, card))
    return LeducPBS(actions=pbs.actions, community=card, ranges=tuple(ranges))
