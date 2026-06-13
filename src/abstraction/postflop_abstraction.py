"""
PostflopAbstraction — per-street EHS-based card bucketing.

Provides the abstraction layer consumed by the subgame solver.
Deep CFR training does NOT use this module — it uses continuous
hand-strength features (NLHEEncoder dims 120–121) directly.

This module exists for tabular subgame solving, where:
  1. The game tree is small enough for explicit traversal.
  2. Information sets must be discrete (not neural-net-continuous).
  3. Hands with similar EHS are merged into the same info set.

EHS (Expected Hand Strength) computation per street:
    Preflop:  not handled here — use CardAbstraction.
    Flop:     Monte Carlo vs random opponent + random turn/river.
    Turn:     Monte Carlo vs random opponent + random river.
    River:    Exact — evaluate_7card deterministically.

EHS here is a WIN PROBABILITY in [0, 1] (wins + 0.5·ties over trials), so it
needs no score normalisation — there is intentionally no MAX-score constant in
this module. The only place a packed-evaluator score is normalised into [0, 1]
is the encoder's board-strength feature, which uses MAX_HAND_SCORE from
abstraction.equity; this module and that feature must not invent separate
magic maxima.

Bucketing:
    Equal-width bins on EHS ∈ [0, 1].  k bins per street.
    This is the standard Billings et al. (2003) approach,
    appropriate when EHS is approximately uniform (which it is
    for postflop boards in practice).

Range representation:
    range_to_buckets() converts a probability distribution over
    specific hole-card pairs into a distribution over buckets.
    This is the key operation for initialising subgame solving:
    the opponent's range (maintained during blueprint traversal)
    maps to a bucket distribution at the subgame root.

Usage:
    abstraction = PostflopAbstraction(num_buckets=8, n_sims=200)

    # Single hand query
    bucket = abstraction.bucket((48, 49), (0, 5, 10))   # AA on A23 flop

    # Range → bucket distribution (for subgame root)
    range_dict = {(48, 49): 0.04, (44, 45): 0.04, ...}  # combo → prob
    bucket_probs = abstraction.range_to_buckets(range_dict, board=(0, 5, 10))
"""

from __future__ import annotations

import numpy as np

from .equity import evaluate_7card


# ── Constants ─────────────────────────────────────────────────────────────────

_STREET_BY_BOARD_LEN: dict[int, str] = {3: "flop", 4: "turn", 5: "river"}


# ── EHS computation ───────────────────────────────────────────────────────────

def ehs(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
    n_sims: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Expected Hand Strength of hole_cards given board, vs a random opponent.

    For flop/turn: Monte Carlo over remaining deck.
    For river (len(board) == 5): exact evaluation, n_sims ignored.

    Args:
        hole_cards: (card1, card2), card = rank * 4 + suit.
        board:      visible community cards (3, 4, or 5 ints).
        n_sims:     Monte Carlo simulations (ignored on river).
        rng:        optional seeded RNG for reproducibility.

    Returns:
        float in [0, 1] — P(win + 0.5 * tie) vs random opponent.

    Note:
        EHS here is equity vs a *uniform random* opponent range, not
        vs the opponent's actual range.  This is standard practice for
        card abstraction (Billings 2003, Johanson 2013) — range-based
        reasoning is handled by the solver, not the abstraction.
    """
    board_len = len(board)
    assert board_len in (3, 4, 5), f"Expected 3–5 board cards, got {board_len}"

    # River: exact (deterministic)
    if board_len == 5:
        return _river_ehs_exact(hole_cards, board)

    # Flop / Turn: Monte Carlo
    if rng is None:
        rng = np.random.default_rng()

    dead = set(hole_cards) | set(board)
    deck = [c for c in range(52) if c not in dead]

    cards_needed = 2 + (5 - board_len)   # 2 opp cards + remaining board cards

    wins = 0.0
    total = 0

    for _ in range(n_sims):
        drawn = rng.choice(deck, size=cards_needed, replace=False)
        opp = (int(drawn[0]), int(drawn[1]))
        runout = tuple(int(drawn[i + 2]) for i in range(5 - board_len))

        full_board = board + runout
        hero    = evaluate_7card(hole_cards + full_board)
        villain = evaluate_7card(opp + full_board)

        if hero > villain:
            wins += 1.0
        elif hero == villain:
            wins += 0.5
        total += 1

    return wins / total


def _river_ehs_exact(
    hole_cards: tuple[int, int],
    board: tuple[int, ...],
) -> float:
    """
    Exact EHS on the river via enumeration over all remaining opponent hands.

    C(remaining_deck, 2) opponent combos — at most C(45, 2) = 990 combos.
    Fast enough to run inline during subgame traversal.
    """
    dead = set(hole_cards) | set(board)
    deck = [c for c in range(52) if c not in dead]
    hero_score = evaluate_7card(hole_cards + board)

    wins = 0.0
    total = 0

    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            opp = (deck[i], deck[j])
            villain_score = evaluate_7card(opp + board)
            if hero_score > villain_score:
                wins += 1.0
            elif hero_score == villain_score:
                wins += 0.5
            total += 1

    return wins / total if total > 0 else 0.5


# ── PostflopAbstraction ───────────────────────────────────────────────────────

class PostflopAbstraction:
    """
    Per-street EHS bucketing for tabular subgame solving.

    Maintains a lazy cache of computed EHS values keyed on
    (hole_cards, board) — safe for reuse across subgame roots.

    Args:
        num_buckets:        number of buckets per street (same for all streets)
                            or a dict {'flop': k1, 'turn': k2, 'river': k3}.
        n_sims:             Monte Carlo simulations per EHS query (flop/turn).
        seed:               RNG seed for reproducibility.
        cache_size:         max EHS cache entries.  0 = unlimited.
    """

    def __init__(
        self,
        num_buckets: int | dict[str, int] = 8,
        n_sims: int = 200,
        seed: int = 42,
        cache_size: int = 100_000,
    ):
        if isinstance(num_buckets, int):
            self._buckets: dict[str, int] = {
                "flop": num_buckets,
                "turn": num_buckets,
                "river": num_buckets,
            }
        else:
            required = {"flop", "turn", "river"}
            missing = required - set(num_buckets)
            if missing:
                raise ValueError(f"num_buckets missing keys: {missing}")
            self._buckets = dict(num_buckets)

        self._n_sims     = n_sims
        self._rng        = np.random.default_rng(seed)
        self._cache_size = cache_size

        # EHS cache: (hole_cards, board) → float
        self._ehs_cache: dict[tuple, float] = {}

    # ── Core API ──────────────────────────────────────────────────────────────

    def ehs(
        self,
        hole_cards: tuple[int, int],
        board: tuple[int, ...],
    ) -> float:
        """
        EHS of hole_cards given board, with caching.

        Cache key: (sorted(hole_cards), board) — suit/order independent
        for hole cards (AA♠A♥ == AA♥A♠), but board order matters.
        """
        key = (tuple(sorted(hole_cards)), board)
        if key in self._ehs_cache:
            return self._ehs_cache[key]

        value = ehs(hole_cards, board, n_sims=self._n_sims, rng=self._rng)

        if self._cache_size == 0 or len(self._ehs_cache) < self._cache_size:
            self._ehs_cache[key] = value
        # If cache full, evict nothing (simple dict, not true LRU).
        # For subgame solving, the working set is small per subgame root.

        return value

    def bucket(
        self,
        hole_cards: tuple[int, int],
        board: tuple[int, ...],
    ) -> int:
        """
        Map (hole_cards, board) to a bucket index.

        Args:
            hole_cards: (card1, card2).
            board:      3, 4, or 5 community cards.

        Returns:
            Bucket index in [0, num_buckets_for_this_street).
        """
        street = _STREET_BY_BOARD_LEN.get(len(board))
        if street is None:
            raise ValueError(
                f"Board length {len(board)} invalid for postflop. "
                "Use 3 (flop), 4 (turn), or 5 (river)."
            )
        k = self._buckets[street]
        strength = self.ehs(hole_cards, board)
        return min(int(strength * k), k - 1)

    def bucket_ehs_range(self, street: str) -> list[tuple[float, float]]:
        """
        Return (lo, hi) EHS boundaries for each bucket on a given street.
        Useful for human-readable strategy analysis.
        """
        k = self._buckets[street]
        return [(i / k, (i + 1) / k) for i in range(k)]

    # ── Range operations (subgame solver interface) ───────────────────────────

    def range_to_buckets(
        self,
        range_dict: dict[tuple[int, int], float],
        board: tuple[int, ...],
    ) -> np.ndarray:
        """
        Convert a hand-level range to a bucket probability distribution.

        Args:
            range_dict: {hole_cards: probability} — need not sum to 1
                        (will be normalised).  Hole cards that conflict
                        with board cards are automatically excluded.
            board:      visible community cards.

        Returns:
            np.ndarray of shape [num_buckets_for_this_street],
            summing to 1 (or all-zero if range_dict is empty / all dead).

        This is the key operation at a subgame root:
            opponent_range_at_root  →  bucket_probs
        The subgame solver initialises reach probabilities from this
        distribution rather than tracking individual combos.
        """
        street = _STREET_BY_BOARD_LEN.get(len(board))
        if street is None:
            raise ValueError(f"Board length {len(board)} invalid.")

        k = self._buckets[street]
        board_set = set(board)
        bucket_probs = np.zeros(k, dtype=np.float64)

        for hole_cards, prob in range_dict.items():
            if prob <= 0.0:
                continue
            # Skip combos that conflict with board (dead cards)
            if set(hole_cards) & board_set:
                continue
            b = self.bucket(hole_cards, board)
            bucket_probs[b] += prob

        total = bucket_probs.sum()
        if total > 0:
            bucket_probs /= total

        return bucket_probs.astype(np.float32)

    def buckets_to_range(
        self,
        bucket_probs: np.ndarray,
        board: tuple[int, ...],
        possible_combos: list[tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], float]:
        """
        Inverse of range_to_buckets: distribute bucket probabilities
        uniformly over the combos that map to each bucket.

        Used when translating a subgame solution back to specific hands
        (Abstraction-Solving-Translation pipeline, step 3).

        Args:
            bucket_probs:    [num_buckets] probability array.
            board:           visible community cards.
            possible_combos: specific combos to consider.  Defaults to
                             all 1326 two-card combinations not in board.

        Returns:
            {hole_cards: probability} dict, summing to 1.
        """
        board_set = set(board)
        if possible_combos is None:
            deck = [c for c in range(52) if c not in board_set]
            possible_combos = [
                (deck[i], deck[j])
                for i in range(len(deck))
                for j in range(i + 1, len(deck))
            ]

        # Group combos by bucket
        bucket_combos: dict[int, list[tuple[int, int]]] = {}
        for combo in possible_combos:
            if set(combo) & board_set:
                continue
            b = self.bucket(combo, board)
            bucket_combos.setdefault(b, []).append(combo)

        result: dict[tuple[int, int], float] = {}
        for b, prob in enumerate(bucket_probs):
            if prob <= 0 or b not in bucket_combos:
                continue
            combos = bucket_combos[b]
            per_combo = float(prob) / len(combos)
            for combo in combos:
                result[combo] = per_combo

        # Normalise
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear the EHS cache (useful between subgame roots)."""
        self._ehs_cache.clear()

    def cache_stats(self) -> dict:
        return {
            "size":     len(self._ehs_cache),
            "capacity": self._cache_size,
        }

    def summary(self) -> str:
        lines = ["PostflopAbstraction:"]
        for street in ("flop", "turn", "river"):
            k = self._buckets[street]
            ranges = self.bucket_ehs_range(street)
            lines.append(f"  {street:<6s}: {k} buckets, "
                         f"EHS bins [{ranges[0][0]:.2f}, {ranges[-1][1]:.2f}]")
        lines.append(f"  MC sims per query: {self._n_sims}")
        lines.append(f"  Cache: {len(self._ehs_cache)}/{self._cache_size} entries")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PostflopAbstraction("
                f"buckets={self._buckets}, "
                f"n_sims={self._n_sims})")