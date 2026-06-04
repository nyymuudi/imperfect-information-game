"""
Preflop equity calculator using Monte Carlo simulation.

Computes hand equity by dealing random boards and evaluating
showdown outcomes. Used for card abstraction — grouping hands
with similar equity profiles into buckets — and as an explicit
input feature for the Deep CFR state encoder (dim 120).

Card encoding:
    card = rank * 4 + suit
    rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    suit: 0=♣, 1=♦, 2=♥, 3=♠

DETERMINISM NOTE
----------------
`canonical_preflop_equity(hand_class)` is the single source of truth for the
encoder's preflop-equity feature. It is DETERMINISTIC per hand class: the RNG
is seeded from a fixed base seed combined with the hand class itself, so the
same hand always yields the same equity regardless of computation order or
which process computes it. This is required so that:

  * the encoder's dim-120 feature is reproducible across training runs, and
  * the C++ engine can replicate the SAME value (it reproduces this table with
    an identical per-class seeding scheme), giving cross-implementation parity
    on dim 120 rather than the previous heuristic divergence.

The older `equity_vs_random(hole, rng=...)` is retained for ad-hoc / abstraction
use, but the encoder must go through `canonical_preflop_equity` so the feature
is stable and parity-checkable.
"""

import numpy as np
from itertools import combinations


NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

RANK_NAMES = "23456789TJQKA"
SUIT_NAMES = "cdhs"

# Fixed base seed for the canonical preflop-equity table. The C++ engine uses
# the same constant + the same per-class derivation so both sides agree.
PREFLOP_EQUITY_BASE_SEED = 1_550_000_000
# Number of Monte Carlo board+opponent samples per canonical hand class.
# 2000 keeps the standard error ~0.011, small relative to the 169-class spread.
PREFLOP_EQUITY_SIMS = 2000


def card_to_str(card: int) -> str:
    return RANK_NAMES[card // 4] + SUIT_NAMES[card % 4]


def str_to_card(s: str) -> int:
    return RANK_NAMES.index(s[0]) * 4 + SUIT_NAMES.index(s[1])


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


# ── Hand Evaluator ──────────────────────────────────────────────
#
# Returns a numeric hand strength value. Higher = better.
# Encoding: (category, *tiebreakers) packed into a single int.
#
# Categories (0-8): High Card, Pair, Two Pair, Trips, Straight,
#                   Flush, Full House, Quads, Straight Flush


def evaluate_5card(cards: tuple[int, ...]) -> int:
    """
    Evaluate a 5-card hand. Returns integer rank (higher = better).
    """
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight
    unique_ranks = sorted(set(ranks))
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[-1] - unique_ranks[0] == 4:
            is_straight = True
            straight_high = unique_ranks[-1]
        # Wheel: A-2-3-4-5
        elif unique_ranks == [0, 1, 2, 3, 12]:
            is_straight = True
            straight_high = 3  # 5-high straight

    # Count rank frequencies
    from collections import Counter
    rank_counts = Counter(ranks)
    # Sort by (count desc, rank desc)
    groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [g[1] for g in groups]
    group_ranks = [g[0] for g in groups]

    if is_straight and is_flush:
        return _pack(8, straight_high)
    if counts == [4, 1]:
        return _pack(7, group_ranks[0], group_ranks[1])
    if counts == [3, 2]:
        return _pack(6, group_ranks[0], group_ranks[1])
    if is_flush:
        return _pack(5, *ranks)
    if is_straight:
        return _pack(4, straight_high)
    if counts == [3, 1, 1]:
        return _pack(3, group_ranks[0], group_ranks[1], group_ranks[2])
    if counts == [2, 2, 1]:
        return _pack(2, group_ranks[0], group_ranks[1], group_ranks[2])
    if counts == [2, 1, 1, 1]:
        return _pack(1, group_ranks[0], group_ranks[1], group_ranks[2], group_ranks[3])
    # High card
    return _pack(0, *ranks)


def _pack(*values: int) -> int:
    """
    Pack category + tiebreakers into a single comparable integer.

    CRITICAL: always pad to exactly 6 values to ensure consistent
    comparison across different hand categories.
    """
    padded = list(values) + [0] * (6 - len(values))
    result = 0
    for v in padded:
        result = result * 15 + v
    return result


def evaluate_7card(cards: tuple[int, ...]) -> int:
    """
    Evaluate best 5-card hand from 7 cards.
    Exhaustive search over C(7,5) = 21 combinations.
    """
    best = 0
    for combo in combinations(cards, 5):
        val = evaluate_5card(combo)
        if val > best:
            best = val
    return best


# Maximum achievable evaluate_7card / evaluate_5card score: straight flush
# (category 8) with high card Ace (rank 12) → _pack(8, 12) = 8*15^5 + 12*15^4.
# Used to normalise board-strength into [0, 1]. This is the SAME constant the
# C++ encoder uses (see torch_model.cpp::board_strength), so dim 121 matches.
MAX_HAND_SCORE = float(_pack(8, 12))


# ── Equity Calculator ───────────────────────────────────────────


def equity_vs_random(
    hole_cards: tuple[int, int],
    num_simulations: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute equity of hole_cards vs a random opponent hand.

    Returns: float in [0, 1] — probability of winning (ties count 0.5).

    NOTE: for the encoder feature use `canonical_preflop_equity` instead;
    this function is non-deterministic unless you pass a per-call seeded rng.
    """
    if rng is None:
        rng = np.random.default_rng()

    deck = [c for c in range(NUM_CARDS) if c not in hole_cards]
    wins = 0.0
    total = 0

    for _ in range(num_simulations):
        # Draw 2 opponent cards + 5 community = 7 cards
        drawn = rng.choice(deck, size=7, replace=False)
        opp = (int(drawn[0]), int(drawn[1]))
        board = tuple(int(x) for x in drawn[2:7])

        hero = evaluate_7card(hole_cards + board)
        villain = evaluate_7card(opp + board)

        if hero > villain:
            wins += 1.0
        elif hero == villain:
            wins += 0.5
        total += 1

    return wins / total


# ── Canonical Hand Classes ──────────────────────────────────────
#
# 169 classes: 13 pairs + 78 suited + 78 offsuit
# Encoding: "AKs", "AKo", "AA", etc.


def canonical_hand_class(card1: int, card2: int) -> str:
    """Convert two cards to canonical hand class (e.g., 'AKs', '77')."""
    r1, r2 = card_rank(card1), card_rank(card2)
    s1, s2 = card_suit(card1), card_suit(card2)

    high, low = max(r1, r2), min(r1, r2)
    h_name = RANK_NAMES[high]
    l_name = RANK_NAMES[low]

    if high == low:
        return f"{h_name}{l_name}"
    elif s1 == s2:
        return f"{h_name}{l_name}s"
    else:
        return f"{h_name}{l_name}o"


def all_169_classes() -> list[str]:
    """Return all 169 canonical preflop hand classes."""
    classes = []
    for i in range(NUM_RANKS - 1, -1, -1):
        for j in range(NUM_RANKS - 1, -1, -1):
            if i == j:
                classes.append(f"{RANK_NAMES[i]}{RANK_NAMES[j]}")
            elif i > j:
                classes.append(f"{RANK_NAMES[i]}{RANK_NAMES[j]}s")
            else:
                classes.append(f"{RANK_NAMES[j]}{RANK_NAMES[i]}o")
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in classes:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def representative_hand(hand_class: str) -> tuple[int, int]:
    """
    Return a representative card pair for a hand class.
    Uses suit 0 and suit 1 for offsuit, suit 0 for suited/pair.
    """
    r1 = RANK_NAMES.index(hand_class[0])
    r2 = RANK_NAMES.index(hand_class[1])

    if len(hand_class) == 2:
        # Pair: e.g., "AA" → A♣, A♦
        return (r1 * 4 + 0, r2 * 4 + 1)
    elif hand_class[2] == 's':
        # Suited: same suit
        return (r1 * 4 + 0, r2 * 4 + 0)
    else:
        # Offsuit: different suits
        return (r1 * 4 + 0, r2 * 4 + 1)


def _class_seed(hand_class: str) -> int:
    """
    Derive a stable per-class RNG seed from the hand class string.

    Deterministic and order-independent: the same class always seeds the same
    way, so its equity is reproducible across runs and across the C++ port
    (which derives the seed identically from the class index).
    """
    # Index into the canonical 169 list gives a small, stable integer that the
    # C++ side reproduces from the same ordering.
    idx = _CLASS_INDEX[hand_class]
    return PREFLOP_EQUITY_BASE_SEED + idx


_CLASS_INDEX = {hc: i for i, hc in enumerate(all_169_classes())}


def canonical_preflop_equity(
    hand_class: str,
    num_simulations: int = PREFLOP_EQUITY_SIMS,
) -> float:
    """
    Deterministic preflop equity of a canonical hand class vs a random hand.

    The RNG is seeded PER CLASS, so the result depends only on the class and
    the (fixed) base seed + sim count — never on computation order. This is the
    value the encoder writes to dim 120 and the value the C++ encoder mirrors.
    """
    cards = representative_hand(hand_class)
    rng = np.random.default_rng(_class_seed(hand_class))
    return equity_vs_random(cards, num_simulations=num_simulations, rng=rng)


def _equity_table_cache_path(num_simulations: int) -> "object":
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parent / "_equity_cache"
    return cache_dir / f"preflop_equity_{num_simulations}.json"


def preflop_equity_table(
    num_simulations: int = PREFLOP_EQUITY_SIMS,
    seed: int | None = None,
    use_disk_cache: bool = True,
) -> dict[str, float]:
    """
    Compute equity vs random for all 169 canonical preflop hands.

    Deterministic per class (the `seed` argument is accepted for backward
    compatibility but ignored — each class is seeded from its own stable seed).

    With use_disk_cache=True the full table is memoised to a JSON file keyed by
    sim count. The first call computes and writes it (~minutes at high sim
    counts); subsequent calls and other processes load it instantly. The C++
    engine can read the SAME JSON, giving exact dim-120 parity without rerunning
    Monte Carlo on its side.

    Returns dict: hand_class → equity (e.g., {"AA": 0.852, "72o": 0.345}).
    """
    if use_disk_cache:
        import json
        from pathlib import Path
        path = _equity_table_cache_path(num_simulations)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if len(data) == 169:
                    return data
            except (json.JSONDecodeError, OSError):
                pass  # fall through to recompute

    table = {hc: canonical_preflop_equity(hc, num_simulations)
             for hc in all_169_classes()}

    if use_disk_cache:
        import json
        from pathlib import Path
        path = _equity_table_cache_path(num_simulations)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(table, indent=0))
        except OSError:
            pass  # caching is best-effort; never fail the caller

    return table