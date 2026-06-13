"""
Card abstraction for NLHE.

Groups the 169 canonical preflop hand classes into k buckets
based on equity similarity. Uses k-means clustering on equity
vs random opponent.

This is the standard approach from Billings et al. (2003) and
the Abstraction-Solving-Translation pipeline used in all
superhuman poker AIs.

The abstraction allows CFR to operate on ~k info sets per decision
point instead of ~169, making the game tree tractable.
"""

import numpy as np
from dataclasses import dataclass
from .equity import preflop_equity_table, all_169_classes


@dataclass
class CardAbstraction:
    """
    Maps 169 canonical preflop hands to k equity buckets.
    
    Bucket 0 = weakest hands, bucket k-1 = strongest hands.
    """
    num_buckets: int
    hand_to_bucket: dict[str, int]
    bucket_ranges: list[tuple[float, float]]  # (min_eq, max_eq) per bucket
    bucket_exemplars: list[str]                # Representative hand per bucket

    @classmethod
    def from_equity(
        cls,
        num_buckets: int = 10,
        num_simulations: int = 3000,
        seed: int = 42,
    ) -> "CardAbstraction":
        """
        Build abstraction by clustering hands on preflop equity.
        
        Simple approach: equal-width equity bins. K-means would be
        more sophisticated but for preflop equity (which is roughly
        uniform), equal-width bins are nearly equivalent.
        """
        print(f"Computing preflop equity table ({num_simulations} sims/hand)...")
        equity = preflop_equity_table(num_simulations=num_simulations, seed=seed)
        
        classes = all_169_classes()
        equities = np.array([equity[h] for h in classes])
        
        # Equal-width binning on equity
        eq_min, eq_max = equities.min(), equities.max()
        bin_width = (eq_max - eq_min) / num_buckets
        
        hand_to_bucket = {}
        buckets_hands: dict[int, list[tuple[str, float]]] = {i: [] for i in range(num_buckets)}
        
        for hand_class, eq in zip(classes, equities):
            bucket = min(int((eq - eq_min) / bin_width), num_buckets - 1)
            hand_to_bucket[hand_class] = bucket
            buckets_hands[bucket].append((hand_class, eq))
        
        # Compute bucket ranges and exemplars
        bucket_ranges = []
        bucket_exemplars = []
        for i in range(num_buckets):
            hands = buckets_hands[i]
            if hands:
                eqs = [e for _, e in hands]
                bucket_ranges.append((min(eqs), max(eqs)))
                # Exemplar = hand closest to bucket mean
                mean_eq = np.mean(eqs)
                best_hand = min(hands, key=lambda x: abs(x[1] - mean_eq))
                bucket_exemplars.append(best_hand[0])
            else:
                bucket_ranges.append((0.0, 0.0))
                bucket_exemplars.append("??")
        
        return cls(
            num_buckets=num_buckets,
            hand_to_bucket=hand_to_bucket,
            bucket_ranges=bucket_ranges,
            bucket_exemplars=bucket_exemplars,
        )

    def get_bucket(self, hand_class: str) -> int:
        """Return bucket index for a hand class."""
        return self.hand_to_bucket[hand_class]

    def summary(self) -> str:
        """Human-readable abstraction summary."""
        lines = [
            f"Card Abstraction: {self.num_buckets} buckets",
            f"{'=' * 55}",
        ]
        
        # Count hands per bucket
        bucket_counts: dict[int, list[str]] = {i: [] for i in range(self.num_buckets)}
        for hand, bucket in self.hand_to_bucket.items():
            bucket_counts[bucket].append(hand)
        
        for i in range(self.num_buckets):
            lo, hi = self.bucket_ranges[i]
            count = len(bucket_counts[i])
            hands_sample = ", ".join(sorted(bucket_counts[i])[:5])
            if count > 5:
                hands_sample += f", ... (+{count - 5})"
            lines.append(
                f"  Bucket {i:2d}: equity [{lo:.3f}-{hi:.3f}] "
                f"| {count:3d} hands | e.g. {hands_sample}"
            )
        
        return "\n".join(lines)