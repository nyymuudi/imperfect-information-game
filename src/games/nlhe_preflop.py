"""
Heads-Up Preflop No-Limit Hold'em with abstraction.

This is the first game in the project that REQUIRES abstraction —
the unabstracted game tree is intractable for vanilla CFR.

Abstraction layers:
    1. Card abstraction: 169 canonical hands → k equity buckets
    2. Action abstraction: continuous bet sizing → discrete actions
    
Architecture:
    - SB (P0) posts 0.5BB, BB (P1) posts 1BB
    - Effective stack: configurable (default 100BB)
    - Actions: fold, call, raise (fixed sizes), all-in
    - Maximum raises per street: configurable (default 4)
    
The game interfaces with CFR through the standard ExtensiveFormGame
interface. The solver sees buckets, not cards — this is the
Abstraction-Solving-Translation pipeline.
"""

from dataclasses import dataclass
from .base import ExtensiveFormGame, History, Action, InfoSetKey
from ..abstraction.card_abstraction import CardAbstraction
from ..abstraction.equity import all_169_classes


# Number of combos for each canonical hand type
def _hand_combos(hand_class: str) -> int:
    """Number of specific card combinations for a canonical hand class."""
    if len(hand_class) == 2:
        return 6     # Pair: C(4,2)
    elif hand_class[2] == 's':
        return 4     # Suited: one per suit
    else:
        return 12    # Offsuit: 4×3


@dataclass
class PreflopNLHE(ExtensiveFormGame):
    """
    Abstracted heads-up preflop NLHE.
    
    History encoding:
        (bucket_p0, bucket_p1, action_0, action_1, ...)
        
    First two elements are the card abstraction bucket indices.
    Subsequent elements are player actions.
    
    Actions:
        'f' = fold
        'c' = call/check
        'r' = raise (to next size in raise schedule)
        'a' = all-in
    """
    abstraction: CardAbstraction
    stack_bb: float = 100.0
    
    # Raise schedule: each entry is total bet size in BB
    # Open → 3bet → 4bet → 5bet
    raise_sizes: tuple[float, ...] = (3.0, 9.0, 22.0, 55.0)
    
    def __post_init__(self):
        """Precompute bucket pair probabilities."""
        self._bucket_pair_probs = self._compute_bucket_pair_probs()

    def _compute_bucket_pair_probs(self) -> dict[tuple[int, int], float]:
        """
        Compute probability of each (bucket_p0, bucket_p1) pair.
        
        Approximation: weight by combo count, ignore card blocking
        between hands (first-order model, standard in poker AI).
        """
        classes = all_169_classes()
        
        # Total combos for each bucket
        bucket_combos: dict[int, float] = {}
        for hc in classes:
            b = self.abstraction.get_bucket(hc)
            bucket_combos[b] = bucket_combos.get(b, 0) + _hand_combos(hc)
        
        # Total possible deals (approximate: 1326 × 1225 ≈ 1.6M)
        total_hero = sum(bucket_combos.values())  # = 1326
        
        pair_probs = {}
        total_weight = 0.0
        for b0 in range(self.abstraction.num_buckets):
            for b1 in range(self.abstraction.num_buckets):
                c0 = bucket_combos.get(b0, 0)
                c1 = bucket_combos.get(b1, 0)
                weight = c0 * c1
                pair_probs[(b0, b1)] = weight
                total_weight += weight
        
        # Normalize
        for key in pair_probs:
            pair_probs[key] /= total_weight
        
        return pair_probs

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        """Each possible (bucket_p0, bucket_p1) pair with its probability."""
        histories = []
        for (b0, b1), prob in self._bucket_pair_probs.items():
            if prob > 0:
                histories.append(((b0, b1), prob))
        return histories

    def _get_actions(self, history: History) -> tuple:
        return history[2:]

    def _betting_state(self, history: History) -> dict:
        """
        Parse betting state from history.
        
        Returns:
            invested: [p0_total, p1_total] in BB
            num_raises: total raises so far
            last_raise_idx: index into raise_sizes for the last raise
            current_player: whose turn
            is_terminal: bool
            terminal_reason: 'fold' | 'call' | 'allin_call' | None
        """
        actions = self._get_actions(history)
        invested = [0.5, 1.0]  # SB, BB
        num_raises = 0
        last_aggressor = -1
        
        player = 0  # SB acts first preflop
        
        for a in actions:
            if a == 'f':
                return {
                    "invested": invested,
                    "num_raises": num_raises,
                    "current_player": player,
                    "is_terminal": True,
                    "terminal_reason": "fold",
                    "folder": player,
                }
            elif a == 'c':
                # Call = match opponent's bet
                opp = 1 - player
                invested[player] = invested[opp]
                
                # Check if this closes the action
                # Action closes after: limp→check, raise→call
                if num_raises > 0 or (len(actions) >= 2 and actions[-2] == 'c'):
                    # This call/check closes the action
                    pass
                    
            elif a == 'r':
                if num_raises < len(self.raise_sizes):
                    new_total = self.raise_sizes[num_raises]
                    invested[player] = min(new_total, self.stack_bb)
                    num_raises += 1
                    last_aggressor = player
                    
            elif a == 'a':
                invested[player] = self.stack_bb
                num_raises += 1
                last_aggressor = player
            
            player = 1 - player
        
        # Check if terminal
        is_terminal = False
        terminal_reason = None
        
        if len(actions) >= 2:
            last = actions[-1]
            second_last = actions[-2]
            
            if last == 'c' and (num_raises > 0 or second_last == 'c'):
                is_terminal = True
                terminal_reason = "call" if invested[0] == invested[1] else "call"
            
            if last == 'c' and second_last == 'a':
                is_terminal = True
                terminal_reason = "allin_call"
        
        # Special: SB limps (calls), BB checks
        if actions == ('c', 'c'):
            is_terminal = True
            terminal_reason = "call"
        
        # Special: raise/allin then call
        if len(actions) >= 2 and actions[-1] == 'c' and actions[-2] in ('r', 'a'):
            is_terminal = True
            terminal_reason = "call"
        
        return {
            "invested": invested,
            "num_raises": num_raises,
            "current_player": player,
            "is_terminal": is_terminal,
            "terminal_reason": terminal_reason,
            "folder": None,
        }

    def is_terminal(self, history: History) -> bool:
        actions = self._get_actions(history)
        if not actions:
            return False
        
        # Quick check: fold
        if actions[-1] == 'f':
            return True
        
        state = self._betting_state(history)
        return state["is_terminal"]

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal: {history}")

        state = self._betting_state(history)
        invested = state["invested"]
        
        # Fold: folder loses their investment
        if state.get("folder") is not None:
            folder = state["folder"]
            winner = 1 - folder
            return (
                -invested[0] if folder == 0 else invested[1],
                -invested[1] if folder == 1 else invested[0],
            )
        
        # Showdown: compare buckets (higher bucket = stronger hand)
        b0, b1 = history[0], history[1]
        if b0 > b1:
            # P0 wins
            return (invested[1], -invested[1])
        elif b1 > b0:
            # P1 wins
            return (-invested[0], invested[0])
        else:
            # Same bucket → tie (split pot)
            return (0.0, 0.0)

    def current_player(self, history: History) -> int:
        actions = self._get_actions(history)
        if not actions:
            return 0  # SB acts first
        state = self._betting_state(history)
        return state["current_player"]

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Info set = player's bucket + action sequence.
        Opponent's bucket is hidden.
        """
        bucket = history[player]
        actions = "".join(self._get_actions(history))
        return f"B{bucket}|{actions}"

    def legal_actions(self, history: History) -> list[Action]:
        state = self._betting_state(history)
        actions_list = self._get_actions(history)
        invested = state["invested"]
        num_raises = state["num_raises"]
        player = state["current_player"]
        
        result = []
        
        # Can always fold (unless checking is free)
        opp = 1 - player
        must_pay = invested[opp] - invested[player]
        
        if must_pay > 0:
            result.append("f")  # Fold only if facing a bet
        
        # Call/check
        result.append("c")
        
        # Raise (if under max raises and not already all-in)
        if num_raises < len(self.raise_sizes) and invested[player] < self.stack_bb:
            # Check if raise size is meaningful (more than a call)
            next_raise = self.raise_sizes[num_raises]
            if next_raise > invested[opp] and next_raise < self.stack_bb:
                result.append("r")
        
        # All-in (if not already all-in and different from raise)
        if invested[player] < self.stack_bb:
            result.append("a")
        
        return result

    def apply_action(self, history: History, action: Action) -> History:
        return history + (action,)

    def bucket_name(self, bucket: int) -> str:
        """Human-readable bucket description."""
        exemplar = self.abstraction.bucket_exemplars[bucket]
        lo, hi = self.abstraction.bucket_ranges[bucket]
        return f"B{bucket}({exemplar}, eq={lo:.2f}-{hi:.2f})"

    def strategy_summary(
        self, strategy: dict[str, 'np.ndarray']
    ) -> str:
        """
        Format strategy as a readable range chart.
        Shows action frequencies per bucket at each decision point.
        """
        lines = [
            "Preflop NLHE Strategy",
            "=" * 60,
        ]
        
        # Group info sets by action sequence
        from collections import defaultdict
        by_sequence: dict[str, list[tuple[int, 'np.ndarray']]] = defaultdict(list)
        
        for key, strat in sorted(strategy.items()):
            parts = key.split("|")
            bucket = int(parts[0][1:])
            action_seq = parts[1] if len(parts) > 1 else ""
            by_sequence[action_seq].append((bucket, strat))
        
        for seq in sorted(by_sequence.keys()):
            if seq == "":
                desc = "SB opening action"
            else:
                desc = f"After: {seq}"
            
            lines.append(f"\n{desc}")
            lines.append("-" * 50)
            
            for bucket, strat in sorted(by_sequence[seq]):
                name = self.bucket_name(bucket)
                actions = self.legal_actions((bucket, 0) + tuple(seq))
                action_strs = [
                    f"{a}={s:.2f}" for a, s in zip(actions, strat)
                ]
                lines.append(f"  {name:>35s}: {', '.join(action_strs)}")
        
        return "\n".join(lines)