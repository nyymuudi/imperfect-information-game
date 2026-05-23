"""
Leduc Hold'em — the standard mid-complexity poker benchmark.

Rules (Southey et al., 2005):
    - Deck: 6 cards — J♠ J♦ Q♠ Q♦ K♠ K♦ (2 suits × 3 ranks)
    - 2 players, each antes 1 chip
    - Round 1: each dealt 1 private card, then betting round
    - Round 2: 1 community card dealt face-up, then betting round
    - Betting: check or raise (fixed size per round)
        - Round 1 raise = 2 chips
        - Round 2 raise = 4 chips
        - Maximum 2 raises per round
    - Hand ranking: pair (private == community) > high card > low card
    - Ties split the pot

Design: all chance nodes (private cards + community card) are
pre-expanded into initial_histories(). The community card exists
in the history from the start but info_set_key only reveals it
after round 1 betting completes. This keeps the ExtensiveFormGame
interface unchanged (no mid-game chance nodes).

Total initial histories: 30 dealings × 4 remaining cards = 120.
"""

from .base import ExtensiveFormGame, History, Action, InfoSetKey


RANKS = ["J", "Q", "K"]
RANK_VALUE = {"J": 0, "Q": 1, "K": 2}
RAISE_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2


class LeducHoldem(ExtensiveFormGame):
    """
    Leduc Hold'em.
    
    History encoding:
        (p0_card, p1_card, community_card, action_0, action_1, ...)
    
    All three cards are present from the start (pre-expanded chance).
    Cards are identifiers like 'J1', 'Q2' etc. Ranks extracted with [0].
    
    Actions: 'c' check, 'r' raise, 'f' fold, 'k' call.
    """

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        """30 dealings × 4 community cards = 120 initial states."""
        cards = ["J1", "J2", "Q1", "Q2", "K1", "K2"]
        results = []
        for i, c0 in enumerate(cards):
            for j, c1 in enumerate(cards):
                if i == j:
                    continue
                remaining = [c for c in cards if c != c0 and c != c1]
                for comm in remaining:
                    # prob = (1/30) * (1/4) = 1/120
                    results.append(((c0, c1, comm), 1.0 / 120))
        return results

    def _actions(self, h: History) -> tuple:
        return h[3:]

    def _split_rounds(self, actions: tuple) -> tuple[list, list, bool]:
        """
        Split action sequence into round 1 and round 2 actions.
        
        Returns (r1_actions, r2_actions, r1_complete).
        Round 1 ends on: check-check, or any call after raise.
        """
        r1, r2 = [], []
        r1_done = False
        target = r1
        raises = 0

        for a in actions:
            target.append(a)
            if a == 'f':
                return r1 if not r1_done else r1, r2, r1_done
            if not r1_done:
                if a == 'r':
                    raises += 1
                elif a == 'k':
                    r1_done = True
                    target = r2
                    raises = 0
                    continue
                elif a == 'c' and len(r1) >= 2:
                    r1_done = True
                    target = r2
                    raises = 0
                    continue
            # Round 2 handled implicitly (stays in r2)

        return r1, r2, r1_done

    def _round_complete(self, round_actions: list) -> bool:
        """Check if a betting round's action sequence is complete."""
        if len(round_actions) < 2:
            return False
        if round_actions[-1] == 'f':
            return True
        if round_actions[-1] == 'k':
            return True
        if round_actions == ['c', 'c']:
            return True
        return False

    def _count_raises(self, round_actions: list) -> int:
        return sum(1 for a in round_actions if a == 'r')

    def is_terminal(self, history: History) -> bool:
        actions = self._actions(history)
        if not actions:
            return False
        if 'f' in actions:
            return True
        r1, r2, r1_done = self._split_rounds(actions)
        if not r1_done:
            return False
        return self._round_complete(r2)

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal: {history}")

        actions = self._actions(history)
        r1, r2, r1_done = self._split_rounds(actions)

        # Compute pot contributions
        pot = [1, 1]  # Antes
        player = 0
        for rnd, round_actions in enumerate([r1, r2], start=1):
            for a in round_actions:
                if a == 'f':
                    winner = 1 - player
                    return (
                        (-pot[0], pot[0]) if winner == 1 else (pot[1], -pot[1])
                    )
                elif a == 'r':
                    opp = 1 - player
                    pot[player] = pot[opp] + RAISE_SIZE[rnd]
                elif a == 'k':
                    opp = 1 - player
                    pot[player] = pot[opp]
                player = 1 - player
            player = 0  # Reset for round 2

        # Showdown
        r0, r1_rank = history[0][0], history[1][0]
        comm = history[2][0]
        winner = self._compare(r0, r1_rank, comm)
        if winner == 0:
            return (float(pot[1]), float(-pot[1]))
        elif winner == 1:
            return (float(-pot[0]), float(pot[0]))
        return (0.0, 0.0)

    def _compare(self, r0: str, r1: str, comm: str) -> int:
        p0_pair = (r0 == comm)
        p1_pair = (r1 == comm)
        if p0_pair and not p1_pair:
            return 0
        if p1_pair and not p0_pair:
            return 1
        v0, v1 = RANK_VALUE[r0], RANK_VALUE[r1]
        if v0 > v1:
            return 0
        if v1 > v0:
            return 1
        return -1

    def current_player(self, history: History) -> int:
        actions = self._actions(history)
        if not actions:
            return 0
        r1, r2, r1_done = self._split_rounds(actions)
        if r1_done:
            return len(r2) % 2
        return len(r1) % 2

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Private card rank + action sequence.
        Community card revealed ONLY after round 1 completes.
        """
        private = history[player][0]  # Rank
        actions = self._actions(history)
        r1, r2, r1_done = self._split_rounds(actions)
        community = history[2][0] if r1_done else ""
        action_str = "".join(actions)
        return f"{private}|{community}|{action_str}"

    def legal_actions(self, history: History) -> list[Action]:
        actions = self._actions(history)
        r1, r2, r1_done = self._split_rounds(actions)

        if r1_done:
            # Round 2
            raises = self._count_raises(r2)
            if not r2:
                return ["c", "r"] if raises < MAX_RAISES else ["c"]
            if r2[-1] == 'r':
                return (["f", "k", "r"] if raises < MAX_RAISES
                        else ["f", "k"])
            return ["c", "r"] if raises < MAX_RAISES else ["c"]

        # Round 1
        raises = self._count_raises(r1)
        if not r1:
            return ["c", "r"] if raises < MAX_RAISES else ["c"]
        if r1[-1] == 'r':
            return (["f", "k", "r"] if raises < MAX_RAISES
                    else ["f", "k"])
        return ["c", "r"] if raises < MAX_RAISES else ["c"]

    def apply_action(self, history: History, action: Action) -> History:
        return history + (action,)