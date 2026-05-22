"""
Kuhn Poker — the canonical minimal imperfect-information poker game.

Rules (Kuhn, 1950):
    - Deck: 3 cards {J, Q, K} (values 0, 1, 2)
    - 2 players, each antes 1 chip
    - Each dealt 1 card
    - Player 0 acts first: check or bet(1)
    - If check → Player 1: check (showdown) or bet(1)
    - If Player 1 bets after check → Player 0: fold or call(1)
    - If bet → Player 1: fold or call(1)
    - Showdown: higher card wins the pot

Game tree has 12 information sets and a known analytical Nash equilibrium:
    Player 0:
        - K: bet with probability 3α (0 ≤ α ≤ 1/3)
        - Q: always check; if facing bet, always fold
        - J: bet with probability (1/3 + α); if checked through, check
    Player 1:
        - K: always call/bet
        - Q: call with probability (1/3 + α)
        - J: always fold

At α = 0 the equilibrium is unique in exploitability-relevant structure.
Game value: Player 0's EV = -1/18 ≈ -0.0556 (slight disadvantage from acting first).
"""

from itertools import permutations
from .base import ExtensiveFormGame, History, Action, InfoSetKey


CARDS = ["J", "Q", "K"]
CARD_RANK = {"J": 0, "Q": 1, "K": 2}


class KuhnPoker(ExtensiveFormGame):
    """
    Kuhn Poker implementation.
    
    History encoding:
        (card_p0, card_p1, action_0, action_1, ...)
        
    First two elements are the dealt cards (chance node outcome).
    Subsequent elements are player actions: 'c' (check), 'b' (bet),
    'f' (fold), 'k' (call).
    """

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        """All 6 possible dealings, each with probability 1/6."""
        dealings = list(permutations(CARDS, 2))
        prob = 1.0 / len(dealings)
        return [(deal, prob) for deal in dealings]

    def is_terminal(self, history: History) -> bool:
        if len(history) < 3:
            return False
        actions = history[2:]
        # Two checks → showdown
        if actions == ("c", "c"):
            return True
        # Bet then fold
        if len(actions) >= 2 and actions[-1] == "f":
            return True
        # Bet then call
        if len(actions) >= 2 and actions[-2] == "b" and actions[-1] == "k":
            return True
        # Check, bet, fold
        if actions == ("c", "b", "f"):
            return True
        # Check, bet, call
        if actions == ("c", "b", "k"):
            return True
        return False

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal history: {history}")

        card_p0, card_p1 = history[0], history[1]
        actions = history[2:]

        winner_sign = 1 if CARD_RANK[card_p0] > CARD_RANK[card_p1] else -1

        # Check-check → showdown, pot = 2 (1 ante each)
        if actions == ("c", "c"):
            return (winner_sign * 1.0, -winner_sign * 1.0)

        # Bet-fold → bettor wins ante
        if actions == ("b", "f"):
            return (1.0, -1.0)  # P0 bet, P1 folded

        # Bet-call → showdown, pot = 4
        if actions == ("b", "k"):
            return (winner_sign * 2.0, -winner_sign * 2.0)

        # Check-bet-fold → P1 wins ante
        if actions == ("c", "b", "f"):
            return (-1.0, 1.0)

        # Check-bet-call → showdown, pot = 4
        if actions == ("c", "b", "k"):
            return (winner_sign * 2.0, -winner_sign * 2.0)

        raise ValueError(f"Unrecognized terminal: {history}")

    def current_player(self, history: History) -> int:
        actions = history[2:]
        if len(actions) == 0:
            return 0
        if len(actions) == 1:
            return 1
        # Only reached in check-bet sequence → P0 responds
        if actions[0] == "c" and actions[1] == "b":
            return 0
        raise ValueError(f"Should be terminal: {history}")

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Information set = player's card + visible action sequence.
        Opponent's card is hidden.
        """
        card = history[player]
        actions = "".join(history[2:])
        return f"{card}:{actions}"

    def legal_actions(self, history: History) -> list[Action]:
        actions = history[2:]
        if len(actions) == 0:
            return ["c", "b"]  # Check or bet
        if actions == ("c",):
            return ["c", "b"]  # Check or bet
        if actions == ("b",):
            return ["f", "k"]  # Fold or call
        if actions == ("c", "b"):
            return ["f", "k"]  # Fold or call
        raise ValueError(f"No legal actions at: {history}")

    def apply_action(self, history: History, action: Action) -> History:
        return history + (action,)

    @staticmethod
    def known_game_value() -> float:
        """Analytical game value: P0's EV at Nash equilibrium."""
        return -1.0 / 18.0

    @staticmethod
    def known_nash_description() -> str:
        return (
            "Nash equilibrium family (parameterized by α ∈ [0, 1/3]):\n"
            "  P0: K→bet 3α, Q→always check, J→bet α\n"
            "  P0 facing check-bet: K→call, Q→indifferent (call 1/3 makes J indifferent), J→fold\n"
            "  P1 after check: K→always bet, Q→never bet, J→bet 1/3\n"
            "  P1 facing bet: K→always call, Q→call 1/3, J→always fold\n"
            "  Game value: P0 EV = -1/18 ≈ -0.0556"
        )