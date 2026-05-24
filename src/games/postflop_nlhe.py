"""
Heads-Up No-Limit Hold'em — Preflop through River.

Designed for Deep CFR (not tabular CFR). Chance nodes are sampled
during MCCFR traversal rather than pre-expanded, since the game
tree is too large to enumerate (~10^14 information sets).

History encoding:
    (p0_cards, p1_cards, board, action1, action2, ...)
    
    p0_cards: tuple of 2 card ints (0-51)
    p1_cards: tuple of 2 card ints (0-51)
    board:    tuple of 0-5 card ints (grows as streets are dealt)
    actions:  'f' fold, 'c' check, 'k' call, 'r' raise, 'a' all-in

Streets determined by board length:
    0 cards = preflop
    3 cards = flop
    4 cards = turn
    5 cards = river

Card encoding: card = rank * 4 + suit
    rank: 0=2, 1=3, ..., 12=A
    suit: 0=♣, 1=♦, 2=♥, 3=♠
"""

import numpy as np
from dataclasses import dataclass
from ..games.base import ExtensiveFormGame, History, Action, InfoSetKey
from ..abstraction.equity import evaluate_7card


STREET_NAMES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@dataclass
class PostflopNLHE(ExtensiveFormGame):
    """
    Full HU NLHE for Deep CFR.
    
    Not compatible with tabular CFR (game tree too large).
    Use DeepCFRSolver which samples chance nodes.
    """
    starting_stack: float = 200.0  # In chips (BB = 2, SB = 1)
    max_raises_per_street: int = 3
    # Raise sizes as fraction of pot
    raise_fractions: tuple[float, ...] = (0.75,)

    def num_players(self) -> int:
        return 2

    def initial_histories(self):
        """Not used by Deep CFR. Raises error for tabular solvers."""
        raise NotImplementedError(
            "PostflopNLHE is too large for tabular CFR. "
            "Use DeepCFRSolver with sample_deal()."
        )

    def sample_deal(self, rng: np.random.Generator) -> History:
        """Sample a random initial deal (hole cards + full board)."""
        cards = rng.choice(52, size=9, replace=False)
        p0 = (int(cards[0]), int(cards[1]))
        p1 = (int(cards[2]), int(cards[3]))
        # Pre-deal the full board (revealed street by street)
        full_board = tuple(int(c) for c in cards[4:9])
        return (p0, p1, full_board)

    def _get_actions(self, history: History) -> tuple:
        return history[3:]

    def _visible_board(self, history: History) -> tuple:
        """Board cards visible at the current street."""
        full_board = history[2]
        street = self._current_street(history)
        if street == "preflop":
            return ()
        elif street == "flop":
            return full_board[:3]
        elif street == "turn":
            return full_board[:4]
        else:  # river
            return full_board[:5]

    def _current_street(self, history: History) -> str:
        """Determine current street from action sequence."""
        actions = self._get_actions(history)
        streets_completed = 0
        street_actions = []
        
        for a in actions:
            if a == 'f':
                break
            street_actions.append(a)
            if self._street_betting_complete(street_actions):
                streets_completed += 1
                street_actions = []

        return ["preflop", "flop", "turn", "river"][min(streets_completed, 3)]

    def _street_betting_complete(self, street_actions: list) -> bool:
        """Check if a single street's betting is complete."""
        if len(street_actions) < 2:
            return False
        # Track if there's a pending bet/raise
        pending = False
        for a in street_actions:
            if a == 'r' or a == 'a':
                pending = True
            elif a == 'k':
                pending = False
        # Street ends when both acted AND no pending action
        last = street_actions[-1]
        if last == 'k':
            return True   # Call resolves a bet → done
        if last == 'c' and not pending:
            return True   # Check with nothing pending → done
        return False

    def _parse_state(self, history: History) -> dict:
        """Full state parse: stacks, pot, street, who acts."""
        actions = self._get_actions(history)
        stacks = [self.starting_stack - 1.0, self.starting_stack - 2.0]
        invested = [1.0, 2.0]  # Blinds
        street_idx = 0
        street_actions = []
        raises_this_street = 0
        current_player = 0  # SB acts first preflop
        folded = False
        all_in = False

        for a in actions:
            if a == 'f':
                folded = True
                break
            
            street_actions.append(a)
            
            if a == 'c':
                pass  # Check
            elif a == 'k':
                # Call
                call_amt = invested[1 - current_player] - invested[current_player]
                invested[current_player] += call_amt
                stacks[current_player] -= call_amt
            elif a == 'r':
                # Raise: to (opponent's invested + raise_size)
                pot = sum(invested)
                raise_size = pot * self.raise_fractions[0]
                # Must first call
                call_amt = invested[1 - current_player] - invested[current_player]
                total = call_amt + raise_size
                total = min(total, stacks[current_player])
                invested[current_player] += total
                stacks[current_player] -= total
                raises_this_street += 1
            elif a == 'a':
                allin = stacks[current_player]
                invested[current_player] += allin
                stacks[current_player] = 0
                all_in = True
                raises_this_street += 1

            # Check if street complete
            if self._street_betting_complete(street_actions):
                street_idx += 1
                street_actions = []
                raises_this_street = 0
                current_player = 0  # P0 acts first postflop
                continue

            current_player = 1 - current_player

        pot = sum(invested)
        to_call = invested[1 - current_player] - invested[current_player]
        
        return {
            "stacks": stacks,
            "invested": invested,
            "pot": pot,
            "to_call": max(0, to_call),
            "street_idx": street_idx,
            "street_name": ["preflop", "flop", "turn", "river"][min(street_idx, 3)],
            "raises_this_street": raises_this_street,
            "current_player": current_player,
            "folded": folded,
            "all_in": all_in,
            "street_actions": street_actions,
        }

    def is_terminal(self, history: History) -> bool:
        actions = self._get_actions(history)
        if not actions:
            return False
        if actions[-1] == 'f':
            return True

        state = self._parse_state(history)
        if state["folded"]:
            return True
        
        # All-in called → terminal (run out remaining board)
        if state["all_in"] and state["to_call"] == 0 and len(state["street_actions"]) >= 2:
            return True

        # River betting complete → showdown
        if state["street_idx"] >= 4:
            return True

        return False

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal: {history}")

        state = self._parse_state(history)
        invested = state["invested"]

        # Fold: folder loses their investment, winner gains it
        if state["folded"]:
            # Determine who folded from action sequence
            actions = self._get_actions(history)
            current = 0
            street_actions_local = []
            for a in actions:
                if a == 'f':
                    break
                street_actions_local.append(a)
                if self._street_betting_complete(street_actions_local):
                    street_actions_local = []
                    current = 0
                    continue
                current = 1 - current
            # 'current' is the player who folded
            folder = current
            winner = 1 - folder
            payoffs = [0.0, 0.0]
            payoffs[folder] = -invested[folder]
            payoffs[winner] = invested[folder]
            return tuple(payoffs)

        # Showdown
        p0_cards = history[0]
        p1_cards = history[1]
        board = history[2][:5]  # Full board

        h0 = evaluate_7card(p0_cards + board)
        h1 = evaluate_7card(p1_cards + board)

        if h0 > h1:
            return (float(invested[1]), float(-invested[1]))
        elif h1 > h0:
            return (float(-invested[0]), float(invested[0]))
        else:
            return (0.0, 0.0)  # Tie

    def current_player(self, history: History) -> int:
        state = self._parse_state(history)
        return state["current_player"]

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        For Deep CFR this isn't used for table lookup —
        the neural network generalizes across info sets.
        But we implement it for compatibility.
        """
        my_cards = history[player]
        board = self._visible_board(history)
        actions = "".join(str(a) for a in self._get_actions(history))
        return f"{my_cards}|{board}|{actions}"

    def legal_actions(self, history: History) -> list[Action]:
        state = self._parse_state(history)
        result = []

        if state["to_call"] > 0:
            result.append("f")  # Can fold if facing bet
            result.append("k")  # Call
        else:
            result.append("c")  # Check

        # Raise (if under limit and has chips)
        if (state["raises_this_street"] < self.max_raises_per_street
                and state["stacks"][state["current_player"]] > state["to_call"]):
            result.append("r")

        # All-in (always available if has chips)
        if state["stacks"][state["current_player"]] > 0:
            if "r" not in result:  # Avoid duplicate when raise = all-in
                result.append("a")
            elif state["stacks"][state["current_player"]] > state["to_call"] + state["pot"] * self.raise_fractions[0]:
                result.append("a")

        return result

    def apply_action(self, history: History, action: Action) -> History:
        new_h = history + (action,)

        # Check if we need to advance to next street
        # (street betting complete but game not terminal)
        if not self.is_terminal(new_h):
            state = self._parse_state(new_h)
            # If street just completed and we haven't run out of streets
            actions = self._get_actions(new_h)
            if (len(state["street_actions"]) == 0 
                    and state["street_idx"] > 0
                    and state["street_idx"] <= 3):
                pass  # Next street starts, board already pre-dealt

        return new_h