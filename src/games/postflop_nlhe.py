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

IMPLEMENTATION NOTE — state machine parity with the C++ engine
--------------------------------------------------------------
_parse_state is a deterministic replay of the SAME state machine implemented
in C++ (NLHEGame::apply_action in nlhe_game.cpp). This is required so that the
Python game tree is bit-identical to the C++ tree used to train the blueprints.
The key behaviours that distinguish this from a naive "betting round" model:

  * Position: SB (player 0) acts first preflop; BB (player 1) acts first
    postflop (the button/SB is in position).
  * CALL always ends the street immediately (advance to next street), including
    a preflop SB call — the BB does NOT get a subsequent option. This matches
    the C++ engine (it is non-standard poker, but it is what the engine does,
    and the engine defines the trained game).
  * CHECK uses last_aggressor: the first check of a street passes action to the
    opponent (last_aggressor -1 -> -2); the second check ends the street.
  * Raise sizing is the standard pot-sized raise on the pot AFTER the call:
    raise_add = (pot + owe) * frac, then total = owe + raise_add, capped at the
    stack. bet_amount mirrors NLHEGame::bet_amount.
"""

import numpy as np
from dataclasses import dataclass
from ..games.base import ExtensiveFormGame, History, Action, InfoSetKey
from ..abstraction.equity import evaluate_7card


STREET_NAMES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}

# Action char <-> meaning. Same external alphabet as before:
#   'f' fold, 'c' check, 'k' call, 'r' raise, 'a' all-in
# Internally 'f'/'c' both map to the C++ FOLD_OR_CHECK action (slot 0); which
# one it is depends on whether there is an outstanding bet (owe > 0).


@dataclass
class PostflopNLHE(ExtensiveFormGame):
    """
    Full HU NLHE for Deep CFR. Tree-identical to the C++ NLHEGame engine.

    Not compatible with tabular CFR (game tree too large).
    Use DeepCFRSolver which samples chance nodes.
    """
    starting_stack: float = 200.0  # In chips (BB = 2, SB = 1)
    max_raises_per_street: int = 2  # matches C++ NLHEGameConfig.max_raises
    sb: float = 1.0
    bb: float = 2.0
    # Raise sizes as fraction of pot (pot AFTER the call)
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
        full_board = tuple(int(c) for c in cards[4:9])
        return (p0, p1, full_board)

    def _get_actions(self, history: History) -> tuple:
        return history[3:]

    # ── Core state machine (mirrors C++ NLHEGame::apply_action) ──────────────
    def _parse_state(self, history: History) -> dict:
        """
        Deterministic replay of the C++ state machine. Returns the live state
        AFTER applying every action in the history. The keys returned match the
        previous public contract (stacks, invested, pot, to_call, street_idx,
        street_name, raises_this_street, current_player, folded, all_in).

        'invested' is reported as TOTAL invested across the hand
        (starting_stack - stack), so payoffs and the encoder see cumulative
        contributions, while 'street_invest' drives call sizing within a street.
        """
        actions = self._get_actions(history)

        stacks = [self.starting_stack - self.sb, self.starting_stack - self.bb]
        street_invest = [self.sb, self.bb]
        pot = self.sb + self.bb
        street = 0
        current_player = 0      # SB acts first preflop
        raises_this_street = 0
        last_aggressor = -1     # -1 = no bet yet this street, -2 = one check done
        folded = [False, False]
        all_in = False
        terminal = False

        def advance_street():
            nonlocal street, raises_this_street, last_aggressor, street_invest
            nonlocal current_player, terminal
            street += 1
            raises_this_street = 0
            last_aggressor = -1
            street_invest = [0.0, 0.0]
            if street >= 4:
                terminal = True          # showdown
            else:
                current_player = 1        # BB acts first postflop

        def bet_amount():
            p = current_player
            owe = street_invest[1 - p] - street_invest[p]
            effective_pot = pot + owe     # pot after calling
            raise_add = effective_pot * self.raise_fractions[0]
            return min(raise_add, stacks[p] - owe)

        for a in actions:
            if terminal:
                break
            p = current_player
            opp = 1 - p
            owe = street_invest[opp] - street_invest[p]

            if a == 'f':
                # Fold (only meaningful when facing a bet)
                folded[p] = True
                terminal = True
                break
            elif a == 'c':
                # Check (owe == 0)
                if last_aggressor == -1:
                    last_aggressor = -2
                    current_player = opp
                else:
                    advance_street()
            elif a == 'k':
                # Call — always ends the street (advance immediately)
                call_amt = min(owe, stacks[p])
                stacks[p] -= call_amt
                street_invest[p] += call_amt
                pot += call_amt
                advance_street()
            elif a == 'r':
                raise_add = bet_amount()
                total = owe + raise_add
                total = min(total, stacks[p])
                stacks[p] -= total
                street_invest[p] += total
                pot += total
                raises_this_street += 1
                last_aggressor = p
                current_player = opp
            elif a == 'a':
                allin = stacks[p]
                stacks[p] = 0.0
                street_invest[p] += allin
                pot += allin
                all_in = True
                raises_this_street += 1
                last_aggressor = p
                current_player = opp

        p = current_player
        opp = 1 - p
        to_call = street_invest[opp] - street_invest[p]
        invested = [self.starting_stack - stacks[0],
                    self.starting_stack - stacks[1]]

        return {
            "stacks": stacks,
            "invested": invested,
            "street_invest": street_invest,
            "pot": pot,
            "to_call": max(0.0, to_call),
            "street_idx": street,
            "street_name": ["preflop", "flop", "turn", "river"][min(street, 3)],
            "raises_this_street": raises_this_street,
            "current_player": current_player,
            "last_aggressor": last_aggressor,
            "folded": folded,
            "all_in": all_in,
            "terminal": terminal,
        }

    def _current_street(self, history: History) -> str:
        return self._parse_state(history)["street_name"]

    def _visible_board(self, history: History) -> tuple:
        """Board cards visible at the current street."""
        full_board = history[2]
        street = self._parse_state(history)["street_idx"]
        n = [0, 3, 4, 5][min(street, 3)]
        return full_board[:n]

    def is_terminal(self, history: History) -> bool:
        actions = self._get_actions(history)
        if not actions:
            return False
        return self._parse_state(history)["terminal"]

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal: {history}")

        state = self._parse_state(history)
        invested = state["invested"]

        # Fold: folder loses their total investment, winner gains it.
        if state["folded"][0] or state["folded"][1]:
            folder = 0 if state["folded"][0] else 1
            winner = 1 - folder
            payoffs = [0.0, 0.0]
            payoffs[folder] = -invested[folder]
            payoffs[winner] = invested[folder]
            return tuple(payoffs)

        # Showdown — compare full 7-card hands, winner takes the loser's stake.
        p0_cards = history[0]
        p1_cards = history[1]
        board = history[2][:5]
        h0 = evaluate_7card(p0_cards + board)
        h1 = evaluate_7card(p1_cards + board)
        if h0 > h1:
            return (float(invested[1]), float(-invested[1]))
        elif h1 > h0:
            return (float(-invested[0]), float(invested[0]))
        else:
            return (0.0, 0.0)  # Tie

    def current_player(self, history: History) -> int:
        return self._parse_state(history)["current_player"]

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
        if state["terminal"]:
            return []

        p = state["current_player"]
        owe = state["to_call"]
        bet = owe > 0.001
        can_raise = state["raises_this_street"] < self.max_raises_per_street
        my_stack = state["stacks"][p]
        pot = state["pot"]

        def bet_amount():
            effective_pot = pot + owe
            raise_add = effective_pot * self.raise_fractions[0]
            return min(raise_add, my_stack - owe)

        result = []
        if bet:
            result.append("f")  # fold
            result.append("k")  # call
        else:
            result.append("c")  # check

        if can_raise and my_stack > owe + 0.01:
            if bet_amount() > 0.01:
                result.append("r")

        # All-in: available if it differs from the (capped) raise, or no raise.
        if my_stack > owe + 0.01:
            allin_add = my_stack - owe
            raise_add = bet_amount() if (can_raise and my_stack > owe + 0.01) else -1.0
            if (not can_raise) or allin_add > raise_add + 0.01:
                result.append("a")

        return result

    def apply_action(self, history: History, action: Action) -> History:
        return history + (action,)