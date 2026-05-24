"""
State encoder for Deep CFR.

Converts game histories into fixed-size numeric vectors suitable
for neural network input. Each game requires a specific encoder
because the state representation depends on the game structure.

Design principle: the encoder must use ONLY information available
to the current player (info set), never the full game state.
This prevents abstraction leakage.
"""

import numpy as np
from abc import ABC, abstractmethod
from ..games.base import History


class StateEncoder(ABC):
    """Abstract state encoder interface."""

    @abstractmethod
    def state_size(self) -> int:
        """Return the fixed size of the output state vector."""
        ...

    @abstractmethod
    def encode(self, history: History, player: int) -> np.ndarray:
        """
        Encode a history from a player's perspective.
        
        MUST use only information visible to the player:
        - Their own private cards
        - Public information (community cards, action sequence)
        - NOT the opponent's cards
        
        Returns a 1D numpy array of size state_size().
        """
        ...


class LeducEncoder(StateEncoder):
    """
    Leduc Hold'em state encoder.
    
    State vector (20 dimensions):
        [0:3]   Private card rank (one-hot: J=0, Q=1, K=2)
        [3:6]   Community card rank (one-hot, zeros if not revealed)
        [6]     Community revealed (0 or 1)
        [7]     Pot size (normalized by max pot = 26)
        [8]     Amount to call (normalized by max pot)
        [9]     Current round (0 or 1)
        [10:20] Action history (last 10 actions, encoded as:
                check=0.0, raise=0.5, fold=-1.0, call=1.0, empty=0.0)
    """

    RANK_IDX = {"J": 0, "Q": 1, "K": 2}
    ACTION_ENC = {"c": 0.0, "r": 0.5, "f": -1.0, "k": 1.0}
    MAX_POT = 26.0
    HISTORY_LEN = 10

    def state_size(self) -> int:
        return 20

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(self.state_size(), dtype=np.float32)
        private_rank = history[player][0]
        state[self.RANK_IDX[private_rank]] = 1.0
        actions = history[3:]
        r1_done = self._is_round1_done(actions)
        if r1_done:
            comm_rank = history[2][0]
            state[3 + self.RANK_IDX[comm_rank]] = 1.0
            state[6] = 1.0
        pot, to_call = self._compute_pot_info(actions, player)
        state[7] = pot / self.MAX_POT
        state[8] = to_call / self.MAX_POT
        state[9] = 1.0 if r1_done else 0.0
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            state[10 + i] = self.ACTION_ENC.get(a, 0.0)
        return state

    def _is_round1_done(self, actions: tuple) -> bool:
        if len(actions) < 2:
            return False
        for i, a in enumerate(actions):
            if a == 'f':
                return False
            if a == 'k':
                return True
            if a == 'c' and i >= 1:
                if actions[i-1] == 'c' or (i == 1 and actions[0] == 'c'):
                    return True
        return False

    def _compute_pot_info(self, actions: tuple, player: int) -> tuple[float, float]:
        pot = [1.0, 1.0]
        r1_done = False
        current = 0
        raise_sizes = {False: 2.0, True: 4.0}
        for a in actions:
            if a == 'f':
                break
            elif a == 'r':
                opp = 1 - current
                pot[current] = pot[opp] + raise_sizes[r1_done]
            elif a == 'k':
                pot[current] = pot[1 - current]
                if not r1_done:
                    r1_done = True
                    current = 0
                    continue
            current = 1 - current
        to_call = max(0, pot[1 - player] - pot[player])
        return sum(pot), to_call


class NLHEEncoder(StateEncoder):
    """
    No-Limit Hold'em state encoder for Deep CFR.
    
    State vector (120 dimensions):
        [0:52]    Private cards (one-hot, 2 bits set)
        [52:104]  Board cards (one-hot, 0-5 bits set)
        [104:108] Street (one-hot: preflop/flop/turn/river)
        [108]     Pot size (normalized by 2× starting stack)
        [109]     Amount to call (normalized)
        [110]     Own stack remaining (normalized)
        [111]     Opponent stack remaining (normalized)
        [112:120] Action history (last 8 actions encoded)
    
    Card encoding: card = rank * 4 + suit
        rank: 0=2, 1=3, ..., 12=A
        suit: 0=♣, 1=♦, 2=♥, 3=♠
    """

    ACTION_ENC = {"f": -1.0, "c": 0.0, "k": 0.25, "r": 0.5, "b": 0.75, "a": 1.0}
    HISTORY_LEN = 8

    _shared_equity_cache = None

    def __init__(self, starting_stack: float = 200.0, equity_sims: int = 500):
        self.starting_stack = starting_stack
        self.norm = 2 * starting_stack  # Max pot ≈ 2× stacks
        self._equity_sims = equity_sims
        if NLHEEncoder._shared_equity_cache is None:
            NLHEEncoder._shared_equity_cache = {}
            self._build_equity_cache()
        self._equity_cache = NLHEEncoder._shared_equity_cache

    def _build_equity_cache(self):
        """Pre-compute preflop equity for all 169 canonical hands."""
        from ..abstraction.equity import (
            canonical_hand_class, representative_hand,
            equity_vs_random, all_169_classes,
        )
        rng = np.random.default_rng(42)
        for hc in all_169_classes():
            cards = representative_hand(hc)
            eq = equity_vs_random(cards, num_simulations=self._equity_sims, rng=rng)
            NLHEEncoder._shared_equity_cache[hc] = eq

    def _get_preflop_equity(self, card1: int, card2: int) -> float:
        from ..abstraction.equity import canonical_hand_class
        hc = canonical_hand_class(card1, card2)
        return self._equity_cache.get(hc, 0.5)

    def _get_board_strength(self, hole: tuple, board: tuple) -> float:
        if not board:
            return 0.0
        from ..abstraction.equity import evaluate_7card, evaluate_5card
        all_cards = hole + board
        if len(all_cards) >= 5:
            val = evaluate_7card(all_cards[:7]) if len(all_cards) >= 7 else evaluate_5card(all_cards[:5])
            return min(val / 6_000_000, 1.0)
        return 0.0

    def state_size(self) -> int:
        return 122

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(122, dtype=np.float32)

        # Private cards (one-hot in 52-dim)
        p0_cards = history[0]  # (card1, card2)
        p1_cards = history[1]
        my_cards = p0_cards if player == 0 else p1_cards
        state[my_cards[0]] = 1.0
        state[my_cards[1]] = 1.0

        # Board cards (only visible ones based on street)
        board = history[2]  # Full pre-dealt board
        actions = history[3:]
        n_visible = self._visible_board_count(actions)
        for card in board[:n_visible]:
            state[52 + card] = 1.0

        # Street (one-hot)
        street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_visible, 0)
        state[104 + street] = 1.0

        # Betting info
        actions = history[3:]
        pot, to_call, my_stack, opp_stack = self._parse_betting(
            actions, player, n_visible
        )
        state[108] = pot / self.norm
        state[109] = to_call / self.norm
        state[110] = my_stack / self.starting_stack
        state[111] = opp_stack / self.starting_stack

        # Action history
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            if isinstance(a, str):
                state[112 + i] = self.ACTION_ENC.get(a, 0.0)
            else:
                state[112 + i] = min(a / self.starting_stack, 1.0)

        # Hand strength features (key for convergence)
        state[120] = self._get_preflop_equity(my_cards[0], my_cards[1])
        visible_board = history[2][:n_visible] if n_visible > 0 else ()
        state[121] = self._get_board_strength(my_cards, visible_board)

        return state

    def _visible_board_count(self, actions: tuple) -> int:
        """Count how many board cards are visible based on completed streets."""
        streets_done = 0
        street_actions = []
        pending = False
        for a in actions:
            if a == 'f':
                break
            street_actions.append(a)
            if a in ('r', 'a'):
                pending = True
            elif a == 'k':
                pending = False
            # Check if street complete
            if len(street_actions) >= 2:
                last = street_actions[-1]
                if last == 'k' or (last == 'c' and not pending):
                    streets_done += 1
                    street_actions = []
                    pending = False
        return {0: 0, 1: 3, 2: 4, 3: 5}.get(streets_done, 5)

    def _parse_betting(
        self, actions: tuple, player: int, n_board: int
    ) -> tuple[float, float, float, float]:
        """Parse actions to get pot, to_call, and stack info."""
        stacks = [self.starting_stack - 1.0, self.starting_stack - 2.0]  # After blinds
        pot = 3.0  # SB(1) + BB(2)
        current = 0  # SB acts first preflop

        for a in actions:
            if a == 'f':
                break
            elif a == 'c':
                pass  # Check, no money change
            elif a == 'k':
                # Call: match opponent
                opp = 1 - current
                call_amt = max(0, (self.starting_stack - stacks[opp]) -
                               (self.starting_stack - stacks[current]))
                stacks[current] -= call_amt
                pot += call_amt
            elif isinstance(a, str) and a == 'a':
                # All-in
                allin_amt = stacks[current]
                pot += allin_amt
                stacks[current] = 0
            elif isinstance(a, str) and a.startswith('r'):
                # Raise: 'r' followed by amount or fixed
                raise_amt = min(pot * 0.75, stacks[current])
                pot += raise_amt
                stacks[current] -= raise_amt
            current = 1 - current

        to_call = max(0, (self.starting_stack - stacks[1-player]) -
                       (self.starting_stack - stacks[player]))
        return pot, to_call, stacks[player], stacks[1 - player]