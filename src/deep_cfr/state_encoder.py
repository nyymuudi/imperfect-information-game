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
    No-Limit Hold'em state encoder for Deep CFR — card-abstracted.

    State vector (36 dimensions):
        [0:8]   Preflop hand bucket (one-hot, K=8 equal-width bins on equity)
        [8:16]  Board EHS bucket (one-hot, K=8; all zeros preflop)
        [16:20] Street (one-hot: preflop/flop/turn/river)
        [20]    Pot size (normalized by 2× starting stack)
        [21]    Amount to call (normalized)
        [22]    Own stack remaining (normalized)
        [23]    Opponent stack remaining (normalized)
        [24:32] Action history (last 8 actions encoded)
        [32]    Preflop equity of own hand (continuous, in [0,1])
        [33]    Board strength (evaluate_7card / MAX_HAND_SCORE, in [0,1])
        [34]    Pot odds = to_call / (pot + to_call)
        [35]    SPR = min(stacks) / pot (capped at 10, normalised)

    Card abstraction:
        Multiple concrete hands map to the same bucket → the same state
        vector → the buffer accumulates dense samples per abstract state
        rather than sparse samples spread across 1326 unique hand pairs.
        Preflop bucket: min(int(equity * K), K-1), K=8.
        Board bucket:   min(int(board_strength * K), K-1), K=8.
        Dims 32-33 retain the continuous signal within the bucket so the
        network can interpolate between adjacent buckets.

    Betting/street: read from PostflopNLHE._parse_state (mirrors C++ NLHEGame).
    Quantisation: dims 20-35 rounded to 1e-6 grid for stable grouping.
    """

    ACTION_ENC = {"f": -1.0, "c": 0.0, "k": 0.25, "r": 0.5, "b": 0.75, "a": 1.0}
    HISTORY_LEN = 8
    FEATURE_QUANT = 1e-6
    K_PREFLOP = 8
    K_BOARD   = 8

    _shared_equity_cache = None

    def __init__(self, starting_stack: float = 200.0, equity_sims: int = 2000):
        self.starting_stack = starting_stack
        self.norm = 2 * starting_stack
        self._equity_sims = equity_sims
        self._game_cache = None
        if NLHEEncoder._shared_equity_cache is None:
            NLHEEncoder._shared_equity_cache = {}
            self._build_equity_cache()
        self._equity_cache = NLHEEncoder._shared_equity_cache

    def _game(self):
        """Lazily build a PostflopNLHE matching this encoder's stack, so the
        encoder reads pot/to_call/street/stacks from the SAME (corrected) state
        machine the C++ engine uses."""
        if self._game_cache is None:
            from ..games.postflop_nlhe import PostflopNLHE
            self._game_cache = PostflopNLHE(
                starting_stack=self.starting_stack,
                max_raises_per_street=2,
                raise_fractions=(0.75,),
            )
        return self._game_cache

    def _build_equity_cache(self):
        from ..abstraction.equity import preflop_equity_table
        table = preflop_equity_table(num_simulations=self._equity_sims)
        NLHEEncoder._shared_equity_cache.update(table)

    def _get_preflop_equity(self, card1: int, card2: int) -> float:
        from ..abstraction.equity import canonical_hand_class
        hc = canonical_hand_class(card1, card2)
        return self._equity_cache.get(hc, 0.5)

    def _get_board_strength(self, hole: tuple, board: tuple) -> float:
        if not board:
            return 0.0
        from ..abstraction.equity import (
            evaluate_7card, evaluate_5card, MAX_HAND_SCORE,
        )
        all_cards = hole + board
        if len(all_cards) >= 5:
            val = (evaluate_7card(all_cards[:7]) if len(all_cards) >= 7
                   else evaluate_5card(all_cards[:5]))
            return min(val / MAX_HAND_SCORE, 1.0)
        return 0.0

    def state_size(self) -> int:
        return 36

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(36, dtype=np.float32)

        my_cards = history[0] if player == 0 else history[1]

        # Preflop equity + bucket [0:8]
        equity = self._get_preflop_equity(my_cards[0], my_cards[1])
        pf_bucket = min(int(equity * self.K_PREFLOP), self.K_PREFLOP - 1)
        state[pf_bucket] = 1.0

        # Betting/street state from corrected state machine
        st = self._game()._parse_state(history)
        street_idx = st["street_idx"]
        n_visible = [0, 3, 4, 5][min(street_idx, 3)]

        # Board strength + bucket [8:16] (zeros preflop)
        visible_board = history[2][:n_visible] if n_visible > 0 else ()
        board_str = self._get_board_strength(my_cards, visible_board)
        if n_visible >= 3:
            brd_bucket = min(int(board_str * self.K_BOARD), self.K_BOARD - 1)
            state[8 + brd_bucket] = 1.0

        # Street one-hot [16:20]
        street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_visible, 0)
        state[16 + street] = 1.0

        # Betting scalars [20:24]
        pot       = st["pot"]
        to_call   = st["to_call"]
        my_stack  = st["stacks"][player]
        opp_stack = st["stacks"][1 - player]
        state[20] = pot / self.norm
        state[21] = to_call / self.norm
        state[22] = my_stack / self.starting_stack
        state[23] = opp_stack / self.starting_stack

        # Action history [24:32]
        # 'f' encodes as 0.0 to match C++ FOLD_OR_CHECK slot
        actions = history[3:]
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            if isinstance(a, str):
                enc = self.ACTION_ENC.get(a, 0.0)
                if a == 'f':
                    enc = 0.0
                state[24 + i] = enc
            else:
                state[24 + i] = min(a / self.starting_stack, 1.0)

        # Continuous hand features [32:36]
        state[32] = equity
        state[33] = board_str
        pot_plus_call = pot + to_call
        state[34] = to_call / pot_plus_call if pot_plus_call > 1e-6 else 0.0
        eff_stack = min(my_stack, opp_stack)
        state[35] = min(eff_stack / pot, 10.0) / 10.0 if pot > 1e-6 else 1.0

        # Quantise continuous dims to stable grouping grid
        cont = slice(20, 36)
        state[cont] = np.round(state[cont] / self.FEATURE_QUANT) * self.FEATURE_QUANT
        return state.astype(np.float32)