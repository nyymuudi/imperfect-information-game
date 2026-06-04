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

    State vector (124 dimensions):
        [0:52]    Private cards (one-hot, 2 bits set)
        [52:104]  Board cards (one-hot, 0-5 bits set)
        [104:108] Street (one-hot: preflop/flop/turn/river)
        [108]     Pot size (normalized by 2× starting stack)
        [109]     Amount to call (normalized)
        [110]     Own stack remaining (normalized)
        [111]     Opponent stack remaining (normalized)
        [112:120] Action history (last 8 actions encoded)
        [120]     Preflop equity of own hand (deterministic per hand class)
        [121]     Board strength of own hand (evaluate_7card / MAX_HAND_SCORE)
        [122]     Pot odds = to_call / (pot + to_call)
        [123]     SPR = min(stacks) / pot (capped, normalised)

    Card encoding: card = rank * 4 + suit
        rank: 0=2, 1=3, ..., 12=A
        suit: 0=♣, 1=♦, 2=♥, 3=♠

    Betting/street parse: this encoder reads pot, to_call, stacks and street
    from PostflopNLHE._parse_state — the SAME corrected state machine the C++
    NLHEGame engine implements — instead of re-deriving them with a separate
    parser. Keeping a single source of truth is what makes the Python training
    features bit-match the C++ encoder inference features (the historical
    "feature mismatch" that caused strategy collapse).

    Determinism of dims 120-121: the preflop-equity feature comes from
    `canonical_preflop_equity`, which is seeded per hand class and is therefore
    reproducible across runs and across the C++ port. The board-strength
    feature normalises evaluate_7card by `MAX_HAND_SCORE`, the exact maximum of
    the packed evaluator score, so it lands in [0, 1] without a magic constant.
    Both features are quantised to a 1e-6 grid on output so that the exact
    grouping in DeepCFRSolver._collapse_by_state is robust to float noise.
    """

    ACTION_ENC = {"f": -1.0, "c": 0.0, "k": 0.25, "r": 0.5, "b": 0.75, "a": 1.0}
    HISTORY_LEN = 8

    # Quantisation grid for the continuous feature dims. _collapse_by_state in
    # the solver groups rows by exact state-vector equality (np.unique); even a
    # 1-ULP difference would split a group. Rounding the continuous dims to this
    # grid makes the grouping deterministic and matches the C++ state_key, which
    # quantises to the same 1e-6 grid before hashing.
    FEATURE_QUANT = 1e-6

    _shared_equity_cache = None

    def __init__(self, starting_stack: float = 200.0, equity_sims: int = 2000):
        self.starting_stack = starting_stack
        self.norm = 2 * starting_stack  # Max pot ≈ 2× stacks
        self._equity_sims = equity_sims
        self._game_cache = None
        if NLHEEncoder._shared_equity_cache is None:
            NLHEEncoder._shared_equity_cache = {}
            self._build_equity_cache()
        self._equity_cache = NLHEEncoder._shared_equity_cache

    def _game(self):
        """Lazily build a PostflopNLHE matching this encoder's stack, so the
        encoder reads pot/to_call/street/stacks from the SAME (corrected) state
        machine the C++ engine uses, rather than re-parsing the history with a
        separate (and previously divergent) parser."""
        if self._game_cache is None:
            from ..games.postflop_nlhe import PostflopNLHE
            self._game_cache = PostflopNLHE(
                starting_stack=self.starting_stack,
                max_raises_per_street=2,
                raise_fractions=(0.75,),
            )
        return self._game_cache

    def _build_equity_cache(self):
        """Pre-compute deterministic preflop equity for all 169 hand classes.

        Uses the disk-cached table so the (slow) Monte Carlo runs once per sim
        count and is reused across processes; this also lets the C++ engine read
        the same values for dim-120 parity."""
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
        return 124

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(124, dtype=np.float32)

        # Private cards (one-hot in 52-dim)
        p0_cards = history[0]  # (card1, card2)
        p1_cards = history[1]
        my_cards = p0_cards if player == 0 else p1_cards
        state[my_cards[0]] = 1.0
        state[my_cards[1]] = 1.0

        actions = history[3:]

        # Authoritative betting/street state from the corrected state machine
        # (mirrors C++ NLHEGame). Single source of truth for pot/to_call/street.
        st = self._game()._parse_state(history)
        street_idx = st["street_idx"]
        n_visible = [0, 3, 4, 5][min(street_idx, 3)]

        # Board cards (only visible ones based on street)
        board = history[2]  # Full pre-dealt board
        for card in board[:n_visible]:
            state[52 + card] = 1.0

        # Street (one-hot)
        street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_visible, 0)
        state[104 + street] = 1.0

        # Betting info (read straight from the state machine)
        pot      = st["pot"]
        to_call  = st["to_call"]
        my_stack = st["stacks"][player]
        opp_stack = st["stacks"][1 - player]
        state[108] = pot / self.norm
        state[109] = to_call / self.norm
        state[110] = my_stack / self.starting_stack
        state[111] = opp_stack / self.starting_stack

        # Action history. 'f' and 'c' are distinct in the Python alphabet, but
        # the C++ engine stores both as FOLD_OR_CHECK (slot 0) and encodes that
        # slot as 0.0. To keep dim 112-119 bit-identical across implementations,
        # encode a terminal-only 'f' as 0.0 too: a fold ends the hand, so a
        # folded node is never queried for a strategy during traversal, and
        # using 0.0 here removes the only remaining action-history divergence.
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            if isinstance(a, str):
                enc = self.ACTION_ENC.get(a, 0.0)
                if a == 'f':
                    enc = 0.0  # parity with C++ FOLD_OR_CHECK slot
                state[112 + i] = enc
            else:
                state[112 + i] = min(a / self.starting_stack, 1.0)

        # Hand strength features (key for convergence) — deterministic.
        state[120] = self._get_preflop_equity(my_cards[0], my_cards[1])
        visible_board = history[2][:n_visible] if n_visible > 0 else ()
        state[121] = self._get_board_strength(my_cards, visible_board)

        # dim [122]: pot odds = to_call / (pot + to_call)
        # Captures the immediate cost of calling relative to the reward.
        pot_plus_call = pot + to_call
        state[122] = to_call / pot_plus_call if pot_plus_call > 1e-6 else 0.0

        # dim [123]: SPR = min(stacks) / pot, normalised (cap at 10)
        # Stack-to-pot ratio: large SPR → post-flop implied odds matter more.
        eff_stack  = min(my_stack, opp_stack)
        state[123] = min(eff_stack / pot, 10.0) / 10.0 if pot > 1e-6 else 1.0

        # Quantise the continuous dims to a fixed grid so _collapse_by_state's
        # exact grouping (and the C++ state_key) treat identical nodes as equal.
        cont = slice(108, 124)
        state[cont] = np.round(state[cont] / self.FEATURE_QUANT) * self.FEATURE_QUANT
        return state.astype(np.float32)