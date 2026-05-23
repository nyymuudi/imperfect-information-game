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
    MAX_POT = 26.0  # Max possible pot in Leduc (ante 1+1, R1 raise 2+2+2+2, R2 raise 4+4+4+4)
    HISTORY_LEN = 10

    def state_size(self) -> int:
        return 20

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(self.state_size(), dtype=np.float32)

        # Private card (one-hot)
        private_rank = history[player][0]
        state[self.RANK_IDX[private_rank]] = 1.0

        # Community card (one-hot, only if revealed)
        # Determine if round 1 is complete by parsing actions
        actions = history[3:]
        r1_done = self._is_round1_done(actions)

        if r1_done:
            comm_rank = history[2][0]
            state[3 + self.RANK_IDX[comm_rank]] = 1.0
            state[6] = 1.0  # Community revealed flag

        # Pot and betting info
        pot, to_call = self._compute_pot_info(actions, player)
        state[7] = pot / self.MAX_POT
        state[8] = to_call / self.MAX_POT
        state[9] = 1.0 if r1_done else 0.0

        # Action history (last HISTORY_LEN actions)
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            state[10 + i] = self.ACTION_ENC.get(a, 0.0)

        return state

    def _is_round1_done(self, actions: tuple) -> bool:
        """Check if round 1 betting is complete."""
        if len(actions) < 2:
            return False
        # Simulate round 1
        for i, a in enumerate(actions):
            if a == 'f':
                return False
            if a == 'k':
                return True  # Call ends round 1
            if a == 'c' and i >= 1:
                # Check after check (or check after any non-raise) ends round
                if actions[i-1] == 'c' or (i == 1 and actions[0] == 'c'):
                    return True
        return False

    def _compute_pot_info(self, actions: tuple, player: int) -> tuple[float, float]:
        """Compute current pot and amount to call."""
        pot = [1.0, 1.0]  # Antes
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
                opp = 1 - current
                pot[current] = pot[opp]
                if not r1_done:
                    r1_done = True
                    current = 0
                    continue
            elif a == 'c':
                if not r1_done and len([x for x in actions if x != 'f'][:actions.index(a)+1]) >= 2:
                    pass
            current = 1 - current

        opp = 1 - player
        to_call = max(0, pot[opp] - pot[player])
        total_pot = sum(pot)
        return total_pot, to_call