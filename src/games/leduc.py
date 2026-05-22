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

This game has ~936 information sets (depending on exact action
abstraction), making it the standard testbed for comparing CFR
variants before scaling to full NLHE.

Complexity comparison:
    Kuhn:  3 cards, 1 round,  12 info sets
    Leduc: 6 cards, 2 rounds, ~936 info sets  (78× more)
    NLHE:  52 cards, 4 rounds, ~10¹⁴ info sets
"""

from .base import ExtensiveFormGame, History, Action, InfoSetKey


RANKS = ["J", "Q", "K"]
RANK_VALUE = {"J": 0, "Q": 1, "K": 2}

# Bet sizes per round (0-indexed)
RAISE_SIZE = {0: 2, 1: 4}
MAX_RAISES_PER_ROUND = 2


class LeducHoldem(ExtensiveFormGame):
    """
    Leduc Hold'em implementation.
    
    History encoding:
        (card_p0, card_p1, community_card, action_0, action_1, ...)
        
    First two elements: dealt private cards (rank + suit identifier).
    Third element: community card rank (e.g. 'J1', 'K2'). Always present
    (pre‑expanded chance node).
    Subsequent elements: player actions.
    
    Actions:
        'c' = check
        'r' = raise (2 chips round 1, 4 chips round 2)
        'f' = fold
        'k' = call
    """

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        """
        Pre‑expand all chance nodes: deal private cards AND community card.
        
        6 cards, 2 dealt = 30 ordered pairs.
        For each, choose 1 of the 4 remaining as community = 120 total.
        Probability = 1/120 ≈ 0.008333...
        """
        cards = ["J1", "J2", "Q1", "Q2", "K1", "K2"]
        histories = []
        for i, c0 in enumerate(cards):
            for j, c1 in enumerate(cards):
                if i == j:
                    continue
                remaining = [c for c in cards if c != c0 and c != c1]
                for comm in remaining:
                    # History: (p0_card, p1_card, community, (empty actions))
                    histories.append(((c0, c1, comm), 1.0 / 120))
        return histories

    def _card_rank(self, card: str) -> str:
        """Extract rank from card identifier (e.g., 'J1' → 'J')."""
        return card[0]

    def _get_actions(self, history: History) -> tuple:
        """Extract action sequence from history."""
        return history[3:]

    def _parse_betting_state(self, actions: tuple) -> dict:
        pot = [1, 1]  # Antes
        r1_actions = []
        r2_actions = []
        round_1_done = False
        game_folded = False
        raises_r1 = 0
        raises_r2 = 0

        current_round_actions = r1_actions
        current_player = 0

        for a in actions:
            current_round_actions.append(a)

            if a == 'f':
                game_folded = True
                break
            elif a == 'r':
                round_num = 1 if round_1_done else 0
                raise_size = RAISE_SIZE[round_num]
                opponent = 1 - current_player
                pot[current_player] = pot[opponent] + raise_size
                if not round_1_done:
                    raises_r1 += 1
                else:
                    raises_r2 += 1
            elif a == 'k':
                opponent = 1 - current_player
                pot[current_player] = pot[opponent]
                if not round_1_done:
                    round_1_done = True
                    current_round_actions = r2_actions
                    current_player = 1   # vaihdetaan 1:ksi, jotta silmukan lopussa tulee 0
                    # ei continue -ta!
            elif a == 'c':
                if not round_1_done:
                    if len(r1_actions) >= 2 and r1_actions[-1] == 'c':
                        round_1_done = True
                        current_round_actions = r2_actions
                        current_player = 1   # sama kuin yllä
                        # ei continue -ta!

            current_player = 1 - current_player   # tämä hoitaa vuoron vaihdon

        return {
            "round_1_complete": round_1_done,
            "game_folded": game_folded,
            "raises_r1": raises_r1,
            "raises_r2": raises_r2,
            "pot": pot,
            "round_1_actions": tuple(r1_actions),
            "round_2_actions": tuple(r2_actions),
        }

    def _betting_round_terminal(self, round_actions: tuple) -> tuple[bool, str]:
        """
        Check if a betting round is complete.
        
        Returns (is_complete, reason) where reason is:
            'fold' - someone folded
            'showdown' - round ended normally (check-check or call)
            'pending' - round not yet complete
        """
        if not round_actions:
            return False, "pending"
        
        if round_actions[-1] == 'f':
            return True, "fold"
        
        if len(round_actions) < 2:
            return False, "pending"
        
        if round_actions == ('c', 'c'):
            return True, "showdown"
        
        if round_actions[-1] == 'k':
            return True, "showdown"
        
        return False, "pending"

    def is_terminal(self, history: History) -> bool:
        actions = self._get_actions(history)
        if not actions:
            return False
        
        if 'f' in actions:
            return True
        
        state = self._parse_betting_state(actions)
        if state["game_folded"]:
            return True
        
        # If round 1 complete, we are in round 2; check if round 2 is complete
        if state["round_1_complete"]:
            r2 = state["round_2_actions"]
            done, _ = self._betting_round_terminal(r2)
            return done
        
        return False

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if not self.is_terminal(history):
            raise ValueError(f"Non-terminal history: {history}")

        actions = self._get_actions(history)
        state = self._parse_betting_state(actions)
        pot = state["pot"]
        
        if state["game_folded"]:
            # Find who folded
            player = 0
            for a in actions:
                if a == 'f':
                    other = 1 - player
                    return (float(pot[1]), float(-pot[1])) if player == 0 else (float(-pot[0]), float(pot[0]))
                elif a == 'k' and not state["round_1_complete"]:
                    # Round 1 ended by call, reset player for round 2
                    player = 0
                    continue
                player = 1 - player
            raise ValueError(f"Fold not found in folded game: {history}")

        # Showdown
        rank_p0 = self._card_rank(history[0])
        rank_p1 = self._card_rank(history[1])
        community = self._card_rank(history[2])

        winner = self._compare_hands(rank_p0, rank_p1, community)
        
        if winner == 0:
            return (float(pot[1]), float(-pot[1]))
        elif winner == 1:
            return (float(-pot[0]), float(pot[0]))
        else:
            return (0.0, 0.0)

    def _compare_hands(self, rank_p0: str, rank_p1: str, community: str) -> int:
        """Compare hands. Returns 0 if P0 wins, 1 if P1 wins, -1 if tie."""
        p0_pair = (rank_p0 == community)
        p1_pair = (rank_p1 == community)
        
        if p0_pair and not p1_pair:
            return 0
        if p1_pair and not p0_pair:
            return 1
        
        v0, v1 = RANK_VALUE[rank_p0], RANK_VALUE[rank_p1]
        if v0 > v1:
            return 0
        elif v1 > v0:
            return 1
        return -1

    def current_player(self, history: History) -> int:
        """Determine whose turn it is."""
        actions = self._get_actions(history)
        if not actions:
            return 0
        
        state = self._parse_betting_state(actions)
        if state["round_1_complete"]:
            # Round 2: count round 2 actions
            return len(state["round_2_actions"]) % 2
        else:
            # Round 1
            return len(state["round_1_actions"]) % 2

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Info set = private card rank + community card (only if round 1 complete)
        + action sequence.
        """
        private_rank = self._card_rank(history[player])
        actions = self._get_actions(history)
        state = self._parse_betting_state(actions)
        
        if state["round_1_complete"]:
            community = self._card_rank(history[2])
            return f"{private_rank}|{community}|{''.join(actions)}"
        else:
            # Community card hidden
            return f"{private_rank}||{''.join(actions)}"

    def legal_actions(self, history: History) -> list[Action]:
        actions = self._get_actions(history)
        state = self._parse_betting_state(actions)
        
        if state["round_1_complete"]:
            # Round 2
            r2 = state["round_2_actions"]
            if not r2:
                if state["raises_r2"] < MAX_RAISES_PER_ROUND:
                    return ["c", "r"]
                return ["c"]
            if r2[-1] == 'r':
                if state["raises_r2"] < MAX_RAISES_PER_ROUND:
                    return ["f", "k", "r"]
                return ["f", "k"]
            return ["c", "r"] if state["raises_r2"] < MAX_RAISES_PER_ROUND else ["c"]
        
        # Round 1
        if not actions:
            return ["c", "r"]
        
        r1 = state["round_1_actions"]
        if r1[-1] == 'r':
            if state["raises_r1"] < MAX_RAISES_PER_ROUND:
                return ["f", "k", "r"]
            return ["f", "k"]
        return ["c", "r"] if state["raises_r1"] < MAX_RAISES_PER_ROUND else ["c"]

    def apply_action(self, history: History, action: Action) -> History:
        return history + (action,)