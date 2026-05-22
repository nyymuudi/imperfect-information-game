"""
Vanilla / Linear Counterfactual Regret Minimization (CFR).

Reference: Zinkevich, M. et al. (2007). "Regret Minimization in Games
with Incomplete Information." NIPS.

CFR computes a Nash equilibrium approximation for any finite
extensive-form game with imperfect information by iteratively:

1. Traversing the game tree for each player
2. Computing counterfactual values at each information set
3. Accumulating regret for each action not taken
4. Updating strategy proportional to positive regret (regret matching)

The average strategy over all iterations converges to a Nash equilibrium
at rate O(1/√T) in exploitability.

Linear CFR (Brown & Sandholm 2019) weights average strategy by iteration
number, improving convergence to O(1/T).

This implementation is fully domain-agnostic — it operates exclusively
through the ExtensiveFormGame interface.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from ..games.base import ExtensiveFormGame, InfoSetKey, Action, History


@dataclass
class InfoSetData:
    """Accumulated data for a single information set."""
    actions: list[Action]
    cumulative_regret: np.ndarray      # Σ regret per action
    cumulative_strategy: np.ndarray    # Σ (reach_prob * strategy) per action

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def current_strategy(self) -> np.ndarray:
        """Regret-matching: strategy proportional to positive regret."""
        positive = np.maximum(self.cumulative_regret, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(self.num_actions) / self.num_actions

    def average_strategy(self) -> np.ndarray:
        """Average strategy over all iterations — this converges to Nash."""
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return np.ones(self.num_actions) / self.num_actions


@dataclass
class CFRSolver:
    """
    Domain‑agnostic CFR solver.

    Attributes:
        game: The game to solve.
        info_sets: Maps info set keys to accumulated regret/strategy data.
        info_set_players: Maps info set keys to the player who acts at that info set.
        iterations: Number of CFR iterations completed.
        linear_averaging: If True, use Linear CFR weighting (O(1/T)).
    """
    game: ExtensiveFormGame
    info_sets: dict[InfoSetKey, InfoSetData] = field(default_factory=dict)
    info_set_players: dict[InfoSetKey, int] = field(default_factory=dict)
    iterations: int = 0
    linear_averaging: bool = True

    def _get_or_create_info_set(
        self, key: InfoSetKey, actions: list[Action], player: int
    ) -> InfoSetData:
        if key not in self.info_sets:
            n = len(actions)
            self.info_sets[key] = InfoSetData(
                actions=list(actions),
                cumulative_regret=np.zeros(n),
                cumulative_strategy=np.zeros(n),
            )
            self.info_set_players[key] = player
        return self.info_sets[key]

    def _cfr_recursive(
        self,
        history: History,
        reach_probs: np.ndarray,  # Shape: (num_players,)
        traversing_player: int,
    ) -> float:
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player = self.game.current_player(history)
        actions = self.game.legal_actions(history)
        info_key = self.game.info_set_key(history, player)
        info_set = self._get_or_create_info_set(info_key, actions, player)

        strategy = info_set.current_strategy()

        if player == traversing_player:
            weight = (self.iterations + 1) if self.linear_averaging else 1.0
            info_set.cumulative_strategy += weight * reach_probs[player] * strategy

        action_values = np.zeros(len(actions))
        for i, action in enumerate(actions):
            new_history = self.game.apply_action(history, action)
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values[i] = self._cfr_recursive(
                new_history, new_reach, traversing_player
            )

        node_value = (strategy * action_values).sum()

        if player == traversing_player:
            cf_reach = np.prod(
                [reach_probs[p] for p in range(self.game.num_players()) if p != player]
            )
            regret = cf_reach * (action_values - node_value)
            info_set.cumulative_regret += regret

        return node_value

    def solve(
        self,
        iterations: int = 10000,
        callback: Optional[Callable[["CFRSolver", int], Any]] = None,
        callback_freq: int = 100,
        mode: str = "linear",
    ) -> list[dict[InfoSetKey, np.ndarray]]:
        """
        Run CFR.

        Returns:
            [p0_strategy, p1_strategy] where each is a dict
            {info_set_key: np.ndarray of probabilities}.
            Order of array elements matches the order of legal_actions.
        """
        self.linear_averaging = (mode == "linear")
        num_players = self.game.num_players()
        initial_states = self.game.initial_histories()

        for t in range(1, iterations + 1):
            for traversing_player in range(num_players):
                for init_history, chance_prob in initial_states:
                    reach = np.ones(num_players) * chance_prob
                    self._cfr_recursive(init_history, reach, traversing_player)

            self.iterations += 1

            if callback and t % callback_freq == 0:
                callback(self, t)

        # Build per‑player average strategy dicts (values are numpy arrays)
        player_strategies = [{} for _ in range(num_players)]
        for key, data in self.info_sets.items():
            player = self.info_set_players[key]
            player_strategies[player][key] = data.average_strategy()

        return player_strategies

    # ── exploitability (for very small games) ──
    def exploitability(self) -> float:
        total = 0.0
        for player in range(self.game.num_players()):
            total += self._best_response_value(player)
        return total

    def _get_player_info_sets(self, player: int) -> list[InfoSetKey]:
        player_sets = []
        for init_h, _ in self.game.initial_histories():
            self._collect_player_info_sets(init_h, player, player_sets)
        return list(dict.fromkeys(player_sets))

    def _collect_player_info_sets(
        self, history: History, player: int, result: list
    ) -> None:
        if self.game.is_terminal(history):
            return
        current = self.game.current_player(history)
        if current == player:
            key = self.game.info_set_key(history, player)
            result.append(key)
        for action in self.game.legal_actions(history):
            self._collect_player_info_sets(
                self.game.apply_action(history, action), player, result
            )

    def _best_response_value(self, br_player: int) -> float:
        from itertools import product

        player_info_sets = self._get_player_info_sets(br_player)
        if not player_info_sets:
            return 0.0

        actions_per_set = []
        for key in player_info_sets:
            data = self.info_sets[key]
            actions_per_set.append(list(range(data.num_actions)))

        best_value = float("-inf")
        for action_combo in product(*actions_per_set):
            pure_strategy = dict(zip(player_info_sets, action_combo))
            value = self._eval_pure_strategy(br_player, pure_strategy)
            if value > best_value:
                best_value = value
        return best_value

    def _eval_pure_strategy(
        self,
        br_player: int,
        pure_strategy: dict[InfoSetKey, int],
    ) -> float:
        total = 0.0
        for init_history, chance_prob in self.game.initial_histories():
            total += chance_prob * self._eval_recursive(
                init_history, br_player, pure_strategy
            )
        return total

    def _eval_recursive(
        self,
        history: History,
        br_player: int,
        pure_strategy: dict[InfoSetKey, int],
    ) -> float:
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[br_player]

        player = self.game.current_player(history)
        actions = self.game.legal_actions(history)
        info_key = self.game.info_set_key(history, player)

        if player == br_player:
            action_idx = pure_strategy.get(info_key, 0)
            return self._eval_recursive(
                self.game.apply_action(history, actions[action_idx]),
                br_player, pure_strategy,
            )
        else:
            data = self.info_sets.get(info_key)
            if data is None:
                avg = np.ones(len(actions)) / len(actions)
            else:
                avg = data.average_strategy()
            return sum(
                avg[i] * self._eval_recursive(
                    self.game.apply_action(history, a),
                    br_player, pure_strategy,
                )
                for i, a in enumerate(actions)
            )