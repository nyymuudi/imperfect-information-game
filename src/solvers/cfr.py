"""
Vanilla Counterfactual Regret Minimization (CFR).

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

This implementation is fully domain-agnostic — it operates exclusively
through the ExtensiveFormGame interface.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

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
    Vanilla CFR solver for any ExtensiveFormGame.
    
    Attributes:
        game: The game to solve.
        info_sets: Maps info set keys to accumulated regret/strategy data.
        iterations: Number of CFR iterations completed.
    """
    game: ExtensiveFormGame
    info_sets: dict[InfoSetKey, InfoSetData] = field(default_factory=dict)
    iterations: int = 0
    linear_averaging: bool = True  # Linear CFR (Brown & Sandholm 2019)
    cfr_plus: bool = False         # CFR+ (Tammelin 2014) — clamp regrets ≥ 0

    def _get_or_create_info_set(
        self, key: InfoSetKey, actions: list[Action]
    ) -> InfoSetData:
        if key not in self.info_sets:
            n = len(actions)
            self.info_sets[key] = InfoSetData(
                actions=list(actions),
                cumulative_regret=np.zeros(n),
                cumulative_strategy=np.zeros(n),
            )
        return self.info_sets[key]

    def _cfr_recursive(
        self,
        history: History,
        reach_probs: np.ndarray,  # Shape: (num_players,)
        traversing_player: int,
    ) -> float:
        """
        Recursive CFR traversal. Returns counterfactual value
        for the traversing player at this history node.
        """
        # Terminal node → return payoff
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player = self.game.current_player(history)
        actions = self.game.legal_actions(history)
        info_key = self.game.info_set_key(history, player)
        info_set = self._get_or_create_info_set(info_key, actions)

        strategy = info_set.current_strategy()

        # Accumulate strategy weighted by player's reach probability
        # Linear CFR: weight by iteration number for O(1/T) convergence
        if player == traversing_player:
            weight = (self.iterations + 1) if self.linear_averaging else 1
            info_set.cumulative_strategy += weight * reach_probs[player] * strategy

        # Compute counterfactual value for each action
        action_values = np.zeros(len(actions))
        for i, action in enumerate(actions):
            new_history = self.game.apply_action(history, action)

            # Update reach probabilities
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]

            action_values[i] = self._cfr_recursive(
                new_history, new_reach, traversing_player
            )

        # Node value under current strategy
        node_value = (strategy * action_values).sum()

        # Update regret only for the acting player
        if player == traversing_player:
            # Counterfactual reach = product of all opponents' reach probs
            cf_reach = np.prod(
                [reach_probs[p] for p in range(self.game.num_players()) if p != player]
            )
            regret = cf_reach * (action_values - node_value)
            info_set.cumulative_regret += regret

        return node_value

    def solve(
        self,
        iterations: int = 10000,
        callback: Optional[callable] = None,
        callback_freq: int = 100,
    ) -> dict[InfoSetKey, np.ndarray]:
        """
        Run CFR for the specified number of iterations.
        
        Args:
            iterations: Number of full CFR iterations.
            callback: Optional fn(solver, iteration) called every callback_freq iters.
            callback_freq: How often to invoke callback.
            
        Returns:
            Dictionary mapping info set keys to average (Nash-convergent) strategies.
        """
        num_players = self.game.num_players()
        initial_states = self.game.initial_histories()

        for t in range(1, iterations + 1):
            for traversing_player in range(num_players):
                for init_history, chance_prob in initial_states:
                    reach = np.ones(num_players) * chance_prob
                    self._cfr_recursive(init_history, reach, traversing_player)

            self.iterations += 1

            # CFR+: clamp cumulative regrets to ≥ 0 after each iteration
            # This prevents negative regrets from accumulating, yielding
            # faster convergence (Tammelin, 2014)
            if self.cfr_plus:
                for data in self.info_sets.values():
                    np.maximum(data.cumulative_regret, 0, out=data.cumulative_regret)

            if callback and t % callback_freq == 0:
                callback(self, t)

        return {
            key: data.average_strategy()
            for key, data in self.info_sets.items()
        }

    def exploitability(self) -> float:
        """
        Compute exploitability of the current average strategy.
        
        For small games (≤20 info sets/player): exact enumeration.
        For larger games: tree-walk best response (bottom-up).
        """
        total = 0.0
        for player in range(self.game.num_players()):
            total += self._best_response_value(player)
        return total

    def _get_player_info_sets(self, player: int) -> list[InfoSetKey]:
        """Return info set keys belonging to a specific player."""
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
        """
        Compute best response value using appropriate method.
        """
        player_info_sets = self._get_player_info_sets(br_player)
        
        if len(player_info_sets) <= 20:
            return self._br_enumeration(br_player, player_info_sets)
        else:
            return self._br_tree_walk(br_player)

    def _br_enumeration(self, br_player: int, player_info_sets: list) -> float:
        """Exact BR via pure strategy enumeration (small games only)."""
        from itertools import product

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

    def _br_tree_walk(self, br_player: int) -> float:
        """
        Tree-walk best response for larger games.
        
        Bottom-up: accumulates per-info-set action values weighted
        by opponent reach, then picks best action per info set.
        
        Two-pass approach:
        1. Traverse tree, accumulate action values per BR info set
        2. Pick best action per info set, recompute total value
        """
        from collections import defaultdict

        # Pass 1: accumulate action values per info set
        is_action_values: dict[InfoSetKey, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        def _accumulate(history: History, opp_reach: float) -> float:
            """Returns value for br_player at this node."""
            if self.game.is_terminal(history):
                return self.game.terminal_payoffs(history)[br_player] * opp_reach

            player = self.game.current_player(history)
            actions = self.game.legal_actions(history)
            info_key = self.game.info_set_key(history, player)

            if player == br_player:
                # Compute value of each action
                best_val = float("-inf")
                for i, action in enumerate(actions):
                    val = _accumulate(
                        self.game.apply_action(history, action), opp_reach
                    )
                    is_action_values[info_key][i] += val
                    best_val = max(best_val, val)
                return best_val  # Clairvoyant for downstream estimation
            else:
                info_set = self._get_or_create_info_set(info_key, actions)
                strategy = info_set.average_strategy()
                total = 0.0
                for i, action in enumerate(actions):
                    total += _accumulate(
                        self.game.apply_action(history, action),
                        opp_reach * strategy[i],
                    )
                return total

        for init_h, chance_prob in self.game.initial_histories():
            _accumulate(init_h, chance_prob)

        # Pick best action per info set
        best_actions: dict[InfoSetKey, int] = {}
        for key, action_vals in is_action_values.items():
            best_actions[key] = max(action_vals, key=action_vals.get)

        # Pass 2: compute actual BR value with fixed actions
        def _evaluate(history: History, weight: float) -> float:
            if self.game.is_terminal(history):
                return self.game.terminal_payoffs(history)[br_player] * weight

            player = self.game.current_player(history)
            actions = self.game.legal_actions(history)
            info_key = self.game.info_set_key(history, player)

            if player == br_player:
                best_idx = best_actions.get(info_key, 0)
                return _evaluate(
                    self.game.apply_action(history, actions[best_idx]),
                    weight,
                )
            else:
                info_set = self._get_or_create_info_set(info_key, actions)
                strategy = info_set.average_strategy()
                return sum(
                    strategy[i] * _evaluate(
                        self.game.apply_action(history, a), weight * strategy[i]
                    )
                    for i, a in enumerate(actions)
                )

        total = 0.0
        for init_h, chance_prob in self.game.initial_histories():
            total += _evaluate(init_h, chance_prob)
        return total

    def _eval_pure_strategy(
        self,
        br_player: int,
        pure_strategy: dict[InfoSetKey, int],
    ) -> float:
        """Evaluate a pure strategy for br_player against avg strategy of opponents."""
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
            # Play the pure strategy action
            action_idx = pure_strategy.get(info_key, 0)
            return self._eval_recursive(
                self.game.apply_action(history, actions[action_idx]),
                br_player, pure_strategy,
            )
        else:
            # Opponent plays average strategy
            info_set = self._get_or_create_info_set(info_key, actions)
            strategy = info_set.average_strategy()
            return sum(
                strategy[i] * self._eval_recursive(
                    self.game.apply_action(history, a),
                    br_player, pure_strategy,
                )
                for i, a in enumerate(actions)
            )