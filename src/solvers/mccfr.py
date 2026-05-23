"""
External Sampling Monte Carlo CFR (ES-MCCFR).

Reference: Lanctot, M. et al. (2009). "Monte Carlo Sampling for
Regret Minimization in Extensive Games." NIPS.

Unlike vanilla CFR which traverses the ENTIRE game tree each
iteration, ES-MCCFR samples:
    1. One chance outcome (initial state) per iteration
    2. One opponent action per opponent decision node

The traversing player's actions are still fully expanded.
This yields:
    - Much faster iterations (no full tree traversal)
    - Higher variance per iteration (sampling noise)
    - Same convergence guarantee (average strategy → Nash)

Tradeoff: vanilla CFR is O(|tree|) per iteration with zero variance.
ES-MCCFR is O(|player_actions|^depth) per iteration with variance.
For large games, the variance tradeoff is overwhelmingly favorable.

This is the algorithmic family that enabled Libratus and Pluribus.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..games.base import ExtensiveFormGame, InfoSetKey, Action, History
from .cfr import InfoSetData


@dataclass
class MCCFRSolver:
    """
    External Sampling MCCFR solver.
    
    Same interface as CFRSolver but uses Monte Carlo sampling
    for chance nodes and opponent actions.
    """
    game: ExtensiveFormGame
    info_sets: dict[InfoSetKey, InfoSetData] = field(default_factory=dict)
    iterations: int = 0
    linear_averaging: bool = True
    cfr_plus: bool = True  # CFR+ by default for MCCFR

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

    def _mccfr_recursive(
        self,
        history: History,
        traversing_player: int,
        rng: np.random.Generator,
    ) -> float:
        """
        External sampling MCCFR traversal.
        
        - At traversing player nodes: expand ALL actions (compute regret)
        - At opponent nodes: SAMPLE one action from current strategy
        - Returns counterfactual value for traversing player
        """
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player = self.game.current_player(history)
        actions = self.game.legal_actions(history)
        info_key = self.game.info_set_key(history, player)
        info_set = self._get_or_create_info_set(info_key, actions)
        strategy = info_set.current_strategy()

        if player == traversing_player:
            # Traversing player: expand ALL actions
            action_values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                new_h = self.game.apply_action(history, action)
                action_values[i] = self._mccfr_recursive(
                    new_h, traversing_player, rng
                )

            # Node value under current strategy
            node_value = (strategy * action_values).sum()

            # Update regret (no reach probability weighting needed —
            # sampling probability cancels with counterfactual reach)
            regret = action_values - node_value
            info_set.cumulative_regret += regret

            # Update strategy sum
            weight = (self.iterations + 1) if self.linear_averaging else 1
            info_set.cumulative_strategy += weight * strategy

            return node_value

        else:
            # Opponent: SAMPLE one action from strategy
            action_idx = rng.choice(len(actions), p=strategy)
            new_h = self.game.apply_action(history, actions[action_idx])
            return self._mccfr_recursive(new_h, traversing_player, rng)

    def solve(
        self,
        iterations: int = 10000,
        callback: Optional[callable] = None,
        callback_freq: int = 100,
        seed: int = 42,
    ) -> dict[InfoSetKey, np.ndarray]:
        """
        Run ES-MCCFR for the specified number of iterations.
        
        Each iteration:
            1. Sample one initial state (chance outcome)
            2. Traverse for each player using external sampling
        """
        rng = np.random.default_rng(seed)
        num_players = self.game.num_players()
        initial_states = self.game.initial_histories()

        # Precompute chance sampling distribution
        chance_probs = np.array([p for _, p in initial_states])
        chance_histories = [h for h, _ in initial_states]

        for t in range(1, iterations + 1):
            for traversing_player in range(num_players):
                # Sample ONE initial state
                idx = rng.choice(len(chance_histories), p=chance_probs)
                init_h = chance_histories[idx]

                self._mccfr_recursive(init_h, traversing_player, rng)

            self.iterations += 1

            # CFR+: clamp regrets
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
        """Reuse CFR's exploitability computation."""
        from .cfr import CFRSolver
        # Create a temporary CFR solver with the same info sets
        tmp = CFRSolver(game=self.game)
        tmp.info_sets = self.info_sets
        return tmp.exploitability()