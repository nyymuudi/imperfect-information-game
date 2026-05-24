"""
Deep CFR Solver.

Reference: Steinberger, E. (2019). "Single Deep Counterfactual
Regret Minimization." arXiv:1901.07621.

Replaces tabular regret/strategy storage with neural networks,
enabling scaling to games with 10^10+ information sets where
tabular CFR is infeasible.

Training loop:
    1. Run MCCFR traversals using regret network for strategy
    2. Store computed regrets in MR buffer
    3. Store current strategies in MΠ buffer
    4. Periodically retrain regret network from MR
    5. After all iterations: train strategy network from MΠ

The regret network replaces the regret table as the strategy
source during traversal. The strategy network is the final
output — the trained Nash equilibrium approximation.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from ..games.base import ExtensiveFormGame, History, Action, InfoSetKey
from .state_encoder import StateEncoder
from .replay_buffer import ReservoirBuffer
from .networks import (
    RegretNetwork, StrategyNetwork,
    train_regret_network, train_strategy_network,
)


@dataclass
class DeepCFRSolver:
    """
    Deep CFR solver for any ExtensiveFormGame.
    
    Uses neural networks instead of regret tables,
    enabling scaling beyond tabular CFR's memory limits.
    """
    game: ExtensiveFormGame
    encoder: StateEncoder
    max_actions: int = 5          # Maximum legal actions at any node
    buffer_capacity: int = 500000
    hidden_size: int = 256
    train_epochs: int = 100
    train_batch: int = 256
    lr: float = 1e-3
    traversals_per_iter: int = 100  # MCCFR traversals per training step

    def __post_init__(self):
        state_sz = self.encoder.state_size()

        # Neural networks
        self.regret_net = RegretNetwork(state_sz, self.max_actions, self.hidden_size)
        self.strategy_net = StrategyNetwork(state_sz, self.max_actions, self.hidden_size)

        # Replay buffers
        self.regret_buffer = ReservoirBuffer(self.buffer_capacity, state_sz, self.max_actions)
        self.strategy_buffer = ReservoirBuffer(self.buffer_capacity, state_sz, self.max_actions)

        self.iterations = 0
        self._rng = np.random.default_rng(42)

    def _get_regret_strategy(self, state: np.ndarray, num_actions: int) -> np.ndarray:
        """
        Query regret network and apply regret matching.
        
        Returns probability distribution over actions.
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            regrets = self.regret_net(s).squeeze(0).numpy()[:num_actions]

        # Regret matching: proportional to positive regrets
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(num_actions) / num_actions

    def _traverse(
        self,
        history: History,
        traversing_player: int,
    ) -> float:
        """
        External sampling MCCFR traversal using regret network.
        
        Same structure as tabular MCCFR but:
        - Strategy comes from regret network (not table lookup)
        - Computed regrets stored in MR buffer
        - Current strategy stored in MΠ buffer
        """
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player = self.game.current_player(history)
        actions = self.game.legal_actions(history)
        num_actions = len(actions)

        # Encode state from current player's perspective
        state = self.encoder.encode(history, player)
        strategy = self._get_regret_strategy(state, num_actions)

        if player == traversing_player:
            # Store current strategy in MΠ
            self.strategy_buffer.add(
                state, strategy, float(self.iterations + 1)
            )

            # Expand all actions (external sampling)
            action_values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                new_h = self.game.apply_action(history, action)
                action_values[i] = self._traverse(new_h, traversing_player)

            # Compute regrets
            node_value = (strategy * action_values).sum()
            regrets = action_values - node_value

            # Store regrets in MR
            self.regret_buffer.add(
                state, regrets, float(self.iterations + 1)
            )

            return node_value

        else:
            # Opponent: sample one action
            action_idx = self._rng.choice(num_actions, p=strategy)
            new_h = self.game.apply_action(history, actions[action_idx])
            return self._traverse(new_h, traversing_player)

    def solve(
        self,
        iterations: int = 100,
        callback: Optional[callable] = None,
        callback_freq: int = 10,
    ) -> StrategyNetwork:
        """
        Run Deep CFR training loop.
        
        Args:
            iterations: Number of CFR meta-iterations.
                Each meta-iteration does multiple MCCFR traversals
                then retrains the regret network.
            callback: Optional fn(solver, iteration) for monitoring.
            callback_freq: How often to call callback.
            
        Returns:
            Trained StrategyNetwork (the final Nash approximation).
        """
        num_players = self.game.num_players()

        # Detect game type: sample_deal for large games, initial_histories for small
        has_sample_deal = hasattr(self.game, 'sample_deal')
        if not has_sample_deal:
            initial_states = self.game.initial_histories()
            chance_probs = np.array([p for _, p in initial_states])
            chance_histories = [h for h, _ in initial_states]

        for t in range(1, iterations + 1):
            # ── Step 1: Generate data via MCCFR traversals ──
            for _ in range(self.traversals_per_iter):
                for traversing_player in range(num_players):
                    if has_sample_deal:
                        init_h = self.game.sample_deal(self._rng)
                    else:
                        idx = self._rng.choice(len(chance_histories), p=chance_probs)
                        init_h = chance_histories[idx]
                    self._traverse(init_h, traversing_player)

            self.iterations += 1

            # ── Step 2: Retrain regret network ──
            if len(self.regret_buffer) >= self.train_batch:
                loss = train_regret_network(
                    self.regret_net, self.regret_buffer,
                    epochs=self.train_epochs,
                    batch_size=self.train_batch,
                    lr=self.lr,
                )

            if callback and t % callback_freq == 0:
                callback(self, t)

        # ── Step 3: Train final strategy network ──
        print(f"Training strategy network from {len(self.strategy_buffer)} samples...")
        if len(self.strategy_buffer) >= self.train_batch:
            train_strategy_network(
                self.strategy_net, self.strategy_buffer,
                epochs=300,
                batch_size=self.train_batch,
                lr=self.lr,
            )

        return self.strategy_net

    def get_strategy(self, history: History, player: int) -> np.ndarray:
        """
        Get the trained strategy for a game state.
        
        Uses the strategy network (final output).
        """
        state = self.encoder.encode(history, player)
        num_actions = len(self.game.legal_actions(history))

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Create action mask
            mask = torch.zeros(self.max_actions)
            mask[:num_actions] = 1.0
            probs = self.strategy_net(s, mask).squeeze(0).numpy()

        return probs[:num_actions]