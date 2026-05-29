"""
Deep CFR Solver.

Reference: Steinberger, E. (2019). "Single Deep Counterfactual
Regret Minimization." arXiv:1901.07621.

Buffer strategy (Steinberger 2019, Section 3):
    Regret buffer  (MR): SlidingWindowBuffer — only fresh data needed.
        The regret network predicts counterfactual regrets for the
        CURRENT strategy. Old regrets from early iterations corrupt
        the gradient signal. FIFO keeps only the last K samples.
    Strategy buffer (MΠ): ReservoirBuffer — needs full history average.
        The strategy network approximates the time-average strategy
        across all iterations, so reservoir sampling over the full
        history is correct here.
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
from .cpp_backend import CppMCCFRBackend, NLHECppBackend, engine_available


@dataclass
class DeepCFRSolver:
    game: ExtensiveFormGame
    encoder: StateEncoder
    max_actions: int = 5
    buffer_capacity: int = 500000
    strategy_buffer_capacity: int = 0     # 0 = same as buffer_capacity
    hidden_size: int = 256
    train_epochs: int = 100
    train_batch: int = 256
    lr: float = 1e-3
    traversals_per_iter: int = 100
    use_cpp_engine: bool = False
    device: str = "cpu"

    def __post_init__(self):
        state_sz  = self.encoder.state_size()
        strat_cap = self.strategy_buffer_capacity or self.buffer_capacity

        self.regret_net   = RegretNetwork(state_sz, self.max_actions, self.hidden_size).to(self.device)
        self.strategy_net = StrategyNetwork(state_sz, self.max_actions, self.hidden_size).to(self.device)

        # Regret buffer: mode='window' (FIFO — fresh data only).
        #   Steinberger 2019 §3: the regret network predicts regrets for the
        #   CURRENT strategy; old data from early iterations corrupts gradients.
        # Strategy buffer: mode='reservoir' (full history average).
        #   Approximates time-average strategy over all iterations — correct.
        self._current_iter   = 0
        self.regret_buffer   = ReservoirBuffer(self.buffer_capacity, state_sz, self.max_actions, mode='window')
        self.strategy_buffer = ReservoirBuffer(strat_cap, state_sz, self.max_actions, mode='reservoir')

        self.iterations = 0
        self._rng = np.random.default_rng(42)

        self._cpp = None
        if self.use_cpp_engine:
            if engine_available():
                from ..games.postflop_nlhe import PostflopNLHE
                if isinstance(self.game, PostflopNLHE):
                    self._cpp = NLHECppBackend(
                        n_traversals=self.traversals_per_iter,
                        regret_capacity=self.buffer_capacity,
                        strategy_capacity=strat_cap,
                        device=self.device,
                        starting_stack=self.game.starting_stack,
                        raise_fraction=self.game.raise_fractions[0],
                        max_raises=self.game.max_raises_per_street,
                    )
                else:
                    self._cpp = CppMCCFRBackend(
                        n_traversals=self.traversals_per_iter,
                        regret_capacity=self.buffer_capacity,
                        strategy_capacity=strat_cap,
                        device=self.device,
                    )
            else:
                import warnings
                warnings.warn(
                    "use_cpp_engine=True but cfr_engine.so not found.",
                    RuntimeWarning,
                )

    def _get_regret_strategy(self, state: np.ndarray, num_actions: int) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            regrets = self.regret_net(s).squeeze(0).cpu().numpy()[:num_actions]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(num_actions) / num_actions

    def _traverse(self, history: History, traversing_player: int) -> float:
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player      = self.game.current_player(history)
        actions     = self.game.legal_actions(history)
        num_actions = len(actions)
        state       = self.encoder.encode(history, player)
        strategy    = self._get_regret_strategy(state, num_actions)

        if player == traversing_player:
            self.strategy_buffer.add(state, strategy, float(self._current_iter + 1))
            action_values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                new_h = self.game.apply_action(history, action)
                action_values[i] = self._traverse(new_h, traversing_player)
            node_value = (strategy * action_values).sum()
            regrets = action_values - node_value
            self.regret_buffer.add(state, regrets, float(self._current_iter + 1))
            return node_value
        else:
            action_idx = self._rng.choice(num_actions, p=strategy)
            new_h = self.game.apply_action(history, actions[action_idx])
            return self._traverse(new_h, traversing_player)

    def _run_cpp_iteration(self) -> None:
        reg_exp, str_exp = self._cpp.run_iteration(
            self.iterations,
            regret_net=self.regret_net if self.iterations > 0 else None,
        )
        w = float(self._current_iter + 1)

        if len(reg_exp) > 0:
            X, actions, values = self._cpp.to_tensors(reg_exp)
            X_np = X.cpu().numpy()
            a_np = actions.cpu().numpy()
            v_np = values.cpu().numpy().astype(np.float32)
            mask = a_np < self.max_actions
            X_np, a_np, v_np = X_np[mask], a_np[mask], v_np[mask]
            n = len(X_np)
            if n > 0:
                reg_mat = np.zeros((n, self.max_actions), dtype=np.float32)
                np.add.at(reg_mat, (np.arange(n), a_np), v_np)
                self.regret_buffer.add_batch(X_np, reg_mat, np.full(n, w, dtype=np.float32))

        if len(str_exp) > 0:
            X, actions, values = self._cpp.to_tensors(str_exp)
            X_np = X.cpu().numpy()
            a_np = actions.cpu().numpy()
            v_np = values.cpu().numpy().astype(np.float32)
            mask = a_np < self.max_actions
            X_np, a_np, v_np = X_np[mask], a_np[mask], v_np[mask]
            n = len(X_np)
            if n > 0:
                str_mat = np.zeros((n, self.max_actions), dtype=np.float32)
                np.add.at(str_mat, (np.arange(n), a_np), v_np)
                self.strategy_buffer.add_batch(X_np, str_mat, np.full(n, w, dtype=np.float32))

    def solve(
        self,
        iterations: int = 100,
        callback: Optional[callable] = None,
        callback_freq: int = 10,
    ) -> StrategyNetwork:
        num_players = self.game.num_players()
        has_sample_deal = hasattr(self.game, 'sample_deal')
        if not has_sample_deal:
            initial_states   = self.game.initial_histories()
            chance_probs     = np.array([p for _, p in initial_states])
            chance_histories = [h for h, _ in initial_states]

        for t in range(1, iterations + 1):
            if self._cpp is not None:
                self._run_cpp_iteration()
            else:
                for _ in range(self.traversals_per_iter):
                    for traversing_player in range(num_players):
                        if has_sample_deal:
                            init_h = self.game.sample_deal(self._rng)
                        else:
                            idx    = self._rng.choice(len(chance_histories), p=chance_probs)
                            init_h = chance_histories[idx]
                        self._traverse(init_h, traversing_player)

            self.iterations += 1

            if len(self.regret_buffer) >= self.train_batch:
                # Fresh Adam optimizer each iteration — avoids momentum lock-in
                self._last_regret_loss = train_regret_network(
                    self.regret_net, self.regret_buffer,
                    epochs=self.train_epochs,
                    batch_size=self.train_batch,
                    lr=self.lr,
                )

            if callback and t % callback_freq == 0:
                callback(self, t)

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
        state       = self.encoder.encode(history, player)
        num_actions = len(self.game.legal_actions(history))
        with torch.no_grad():
            s    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = torch.zeros(self.max_actions).to(self.device)
            mask[:num_actions] = 1.0
            probs = self.strategy_net(s, mask).squeeze(0).cpu().numpy()
        return probs[:num_actions]