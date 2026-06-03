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

Weighting (corrected):
    Regret samples enter the buffer UNWEIGHTED (weight = 1.0). The regret
    network must fit the regrets of the CURRENT strategy; iteration-weighting
    regrets has no theoretical basis in Deep CFR and, combined with a
    non-rotating window buffer, lets stale data dominate the gradient.
    Linear-CFR iteration weighting applies ONLY to the strategy buffer,
    which approximates the time-average strategy (Brown & Sandholm 2019).
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
        #   NOTE: window mode only helps if capacity is SMALL relative to total
        #   samples produced over the run. If capacity > total samples, nothing
        #   is ever evicted and the window degenerates into an unbounded
        #   accumulating buffer. Follow the repo guideline:
        #       buffer ≈ 10 × traversals_per_iter × 2
        #   e.g. 500 traversals → ~10_000. A 2_000_000 capacity defeats the
        #   purpose entirely.
        # Regret (value) buffer: mode='reservoir' (Deep CFR / SD-CFR, Brown 2019
        #   & Steinberger 2019). The value network is RE-FITTED each iteration on
        #   a reservoir over samples from ALL iterations; that reservoir is how
        #   the network comes to approximate CUMULATIVE counterfactual regret
        #   without any explicit summation. A 'window' buffer keeps only the
        #   freshest samples, so the network fits only the latest iteration's
        #   INSTANTANEOUS regrets — that is not CFR and does not converge
        #   (verified: flat exploitability on Leduc). Reservoir is required, and
        #   it should be LARGE (cover many iterations), not the small
        #   ~10×traversals window size used previously.
        # Strategy buffer: mode='reservoir' (full history average).
        #   Approximates time-average strategy over all iterations — correct.
        self._current_iter   = 0
        self.regret_buffer   = ReservoirBuffer(self.buffer_capacity, state_sz, self.max_actions, mode='reservoir')
        self.strategy_buffer = ReservoirBuffer(strat_cap, state_sz, self.max_actions, mode='reservoir')

        # CFR+-clipped cumulative regret target (DEEPCFR_TARGET=cfrplus, default).
        # Per-infoset R <- max(R + r^t, 0): the quantity tabular CFR+ regret-
        # matches on, which CONVERGES on Leduc where the instantaneous target
        # oscillates. Keyed by info_set_key (string). Buffer stores R+/t
        # (bounded, sign-preserving; measured tabular scale ~[0.01,0.06]).
        # DEEPCFR_TARGET=instant restores the previous instantaneous target.
        import os as _os
        self._target_mode = _os.environ.get("DEEPCFR_TARGET", "cfrplus").strip().lower()
        self._cfrplus_regret = {}

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
        # Read the network's ACTUAL device from its parameters rather than
        # trusting self.device. export_for_libtorch and other helpers can leave
        # the regret network on a different device than self.device records,
        # which caused a cpu/mps tensor mismatch during exploitability rollouts.
        net_device = next(self.regret_net.parameters()).device
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(net_device)
            regrets = self.regret_net(s).squeeze(0).cpu().numpy()[:num_actions]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        # Deep CFR (Brown et al. 2019, Fig. 4): when all regrets are <= 0, play
        # the SINGLE highest-regret action with probability 1, NOT a uniform
        # distribution. Uniform fallback raises final exploitability ~50%.
        strat = np.zeros(num_actions, dtype=np.float64)
        strat[int(np.argmax(regrets))] = 1.0
        return strat

    def _traverse(self, history: History, traversing_player: int) -> float:
        if self.game.is_terminal(history):
            return self.game.terminal_payoffs(history)[traversing_player]

        player      = self.game.current_player(history)
        actions     = self.game.legal_actions(history)
        num_actions = len(actions)
        state       = self.encoder.encode(history, player)
        strategy    = self._get_regret_strategy(state, num_actions)

        if player == traversing_player:
            # Strategy buffer: iteration-weighted (Linear CFR — time-average).
            self.strategy_buffer.add(state, strategy, float(self.iterations + 1))
            action_values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                new_h = self.game.apply_action(history, action)
                action_values[i] = self._traverse(new_h, traversing_player)
            node_value = (strategy * action_values).sum()
            regrets = action_values - node_value

            if self._target_mode == "instant":
                target = regrets
            else:
                # CFR+-clipped cumulative regret, keyed by info_set_key so the
                # same infoset accumulates across traversals. max(R+r^t,0) gives
                # the correct sign (the instantaneous mean had the wrong sign:
                # measured K[-1.1,+0.05] vs tabular cum [+10.8,+60.1]). R+/t keeps
                # the regression target at the network-fittable scale measured
                # from tabular CFR+ (~[0.01,0.06] for K), preserving sign/ratio.
                key = self.game.info_set_key(history, player)
                entry = self._cfrplus_regret.get(key)
                if entry is None:
                    entry = [np.zeros(self.max_actions, dtype=np.float64), 0]
                R, n = entry
                R[:num_actions] = np.maximum(R[:num_actions] + regrets, 0.0)
                n += 1
                self._cfrplus_regret[key] = [R, n]
                # Divide by PER-INFOSET visit count n (not iteration t): R+ and n
                # grow at the same rate, so R+/n is a true mean regret that lands
                # at the tabular scale ~[0.01,0.06] regardless of how often this
                # infoset is visited per iteration. A global t-power cannot do
                # this because visit density varies across infosets.
                target = (R[:num_actions] / float(n)).astype(np.float32)

            self.regret_buffer.add(state, target, 1.0)
            return node_value
        else:
            action_idx = self._rng.choice(num_actions, p=strategy)
            new_h = self.game.apply_action(history, actions[action_idx])
            return self._traverse(new_h, traversing_player)

    def _collapse_by_state(self, X_np, a_np, v_np):
        """
        Collapse per-(state, action) samples into one row per unique state
        with a FULL action-vector target.

        The C++ engine emits one sample per (info-set, action): each carries
        an identical 124-dim state vector but a single action index + that
        action's regret (or strategy probability). Scattering these onto one
        row each (np.arange(n)) produces targets with a single non-zero slot
        and artificial zeros elsewhere — so the network is trained to map the
        SAME input to several conflicting one-hot-ish targets at once. That
        prevents the regret network from ever learning a coherent per-action
        regret vector, which manifests as non-converging / degenerate play.

        Fix: group rows by their exact state vector (the C++ encoder produces
        bit-identical floats for the same node, so exact grouping is safe) and
        sum each action's value into the correct slot, yielding one row per
        state with the complete [r0, r1, r2, r3] target.

        Returns (states_unique [m, S], targets [m, A]).
        """
        n = len(X_np)
        if n == 0:
            return X_np, np.zeros((0, self.max_actions), dtype=np.float32)

        # Group by exact state vector. np.unique on rows gives an inverse index
        # mapping each original row to its unique-state group.
        uniq_states, inverse = np.unique(X_np, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)   # 1D across numpy versions
        m = uniq_states.shape[0]

        targets = np.zeros((m, self.max_actions), dtype=np.float32)
        # Scatter each sample's value into (its group row, its action slot).
        np.add.at(targets, (inverse, a_np), v_np)
        return uniq_states.astype(np.float32), targets

    def _run_cpp_iteration(self) -> None:
        reg_exp, str_exp = self._cpp.run_iteration(
            self.iterations,
            regret_net=self.regret_net if self.iterations > 0 else None,
        )
        # Strategy weight: Linear-CFR iteration weight (time-average strategy).
        strat_w = float(self.iterations + 1)

        if len(reg_exp) > 0:
            X, actions, values = self._cpp.to_tensors(reg_exp)
            X_np = X.cpu().numpy()
            a_np = actions.cpu().numpy()
            v_np = values.cpu().numpy().astype(np.float32)
            mask = a_np < self.max_actions
            X_np, a_np, v_np = X_np[mask], a_np[mask], v_np[mask]
            if len(X_np) > 0:
                states, reg_mat = self._collapse_by_state(X_np, a_np, v_np)
                m = len(states)
                # Regrets UNWEIGHTED — fit the current strategy only.
                self.regret_buffer.add_batch(
                    states, reg_mat, np.ones(m, dtype=np.float32)
                )

        if len(str_exp) > 0:
            X, actions, values = self._cpp.to_tensors(str_exp)
            X_np = X.cpu().numpy()
            a_np = actions.cpu().numpy()
            v_np = values.cpu().numpy().astype(np.float32)
            mask = a_np < self.max_actions
            X_np, a_np, v_np = X_np[mask], a_np[mask], v_np[mask]
            if len(X_np) > 0:
                states, str_mat = self._collapse_by_state(X_np, a_np, v_np)
                m = len(states)
                self.strategy_buffer.add_batch(
                    states, str_mat, np.full(m, strat_w, dtype=np.float32)
                )

    def current_strategy_blueprint(self):
        """
        Wrap the CURRENT regret-matching strategy as a query interface
        compatible with estimate_exploitability().

        This is the strategy CFR actually iterates (positive-regret
        matching on the regret network), and it updates every iteration —
        unlike strategy_net, which is trained only once at the end.
        Use this for mid-training convergence measurement.
        """
        solver = self

        class _CurrentStrategy:
            class _Meta:
                def __init__(self, state_size):
                    self.state_size = state_size
            def __init__(self):
                self.metadata = self._Meta(solver.encoder.state_size())
            def query(self, state_vec, num_actions):
                return solver._get_regret_strategy(
                    np.asarray(state_vec, dtype=np.float32), num_actions
                )

        return _CurrentStrategy()

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
            # Keep _current_iter in sync with the loop so any iteration-weighted
            # logic (strategy buffer) sees the true current iteration, not a
            # constant. Previously _current_iter stayed at 0 for the whole run.
            self._current_iter = self.iterations

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
                # Deep CFR (Brown et al. 2019): the value/regret network is
                # trained FROM SCRATCH each iteration, starting from a random
                # initialization. Fine-tuning from the previous iteration's
                # weights raises final exploitability ~50% (their Fig. 4) and,
                # combined with a changing target, causes the drift we observed
                # (flat/oscillating exploitability on Leduc). Re-initialise here.
                state_sz = self.encoder.state_size()
                net_device = next(self.regret_net.parameters()).device
                self.regret_net = RegretNetwork(
                    state_sz, self.max_actions, self.hidden_size
                ).to(net_device)
                # Fresh Adam optimizer each iteration (created inside trainer).
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