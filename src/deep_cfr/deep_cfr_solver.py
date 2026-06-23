"""
Deep CFR Solver.

Reference: Steinberger, E. (2019). "Single Deep Counterfactual
Regret Minimization." arXiv:1901.07621.

Buffer strategy (Brown et al. 2019 / Steinberger 2019):
    Regret (value) buffer (MR): RESERVOIR over many iterations.
        The value network is RE-FITTED from scratch each iteration on a
        reservoir drawn from ALL past iterations; that reservoir is how the
        network comes to approximate CUMULATIVE counterfactual regret without
        explicit summation. It must be LARGE (cover many iterations).
        A small FIFO 'window' keeps only the freshest samples, so the network
        fits only the latest iteration's INSTANTANEOUS regrets — that is not
        CFR and does not converge (verified: flat exploitability on Leduc).
    Strategy buffer (MΠ): RESERVOIR over the full history.
        Approximates the time-average strategy across all iterations, which is
        the quantity that converges to Nash. Reservoir is correct here too.

Weighting:
    Regret samples enter the buffer UNWEIGHTED (weight = 1.0). The value
    network fits the regret targets directly; iteration-weighting the regret
    targets has no basis in Deep CFR. Linear-CFR iteration weighting applies
    ONLY to the strategy buffer, which approximates the time-average strategy
    (Brown & Sandholm 2019).

Regret target (DEEPCFR_TARGET, default 'cfrplus'):
    CFR+-clipped cumulative regret R <- max(R + r^t, 0), normalised by the
    per-infoset visit count (R+/visits). This is the quantity tabular CFR+
    regret-matches on; it converges on Leduc where the instantaneous target
    oscillates. DEEPCFR_TARGET=instant restores the legacy instantaneous form
    for A/B comparison.
"""

import os as _os

import numpy as np
import torch
from dataclasses import dataclass

from typing import Optional

from ..games.base import ExtensiveFormGame, History
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
    buffer_capacity: int = 1_000_000
    strategy_buffer_capacity: int = 0     # 0 = same as buffer_capacity
    hidden_size: int = 256
    train_epochs: int = 100
    train_batch: int = 256
    lr: float = 1e-3
    traversals_per_iter: int = 100
    use_cpp_engine: bool = False
    device: str = "cpu"
    # Warm-start: kun True, regret-verkko EI alusteta nollista joka iteraatiolla.
    # Sen sijaan fitataan olemassaolevista painoista pienemmällä LR:llä
    # (lr / warm_start_lr_factor). Tämä antaa verkon hyödyntää edellisen
    # iteraation opittua regrettiä eikä aloita tyhjästä 500k-näytteen kanssa.
    # Brown et al. 2019 suosittelee cold-startia isoilla peleillä, mutta
    # postflop NLHE:n monimutkaisuudella warm-start konvergoi nopeammin
    # kun train_epochs on rajallinen (< 100).
    warm_start: bool = True
    warm_start_lr_factor: float = 5.0
    # DCFR temporal weighting exponent γ (Brown & Sandholm 2019).
    # 0.0 = vanilla Deep CFR (uniform reservoir).
    # 2.0 = DCFR (suositeltu regret-bufferille): näytteet painotetaan t^γ
    #       näytteistysvaiheessa jolloin myöhemmät iteraatiot dominoivat.
    # Strategy-bufferi käyttää Linear-CFR -painotusta (strat_w) loss-funktion
    # kautta eikä tarvitse dcfr_gamma-painotusta.
    dcfr_gamma: float = 2.0
    # Regret-kohde: 'instant' = per-traversaali hetkellinen regret (Brown et al.
    # 2019 Algorithm 1, oikea valinta jatkuvalle tilavektorille).
    # 'cfrplus' = CFR+/visits kumulatiivisena kohteena (toimii vain kun
    # visits(I)>>1, eli diskreetti infoset-avain kuten Leducissa).
    regret_target: str = "instant"
    # DCFR α/β-diskontaus (Brown & Sandholm 2019).
    # Iteraatiossa t: regret-näytteen paino = t^α / (t^α + 1) (α=1.5, β=0).
    # Tämä diskontaa varhaisten iteraatioiden kohinaisia regrettejä
    # ilman että ne poistetaan kokonaan reservoirista.
    # 0.0 = ei diskontausta (vanilla Deep CFR).
    dcfr_alpha: float = 1.5
    dcfr_beta:  float = 0.0
    # Pluribuksen empiirinen havainto: Linear CFR (α=1.0) auttaa varhaisissa
    # iteraatioissa mutta hyöty katoaa myöhemmin — jopa hidastaa konvergenssia
    # myöhäisillä iteraatioilla. linear_cfr_iters > 0 → käytä α=1.0 ensimmäiset
    # N iteraatiota, sitten vaihda self.dcfr_alpha:han.
    # 0 = pois päältä (vanha käytös: dcfr_alpha kaikilla iteraatioilla).
    linear_cfr_iters: int = 0
    # Pluribus-style dynamic pruning. C++ MCCFR -enginelle: actionit, joiden
    # blueprint-prob < prune_threshold, jätetään traversoimatta ja niistä ei
    # päivitetä regretiä. Säästää compute → enemmän iteraatioita samalla
    # budjetilla. Aktivoituu vasta iter >= prune_after_iter (alkuiteraatiot
    # ovat liian uniformeja prune:lle). 0.0 = pois.
    prune_threshold:  float = 0.0
    prune_after_iter: int   = 0
    # Position-bit ablation (v15). True = standard SB/BB signal in last dim
    # (kept for back-compat with v12-v14). False = constant 0.0 in slot →
    # tests whether the position signal was contributing to the v14 regression.
    include_position_bit: bool = True
    # Iteraation jälkeen lr_decay_start regret-LR kerrotaan lr_decay_factor:lla
    # joka iteraatio. 1.0 = ei decay:tä. Auttaa kun buffer saturoi ja kohde
    # muuttuu hitaammin — pienempi LR vähentää kohinaa myöhäisissä iteraatioissa.
    lr_decay_start:  int   = 100
    lr_decay_factor: float = 1.0
    # Seed C++ MCCFR -enginelle (samplaus-RNG). Multi-run baseline:lla eri
    # seed tuottaa eri näytteet samoilla hyperparametreilla → eri blueprint.
    seed: int = 42
    # Strategy-verkon fine-tune: pääajon jälkeen ekstra-epochit pienemmällä
    # LR:llä. 0 epochs = pois päältä (vanha käytös). Käyttää strategy_bufferia.
    finetune_epochs: int   = 0
    finetune_lr:     float = 1e-4
    # Auxiliary EV-prediction loss: when > 0 AND the encoder has a CFR
    # cache attached (state_size 49 with advisor dims at [37:49]), the
    # regret network gets a small EV-prediction head trained on the
    # cache's per-action EVs (slots [43:49]). The shared trunk learns to
    # encode advisor-derived value information, which the main regret
    # head then leverages. 0.0 = disabled (default; v4 behaviour).
    # Suggested starting value: 0.1.
    aux_ev_weight: float = 0.0
    # Teacher-student distillation: when ``teacher_net`` (a StrategyNetwork or
    # compatible callable, e.g. a Blueprint's underlying net) is provided
    # AND teacher_kl_weight > 0, the final strategy-network training adds
    # KL(teacher || student) to the cross-entropy loss. Used for self-
    # distillation from a previous production blueprint to reduce seed
    # variance and inherit calibrated mixing.
    teacher_net = None
    teacher_kl_weight: float = 0.0
    # Counterfactual hand-bucket augmentation (Path A). With probability
    # ``aug_bucket_prob`` per sample, the hand-bucket one-hot is shifted
    # by ±aug_bucket_radius and the cache is re-queried to refresh the
    # advisor dims. Forces the regret network to be smooth across the
    # equity-bucket abstraction. 0 = disabled (default).
    aug_bucket_prob:   float = 0.0
    aug_bucket_radius: int   = 1
    # Predictive CFR+ (Brown 2020) momentum coefficient. > 0 enables the
    # accelerated update rule on the C++ CFR+ accumulator. 0.0 = vanilla
    # CFR+ (default; safe fallback). Only meaningful when regret_target
    # == 'cfrplus' (the C++ engine's default with cache).
    predictive_alpha: float = 0.0
    # Value head (DeepStack / ReBeL style): scalar V(s) prediction trained
    # to match Σ_a probs[a] * EVs[a] from the cache. When > 0, the
    # RegretNetwork gains a value_head and the loss adds value_head_weight
    # × MSE(V_pred, V_target). Forces the trunk to encode state-level
    # expected value as a separate signal alongside per-action regrets.
    value_head_weight: float = 0.0
    # BRD (best-response-defense) sample weighting: when an exploit-gap
    # map is provided, each regret-buffer sample's loss weight is multi-
    # plied by (1 + lambda * gap/median). High-exploitation spots get
    # amplified gradient → blueprint preferentially patches its biggest
    # leaks. ``exploit_gap_map`` = dict[uint64 key → gap_mbb]; mined by
    # scripts/mine_exploit_gaps.py. lambda=0 disables.
    exploit_gap_map: dict | None = None
    exploit_gap_lambda: float    = 0.0

    def __post_init__(self):
        # Buffer / network width must match what the SAMPLE EMITTER
        # produces. The C++ NLHEStateEncoder always emits 49-dim vectors
        # (advisor slots zero when no cache). The Python encoder returns
        # 37 or 49 depending on cache attachment. On C++ path, pin to 49
        # so buffer + network + emitter all agree (advisor stays zero
        # when no cache; behaviour matches legacy 37-dim training because
        # the network's input slots [37:49] all see constant zero and
        # collapse the corresponding weights to a constant bias term).
        if self.use_cpp_engine:
            try:
                from .cpp_backend import NLHECppBackend as _CppBp
                state_sz = _CppBp.STATE_SIZE
            except Exception:
                state_sz = self.encoder.state_size()
        else:
            state_sz = self.encoder.state_size()
        strat_cap = self.strategy_buffer_capacity or self.buffer_capacity

        # Aux EV head is meaningful only when the encoder has 49 dims
        # (BASE_STATE_SIZE + 12 advisor slots). Otherwise the EV slots
        # don't exist in the state vector. The value head shares the
        # same precondition (its target is derived from the same advisor
        # slots).
        ev_head_dim = 6 if (self.aux_ev_weight > 0 and state_sz == 49) else 0
        use_value   = self.value_head_weight > 0 and state_sz == 49
        self.regret_net   = RegretNetwork(
            state_sz, self.max_actions, self.hidden_size,
            ev_head_dim=ev_head_dim,
            value_head=use_value,
        ).to(self.device)
        self.strategy_net = StrategyNetwork(state_sz, self.max_actions, self.hidden_size).to(self.device)

        # Both buffers are reservoirs (see module docstring). The regret buffer
        # must be LARGE: it is the implicit cumulative-regret estimator, not a
        # fresh-data window. The strategy buffer is the time-average estimator.
        self._current_iter   = 0
        self.regret_buffer   = ReservoirBuffer(
            self.buffer_capacity, state_sz, self.max_actions,
            mode='reservoir', dcfr_gamma=self.dcfr_gamma,
        )
        # Strategy-bufferille ei DCFR-painotusta — Linear-CFR strat_w
        # hoitaa iteraatiopainotuksen loss-funktion kautta.
        self.strategy_buffer = ReservoirBuffer(strat_cap, state_sz, self.max_actions, mode='reservoir')

        # CFR+-clipped cumulative regret target (default). Keyed by info_set_key
        # so the same infoset accumulates across traversals. Buffer stores
        # R+/visits (bounded, sign-preserving; tabular scale ~[0.01, 0.06]).
        self._target_mode = _os.environ.get("DEEPCFR_TARGET", "cfrplus").strip().lower()
        self._cfrplus_regret = {}

        self.iterations = 0
        self._rng = np.random.default_rng(self.seed)

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
                        raise_fractions=self.game.raise_fractions,
                        max_actions=self.max_actions,
                        max_raises=self.game.max_raises_per_street,
                        regret_target=self.regret_target,
                        seed=self.seed,
                        prune_threshold=self.prune_threshold,
                        prune_after_iter=self.prune_after_iter,
                        include_position_bit=self.include_position_bit,
                        predictive_alpha=self.predictive_alpha,
                    )
                    # When the encoder carries a CFR advisor cache, wire it
                    # into cpp_backend.to_tensors so advisor dims get
                    # backfilled into each sample (Vaihtoehto 1 design).
                    enc_cache = getattr(self.encoder, "cfr_cache", None)
                    if enc_cache is not None and hasattr(self._cpp, "set_cache_context"):
                        self._cpp.set_cache_context(enc_cache, self.encoder)
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
                # the correct sign; R+/visits keeps the target at the tabular
                # scale (~[0.01,0.06]) regardless of per-infoset visit density.
                key = self.game.info_set_key(history, player)
                entry = self._cfrplus_regret.get(key)
                if entry is None:
                    entry = [np.zeros(self.max_actions, dtype=np.float64), 0]
                R, n = entry
                R[:num_actions] = np.maximum(R[:num_actions] + regrets, 0.0)
                n += 1
                self._cfrplus_regret[key] = [R, n]
                target = (R[:num_actions] / float(n)).astype(np.float32)

            self.regret_buffer.add(state, target, 1.0, iteration=self.iterations + 1)
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
        row each produces targets with a single non-zero slot and artificial
        zeros elsewhere — so the network is trained to map the SAME input to
        several conflicting near-one-hot targets at once, which prevents the
        regret network from learning a coherent per-action regret vector.

        Fix: group rows by their exact state vector and sum each action's value
        into the correct slot, yielding one row per state with the complete
        [r0, r1, r2, r3] target. Both sides quantise the continuous feature dims
        to a fixed 1e-6 grid (encoder + C++ state_key), so exact grouping is
        robust to float noise and the two implementations group identically.

        Returns (states_unique [m, S], targets [m, A]).
        """
        n = len(X_np)
        if n == 0:
            return X_np, np.zeros((0, self.max_actions), dtype=np.float32)

        uniq_states, inverse = np.unique(X_np, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)   # 1D across numpy versions

        targets = np.zeros((uniq_states.shape[0], self.max_actions), dtype=np.float32)
        np.add.at(targets, (inverse, a_np), v_np)
        return uniq_states.astype(np.float32), targets

    def _dcfr_regret_weight(self, t: int) -> float:
        """DCFR α/β-paino iteraatiolle t (Brown & Sandholm 2019).

        regret_weight = t^α / (t^α + 1)   kun α > 0
        Lähestyy 1.0:aa kun t → ∞, joten myöhemmät (tarkemmat) iteraatiot
        dominoivat näytteistystä. Varhaiset iteraatiot diskontautuvat mutta
        eivät poistu kokonaan reservoirista.
        dcfr_beta ei vaikuta regret-painoon (vaikuttaa strategy-painoon
        paperissa, mutta Linear-CFR strat_w hoitaa sen jo).

        Jos linear_cfr_iters > 0 ja t < linear_cfr_iters, käytetään α=1.0
        (puhdas Linear CFR). Tämän jälkeen vaihdetaan self.dcfr_alpha:han.
        """
        if self.linear_cfr_iters > 0 and t < self.linear_cfr_iters:
            alpha = 1.0
        else:
            alpha = self.dcfr_alpha
        if alpha <= 0.0:
            return 1.0
        ta = float(t + 1) ** alpha
        return ta / (ta + 1.0)

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
                # INSTANT-mode: normalisoi regretit 2*stack:lla jotta ne ovat
                # samassa skaalassa kuin CFR+/visits (~[-1,1]).
                # INSTANT-regretit ovat raakoja pot-skaalattuja arvoja (action_ev
                # - node_ev), jotka voivat olla satoja BB:tä — ilman normalisointia
                # loss räjähtää (3-4 vs 0.04) ja verkko ei konvergoi.
                if self.regret_target == "instant":
                    norm = 2.0 * getattr(self.game, 'starting_stack', 200.0)
                    reg_mat = reg_mat / (norm + 1e-8)
                # DCFR α-paino: t^α/(t^α+1) diskontaa varhaisten iteraatioiden
                # kohinaisia regrettejä. γ-painotus hoitaa näytteistysvaiheen.
                reg_w = self._dcfr_regret_weight(self.iterations)
                self.regret_buffer.add_batch(
                    states, reg_mat,
                    np.full(m, reg_w, dtype=np.float32),
                    iteration=self.iterations + 1,
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
                    states, str_mat, np.full(m, strat_w, dtype=np.float32),
                    iteration=self.iterations + 1,
                )

    def current_strategy_blueprint(self):
        """
        Wrap the CURRENT regret-matching strategy as a query interface
        compatible with estimate_exploitability().

        This is the strategy CFR actually iterates (positive-regret matching on
        the regret network), and it updates every iteration — unlike
        strategy_net, which is trained only at the end. Use this for
        mid-training convergence measurement.
        """
        solver = self

        class _CurrentStrategy:
            class _Meta:
                def __init__(self, state_size, action_size):
                    self.state_size  = state_size
                    self.action_size = action_size
            def __init__(self):
                self.metadata = self._Meta(
                    solver.encoder.state_size(),
                    solver.max_actions,
                )
            def query(self, state_vec, num_actions):
                return solver._get_regret_strategy(
                    np.asarray(state_vec, dtype=np.float32), num_actions
                )
            def query_by_slots(self, state_vec, slot_indices):
                # Read regret-matched probabilities at specific network slots,
                # renormalised. Mirrors Blueprint.query_by_slots so the LBR
                # estimator can use the same slot-mapping path on the
                # mid-training current strategy.
                state = np.asarray(state_vec, dtype=np.float32)
                n_legal = len(slot_indices)
                # _get_regret_strategy expects "num_actions" = the position
                # of the highest legal slot + 1, so the regret network
                # output is sliced wide enough to cover all requested slots.
                max_slot = max(slot_indices) if slot_indices else 0
                full = solver._get_regret_strategy(state, max_slot + 1)
                # Reindex by slot, renormalise over the legal subset.
                vals = np.asarray(
                    [full[s] if s < len(full) else 0.0 for s in slot_indices],
                    dtype=np.float64,
                )
                v = vals.sum()
                if v > 1e-9:
                    return vals / v
                return np.ones(n_legal, dtype=np.float64) / max(n_legal, 1)

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
            # logic sees the true current iteration, not a constant.
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
                if self.use_cpp_engine:
                    try:
                        from .cpp_backend import NLHECppBackend as _CppBp
                        state_sz = _CppBp.STATE_SIZE
                    except Exception:
                        state_sz = self.encoder.state_size()
                else:
                    state_sz = self.encoder.state_size()
                net_device = next(self.regret_net.parameters()).device
                # INSTANT-mode: aina cold-start (Brown et al. 2019 Algorithm 1).
                # Warm-start on haitallinen INSTANT-moden kanssa koska kohdejakauma
                # muuttuu merkittävästi iteraatioiden välillä — Adam-momentum
                # ajautuu väärään suuntaan ja loss kasvaa. CFR+-mode hyötyy
                # warm-startista koska kumulatiivinen kohde muuttuu hitaasti.
                use_warm = self.warm_start and self.iterations > 1 \
                           and self.regret_target != "instant"
                if use_warm:
                    # Warm-start: fitataan olemassaolevista painoista
                    # pienemmällä LR:llä. Verkko hyödyntää edellisen
                    # iteraation opittua regrettiä — ei aloita tyhjästä
                    # 500k-näytteen kanssa per iteraatio.
                    train_lr = self.lr / self.warm_start_lr_factor
                else:
                    # Cold-start: alustetaan nollista Brown et al. 2019
                    # -suosituksen mukaisesti.
                    ev_head_dim = 6 if (self.aux_ev_weight > 0 and state_sz == 49) else 0
                    use_value_cs = self.value_head_weight > 0 and state_sz == 49
                    self.regret_net = RegretNetwork(
                        state_sz, self.max_actions, self.hidden_size,
                        ev_head_dim=ev_head_dim,
                        value_head=use_value_cs,
                    ).to(net_device)
                    train_lr = self.lr
                # LR decay buffer-saturaation jälkeen: vähentää myöhäisten
                # iteraatioiden kohinaa kun kohde muuttuu hitaammin.
                if self.lr_decay_factor < 1.0 and self.iterations > self.lr_decay_start:
                    decay_steps = self.iterations - self.lr_decay_start
                    train_lr *= self.lr_decay_factor ** decay_steps
                self._last_regret_loss = train_regret_network(
                    self.regret_net, self.regret_buffer,
                    epochs=self.train_epochs,
                    batch_size=self.train_batch,
                    lr=train_lr,
                    aux_ev_weight=self.aux_ev_weight,
                    exploit_gap_map=self.exploit_gap_map,
                    exploit_gap_lambda=self.exploit_gap_lambda,
                    exploit_gap_encoder=(self.encoder
                                          if self.exploit_gap_map is not None
                                          else None),
                    aug_bucket_prob=self.aug_bucket_prob,
                    aug_bucket_radius=self.aug_bucket_radius,
                    aug_cache=(self.encoder.cfr_cache
                                if self.aug_bucket_prob > 0
                                   and getattr(self.encoder, "cfr_cache", None) is not None
                                else None),
                    aug_encoder=(self.encoder
                                  if self.aug_bucket_prob > 0
                                  else None),
                    value_head_weight=self.value_head_weight,
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
                teacher_net=self.teacher_net,
                teacher_kl_weight=self.teacher_kl_weight,
            )
            # Fine-tune phase: ekstra-epochit pienemmällä LR:llä.
            # Painottuu strategy-bufferin Linear-CFR-painojen kautta
            # automaattisesti myöhäisiin iteraatioihin.
            if self.finetune_epochs > 0:
                print(f"Fine-tuning strategy net "
                      f"({self.finetune_epochs} epochs @ lr={self.finetune_lr})")
                train_strategy_network(
                    self.strategy_net, self.strategy_buffer,
                    epochs=self.finetune_epochs,
                    batch_size=self.train_batch,
                    lr=self.finetune_lr,
                    teacher_net=self.teacher_net,
                    teacher_kl_weight=self.teacher_kl_weight,
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