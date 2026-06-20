"""
Subgame Solver — re-solving a PostflopNLHE subtree at runtime.

Two solvers with a shared game abstraction:

    UnsafeSubgameSolver (Burch et al., 2014)
        Re-solve a subgame subtree using tabular CFR. Terminal values
        are exact hand evaluations — no blueprint needed at leaf nodes.
        'Unsafe' means there is no guarantee the full-game exploitability
        doesn't increase vs the blueprint; in practice it improves play.

    SafeSubgameSolver (Moravčík et al., 2016)
        Extends unsafe solving with the gadget-game trick: the opponent
        may 'opt out' of the subgame and take their blueprint expected
        value at every information set. The Nash equilibrium of the
        gadget game is provably at least as non-exploitable as the
        blueprint (Theorem 1, Burch et al. 2014).

Architecture:

    SubgameGame(ExtensiveFormGame)
        Wraps PostflopNLHE. Generates initial_histories() from the
        cross-product of hero_range × opponent_range, filtered for
        card conflicts, weighted by joint probability. Info set keys
        are LOCAL to the subgame — the root-prefix actions are stripped
        so that identical subgame trajectories share an info set even
        when reached via different preflop lines.

    GadgetGame(SubgameGame)
        Adds a terminal 'opt-out' action for the opponent at the
        subgame root. The terminal payoff = blueprint EV for the
        opponent's specific hole cards at that root.

    SubgameStrategy
        Query interface over the solved strategy dictionary. Falls back
        to uniform when an info set was not visited.

Usage:
    game = PostflopNLHE(...)
    encoder = NLHEEncoder(...)

    # Unsafe
    solver = UnsafeSubgameSolver(game)
    strategy = solver.solve(
        root_history=...,
        hero_player=0,
        hero_range={...},
        opponent_range={...},
        iterations=200,
    )
    probs = strategy.query(current_history, player=0)

    # Safe
    safe_solver = SafeSubgameSolver(game, blueprint=bp, encoder=encoder)
    strategy = safe_solver.solve(
        root_history=...,
        hero_player=0,
        hero_range={...},
        opponent_range={...},
        iterations=200,
    )
"""

from __future__ import annotations

import numpy as np

from ..games.base import ExtensiveFormGame, History, Action, InfoSetKey
from ..games.postflop_nlhe import PostflopNLHE
from ..solvers.cfr import CFRSolver


# ── SubgameGame ───────────────────────────────────────────────────────────────

class SubgameGame(ExtensiveFormGame):
    """
    PostflopNLHE restricted to a subgame rooted at root_history.

    initial_histories():
        Cross-product of hero_range × opponent_range, filtered for
        card conflicts with each other and with the board.
        Each (hero_cards, opp_cards) pair is one 'deal' in the
        subgame — the board is taken from root_history (pre-dealt).

    info_set_key():
        (my_cards, visible_board, subgame_local_actions)
        Strips root-prefix actions so that the CFR solver builds
        a strategy over the subgame action space, not the full game.

    All other methods delegate to base_game (PostflopNLHE).
    """

    def __init__(
        self,
        base_game: PostflopNLHE,
        root_history: History,
        hero_player: int,
        hero_range: dict[tuple[int, int], float],
        opponent_range: dict[tuple[int, int], float],
        max_deals: int = 300,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            base_game:        PostflopNLHE instance.
            root_history:     Game state at the subgame entry point.
                              history[2] is the full pre-dealt board.
            hero_player:      Player performing the re-solve (0 or 1).
            hero_range:       {hole_cards: probability} for the hero.
            opponent_range:   {hole_cards: probability} for the opponent.
            max_deals:        Cap on initial_histories() for tractability.
            rng:              RNG for deal sub-sampling when > max_deals.
        """
        self.base_game     = base_game
        self.root_history  = root_history
        self.hero_player   = hero_player
        self._hero_range   = hero_range
        self._opp_range    = opponent_range
        self._max_deals    = max_deals
        self._rng          = rng or np.random.default_rng(42)

        # Number of actions already played before the subgame root.
        # root_history = (p0_cards, p1_cards, board, *prefix_actions)
        # so prefix_len = len(root_history) - 3.
        self._prefix_len: int = len(root_history) - 3

        # Precompute (history, prob) pairs — done once on construction.
        self._initial: list[tuple[History, float]] = self._build_initial()

    # ── ExtensiveFormGame interface ───────────────────────────────────────────

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        return self._initial

    def is_terminal(self, history: History) -> bool:
        return self.base_game.is_terminal(history)

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        return self.base_game.terminal_payoffs(history)

    def current_player(self, history: History) -> int:
        return self.base_game.current_player(history)

    def legal_actions(self, history: History) -> list[Action]:
        return self.base_game.legal_actions(history)

    def apply_action(self, history: History, action: Action) -> History:
        return self.base_game.apply_action(history, action)

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Local info set key: strips root-prefix so the subgame CFR
        operates on a fresh strategy table.

        Key = (my_cards, visible_board_as_tuple, subgame_local_actions)
        """
        my_cards      = history[0] if player == 0 else history[1]
        visible_board = self.base_game._visible_board(history)

        # Only actions AFTER the root prefix belong to the subgame.
        subgame_actions = history[3:][self._prefix_len:]

        return str((my_cards, visible_board, subgame_actions))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_initial(self) -> list[tuple[History, float]]:
        """
        Enumerate (history, prob) pairs from hero_range × opp_range.
        Filters card conflicts; normalises; sub-samples to max_deals.
        """
        board_set = set(self.root_history[2])
        result: list[tuple[History, float]] = []

        for h_cards, h_prob in self._hero_range.items():
            if h_prob <= 0 or (set(h_cards) & board_set):
                continue
            for o_cards, o_prob in self._opp_range.items():
                if o_prob <= 0:
                    continue
                if set(o_cards) & board_set:
                    continue
                if set(h_cards) & set(o_cards):       # hole card conflict
                    continue

                p0 = h_cards if self.hero_player == 0 else o_cards
                p1 = o_cards if self.hero_player == 0 else h_cards

                # Build deal: substitute cards into root_history.
                deal_history = (p0, p1) + self.root_history[2:]
                result.append((deal_history, h_prob * o_prob))

        if not result:
            return result

        # Normalise
        total = sum(p for _, p in result)
        result = [(h, p / total) for h, p in result]

        # Sub-sample if over max_deals
        if len(result) > self._max_deals:
            probs  = np.array([p for _, p in result])
            probs /= probs.sum()
            idx    = self._rng.choice(
                len(result), size=self._max_deals, replace=False, p=probs
            )
            result = [result[i] for i in idx]
            total  = sum(p for _, p in result)
            result = [(h, p / total) for h, p in result]

        return result


# ── GadgetGame ────────────────────────────────────────────────────────────────

_OPT_OUT = "__opt_out__"   # sentinel action string


class GadgetGame(SubgameGame):
    """
    Gadget game for safe subgame solving (Moravčík et al., 2016).

    At the subgame root (before any subgame action), the OPPONENT
    has an additional 'opt-out' action whose terminal payoff equals
    their blueprint expected value for those specific hole cards.

    Solving this game to Nash equilibrium yields a hero strategy
    that is provably at least as non-exploitable as the blueprint
    (Theorem 1, Burch et al. 2014).

    The opt-out payoffs are pre-computed via blueprint rollouts;
    see estimate_blueprint_ev().
    """

    def __init__(
        self,
        blueprint_ev_by_opp_cards: dict[tuple[int, int], float],
        **kwargs,
    ):
        """
        Args:
            blueprint_ev_by_opp_cards:
                {opp_hole_cards: expected_payoff_FOR_OPPONENT}
                computed by estimate_blueprint_ev() before constructing
                this game.
            **kwargs: forwarded to SubgameGame.__init__().
        """
        super().__init__(**kwargs)
        self._bp_ev = blueprint_ev_by_opp_cards
        self._opp_player = 1 - self.hero_player

    # ── Override three methods to inject the gadget ───────────────────────────

    def is_terminal(self, history: History) -> bool:
        if len(history) > 3 and history[-1] == _OPT_OUT:
            return True
        return super().is_terminal(history)

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        if len(history) > 3 and history[-1] == _OPT_OUT:
            opp_cards = (
                history[1] if self.hero_player == 0 else history[0]
            )
            opp_ev = self._bp_ev.get(opp_cards, 0.0)
            # Zero-sum: hero gets -(opp_ev)
            if self.hero_player == 0:
                return (-opp_ev, opp_ev)
            else:
                return (opp_ev, -opp_ev)
        return super().terminal_payoffs(history)

    def legal_actions(self, history: History) -> list[Action]:
        subgame_actions = history[3 + self._prefix_len :]
        # Opt-out is available to the opponent ONLY at the subgame root.
        if (
            len(subgame_actions) == 0
            and self.base_game.current_player(history) == self._opp_player
        ):
            return [_OPT_OUT] + super().legal_actions(history)
        return super().legal_actions(history)


# ── SubgameStrategy ───────────────────────────────────────────────────────────

class SubgameStrategy:
    """
    Query interface over a solved subgame strategy dictionary.

    Wraps the {InfoSetKey → average_strategy} dict produced by CFRSolver.
    Falls back to a uniform distribution for unseen info sets.
    """

    def __init__(
        self,
        strategy_dict: dict[InfoSetKey, np.ndarray],
        game: SubgameGame,
    ):
        self._dict = strategy_dict
        self._game = game

    def query(self, history: History, player: int) -> np.ndarray:
        """
        Return action probabilities for (history, player).

        Args:
            history: Current game state (PostflopNLHE history tuple).
            player:  Acting player index (0 or 1).

        Returns:
            np.ndarray of shape [num_legal_actions], summing to 1.
        """
        key     = self._game.info_set_key(history, player)
        actions = self._game.legal_actions(history)
        n       = len(actions)

        if key in self._dict:
            return self._dict[key][:n]

        # Unseen info set — uniform fallback
        return np.ones(n) / n

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"SubgameStrategy({len(self._dict)} info sets)"


# ── Blueprint rollout helpers ─────────────────────────────────────────────────

def _rollout_expected_value(
    blueprint,
    encoder,
    game: PostflopNLHE,
    history: History,
    payoff_fn=None,
) -> tuple[float, float]:
    """
    Compute expected payoffs (p0_ev, p1_ev) for a specific deal by
    rolling out both players' blueprint strategies recursively.

    Recursive tree traversal — tractable for subgame subtrees
    (typically ≤ 5 betting rounds deep with ≤ 4 actions each).

    payoff_fn: optional callable(history, game) -> (p0, p1) used to compute
        terminal values instead of game.terminal_payoffs. Pass
        head_to_head.ev_adjusted_payoffs to integrate over all-in runouts
        instead of using the deal's predetermined cards — variance-reducing
        for noisy mid-training exploitability estimates.
    """
    if game.is_terminal(history):
        if payoff_fn is not None:
            payoffs = payoff_fn(history, game)
        else:
            payoffs = game.terminal_payoffs(history)
        return float(payoffs[0]), float(payoffs[1])

    player  = game.current_player(history)
    actions = game.legal_actions(history)

    state_vec = encoder.encode(history, player)
    # Slot-indexed query — correct for postflop no-bet (legal=['c','r','a']
    # → slots [0, 2, 3] not [0, 1, 2]) and single-raise ALL_IN remap.
    from ..deep_cfr.action_slots import legal_actions_to_slots
    slots     = legal_actions_to_slots(actions, blueprint.metadata.action_size)
    probs     = blueprint.query_by_slots(state_vec, slots)

    ev0, ev1 = 0.0, 0.0
    for prob, action in zip(probs, actions):
        if prob < 1e-9:
            continue
        next_h      = game.apply_action(history, action)
        next0, next1 = _rollout_expected_value(blueprint, encoder, game, next_h,
                                                payoff_fn=payoff_fn)
        ev0 += prob * next0
        ev1 += prob * next1

    return ev0, ev1


def estimate_blueprint_ev(
    blueprint,
    encoder,
    base_game: PostflopNLHE,
    root_history: History,
    hero_player: int,
    hero_range: dict[tuple[int, int], float],
    opponent_range: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Estimate the blueprint expected value for each opponent hand at
    the subgame root. Used to initialise GadgetGame terminal payoffs.

    For each opp_cards in opponent_range:
        EV = Σ_{hero_cards} P(hero_cards) × blueprint_payoff(opp_cards, hero_cards)

    Args:
        blueprint:        Trained Blueprint instance.
        encoder:          NLHEEncoder matching blueprint's state_size.
        base_game:        PostflopNLHE game object.
        root_history:     Subgame root (contains pre-dealt board).
        hero_player:      Index of the re-solving player.
        hero_range:       {hero_cards: probability}.
        opponent_range:   {opp_cards: probability}.

    Returns:
        {opp_cards: expected_payoff_for_opponent_at_subgame_root}
    """
    opp_player  = 1 - hero_player
    board_set   = set(root_history[2])
    ev_by_opp: dict[tuple[int, int], float] = {}

    for opp_cards, opp_prob in opponent_range.items():
        if opp_prob <= 0 or (set(opp_cards) & board_set):
            continue

        total_ev     = 0.0
        total_weight = 0.0

        for hero_cards, hero_prob in hero_range.items():
            if hero_prob <= 0 or (set(hero_cards) & board_set):
                continue
            if set(hero_cards) & set(opp_cards):    # conflict
                continue

            p0 = hero_cards if hero_player == 0 else opp_cards
            p1 = opp_cards  if hero_player == 0 else hero_cards

            deal_history = (p0, p1) + root_history[2:]
            ev0, ev1     = _rollout_expected_value(
                blueprint, encoder, base_game, deal_history
            )
            opp_ev = ev1 if opp_player == 1 else ev0

            total_ev     += hero_prob * opp_ev
            total_weight += hero_prob

        if total_weight > 0:
            ev_by_opp[opp_cards] = total_ev / total_weight

    return ev_by_opp


# ── UnsafeSubgameSolver ───────────────────────────────────────────────────────

class UnsafeSubgameSolver:
    """
    Re-solves a PostflopNLHE subtree without safety guarantees.

    Uses tabular linear CFR (CFRSolver with linear_averaging=True)
    on a SubgameGame. Terminal payoffs are exact hand evaluations —
    the blueprint is not consulted during solving.

    The 'unsafe' qualifier: there is no proof that this improves
    overall exploitability vs the blueprint. In Libratus-style
    deployment it is used as a fallback for non-critical streets,
    while SafeSubgameSolver handles the final street.
    """

    def __init__(self, base_game: PostflopNLHE):
        """
        Args:
            base_game: PostflopNLHE game object (shared, not mutated).
        """
        self.base_game = base_game

    def solve(
        self,
        root_history: History,
        hero_player: int,
        hero_range: dict[tuple[int, int], float],
        opponent_range: dict[tuple[int, int], float],
        iterations: int = 100,
        max_deals: int = 200,
        rng: np.random.Generator | None = None,
    ) -> SubgameStrategy:
        """
        Re-solve the subgame rooted at root_history.

        Args:
            root_history:     Entry point into the subgame.
            hero_player:      Player performing the re-solve.
            hero_range:       {hole_cards: probability} for hero at root.
            opponent_range:   {hole_cards: probability} for opp at root.
            iterations:       CFR iterations to run.
            max_deals:        Cap on enumerated deals for tractability.
            rng:              RNG for deal sub-sampling.

        Returns:
            SubgameStrategy with query() interface.
        """
        subgame = self._build_game(
            root_history, hero_player,
            hero_range, opponent_range,
            max_deals, rng,
        )

        if not subgame.initial_histories():
            # No valid deals — return uniform strategy stub
            return SubgameStrategy({}, subgame)

        solver = CFRSolver(game=subgame, linear_averaging=True)
        solver.solve(iterations=iterations)

        # Convert InfoSetData → average_strategy arrays
        strategy_dict = {
            key: info.average_strategy()
            for key, info in solver.info_sets.items()
        }

        return SubgameStrategy(strategy_dict, subgame)

    def _build_game(
        self,
        root_history, hero_player,
        hero_range, opponent_range,
        max_deals, rng,
    ) -> SubgameGame:
        return SubgameGame(
            base_game=self.base_game,
            root_history=root_history,
            hero_player=hero_player,
            hero_range=hero_range,
            opponent_range=opponent_range,
            max_deals=max_deals,
            rng=rng,
        )


# ── SafeSubgameSolver ─────────────────────────────────────────────────────────

class SafeSubgameSolver(UnsafeSubgameSolver):
    """
    Safe subgame solving via the gadget game (Moravčík et al., 2016).

    Guarantees: the Nash equilibrium of the gadget game is a strategy
    for the hero that is at least as non-exploitable as the blueprint
    (Theorem 1, Burch et al. 2014).

    The safety comes from the opponent's opt-out option: if the re-solved
    subgame strategy is worse for the opponent than the blueprint, they
    will opt out — which forces the solution to match the blueprint value.
    Conversely, if the re-solve finds a better strategy, it will be used.

    Requires a trained Blueprint and matching NLHEEncoder.
    """

    def __init__(
        self,
        base_game: PostflopNLHE,
        blueprint,
        encoder,
    ):
        """
        Args:
            base_game:  PostflopNLHE game object.
            blueprint:  Trained Blueprint instance (for EV estimation).
            encoder:    NLHEEncoder instance matching blueprint's state_size.
        """
        super().__init__(base_game)
        self.blueprint = blueprint
        self.encoder   = encoder

    def solve(
        self,
        root_history: History,
        hero_player: int,
        hero_range: dict[tuple[int, int], float],
        opponent_range: dict[tuple[int, int], float],
        iterations: int = 100,
        max_deals: int = 200,
        rng: np.random.Generator | None = None,
    ) -> SubgameStrategy:
        """
        Re-solve with safety guarantee.

        Computes blueprint EVs for all opponent hands, constructs the
        gadget game, then solves it with tabular CFR.  The returned
        strategy excludes the opt-out action (which is internal to
        the gadget game and never played in real games).
        """
        # Step 1: estimate blueprint EV per opponent hand
        bp_ev = estimate_blueprint_ev(
            blueprint=self.blueprint,
            encoder=self.encoder,
            base_game=self.base_game,
            root_history=root_history,
            hero_player=hero_player,
            hero_range=hero_range,
            opponent_range=opponent_range,
        )

        # Step 2: build gadget game
        gadget = GadgetGame(
            blueprint_ev_by_opp_cards=bp_ev,
            base_game=self.base_game,
            root_history=root_history,
            hero_player=hero_player,
            hero_range=hero_range,
            opponent_range=opponent_range,
            max_deals=max_deals,
            rng=rng,
        )

        if not gadget.initial_histories():
            return SubgameStrategy({}, gadget)

        # Step 3: solve gadget game with CFR
        solver   = CFRSolver(game=gadget, linear_averaging=True)
        solver.solve(iterations=iterations)

        # Step 4: extract hero's strategy, strip opt-out from info sets
        strategy_dict: dict[InfoSetKey, np.ndarray] = {}
        for key, info in solver.info_sets.items():
            avg = info.average_strategy()
            actions = info.actions

            if _OPT_OUT in actions:
                # This info set belongs to the opponent at the gadget root.
                # Drop the opt-out slot — it is never played in real games.
                real_mask = [a != _OPT_OUT for a in actions]
                real_probs = avg[real_mask]
                total = real_probs.sum()
                strategy_dict[key] = (
                    real_probs / total if total > 0
                    else np.ones(real_probs.shape) / len(real_probs)
                )
            else:
                strategy_dict[key] = avg

        # Use gadget game's info_set_key method (same as SubgameGame's)
        return SubgameStrategy(strategy_dict, gadget)

    def _build_game(self, *args, **kwargs) -> GadgetGame:
        # Not used directly in SafeSubgameSolver.solve() —
        # override to prevent accidental use of parent's _build_game.
        raise NotImplementedError(
            "SafeSubgameSolver builds GadgetGame directly in solve()."
        )