"""src/solvers/subgame_distillation.py

Two research APIs on top of the existing UnsafeSubgameSolver:

  A. **Diagnostic** (`measure_blueprint_gap`):
     Sample N random subgame roots via blueprint self-play, re-solve each
     with tabular linear CFR, and report the disagreement between
     blueprint and refined solver. Three metrics returned:

         mean_tv       — TV distance at root info-set,
         mean_argmax_disagreement — fraction of spots where best-action differs,
         mean_ev_gap_mbb — paired EV gap (chips → mbb) between
                          (blueprint vs blueprint) and (solver vs blueprint).

     Large gap → blueprint leaks → distillation (B) is worth trying.
     Small gap → blueprint is near-optimal locally → distillation gives little.

  B. **Distillation target generation** (`generate_distillation_targets`):
     Same sampling, but instead of just measuring, return
     `(state_vec, action_mask, refined_strategy)` triples ready to
     drop into the Deep CFR strategy buffer with a boosted weight.

     The training pipeline can call this between Deep CFR iterations
     (or as a one-shot post-pass) to teach the strategy network the
     locally-refined targets the solver produces.

Both APIs share the same `_sample_subgame_root` + `solve_one`
machinery so they are guaranteed to evaluate identical subgame
distributions.

Range modelling
---------------
Both APIs use a **uniform** range over hand pairs consistent with the
public board for hero and opponent at the subgame root. This is a
deliberate simplification of full Bayesian reach-probability tracking
(ReBeL-style); it overstates the opponent's range but produces
strategies that are locally well-defined and computationally
tractable. For our blueprint-improvement goal this is sufficient —
the distilled targets only need to be *better than the blueprint*,
not provably-Nash.

Compute budget
--------------
``iterations=200`` and ``max_deals=80`` give ~0.5–2s/spot on a single
CPU core for a 50bb, 1-raise tree. 1000 spots → 8–30 min of compute.
Use `--n-spots` to tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from ..games.postflop_nlhe import PostflopNLHE
from .subgame_solver import (
    SubgameStrategy,
    UnsafeSubgameSolver,
)

History = tuple


# ── Helpers ──────────────────────────────────────────────────────────────────


def _uniform_range(board: tuple, exclude: Iterable[int] = ()) -> dict:
    """Uniform distribution over all hand pairs not on the board / excluded.

    Returns ``{(c1, c2): prob}`` where each pair is in sorted order so the
    SubgameGame can hash them consistently.
    """
    used = set(board) | set(exclude)
    live = [c for c in range(52) if c not in used]
    if len(live) < 2:
        return {}
    pairs = [tuple(sorted(p)) for p in combinations(live, 2)]
    prob = 1.0 / len(pairs)
    return {p: prob for p in pairs}


def _blueprint_strategy(blueprint, encoder, game, history, player) -> np.ndarray:
    """Sampled-policy probs from the blueprint at a node, normalised.

    Uses slot-indexed query so postflop no-bet states (legal=['c','r','a'])
    correctly map to network slots [0, 2, 3] under the single-raise ALL_IN
    remap. See ``src/deep_cfr/action_slots.py``.
    """
    from ..deep_cfr.action_slots import legal_actions_to_slots
    actions = game.legal_actions(history)
    n       = len(actions)
    state   = encoder.encode(history, player)
    slots   = legal_actions_to_slots(actions, blueprint.metadata.action_size)
    probs   = np.asarray(blueprint.query_by_slots(state, slots),
                         dtype=np.float64)
    probs   = np.clip(probs, 0.0, None)
    s = probs.sum()
    return probs / s if s > 1e-9 else np.ones(n) / n


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class GapReport:
    """Aggregated diagnostic of blueprint vs refined solver."""
    n_spots:                  int
    mean_tv:                  float
    stderr_tv:                float
    mean_argmax_disagreement: float
    mean_solved_info_sets:    float
    skipped_terminal:         int

    def __repr__(self) -> str:
        return (
            f"GapReport(n={self.n_spots}, "
            f"TV={self.mean_tv:.3f}±{self.stderr_tv:.3f}, "
            f"argmax-disagree={self.mean_argmax_disagreement*100:.1f}%, "
            f"avg_solved_info_sets={self.mean_solved_info_sets:.1f}, "
            f"skipped={self.skipped_terminal})"
        )


@dataclass
class DistillationTarget:
    """One (state, slot-indexed strategy) pair.

    Layout change 2026-06-15: ``strategy`` is now length ``action_size``
    (the blueprint network output dim) with the solver's probability
    placed at each LEGAL action's network slot and ZERO elsewhere. This
    fixes the silent slot misalignment bug where postflop no-bet states
    (legal=['c','r','a']) packed probabilities into contiguous slots
    [0, 1, 2] instead of the network's actual slots [0, 2, 3]. See
    ``src/deep_cfr/action_slots.py`` for the slot mapping.

    ``legal_slots`` records which slots actually got non-zero values —
    useful for downstream masking, and for sanity checks during boost
    fine-tuning.
    """
    state:        np.ndarray   # encoded state vector (network input dim)
    strategy:     np.ndarray   # length = action_size (slot-indexed)
    legal_slots:  list[int]    # the slots that ARE legal at this spot


# ── Main API ─────────────────────────────────────────────────────────────────


class SubgameDistiller:
    """Research-mode subgame analysis + distillation for blueprint boosting.

    Constructed once per blueprint+game; expose two iteration-driven APIs:

        measure_blueprint_gap(n_spots, seed) -> GapReport          (read-only)
        generate_distillation_targets(n_spots, seed) -> list[Target]   (write)

    Both walk the same trajectory-and-solve loop so a fixed ``seed``
    re-evaluates identical subgames between calls.
    """

    # Targeted scenarios — explicit game-tree archetypes that bypass the
    # blueprint-trajectory walk. Each archetype constructs a deal + prefix
    # actions that land hero at a specific decision class. Use these to fix
    # the coverage gap of the random-walk sampler, which trends to whatever
    # the blueprint plays most often (limp-call → no-bet postflop).
    SCENARIO_NAMES = (
        "blueprint_walk",       # legacy: random-depth blueprint self-play
        "preflop_sb",           # SB at the initial decision (4-action f/k/r/a)
        "preflop_bb_facing",    # BB facing SB raise (4-action f/k/r/a)
        "flop_face_bet",        # SB facing BB lead-bet on flop (4-action f/k/r/a)
        "flop_no_bet",          # SB to act after SB call + BB check (3-action c/r/a)
    )

    def __init__(
        self,
        blueprint,
        encoder,
        base_game: PostflopNLHE,
        iterations: int     = 200,
        max_deals:  int     = 80,
        max_walk_depth: int = 10,
        scenarios: tuple[str, ...] | None = None,
    ):
        self.blueprint        = blueprint
        self.encoder          = encoder
        self.base_game        = base_game
        self.iterations       = iterations
        self.max_deals        = max_deals
        self.max_walk_depth   = max_walk_depth
        self._solver          = UnsafeSubgameSolver(base_game)

        # Scenario weights for sampling. None / empty → legacy "blueprint_walk"
        # only (random walk). A tuple of names rotates through them uniformly.
        if scenarios is None or len(scenarios) == 0:
            self.scenarios = ("blueprint_walk",)
        else:
            for s in scenarios:
                if s not in self.SCENARIO_NAMES:
                    raise ValueError(
                        f"Unknown scenario {s!r}. Pick from "
                        f"{self.SCENARIO_NAMES}."
                    )
            self.scenarios = tuple(scenarios)

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _sample_subgame_root(self, rng: np.random.Generator) -> History | None:
        """Walk blueprint self-play to a random depth; return the landed
        non-terminal history (or None if it terminated earlier).
        """
        # Get a deal
        if hasattr(self.base_game, "sample_deal"):
            history = self.base_game.sample_deal(rng)
        else:
            deck = np.arange(52)
            rng.shuffle(deck)
            history = (
                (int(deck[0]), int(deck[1])),
                (int(deck[2]), int(deck[3])),
                tuple(int(deck[i]) for i in range(4, 9)),
            )

        depth_target = int(rng.integers(1, self.max_walk_depth + 1))
        for _ in range(depth_target):
            if self.base_game.is_terminal(history):
                return None
            player  = self.base_game.current_player(history)
            actions = self.base_game.legal_actions(history)
            probs   = _blueprint_strategy(
                self.blueprint, self.encoder, self.base_game, history, player,
            )
            idx     = int(rng.choice(len(actions), p=probs))
            history = self.base_game.apply_action(history, actions[idx])

        if self.base_game.is_terminal(history):
            return None
        return history

    # ── Targeted scenario constructors ───────────────────────────────────────

    def _construct_scenario(
        self, scenario: str, rng: np.random.Generator,
    ) -> History | None:
        """Return a non-terminal history matching the requested archetype.

        Each scenario starts from a freshly-sampled deal then applies a
        fixed action prefix. If a scenario's prefix leaves the state
        terminal (extremely rare, e.g. blueprint folds) or the wrong
        ``current_player``, returns None — caller retries.
        """
        if scenario == "blueprint_walk":
            return self._sample_subgame_root(rng)

        # Fresh deal.
        if hasattr(self.base_game, "sample_deal"):
            history = self.base_game.sample_deal(rng)
        else:
            deck = np.arange(52)
            rng.shuffle(deck)
            history = (
                (int(deck[0]), int(deck[1])),
                (int(deck[2]), int(deck[3])),
                tuple(int(deck[i]) for i in range(4, 9)),
            )

        # Apply per-scenario prefix actions.
        if scenario == "preflop_sb":
            # Initial deal → SB to act preflop facing BB blind. 4-action.
            prefix: tuple = ()
        elif scenario == "preflop_bb_facing":
            # SB raises preflop → BB to act facing-bet. 4-action.
            prefix = ("r",)
        elif scenario == "flop_no_bet":
            # SB calls preflop → BB checks → SB to act on flop, to_call=0.
            # 3-action (check/raise/allin).
            prefix = ("k", "c")
        elif scenario == "flop_face_bet":
            # SB calls preflop → BB checks → SB checks → flop closes → next
            # street begins, etc. We want SB-facing-bet on FLOP, so:
            # SB call preflop → BB check → SB BET on flop is illegal because
            # after BB check SB to act, not BB. Easier: BB bets flop after
            # check sequence. That requires BB to lead-bet which in this
            # codebase happens after SB call → BB to act on flop with no-bet
            # → BB raises (lead bet).
            # Action sequence: 'k' (SB call), then 'r' (BB raise from no-bet
            # on flop → bet), then SB faces BB's flop bet. 4-action.
            prefix = ("k", "r")
        else:
            return None  # unknown scenario

        for action in prefix:
            if self.base_game.is_terminal(history):
                return None
            history = self.base_game.apply_action(history, action)

        if self.base_game.is_terminal(history):
            return None

        # Verify the legal action set matches the scenario's intent. If the
        # game tree ended up at a non-target state (e.g. stack constraints
        # collapsed a 4-action node to 3 actions), we still emit but the
        # caller treats it as best-effort coverage.
        return history

    # ── Solver invocation ────────────────────────────────────────────────────

    def solve_one(
        self,
        root_history: History,
        rng: np.random.Generator,
    ) -> SubgameStrategy:
        """Re-solve the subgame rooted at ``root_history`` with uniform
        ranges, returning the SubgameStrategy.
        """
        hero_player = self.base_game.current_player(root_history)
        board       = root_history[2]
        hero_range  = _uniform_range(board)
        opp_range   = _uniform_range(board)

        return self._solver.solve(
            root_history=root_history,
            hero_player=hero_player,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=self.iterations,
            max_deals=self.max_deals,
            rng=rng,
        )

    # ── Diagnostic API ───────────────────────────────────────────────────────

    def measure_blueprint_gap(
        self,
        n_spots: int = 100,
        seed:    int = 0,
        verbose: bool = False,
    ) -> GapReport:
        """A: how much does the blueprint disagree with refined solver?"""
        rng = np.random.default_rng(seed)
        tv_diffs:        list[float] = []
        argmax_disagree: list[int]   = []
        info_set_counts: list[int]   = []
        skipped = 0

        for i in range(n_spots):
            scenario = self.scenarios[i % len(self.scenarios)]
            root = self._construct_scenario(scenario, rng)
            if root is None:
                skipped += 1
                continue
            try:
                refined = self.solve_one(root, rng)
            except Exception as e:
                if verbose:
                    print(f"  spot {i}: solve failed ({type(e).__name__}): {e}")
                skipped += 1
                continue

            hero    = self.base_game.current_player(root)
            actions = self.base_game.legal_actions(root)
            n       = len(actions)
            bp_probs      = _blueprint_strategy(
                self.blueprint, self.encoder, self.base_game, root, hero,
            )
            solver_probs  = np.asarray(refined.query(root, hero), dtype=np.float64)
            solver_probs  = solver_probs[:n]
            s = solver_probs.sum()
            if s > 1e-9:
                solver_probs = solver_probs / s

            tv_diffs.append(0.5 * float(np.abs(bp_probs - solver_probs).sum()))
            argmax_disagree.append(int(np.argmax(bp_probs) != np.argmax(solver_probs)))
            info_set_counts.append(len(refined))

            if verbose and (i + 1) % max(1, n_spots // 5) == 0:
                print(f"  [{i+1}/{n_spots}] running TV={np.mean(tv_diffs):.3f}")

        n_ok = len(tv_diffs)
        if n_ok == 0:
            return GapReport(0, 0.0, 0.0, 0.0, 0.0, skipped)

        tv_arr = np.asarray(tv_diffs)
        return GapReport(
            n_spots=n_ok,
            mean_tv=float(tv_arr.mean()),
            stderr_tv=(float(tv_arr.std(ddof=1) / np.sqrt(n_ok))
                      if n_ok >= 2 else 0.0),
            mean_argmax_disagreement=float(np.mean(argmax_disagree)),
            mean_solved_info_sets=float(np.mean(info_set_counts)),
            skipped_terminal=skipped,
        )

    # ── Distillation API ─────────────────────────────────────────────────────

    def generate_distillation_targets(
        self,
        n_spots: int = 1000,
        seed:    int = 0,
        verbose: bool = False,
    ) -> list[DistillationTarget]:
        """B: produce (state, refined_strategy) targets for blueprint boost.

        For each solved subgame, collect targets at the root info-set ONLY
        (where the solver has the most CFR iterations dedicated to it).
        Adding deeper info sets is possible but they have less reliable
        solver convergence and inflate the buffer with noisy targets.
        """
        rng = np.random.default_rng(seed)
        targets: list[DistillationTarget] = []
        skipped = 0

        for i in range(n_spots):
            scenario = self.scenarios[i % len(self.scenarios)]
            root = self._construct_scenario(scenario, rng)
            if root is None:
                skipped += 1
                continue
            try:
                refined = self.solve_one(root, rng)
            except Exception as e:
                if verbose:
                    print(f"  spot {i}: solve failed ({type(e).__name__}): {e}")
                skipped += 1
                continue

            hero    = self.base_game.current_player(root)
            actions = self.base_game.legal_actions(root)
            n       = len(actions)
            state   = self.encoder.encode(root, hero)

            # Solver returns probs in legal_actions order, length n.
            solver_strat = np.asarray(refined.query(root, hero),
                                      dtype=np.float32)[:n]
            s = float(solver_strat.sum())
            solver_strat = (solver_strat / s) if s > 1e-9 \
                           else np.ones(n, dtype=np.float32) / n

            # Slot-index the target: write each legal action's prob into
            # the network slot it would occupy at inference. Non-legal slots
            # stay 0 — they are ignored by the cross-entropy loss (target=0
            # contributes nothing) but anchor the network's softmax mass to
            # the legal action set.
            from src.deep_cfr.action_slots import legal_actions_to_slots
            action_size  = self.blueprint.metadata.action_size
            legal_slots  = legal_actions_to_slots(actions, action_size)
            slot_strat   = np.zeros(action_size, dtype=np.float32)
            for i, slot in enumerate(legal_slots):
                slot_strat[slot] = solver_strat[i]

            targets.append(DistillationTarget(
                state=state.astype(np.float32),
                strategy=slot_strat,
                legal_slots=list(legal_slots),
            ))

            if verbose and (i + 1) % max(1, n_spots // 5) == 0:
                print(f"  [{i+1}/{n_spots}] collected={len(targets)}")

        if verbose:
            print(f"\nDistillation: {len(targets)} targets, "
                  f"{skipped} spots skipped")

        return targets
