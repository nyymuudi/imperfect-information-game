#!/usr/bin/env python3
"""scripts/build_cfr_cache.py

Build a CFR advisor cache for a blueprint:
  1. Self-play the blueprint to discover the most-visited public-state
     abstractions.
  2. Pick the top-N most-visited keys.
  3. For each key, run mini-CFR (tabular linear) on a small subgame
     rooted at one example history that maps to that key.
  4. Extract root action probabilities + per-action EVs.
  5. Save as ``.cache`` (numpy npz under the hood).

The resulting cache is loaded by the augmented NLHEEncoder at training
and inference time — each state's 12-dim advisor signal comes from this
cache when the key hits, or from a live MC EV fallback when it misses.

Usage:
    python3 scripts/build_cfr_cache.py BLUEPRINT \\
        -o blueprints/cache/<name>.cache \\
        --n-trajectories 50000 --n-spots 10000 --iter-per-spot 100

Compute: ~3 hours for n_spots=10000 on a single CPU core.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.deep_cfr.action_slots import legal_actions_to_slots
from src.deep_cfr.blueprint import Blueprint
from src.deep_cfr.cfr_cache import (
    ADVISOR_DIMS, EV_DIMS, PROB_DIMS,
    CFRCache, CFRCacheMeta,
    collect_visit_distribution,
    make_meta,
    public_state_key,
)
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE
from src.solvers.subgame_solver import UnsafeSubgameSolver


# ── Hero-only CFR vs fixed baseline opponent ────────────────────────────────


def _solve_one_vs_baseline(
    history: tuple,
    game,
    blueprint,
    encoder,
    iter_per_spot: int,
    max_deals: int,
    max_actions: int,
    rng: np.random.Generator,
    compute_ev: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a single spot with hero running tabular CFR while opponent
    plays the fixed baseline blueprint strategy.

    Eliminates the information leak inherent in single-hand-hero solving:
    the opponent never learns from the hero's degenerate range because
    its strategy is frozen to the blueprint forward-pass output.

    Returns (probs[6], evs[6]) slot-indexed via legal_actions_to_slots.
    """
    from src.deep_cfr.action_slots import legal_actions_to_slots

    probs_out = np.zeros(PROB_DIMS, dtype=np.float32)
    evs_out   = np.zeros(EV_DIMS,   dtype=np.float32)

    hero_player = game.current_player(history)
    opp_player  = 1 - hero_player
    board       = history[2]
    hero_cards  = history[hero_player]

    # ── Enumerate opp deals (hero hand is fixed) ────────────────────────────
    used  = set(board) | set(hero_cards)
    from itertools import combinations
    all_opp = [tuple(sorted(p)) for p in combinations(
        [c for c in range(52) if c not in used], 2)]
    if not all_opp:
        return probs_out, evs_out
    if len(all_opp) > max_deals:
        idx = rng.choice(len(all_opp), size=max_deals, replace=False)
        opp_deals = [all_opp[i] for i in idx]
    else:
        opp_deals = all_opp
    deal_weight = 1.0 / len(opp_deals)

    # ── Build root histories per deal ───────────────────────────────────────
    # history = (p0_cards, p1_cards, board, *prefix_actions). We substitute
    # opp cards while keeping hero's cards, board, and any prefix actions.
    prefix_len = len(history) - 3
    prefix     = history[3:]
    deal_roots = []
    for oc in opp_deals:
        p0 = hero_cards if hero_player == 0 else oc
        p1 = oc          if hero_player == 0 else hero_cards
        deal_roots.append((p0, p1) + (board,) + prefix)

    # ── Caches ──────────────────────────────────────────────────────────────
    # Hero's regret table: info_key -> length-max_actions array.
    # Hero observes: (own cards, visible board, subgame-local action history).
    # Since hero cards are constant for this spot, the (board, actions) part
    # is what varies, but we still key by all three for safety.
    hero_regrets:      dict = {}
    hero_strategy_sum: dict = {}
    # Opp blueprint output cache: key by state_vec bytes. The same opp state
    # recurs across iterations and deals → first iter is the only cost.
    opp_strat_cache:   dict = {}

    def hero_info_key(h):
        visible = game._visible_board(h)
        actions = h[3:][prefix_len:]
        return (hero_cards, visible, actions)

    def hero_strategy(info_key, n_legal):
        regrets = hero_regrets.get(info_key)
        if regrets is None:
            return np.ones(n_legal) / n_legal
        positive = np.maximum(regrets[:n_legal], 0)
        s = positive.sum()
        if s > 0:
            return positive / s
        return np.ones(n_legal) / n_legal

    def opp_strategy(h, n_legal):
        sv = encoder.encode(h, opp_player)
        cache_key = sv.tobytes()
        cached = opp_strat_cache.get(cache_key)
        if cached is not None:
            return cached
        actions = game.legal_actions(h)
        slots   = legal_actions_to_slots(actions, max_actions)
        raw     = np.asarray(blueprint.query_by_slots(sv, slots), dtype=np.float64)
        raw     = np.clip(raw, 0, None)
        s = raw.sum()
        strat = raw / s if s > 1e-9 else np.ones(n_legal) / n_legal
        opp_strat_cache[cache_key] = strat
        return strat

    # ── Hero-only CFR traversal ─────────────────────────────────────────────
    def traverse(h, hero_reach, opp_reach, iter_weight, depth=0):
        if depth > 40 or game.is_terminal(h):
            if game.is_terminal(h):
                return float(game.terminal_payoffs(h)[hero_player])
            return 0.0
        cur     = game.current_player(h)
        actions = game.legal_actions(h)
        n       = len(actions)

        if cur == hero_player:
            info_key = hero_info_key(h)
            strat    = hero_strategy(info_key, n)
            child_vs = np.zeros(n)
            for i, a in enumerate(actions):
                child_vs[i] = traverse(game.apply_action(h, a),
                                       hero_reach * strat[i], opp_reach,
                                       iter_weight, depth + 1)
            node_v = float((strat * child_vs).sum())
            # Counterfactual regret update (weighted by opp reach).
            regrets = hero_regrets.get(info_key)
            if regrets is None:
                regrets = np.zeros(max_actions, dtype=np.float64)
                hero_regrets[info_key] = regrets
            for i in range(n):
                regrets[i] += opp_reach * (child_vs[i] - node_v)
            # Linear-CFR strategy sum (weighted by hero reach × iter_weight).
            ssum = hero_strategy_sum.get(info_key)
            if ssum is None:
                ssum = np.zeros(max_actions, dtype=np.float64)
                hero_strategy_sum[info_key] = ssum
            for i in range(n):
                ssum[i] += hero_reach * iter_weight * strat[i]
            return node_v

        # Opp: fixed baseline.
        strat = opp_strategy(h, n)
        value = 0.0
        for i, a in enumerate(actions):
            value += strat[i] * traverse(game.apply_action(h, a),
                                          hero_reach, opp_reach * strat[i],
                                          iter_weight, depth + 1)
        return value

    # ── Iterate ─────────────────────────────────────────────────────────────
    try:
        for t in range(iter_per_spot):
            iter_weight = t + 1   # Linear CFR
            for root_h in deal_roots:
                traverse(root_h, hero_reach=1.0, opp_reach=deal_weight,
                         iter_weight=iter_weight)
    except Exception:
        return probs_out, evs_out

    # ── Extract root strategy ──────────────────────────────────────────────
    legal       = game.legal_actions(history)
    legal_slots = legal_actions_to_slots(legal, max_actions)
    root_key    = hero_info_key(history)
    ssum        = hero_strategy_sum.get(root_key)
    if ssum is None or ssum.sum() < 1e-9:
        raw = np.ones(len(legal), dtype=np.float64) / len(legal)
    else:
        raw = ssum[:len(legal)] / ssum[:len(legal)].sum()
    for i, slot in enumerate(legal_slots):
        if 0 <= slot < PROB_DIMS:
            probs_out[slot] = raw[i]

    if not compute_ev:
        return probs_out, evs_out

    # ── Per-action EV (hero plays avg strategy from there, opp baseline) ───
    NORM = 2.0 * float(game.starting_stack)

    def rollout_ev(h, depth=0):
        if depth > 40 or game.is_terminal(h):
            if game.is_terminal(h):
                return float(game.terminal_payoffs(h)[hero_player])
            return 0.0
        cur     = game.current_player(h)
        actions = game.legal_actions(h)
        n       = len(actions)
        if cur == hero_player:
            info_key = hero_info_key(h)
            ssum_local = hero_strategy_sum.get(info_key)
            if ssum_local is None or ssum_local[:n].sum() < 1e-9:
                strat = np.ones(n) / n
            else:
                strat = ssum_local[:n] / ssum_local[:n].sum()
        else:
            strat = opp_strategy(h, n)
        value = 0.0
        for i, a in enumerate(actions):
            if strat[i] < 1e-6:
                continue
            value += strat[i] * rollout_ev(game.apply_action(h, a), depth + 1)
        return value

    for i, action in enumerate(legal):
        slot = legal_slots[i]
        if not (0 <= slot < EV_DIMS):
            continue
        try:
            # Average EV per opp deal at this hero action.
            action_ev = 0.0
            for root_h in deal_roots:
                next_h = game.apply_action(root_h, action)
                action_ev += rollout_ev(next_h)
            evs_out[slot] = float(action_ev / len(deal_roots)) / NORM
        except Exception:
            evs_out[slot] = 0.0

    return probs_out, evs_out


# ── Helpers ──────────────────────────────────────────────────────────────────


def _uniform_range(board: tuple, exclude=()) -> dict:
    used = set(board) | set(exclude)
    live = [c for c in range(52) if c not in used]
    pairs = [tuple(sorted(p)) for p in combinations(live, 2)]
    if not pairs:
        return {}
    prob = 1.0 / len(pairs)
    return {p: prob for p in pairs}


def _solve_one(
    history: tuple,
    game,
    solver: UnsafeSubgameSolver,
    iter_per_spot: int,
    max_deals: int,
    max_actions: int,
    rng: np.random.Generator,
    compute_ev: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Mini-CFR at a public-state root → (probs[6], evs[6]).

    The returned arrays are length-6 (= NLHE network output capacity),
    slot-indexed via ``action_slots.legal_actions_to_slots``. Non-legal
    slots get probability and EV of 0.

    EVs are normalised by 2 × starting_stack so they fit roughly in
    [-1, 1] — keeps the augmented network input stable.
    """
    probs = np.zeros(PROB_DIMS, dtype=np.float32)
    evs   = np.zeros(EV_DIMS,   dtype=np.float32)

    hero_player = game.current_player(history)
    board       = history[2]
    # Pin hero range to the sample hand so the solved strategy_dict is
    # guaranteed to contain the hero's root info-set. With a broad uniform
    # hero_range and max_deals sub-sampling, the sample hand almost never
    # survives → SubgameStrategy.query falls back to uniform → cache fills
    # with [1/n, 1/n, ...]. The cache key already buckets hand strength,
    # so one representative hand per bucket is the intended semantics.
    sample_hero = history[hero_player]
    hero_range  = {sample_hero: 1.0}
    opp_range   = _uniform_range(board, exclude=sample_hero)

    try:
        strat = solver.solve(
            root_history=history,
            hero_player=hero_player,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=iter_per_spot,
            max_deals=max_deals,
            rng=rng,
        )
    except Exception:
        return probs, evs

    # Root strategy
    legal       = game.legal_actions(history)
    raw_probs   = strat.query(history, hero_player)
    legal_slots = legal_actions_to_slots(legal, max_actions)

    s = float(np.sum(raw_probs))
    if s > 1e-9:
        raw_probs = np.asarray(raw_probs, dtype=np.float32) / s
    else:
        raw_probs = np.ones(len(legal), dtype=np.float32) / len(legal)

    for i, slot in enumerate(legal_slots):
        if 0 <= slot < PROB_DIMS:
            probs[slot] = raw_probs[i]

    if not compute_ev:
        return probs, evs

    # Per-action EV: use the SOLVED SubgameStrategy to roll out the
    # post-action subtree. The strategy is heavily concentrated (mini-CFR
    # has converged) so the recursion only follows the high-probability
    # branches — much cheaper than the uniform rollout used in the
    # initial implementation.
    NORM = 2.0 * float(game.starting_stack)

    def _rollout_ev(h, hero_p, depth=0):
        """Recursive EV with the solved strategy. Depth-limited so a
        pathological deep tree can't blow the budget."""
        if depth > 30 or game.is_terminal(h):
            try:
                if game.is_terminal(h):
                    return float(game.terminal_payoffs(h)[hero_p])
            except Exception:
                pass
            return 0.0
        cur_player = game.current_player(h)
        actions = game.legal_actions(h)
        try:
            probs = strat.query(h, cur_player)
        except Exception:
            probs = np.ones(len(actions)) / len(actions)
        probs = np.asarray(probs, dtype=np.float64)
        s = probs.sum()
        probs = probs / s if s > 1e-9 else np.ones(len(actions)) / len(actions)
        ev = 0.0
        for j, a in enumerate(actions):
            if probs[j] < 1e-6:
                continue
            try:
                ev += probs[j] * _rollout_ev(game.apply_action(h, a),
                                              hero_p, depth + 1)
            except Exception:
                pass
        return ev

    for i, action in enumerate(legal):
        slot = legal_slots[i]
        if not (0 <= slot < EV_DIMS):
            continue
        try:
            next_h = game.apply_action(history, action)
            evs[slot] = _rollout_ev(next_h, hero_player) / NORM
        except Exception:
            evs[slot] = 0.0

    return probs, evs


# ── Multiprocess worker ─────────────────────────────────────────────────────


def _worker_solve_chunk(work_items, seed, worker_args):
    """Worker function: rebuild game/solver locally, solve a chunk of spots.

    Returns ``[(key, probs[6], evs[6]), ...]`` for each successfully solved
    spot. Errors in individual spots return zero-filled (key, zeros, zeros)
    so the outer iteration doesn't get confused by missing entries.
    """
    # Imports inside the worker so child processes can pickle the closure
    # (the outer module-level imports already cover this, but explicit is
    # safer on Spawn-based starts like macOS).
    import numpy as np
    from src.games.postflop_nlhe import PostflopNLHE
    from src.solvers.subgame_solver import UnsafeSubgameSolver

    game = PostflopNLHE(
        starting_stack=worker_args["starting_stack"],
        max_raises_per_street=worker_args["max_raises_per_street"],
        raise_fractions=worker_args["raise_fractions"],
    )
    rng = np.random.default_rng(seed)

    # Optional baseline blueprint + encoder for vs-baseline solving.
    blueprint = None
    encoder   = None
    if worker_args.get("vs_baseline"):
        from src.deep_cfr.blueprint import Blueprint
        from src.deep_cfr.state_encoder import NLHEEncoder
        blueprint = Blueprint.load(worker_args["blueprint_path"], device="cpu")
        encoder   = NLHEEncoder(
            starting_stack=worker_args["starting_stack"],
            raise_fractions=worker_args["raise_fractions"],
        )
    else:
        solver = UnsafeSubgameSolver(game)

    out = []
    for key, history in work_items:
        try:
            if worker_args.get("vs_baseline"):
                probs, evs = _solve_one_vs_baseline(
                    history, game, blueprint, encoder,
                    iter_per_spot=worker_args["iter_per_spot"],
                    max_deals=worker_args["max_deals"],
                    max_actions=worker_args["max_actions"],
                    rng=rng,
                    compute_ev=worker_args["compute_ev"],
                )
            else:
                probs, evs = _solve_one(
                    history, game, solver,
                    iter_per_spot=worker_args["iter_per_spot"],
                    max_deals=worker_args["max_deals"],
                    max_actions=worker_args["max_actions"],
                    rng=rng,
                    compute_ev=worker_args["compute_ev"],
                )
        except Exception:
            probs = np.zeros(PROB_DIMS, dtype=np.float32)
            evs   = np.zeros(EV_DIMS,   dtype=np.float32)
        out.append((key, probs, evs))
    return out


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("blueprint",
                   help="Source blueprint directory (used for self-play "
                        "trajectories to discover representative spots).")
    p.add_argument("-o", "--output",   required=True,
                   help="Output .cache path.")
    p.add_argument("--n-trajectories", type=int, default=80000,
                   help="Self-play hands for visit-frequency estimation. "
                        "Higher → broader spot enumeration.")
    p.add_argument("--n-spots",        type=int, default=10000,
                   help="Top-N most-visited abstraction keys to solve.")
    p.add_argument("--iter-per-spot",  type=int, default=150,
                   help="Tabular CFR iterations per spot. Higher → better "
                        "converged solver strategy (= cleaner advisor signal).")
    p.add_argument("--max-deals",      type=int, default=50,
                   help="Subgame deal-enumeration cap per spot. Higher → more "
                        "diverse range coverage in the solver.")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--no-ev",          action="store_true",
                   help="Skip per-action EV computation (cache stores only "
                        "action probabilities; EVs stay at 0). ~3x faster "
                        "but loses ~half the advisor signal.")
    p.add_argument("--workers",        type=int, default=1,
                   help="Number of parallel worker processes. Each worker "
                        "solves spots independently (embarrassingly parallel). "
                        "On an M2 Air try 4-6; expect 3-5x wall-clock speedup "
                        "(GIL-free multiprocess, but per-spot setup overhead).")
    p.add_argument("--vs-baseline",    action="store_true", default=True,
                   help="Solve hero-only CFR with the opponent fixed to the "
                        "source blueprint. Eliminates the info-leak that "
                        "occurs when both players' strategies adapt to a "
                        "degenerate single-hand hero range. Recommended; "
                        "default ON.")
    p.add_argument("--no-vs-baseline", dest="vs_baseline", action="store_false",
                   help="Disable --vs-baseline; fall back to symmetric "
                        "UnsafeSubgameSolver (leaks; only for ablations).")
    p.add_argument("--verbose", "-v",  action="store_true")
    args = p.parse_args()

    if not Path(args.blueprint).exists():
        sys.exit(f"[error] blueprint not found: {args.blueprint}")

    # ── Load blueprint + reconstruct game/encoder ────────────────────────────
    print(f"Loading blueprint: {args.blueprint}")
    bp = Blueprint.load(args.blueprint, device="cpu")
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    encoder = NLHEEncoder(
        starting_stack=bp.metadata.starting_stack,
        raise_fractions=_rfs,
    )
    max_actions = bp.metadata.action_size
    print(f"  state_size={bp.metadata.state_size}, "
          f"action_size={max_actions}, "
          f"raise_fractions={_rfs}")

    # ── Step 1: Self-play to discover top-N most-visited spots ───────────────
    print(f"\nStep 1/2: Collecting visit distribution over "
          f"{args.n_trajectories} self-play trajectories...")
    t0 = time.time()
    counts, samples = collect_visit_distribution(
        bp, encoder, game,
        n_trajectories=args.n_trajectories,
        seed=args.seed,
    )
    print(f"  {len(counts)} unique abstraction keys in "
          f"{time.time() - t0:.1f}s")

    # Top-N by count
    top_keys = sorted(counts.keys(), key=counts.get, reverse=True)
    top_keys = top_keys[: args.n_spots]
    total_visits = sum(counts.values())
    covered = sum(counts[k] for k in top_keys)
    coverage_pct = 100 * covered / max(total_visits, 1)
    print(f"  Top {len(top_keys)} keys cover {coverage_pct:.1f}% of all visits")

    # ── Step 2: Mini-CFR per spot ────────────────────────────────────────────
    print(f"\nStep 2/2: Solving mini-CFR ({args.iter_per_spot} iter, "
          f"max_deals={args.max_deals}) for {len(top_keys)} spots "
          f"with {args.workers} worker(s)...")

    # Build (key, history) work items, dropping keys whose sample is missing.
    work_items = [(int(k), samples[k]) for k in top_keys if k in samples]

    cache_keys  = []
    cache_probs = []
    cache_evs   = []

    t0 = time.time()

    if args.workers <= 1:
        # Single-process path (simpler debugging + low overhead at small N).
        rng    = np.random.default_rng(args.seed + 1)
        solver = None if args.vs_baseline else UnsafeSubgameSolver(game)
        if args.vs_baseline:
            print("  Solver: hero-only CFR vs fixed baseline blueprint")
        for i, (key, history) in enumerate(work_items):
            if args.vs_baseline:
                probs, evs = _solve_one_vs_baseline(
                    history, game, bp, encoder,
                    iter_per_spot=args.iter_per_spot,
                    max_deals=args.max_deals,
                    max_actions=max_actions,
                    rng=rng,
                    compute_ev=not args.no_ev,
                )
            else:
                probs, evs = _solve_one(
                    history, game, solver,
                    iter_per_spot=args.iter_per_spot,
                    max_deals=args.max_deals,
                    max_actions=max_actions,
                    rng=rng,
                    compute_ev=not args.no_ev,
                )
            cache_keys.append(key)
            cache_probs.append(probs)
            cache_evs.append(evs)

            if args.verbose and (i + 1) % max(1, len(work_items) // 20) == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                remaining = (len(work_items) - i - 1) / rate
                print(f"  [{i+1:>5d}/{len(work_items)}] "
                      f"{rate:.1f} spots/s  ETA {remaining/60:.1f} min")
    else:
        # Multi-process path. Each worker rebuilds its own game/solver/RNG
        # locally so we don't pickle solver state. The (key, history) tuple
        # is small; the result fits in a few floats.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        worker_args = dict(
            starting_stack=game.starting_stack,
            max_raises_per_street=game.max_raises_per_street,
            raise_fractions=tuple(game.raise_fractions),
            iter_per_spot=args.iter_per_spot,
            max_deals=args.max_deals,
            max_actions=max_actions,
            compute_ev=not args.no_ev,
            vs_baseline=args.vs_baseline,
            blueprint_path=str(Path(args.blueprint).resolve()),
        )
        if args.vs_baseline:
            print("  Solver: hero-only CFR vs fixed baseline blueprint")

        done = 0
        # Chunk work to reduce IPC overhead.
        chunksize = max(1, len(work_items) // (args.workers * 8))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(_worker_solve_chunk,
                          work_items[s:s + chunksize], args.seed + 1 + i,
                          worker_args)
                for i, s in enumerate(range(0, len(work_items), chunksize))
            ]
            for fut in as_completed(futures):
                chunk_results = fut.result()
                for key, probs, evs in chunk_results:
                    cache_keys.append(key)
                    cache_probs.append(probs)
                    cache_evs.append(evs)
                done += len(chunk_results)

                if args.verbose:
                    elapsed = time.time() - t0
                    rate    = done / max(elapsed, 1e-9)
                    remaining = (len(work_items) - done) / max(rate, 1e-9)
                    print(f"  [{done:>5d}/{len(work_items)}] "
                          f"{rate:.1f} spots/s  ETA {remaining/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n  Solved {len(cache_keys)} spots in {elapsed/60:.1f} min "
          f"({len(cache_keys)/max(elapsed, 1e-9):.1f} spots/s)")

    # ── Save ────────────────────────────────────────────────────────────────
    meta = make_meta(game, args.n_spots, args.iter_per_spot)
    cache = CFRCache(
        keys=np.asarray(cache_keys, dtype=np.uint64),
        probs=np.asarray(cache_probs, dtype=np.float32),
        evs=np.asarray(cache_evs,   dtype=np.float32),
        meta=meta,
    )
    cache.save(args.output)
    print(f"\n[ok] wrote {len(cache)} cache entries → {args.output}")
    print(f"  iter_per_spot={meta.iter_per_spot}, "
          f"max_actions={meta.max_actions}, "
          f"starting_stack={meta.starting_stack}BB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
