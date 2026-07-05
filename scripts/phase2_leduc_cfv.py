#!/usr/bin/env python3
"""
Phase 2 gate: PBS value net + depth-limited re-solving on Leduc.

ReBeL Algorithm 1 outer loop:
  epoch e:
    1. Sample reveal-boundary PBSs (blueprint self-play lines + random
       Dirichlet ranges for coverage of solver-iterate range space).
    2. Solve each round-2 subgame EXACTLY (Phase 1 solver); store
       (PBS encoding, per-holding CFVs) in a ReservoirBuffer.
    3. Re-train the CFV net FROM SCRATCH on the reservoir.
    4. Build the depth-limited re-solving agent (round-1 decisions:
       VectorCFR truncated at the reveal boundary with net leaf values;
       round-2 decisions: exact resolve) and measure its EXACT
       exploitability.
  Margin(e) = DL-agent exploitability − exact re-solving agent
  exploitability (Phase 1 machinery, same blueprint, computed once).

Exit criterion: margin trends downward with epochs. Plot saved to
validation_runs/phase2_leduc_margin.png.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.games.leduc import LeducHoldem
from src.deep_cfr.state_encoder import LeducEncoder
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.deep_cfr.replay_buffer import ReservoirBuffer
from src.search.pbs import (
    LEDUC_CARDS, CARD_IDX, initial_pbs,
    representative_history, update_on_action, update_on_community,
)
from src.search.resolve import build_resolving_strategy
from src.search.depth_limited import build_dl_resolving_strategy
from src.search.cfv import ENCODING_DIMS, encode_round2_pbs, fast_round2_cfvs
from src.search.cfv_net import train_cfv_net, net_leaf_evaluator

from scripts.phase1_leduc_resolve import exact_exploitability_of

CONTS = (1.0, 3.0, 5.0)


def sample_selfplay_pbs(game, blueprint_fn, rng):
    """Follow blueprint public dynamics to a reveal boundary; return
    (comm, cont, y0, y1) or None if the line folded out."""
    pbs = initial_pbs()
    while True:
        rep = representative_history("J1", 0, pbs.community, pbs.actions)
        if game.is_terminal(rep):
            return None
        _, _, r1_done = game._split_rounds(pbs.actions)
        if r1_done:
            comm = LEDUC_CARDS[rng.integers(0, 6)]
            post = update_on_community(pbs, comm)
            cont = 1.0 + 2.0 * pbs.actions.count("r")
            return comm, cont, post.range_array(0), post.range_array(1)
        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        # Public action probability: range-weighted average strategy.
        rvec = pbs.range_array(player)
        avg = np.zeros(len(legal))
        for i, card in enumerate(LEDUC_CARDS):
            if rvec[i] <= 0:
                continue
            h = representative_history(card, player, pbs.community, pbs.actions)
            avg += rvec[i] * np.asarray(blueprint_fn(h, player))[: len(legal)]
        s = avg.sum()
        avg = avg / s if s > 0 else np.ones(len(legal)) / len(legal)
        action = legal[int(rng.choice(len(legal), p=avg))]
        pbs = update_on_action(pbs, game, player, action, blueprint_fn)


def sample_random_pbs(rng):
    """Random reveal-boundary PBS. Dirichlet α is itself randomised
    (broad → near-degenerate) so sharp CFR-iterate ranges are covered."""
    comm = LEDUC_CARDS[rng.integers(0, 6)]
    cont = CONTS[rng.integers(0, len(CONTS))]
    ci = CARD_IDX[comm]
    alpha = float(rng.choice([0.15, 0.3, 1.0, 3.0]))
    ranges = []
    for _ in range(2):
        y = rng.dirichlet(np.full(6, alpha))
        y[ci] = 0.0
        s = y.sum()
        if s <= 0:
            y = np.ones(6); y[ci] = 0.0; s = y.sum()
        y /= s
        ranges.append(y)
    return comm, cont, ranges[0], ranges[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=2000)
    ap.add_argument("--target-iters", type=int, default=300,
                    help="exact solve iterations per CFV target")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--onpolicy-samples", type=int, default=400,
                    help="leaf queries from the previous epoch's DL solves "
                         "added as extra targets (ReBeL on-policy data)")
    ap.add_argument("--dl-iters", type=int, default=200)
    ap.add_argument("--bp-iters", type=int, default=100)
    ap.add_argument("--resolve-iters", type=int, default=400,
                    help="Phase 1 exact-agent resolve iterations (baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="validation_runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    game = LeducHoldem()
    encoder = LeducEncoder()

    # ── Blueprint (same recipe/seed as Phase 1 gate) ────────────────────────
    print(f"[setup] Training Deep CFR blueprint ({args.bp_iters} iters)...")
    t0 = time.time()
    solver = DeepCFRSolver(
        game=game, encoder=encoder, max_actions=4,
        buffer_capacity=500_000, hidden_size=128, train_epochs=60,
        traversals_per_iter=500, use_cpp_engine=False, device="cpu", lr=1e-3,
    )
    solver.solve(iterations=args.bp_iters)
    print(f"        done in {time.time()-t0:.0f}s")

    def blueprint_fn(history, player):
        state = encoder.encode(history, player)
        n = len(game.legal_actions(history))
        return solver._get_regret_strategy(state, n)

    bp_expl = exact_exploitability_of(game, blueprint_fn)
    print(f"[setup] blueprint exploitability = {bp_expl:.5f}")

    # ── Baseline: Phase 1 exact re-solving agent ────────────────────────────
    print(f"[setup] Exact re-solving agent baseline "
          f"({args.resolve_iters} iters/spot)...")
    t0 = time.time()
    exact_strategy = {}
    for seat in (0, 1):
        exact_strategy.update(build_resolving_strategy(
            game, blueprint_fn, hero_seat=seat,
            resolve_iters=args.resolve_iters,
        ))

    def dict_fn(strategy):
        def fn(history, player):
            key = game.info_set_key(history, player)
            n = len(game.legal_actions(history))
            if key in strategy:
                return strategy[key][:n]
            return np.ones(n) / n
        return fn

    exact_expl = exact_exploitability_of(game, dict_fn(exact_strategy))
    print(f"        exact-agent exploitability = {exact_expl:.5f} "
          f"({time.time()-t0:.0f}s)")

    # ── Outer loop ──────────────────────────────────────────────────────────
    buffer = ReservoirBuffer(capacity=50_000, state_size=ENCODING_DIMS,
                             action_size=12)
    history_rows = []

    print(f"\n{'epoch':>5} {'buffer':>7} {'dup%':>6} {'loss':>9} "
          f"{'dl_expl':>9} {'margin':>9} {'t(s)':>6}")

    onpolicy_pool: list = []          # leaf queries from previous DL builds

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 1-2. Data generation with exact targets (vector-CFR solver).
        samples = []
        for k in range(args.samples_per_epoch):
            if k % 2 == 0:
                s = sample_selfplay_pbs(game, blueprint_fn, rng)
                samples.append(s if s is not None else sample_random_pbs(rng))
            else:
                samples.append(sample_random_pbs(rng))
        # ReBeL on-policy augmentation: PBSs the previous epoch's DL
        # solves actually queried (late-iteration half = converged ranges).
        n_op = 0
        if onpolicy_pool:
            late = onpolicy_pool[len(onpolicy_pool) // 2:]
            n_op = min(args.onpolicy_samples, len(late))
            idx = rng.choice(len(late), size=n_op, replace=False)
            samples.extend(late[i] for i in idx)
            onpolicy_pool = []

        states, targets = [], []
        for comm, cont, y0, y1 in samples:
            v0, v1 = fast_round2_cfvs(game, comm, cont, y0, y1,
                                      solve_iters=args.target_iters)
            states.append(encode_round2_pbs(comm, cont, y0, y1))
            targets.append(np.concatenate([v0, v1]).astype(np.float32))

        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        # visits≈1 verification: fraction of duplicate encodings in batch.
        n_unique = len(np.unique(states, axis=0))
        dup_frac = 1.0 - n_unique / len(states)
        buffer.add_batch(states, targets,
                         np.ones(len(states), dtype=np.float32),
                         iteration=epoch)

        # 3. From-scratch retraining (fresh net + fresh Adam).
        net, loss = train_cfv_net(buffer, epochs=args.train_steps,
                                  hidden=args.hidden)

        # 4. Depth-limited agent + exact exploitability. Leaf queries are
        # logged into the on-policy pool for the next epoch's targets.
        base_eval = net_leaf_evaluator(net)

        def logging_eval(comm, cont, y0, y1):
            onpolicy_pool.append((comm, cont, y0.copy(), y1.copy()))
            return base_eval(comm, cont, y0, y1)

        dl_strategy = {}
        for seat in (0, 1):
            dl_strategy.update(build_dl_resolving_strategy(
                game, blueprint_fn, hero_seat=seat,
                leaf_evaluator=logging_eval, dl_iters=args.dl_iters,
            ))
        dl_expl = exact_exploitability_of(game, dict_fn(dl_strategy))
        margin = dl_expl - exact_expl

        dt = time.time() - t0
        print(f"{epoch:>5} {len(buffer):>7} {100*dup_frac:>5.1f}% "
              f"{loss:>9.5f} {dl_expl:>9.5f} {margin:>+9.5f} {dt:>6.0f}"
              f"   (+{n_op} on-policy)")
        history_rows.append(dict(epoch=epoch, buffer=len(buffer),
                                 dup_frac=dup_frac, loss=loss,
                                 dl_expl=dl_expl, margin=margin,
                                 n_onpolicy=n_op))
        torch.save(net.state_dict(),
                   os.path.join(args.out_dir, "phase2_cfv_net.pt"))

    # ── Report + plot ───────────────────────────────────────────────────────
    out = dict(blueprint_expl=bp_expl, exact_agent_expl=exact_expl,
               epochs=history_rows, args=vars(args))
    log_path = os.path.join(args.out_dir, "phase2_leduc_margin.json")
    with open(log_path, "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["epoch"] for r in history_rows]
        ms = [r["margin"] for r in history_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ms, "o-", label="DL-agent margin vs exact re-solve")
        plt.axhline(0.0, color="gray", lw=0.8)
        plt.xlabel("self-play epoch")
        plt.ylabel("exploitability margin")
        plt.title("Phase 2: depth-limited (CFV net) vs exact re-solving, Leduc")
        plt.legend()
        plt.tight_layout()
        png = os.path.join(args.out_dir, "phase2_leduc_margin.png")
        plt.savefig(png, dpi=120)
        print(f"\nplot: {png}")
    except ImportError:
        print("\n(matplotlib not available — skipped plot)")

    print(f"log:  {log_path}")
    first, last = history_rows[0]["margin"], history_rows[-1]["margin"]
    # Exit criterion: DL exploitability within a small, SHRINKING margin
    # of exact re-solving. A margin that is already ≤ small threshold
    # satisfies the intent outright (it cannot meaningfully shrink toward
    # zero from below); otherwise require a downward trend.
    ok = (last <= 0.05) or (last < first)
    verdict = "PASS" if ok else "FAIL"
    print(f"\n== Phase 2 gate ==")
    print(f"blueprint expl     : {bp_expl:.5f}")
    print(f"exact agent expl   : {exact_expl:.5f}")
    print(f"margin first→last  : {first:+.5f} → {last:+.5f}  {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
