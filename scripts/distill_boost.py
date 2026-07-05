#!/usr/bin/env python3
"""scripts/distill_boost.py

ReBeL-lite blueprint booster: take an existing blueprint plus a .npz of
subgame-solver-refined targets, and fine-tune the strategy network so it
moves toward the solver targets WITHOUT catastrophically forgetting the
rest of the time-averaged policy.

Algorithm
---------
1. Load source blueprint and reconstruct its game/encoder.
2. Load boost targets from .npz produced by ``distill_subgame.py distill``.
3. Sample ``preservation_samples`` states by walking blueprint self-play
   and record the blueprint's CURRENT strategy at each — these are
   "preservation" targets that pin the network's behaviour on the rest
   of the state distribution.
4. Combine preservation samples (weight=1) and boost samples
   (weight=``boost_weight``) into a single reservoir buffer.
5. Fine-tune the strategy network on the combined buffer at LOW LR for
   ``epochs`` epochs. Low LR avoids overshooting either signal.
6. Wrap the trained weights back into a ScriptableStrategyNet and save
   the boosted blueprint.

Usage
-----
    python3 scripts/distill_boost.py \\
        blueprints/50bb_v14_winner \\
        --targets blueprints/distill/v14_targets.npz \\
        -o blueprints/50bb_v15_boosted \\
        --boost-weight 10 --preservation-samples 20000 \\
        --epochs 100 --lr 1e-4

Compute: dominated by preservation sampling (~5-15 min for 20k samples
through blueprint forward + game tree walks) + the actual fine-tune
(~1-5 min for 100 epochs at batch 256). Total ~10-30 min on a Mac.

Don't expect huge LBR gains from this step — Pluribus's blueprint was
already strong before search added value. A 5-20% LBR reduction is a
realistic mental model. If the boost makes the blueprint WORSE on LBR,
you've either over-weighted the boost (try W=5) or the targets file
came from a too-small/noisy distillation run (try more --n-spots).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.deep_cfr.blueprint import (
    Blueprint,
    BlueprintMetadata,
    ScriptableStrategyNet,
)
from src.deep_cfr.networks import StrategyNetwork, train_strategy_network
from src.deep_cfr.replay_buffer import ReservoirBuffer
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE


# ── Loading ──────────────────────────────────────────────────────────────────


def _load_blueprint_bundle(path: str, cfr_cache_path: str = ""):
    bp = Blueprint.load(path, device="cpu")
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    cfr_cache = None
    if cfr_cache_path:
        from src.deep_cfr.cfr_cache import CFRCache
        cfr_cache = CFRCache.load(cfr_cache_path)
        print(f"Loaded CFR cache: {len(cfr_cache)} entries from {cfr_cache_path}")
    encoder = NLHEEncoder(
        starting_stack=bp.metadata.starting_stack,
        raise_fractions=_rfs,
        cfr_cache=cfr_cache,
    )
    if bp.metadata.state_size != encoder.state_size():
        sys.exit(
            f"[error] {path}: state_size {bp.metadata.state_size} "
            f"≠ encoder {encoder.state_size()} (pass --cfr-cache for 49-dim blueprints)"
        )
    return bp, game, encoder


def _load_targets(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    # format_version=2 (2026-06-15): slot-indexed strategies + legal_slots.
    # Older format had ``n_legals`` and contiguous-packed strategies; refuse
    # to load it because the slot semantics are silently wrong.
    fmt = int(d["format_version"]) if "format_version" in d.files else 1
    if fmt < 2:
        sys.exit(
            f"[error] {npz_path}: legacy distillation target format "
            f"(version {fmt}) — strategies are packed in legal-action order, "
            f"not network slot order. Regenerate with the current "
            f"distill_subgame.py to fix slot semantics."
        )
    keys = {"states", "strategies", "legal_slots", "action_size"}
    missing = keys - set(d.files)
    if missing:
        sys.exit(f"[error] {npz_path}: missing fields {missing}")
    print(f"  states.shape={d['states'].shape}, "
          f"strategies.shape={d['strategies'].shape} (slot-indexed), "
          f"action_size={int(d['action_size'])}, "
          f"n_targets={len(d['states'])}")
    return {
        "states":      d["states"],
        "strategies":  d["strategies"],
        "legal_slots": d["legal_slots"],
        "action_size": int(d["action_size"]),
    }


# ── Preservation sampling ───────────────────────────────────────────────────


def _collect_preservation_samples(
    blueprint, encoder, game, n_samples: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample states via blueprint self-play; record blueprint's strategy
    at each non-terminal state. Returns (states, strategies, n_legals).
    """
    rng = np.random.default_rng(seed)
    states_out:   list[np.ndarray] = []
    strats_out:   list[np.ndarray] = []
    nlegal_out:   list[int]        = []
    action_dim = blueprint.metadata.action_size

    while len(states_out) < n_samples:
        # Fresh deal + self-play to terminal, recording every decision node
        if hasattr(game, "sample_deal"):
            history = game.sample_deal(rng)
        else:
            deck = np.arange(52)
            rng.shuffle(deck)
            history = (
                (int(deck[0]), int(deck[1])),
                (int(deck[2]), int(deck[3])),
                tuple(int(deck[i]) for i in range(4, 9)),
            )

        while not game.is_terminal(history):
            player  = game.current_player(history)
            actions = game.legal_actions(history)
            n       = len(actions)
            state   = encoder.encode(history, player)
            # Slot-indexed query so non-bet states map to correct network
            # slots and the all-in remap is honoured.
            from src.deep_cfr.action_slots import legal_actions_to_slots
            slots = legal_actions_to_slots(actions, action_dim)
            probs = np.asarray(blueprint.query_by_slots(state, slots),
                               dtype=np.float32)
            s = float(probs.sum())
            probs = probs / s if s > 1e-9 else np.ones(n, dtype=np.float32) / n

            # Place blueprint's policy at the network slots it actually
            # produced — zero in all other slots. Preservation target now
            # uses the same slot semantics as the boost target.
            padded = np.zeros(action_dim, dtype=np.float32)
            for i, slot in enumerate(slots):
                padded[slot] = probs[i]
            states_out.append(state.astype(np.float32))
            strats_out.append(padded)
            nlegal_out.append(n)

            if len(states_out) >= n_samples:
                break

            idx = int(rng.choice(n, p=probs))
            history = game.apply_action(history, actions[idx])

    return (
        np.stack(states_out[:n_samples]),
        np.stack(strats_out[:n_samples]),
        np.asarray(nlegal_out[:n_samples], dtype=np.int32),
    )


# ── Boost training ───────────────────────────────────────────────────────────


def _train_boosted_network(
    src_net: ScriptableStrategyNet,
    state_size:      int,
    action_size:     int,
    hidden_size:     int,
    boost_states:    np.ndarray,
    boost_strats:    np.ndarray,
    preserve_states: np.ndarray,
    preserve_strats: np.ndarray,
    boost_weight:    float,
    epochs:          int,
    lr:              float,
    batch_size:      int,
    device:          str,
) -> StrategyNetwork:
    """Returns a trained StrategyNetwork, initialised from src_net's weights."""
    target_net = StrategyNetwork(state_size, action_size, hidden_size).to(device)

    # Copy weights from ScriptableStrategyNet.net into StrategyNetwork.net.
    # Both share an nn.Sequential of the same shape, so state_dict transfers.
    target_net.net.load_state_dict(src_net.net.state_dict())

    # Boost strategies are already slot-indexed and sized to action_size
    # — no padding/remapping needed. Sanity check the shape.
    if boost_strats.shape[1] != action_size:
        sys.exit(
            f"[error] boost strategies have action_size={boost_strats.shape[1]} "
            f"but blueprint has action_size={action_size}. Distillation target "
            f"file is incompatible — regenerate against this blueprint."
        )
    boost_padded = boost_strats.astype(np.float32)

    n_boost     = len(boost_states)
    n_preserve  = len(preserve_states)
    capacity    = n_boost + n_preserve

    buf = ReservoirBuffer(
        capacity=capacity, state_size=state_size, action_size=action_size,
        mode="reservoir",
    )
    # Bulk-insert: weights are the only differentiator between boost and
    # preservation rows. iteration field is unused here (no DCFR weighting).
    boost_w    = np.full(n_boost,    boost_weight, dtype=np.float32)
    preserve_w = np.full(n_preserve, 1.0,          dtype=np.float32)
    buf.add_batch(boost_states,    boost_padded,    boost_w,    iteration=1)
    buf.add_batch(preserve_states, preserve_strats, preserve_w, iteration=1)
    print(f"  buffer filled: {n_boost} boost (w={boost_weight}) + "
          f"{n_preserve} preserve (w=1) = {buf.size} total")

    final_loss = train_strategy_network(
        target_net, buf,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    print(f"  fine-tune final loss = {final_loss:.4f}")
    return target_net


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("blueprint",          help="Source blueprint directory")
    p.add_argument("--targets",         required=True,
                   help="Path to .npz produced by distill_subgame.py distill")
    p.add_argument("-o", "--output",    required=True,
                   help="Output blueprint directory")
    p.add_argument("--boost-weight",     type=float, default=10.0,
                   help="Per-sample weight for boost targets vs preserve=1.0")
    p.add_argument("--preservation-samples", type=int, default=20000,
                   help="Self-play states whose blueprint output is preserved "
                        "during fine-tune. Higher = more anti-forgetting "
                        "but slower sampling.")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="Fine-tune learning rate. Low to avoid overshooting.")
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--device",          type=str,   default="cpu",
                   help="'cpu' or 'mps'. Bigger preservation samples benefit "
                        "from mps; small runs stay simpler on cpu.")
    p.add_argument("--cfr-cache",       type=str,   default="",
                   help="CFR advisor cache path. Required for blueprints "
                        "trained at state_size=49 (v15_c2_v4+). Same cache "
                        "the blueprint was trained against.")
    args = p.parse_args()

    # Validate paths
    if not Path(args.blueprint).exists():
        sys.exit(f"[error] blueprint not found: {args.blueprint}")
    if not Path(args.targets).exists():
        sys.exit(f"[error] targets file not found: {args.targets}")

    # 1. Load source blueprint
    print(f"Loading source blueprint: {args.blueprint}")
    src_bp, game, encoder = _load_blueprint_bundle(args.blueprint, args.cfr_cache)
    state_size  = src_bp.metadata.state_size
    action_size = src_bp.metadata.action_size
    hidden_size = src_bp.metadata.hidden_size
    print(f"  state_size={state_size}, action_size={action_size}, "
          f"hidden_size={hidden_size}")

    # 2. Load boost targets
    print(f"\nLoading boost targets: {args.targets}")
    targets = _load_targets(args.targets)

    # 3. Collect preservation samples
    print(f"\nCollecting {args.preservation_samples} preservation samples...")
    t0 = time.time()
    preserve_states, preserve_strats, _ = _collect_preservation_samples(
        src_bp, encoder, game,
        n_samples=args.preservation_samples, seed=args.seed,
    )
    print(f"  done in {time.time() - t0:.1f}s "
          f"({args.preservation_samples / max(time.time() - t0, 1e-6):.0f}/s)")

    # 4. + 5. Train boosted network
    print(f"\nFine-tuning strategy network "
          f"(epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})...")
    t0 = time.time()
    trained_net = _train_boosted_network(
        src_bp._net,
        state_size=state_size, action_size=action_size, hidden_size=hidden_size,
        boost_states=targets["states"],
        boost_strats=targets["strategies"],
        preserve_states=preserve_states,
        preserve_strats=preserve_strats,
        boost_weight=args.boost_weight,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"  fine-tune complete in {time.time() - t0:.1f}s")

    # 6. Wrap into ScriptableStrategyNet + Blueprint, save
    scripted = ScriptableStrategyNet.from_strategy_network(trained_net)
    new_meta = BlueprintMetadata(
        state_size=src_bp.metadata.state_size,
        action_size=src_bp.metadata.action_size,
        hidden_size=src_bp.metadata.hidden_size,
        starting_stack=src_bp.metadata.starting_stack,
        sb=src_bp.metadata.sb,
        bb=src_bp.metadata.bb,
        raise_fraction=src_bp.metadata.raise_fraction,
        raise_fractions=list(src_bp.metadata.raise_fractions),
        max_raises=src_bp.metadata.max_raises,
        iterations=src_bp.metadata.iterations,  # source iters; boost is offline
        traversals_per_iter=src_bp.metadata.traversals_per_iter,
        strategy_samples=(args.preservation_samples + len(targets["states"])),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    new_bp = Blueprint(net=scripted, metadata=new_meta, device="cpu")
    new_bp.save(args.output)
    print(f"\n[ok] boosted blueprint → {args.output}")
    print(f"  source = {args.blueprint}")
    print(f"  targets = {args.targets}")
    print(f"  boost_weight = {args.boost_weight}, "
          f"preserve_samples = {args.preservation_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
