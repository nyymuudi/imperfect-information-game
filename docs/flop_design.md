# Flop-level design memo

Date: 2026-07-25. All decisions below are tied to a measurement; sources are
committed logs (`docs/experiments/`, `validation_runs/logs/` reproduced here)
or commit hashes.

## Central measured finding: a turn CFV net is a prerequisite

A flop depth-limited solver truncates at the **turn boundary**; every turn
leaf needs a turn-root CFV. Today only a nested `TurnVectorCFR` (b5 river-net
leaves) can produce those — no turn CFV net exists.

Measured (`flop_throughput.log`): the flop betting tree from a flop decision
node has **17 nodes and 3 turn-boundary lines**; each line fans out to 46 turn
cards. A nested flop solve would run 3 × 46 turn-solves *per CFR iteration* —
at sustained turn-solve cost this is **~3 h per flop solve → infeasible**.
Given a turn net, a flop solve collapses to one batched net query per leaf,
i.e. the same cost class as a turn solve today.

**⇒ The flop build is sequenced as: (1) generate turn-root CFV data
(this memo's launch), (2) train the turn net, (3) build the flop solver
reusing the `TurnVectorCFR` architecture with turn-net leaves.**

## Measurements

### 1. Turn-solve throughput (flop_throughput_probe.py)

Cold (fresh process, median of 3):

| iters | solve time | solves/h |
|------:|-----------:|---------:|
| 1000  | 12.4 s     | 290 |
| 2500  | 29.1 s     | 124 |
| 5000  | 57.4 s     | 63 |

Linear in iterations (~11.6 ms/iter); board-table cache makes setup ≈ 0.

Sustained 60-min window @2500 iters: **48 solves/h** (75.2 s/solve) →
**thermal throttle ≈ 2.6× vs cold** on the fanless M2. (The window's
"first 10 min: 2 solves" reading is contaminated — a Next.js production
build ran concurrently; the sustained figure is from the solo tail.)

### 2. Solve convergence for targets (root_values_check.log)

Turn-root value drift, 500 → 1500 iterations, in-support combos
(own-mass > 1e-3), 2 spots: **mean |Δv| = 0.030 / 0.033 pot** — the same
order as the b5 net's own holdout MAE (0.038 base / 0.062 on-policy,
commit acff67d). Iterating past 500 buys precision the net cannot absorb.

### 3. Bucket sensitivity (flop_bucket_sensitivity.log)

Mass-weighted L1 between exact per-combo river values and their K-bucket
reconstruction (own-range-weighted bucket means), 8 exact river solves:

| K  | mean L1 (pot) | Δ to next K |
|---:|--------------:|------------:|
| 20 | 0.153 | — |
| 30 | 0.115 | 0.038 |
| 50 | 0.084 | 0.031 |
| 80 | 0.060 | 0.024 |

Threshold stated in advance: pick the smallest K whose improvement from
the next-larger K is **< 0.005 pot** (below target-noise level from
measurement 2). **Result: no tested K meets the threshold** — the
representation loss falls near-monotonically through the whole range,
and at K=50 the bucketing floor (0.084) already matches the b5 net's
entire on-policy MAE (0.062): bucketing, not net capacity, is the
binding error source at K=50.

### 4. Dataset sizing (river-net learning curve)

River-net holdout MAE vs data: 0.134 on-policy @5k → 0.068 @19.5k →
0.062 @35.5k (commits 5920077, acff67d, gate_b4_vs_b5.log) — clear
plateau by ~35k. Turn boards are more diverse than river boards →
target the top of that range: **30k base samples**, on-policy round
added later once a flop solver exists to query the net.

## Decisions

1. **Prerequisite: turn CFV net before any flop solver.**
   — nested flop solving measured infeasible (3×46 turn-solves/iter, §1).
2. **Iteration budget per data-gen solve: 500.**
   — 500→1500 drift (0.03 pot) is at the net's own MAE floor (§2);
   halves wall-clock vs the earlier 1000-iter default.
3. **Depth limit: one street everywhere.**
   — flop→turn-net, turn→river-net (b5), river exact; identical to the
   architecture whose gate passed (Δ −36.3 ± 16.4, 20 seeds; replicated
   stratified −30.2 ± 27.6).
4. **Bucket count for the turn net: K=80, provisional.**
   — the pre-stated <0.005 rule was met by NO tested K (§3); the floor
   still drops 0.024 pot from 50→80, so take the largest tested. Raw
   per-combo storage (decision 5) makes this revisable at train time
   without re-solving.
5. **Turn-root bucketing scheme: decided at train time, not collection time.**
   — `evaluate_7card` needs 7 cards; turn-root strength (6 cards) needs its
   own metric. The collector stores RAW per-combo values (1326-dim), so the
   scheme choice costs nothing to revisit.
6. **Dataset: 30k base samples.**
   — river-curve plateau ~35k, turn boards more diverse (§4).
7. **Wall-clock: ~5–7 days sustained.**
   — 500-iter solve ≈ 6.2 s cold → ~16 s throttled (2.6×, §1) →
   ~220 solves/h → 30k ≈ 5.7 days; self-play range computation adds
   overhead on ~30% of samples.

## Run schedule

- Command: `nohup caffeinate -dimsu python3 -u scripts/phase_c1_turn_data.py
  --n-samples 30000 --iters 500 > validation_runs/logs/turn_cfv_datagen.log 2>&1 &`
- Checkpoint: npz shard every 200 samples (~55 min at sustained rate).
- Resume: re-run the same command; the collector counts existing shards and
  continues (`[resume] N samples` — verified by kill/restart before leaving
  the run unattended).
- Thermal plan: run continuous at the measured throttled rate; no duty
  cycling (throttle already priced into the schedule).
- `nohup` mandatory: harness-managed background tasks die with the session
  (measured twice: 9-day and same-day losses before the gate run survived
  under nohup).

## Deferred

- **v16e (3 raise sizes) @5000 iters** — sample-starvation hypothesis for the
  rejected richer tree. Explicitly NOT this session.
- Turn-root bucketing metric (decision 5) — at turn-net train time.
- On-policy turn data round — needs a flop solver to generate queries.
- Flop solver implementation + its gate (whole-strategy LBR, multi-seed,
  CRN-paired) — after the turn net trains.
