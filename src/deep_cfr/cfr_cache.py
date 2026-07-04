"""src/deep_cfr/cfr_cache.py

CFR advisor cache: pre-computed mini-CFR action probabilities + per-action
EV estimates at representative public-state buckets.

Used by the augmented NLHEEncoder to feed a "wise advisor" signal into
the Deep CFR network at every decision node — speeds up convergence in
multi-raise / sparse-action settings where raw regret-matching has too
few samples per action.

Cache layout
------------
The file format is a single ``.npz`` archive with three arrays:
  * ``keys``       — uint64 [N] abstraction-key hashes (sorted)
  * ``probs``      — float32 [N, 6] action probabilities at each key
  * ``evs``        — float32 [N, 6] per-action EVs at each key
                     (normalised by 2 × starting_stack so values lie in
                     roughly [-1, 1])

Plus metadata fields stored as 0-d scalars:
  * ``starting_stack``     — float
  * ``raise_fractions``    — array (variable length)
  * ``max_actions``        — int
  * ``iter_per_spot``      — int
  * ``n_spots_requested``  — int
  * ``timestamp``          — str

Lookup at training/inference time:
  ``CFRCache.lookup(history, game)`` returns ``(probs, evs)`` if the
  abstraction key for that state exists in the cache, else ``None``.
  The encoder appends 12 zeros and lets the live MC EV fallback (option
  A in the design doc) fill the slot when None is returned.

Key construction
----------------
The abstraction key is a 64-bit hash over a small set of public-state
features (street, current player, raise count, pot bucket, SPR bucket,
board texture, hero hand bucket). The same features are computed
identically in C++ so the parity test pins both implementations to a
single source of truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── Abstraction key field widths ────────────────────────────────────────────
#
# Total: 21 bits → ~2M unique abstractions. In practice ~50k-100k are reachable
# via legal play; the cache stores the top-N most-visited (by blueprint
# self-play) below that.

_FIELDS = (
    ("street",         2),  # 0..3
    ("player",         1),  # 0 or 1
    ("raises",         3),  # 0..7  (clamped)
    ("last_aggressor", 2),  # 0=none, 1=hero, 2=opp
    ("pot_bucket",     3),  # 0..7
    ("spr_bucket",     3),  # 0..7
    ("board_bucket",   3),  # 0..7
    ("hand_bucket",    3),  # 0..7
)
# Bit offsets of each field within the 64-bit key.
_OFFSETS = {}
_off = 0
for _name, _width in _FIELDS:
    _OFFSETS[_name] = _off
    _off += _width
KEY_WIDTH_BITS = _off  # 20


def encode_key(
    street: int,
    player: int,
    raises: int,
    last_aggressor_rel: int,
    pot_bucket: int,
    spr_bucket: int,
    board_bucket: int,
    hand_bucket: int,
) -> int:
    """Pack the abstraction fields into a single uint64 key.

    All inputs are clamped into their field width: a value of 9 in a
    3-bit field clamps to 7. Negative values clamp to 0.
    """
    values = {
        "street":         street,
        "player":         player,
        "raises":         raises,
        "last_aggressor": last_aggressor_rel,
        "pot_bucket":     pot_bucket,
        "spr_bucket":     spr_bucket,
        "board_bucket":   board_bucket,
        "hand_bucket":    hand_bucket,
    }
    val = 0
    for name, width in _FIELDS:
        v = max(0, min(int(values[name]), (1 << width) - 1))
        val |= v << _OFFSETS[name]
    return val


# ── State → key extraction ──────────────────────────────────────────────────


def _pot_bucket(pot_chips: float, sb_chips: float) -> int:
    """Bucket pot size into [0..7] given the small blind's chip value."""
    # bb = 2 × sb in our setup; express pot in BBs for stable boundaries
    pot_bb = pot_chips / (2.0 * sb_chips)
    if   pot_bb < 5:     return 0
    elif pot_bb < 8:     return 1
    elif pot_bb < 14:    return 2
    elif pot_bb < 22:    return 3
    elif pot_bb < 35:    return 4
    elif pot_bb < 60:    return 5
    elif pot_bb < 100:   return 6
    else:                return 7


def _spr_bucket(stacks: tuple, pot_chips: float) -> int:
    """Bucket effective stack-to-pot ratio."""
    eff = min(stacks)
    if pot_chips <= 1e-6:
        return 7  # pre-flop / huge SPR
    spr = eff / pot_chips
    if   spr < 0.5:  return 0
    elif spr < 1.5:  return 1
    elif spr < 3:    return 2
    elif spr < 5:    return 3
    elif spr < 9:    return 4
    elif spr < 15:   return 5
    elif spr < 25:   return 6
    else:            return 7


def key_from_state_vector(state_vec, encoder) -> int:
    """Re-derive the abstraction key from an already-encoded state vector.

    Used by the C++ → Python buffer pipeline (Vaihtoehto 1) where the C++
    engine produces state vectors with advisor dims zeroed and Python
    backfills them via cache lookup. The state vector retains enough
    information to reconstruct the key: street/board/hand are one-hot,
    position is a bit, pot/SPR can be re-bucketed from their continuous
    values.

    Layout assumed (matches NLHEEncoder.encode at 2026-06-16):
        [0:8]   preflop bucket one-hot → hand_bucket
        [8:16]  board bucket one-hot   → board_bucket
        [16:20] street one-hot
        [20]    pot / (2 * stack)      (normalised, quantised)
        [21]    to_call / (2 * stack)  (normalised, quantised)
        [22]    my stack / stack       (normalised, quantised)
        [23]    opp stack / stack      (normalised, quantised)
        [24:32] action history (8 slots, each in {-1.0, 0.0, 0.25, 0.4..0.6, 1.0})
        [32]    continuous: own preflop equity
        [33]    continuous: board strength
        [34]    continuous: pot odds
        [35]    continuous: SPR (capped)
        [36]    position bit (1.0 = SB / player 0, 0.0 = BB)
    """
    import numpy as np
    sv = np.asarray(state_vec, dtype=np.float32)

    # Hand bucket: argmax over slots [0:8]
    hand_bucket = int(np.argmax(sv[0:8]))

    # Board bucket: argmax over slots [8:16] (preflop is all-zero → 0)
    if sv[8:16].sum() > 0:
        board_bucket = int(np.argmax(sv[8:16]))
    else:
        board_bucket = 0

    # Street: argmax over slots [16:20]
    street = int(np.argmax(sv[16:20]))

    # Position from slot 36 (last base dim before advisor pad)
    player = 0 if float(sv[36]) > 0.5 else 1

    # Action-history-derived: count raise actions; detect last aggressor by
    # sign and ordering of the last few history slots.
    hist_slots = sv[24:32]
    # 'r' / 'r0' encoded as 0.5; 'r1' = 0.4; 'r2' = 0.6; 'a' = 1.0.
    raise_marks = ((hist_slots > 0.35) & (hist_slots < 0.65)) | (np.isclose(hist_slots, 0.6, atol=0.02))
    allin_marks = hist_slots >= 0.95
    raises = int(np.sum(raise_marks) + np.sum(allin_marks))

    # Last aggressor relative: walk back through history slots to find the
    # most recent raise/allin. The player who took that action is the
    # aggressor — its position relative to ``player`` determines la_rel.
    last_aggressor_rel = 0
    # We don't have direct player attribution from the history slot alone,
    # so use a heuristic: if there are no raises this street the field is
    # 0. Otherwise infer "1 = me made the raise" if my action came last
    # in the history, else "2 = opp". For the slot-based key the exact
    # field value matters less than its consistency between Python and
    # the cache builder — both use the SAME history-derived key. Keep
    # this conservative for now and refine if cache hit-rate is poor.
    if raises > 0:
        last_aggressor_rel = 2  # most often opp raised before our turn

    # Pot bucket: undo the encoder's pot/2*stack normalisation. We don't
    # have the SB chip value here directly, but the encoder norms by
    # 2*starting_stack, and the cache builder uses the same scale.
    pot_norm  = float(sv[20])
    pot_chips = pot_norm * (2.0 * encoder.starting_stack)
    sb_chips  = float(getattr(encoder._game(), "sb", 1.0))
    pb_idx    = _pot_bucket(pot_chips, sb_chips)

    # SPR bucket: stack_22, stack_23 are own/opp normalised stacks.
    my_stack  = float(sv[22]) * encoder.starting_stack
    opp_stack = float(sv[23]) * encoder.starting_stack
    spr_idx   = _spr_bucket((my_stack, opp_stack), max(pot_chips, 1e-6))

    return encode_key(
        street=street,
        player=player,
        raises=raises,
        last_aggressor_rel=last_aggressor_rel,
        pot_bucket=pb_idx,
        spr_bucket=spr_idx,
        board_bucket=board_bucket,
        hand_bucket=hand_bucket,
    )


def public_state_key(history, game, encoder) -> int:
    """Compute the 64-bit abstraction key for a public state.

    The hero's hand is hashed into a bucket too — strictly speaking that
    is NOT public information, but the encoder's network input already
    conditions on the hero's hand bucket, and the cache key has to be
    in 1:1 correspondence with the encoder's view of the state. Including
    the hand bucket means different hero hands at the same public state
    get distinct advisor signals — which is what we want.

    Delegates to ``key_from_state_vector`` on the encoded state so the
    raw-history path and the state-vector path (C++ pipeline, lookup())
    produce identical keys by construction. A hand-rolled _parse_state
    version drifted from the encoder's pot/stack semantics (pot included
    posted blinds differently), silently breaking cache-hit parity.
    """
    player = game.current_player(history)
    saved  = getattr(encoder, "cfr_cache", None)
    encoder.cfr_cache = None
    try:
        sv = encoder.encode(history, player)
    finally:
        encoder.cfr_cache = saved
    return key_from_state_vector(sv, encoder)


# ── Cache container ─────────────────────────────────────────────────────────


# Each cache entry is a fixed-size float vector: 6 probs + 6 EVs.
ADVISOR_DIMS = 12
PROB_DIMS    = 6
EV_DIMS      = 6


@dataclass
class CFRCacheMeta:
    starting_stack:    float
    raise_fractions:   tuple
    max_actions:       int
    iter_per_spot:     int
    n_spots_requested: int
    timestamp:         str


class CFRCache:
    """In-memory CFR advisor cache.

    Use ``CFRCache.load(path)`` to read from disk and
    ``cache.lookup(history, game, encoder)`` to fetch advisor values.
    """

    def __init__(
        self,
        keys: np.ndarray,
        probs: np.ndarray,
        evs:   np.ndarray,
        meta:  CFRCacheMeta,
    ):
        if keys.dtype != np.uint64:
            keys = keys.astype(np.uint64)
        if keys.ndim != 1 or probs.shape != (len(keys), PROB_DIMS) \
                or evs.shape != (len(keys), EV_DIMS):
            raise ValueError(
                f"shape mismatch: keys={keys.shape}, "
                f"probs={probs.shape}, evs={evs.shape}"
            )
        # Build a Python dict for O(1) lookup. (uint64 keys map fine to
        # Python int; numpy uint64 needs ``int()`` conversion at lookup.)
        self._table = {
            int(k): (probs[i].astype(np.float32),
                     evs[i].astype(np.float32))
            for i, k in enumerate(keys)
        }
        self.keys_array  = keys
        self.probs_array = probs.astype(np.float32)
        self.evs_array   = evs.astype(np.float32)
        self.meta        = meta

    def __len__(self) -> int:
        return len(self._table)

    def lookup(self, history, game, encoder) -> Optional[tuple]:
        """Return ``(probs[6], evs[6])`` if cached, else ``None``.

        Key derivation goes through the encoded state vector (not the raw
        history) so that lookups stay consistent with the C++ training
        pipeline — the C++ engine only has access to the state vector and
        re-derives keys via ``key_from_state_vector``. Building the cache
        with the same path guarantees train ↔ inference parity.

        We temporarily detach the cache from the encoder during the helper
        encode() so that the lookup → encode → lookup loop doesn't recurse
        infinitely. The base-37 state vector is all we need for key
        derivation (advisor slots aren't read by key_from_state_vector).
        """
        try:
            player = game.current_player(history)
            saved  = encoder.cfr_cache
            encoder.cfr_cache = None
            try:
                sv = encoder.encode(history, player)
            finally:
                encoder.cfr_cache = saved
            k = key_from_state_vector(sv, encoder)
        except Exception:
            return None
        entry = self._table.get(int(k))
        return entry

    def lookup_by_key(self, key: int) -> Optional[tuple]:
        return self._table.get(int(key))

    # ── Persistence ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "CFRCache":
        # numpy.savez_compressed appends ``.npz`` to the path if missing.
        # Mirror that here so callers don't need to add the suffix.
        path = Path(path)
        if not path.exists() and path.with_suffix(path.suffix + ".npz").exists():
            path = path.with_suffix(path.suffix + ".npz")
        elif not path.exists() and Path(str(path) + ".npz").exists():
            path = Path(str(path) + ".npz")
        d = np.load(path, allow_pickle=True)
        meta = CFRCacheMeta(
            starting_stack    = float(d["starting_stack"]),
            raise_fractions   = tuple(float(x) for x in d["raise_fractions"]),
            max_actions       = int(d["max_actions"]),
            iter_per_spot     = int(d["iter_per_spot"]),
            n_spots_requested = int(d["n_spots_requested"]),
            timestamp         = str(d["timestamp"]),
        )
        return cls(
            keys=np.asarray(d["keys"], dtype=np.uint64),
            probs=np.asarray(d["probs"], dtype=np.float32),
            evs=np.asarray(d["evs"], dtype=np.float32),
            meta=meta,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keys              = self.keys_array.astype(np.uint64),
            probs             = self.probs_array.astype(np.float32),
            evs               = self.evs_array.astype(np.float32),
            starting_stack    = float(self.meta.starting_stack),
            raise_fractions   = np.asarray(self.meta.raise_fractions, dtype=np.float32),
            max_actions       = int(self.meta.max_actions),
            iter_per_spot     = int(self.meta.iter_per_spot),
            n_spots_requested = int(self.meta.n_spots_requested),
            timestamp         = str(self.meta.timestamp),
        )

    def save_binary(self, path: str | Path) -> None:
        """Export to compact binary for C++ ``CFRCacheLoader``.

        Format (little-endian, packed):
            uint32  magic   = 0x43464341 ('CFCA')
            uint32  version = 1
            uint32  n_entries
            uint32  prob_dim (=6)
            uint32  ev_dim   (=6)
            float32 ev_norm  (= 2*starting_stack)
            uint8   reserved[12]
            entries[]: uint64 key, float32 probs[6], float32 evs[6]
                       sorted by key ascending
        """
        import struct
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by key for binary search on C++ side.
        order = np.argsort(self.keys_array)
        keys  = self.keys_array[order].astype(np.uint64)
        probs = self.probs_array[order].astype(np.float32)
        evs   = self.evs_array[order].astype(np.float32)
        n     = len(keys)
        ev_norm = 2.0 * float(self.meta.starting_stack)

        with open(path, "wb") as f:
            # Header: 12 bytes (3 × uint32) + 8 bytes (prob_dim, ev_dim)
            #         + 4 (ev_norm) + 12 (reserved) = 36 bytes
            f.write(struct.pack("<III", 0x43464341, 1, n))
            f.write(struct.pack("<II", PROB_DIMS, EV_DIMS))
            f.write(struct.pack("<f",  ev_norm))
            f.write(b"\x00" * 12)
            # Entries
            for i in range(n):
                f.write(struct.pack("<Q", int(keys[i])))
                f.write(probs[i].tobytes(order="C"))
                f.write(evs[i].tobytes(order="C"))


# ── Visit-frequency collection (for "top-N most-visited spots" strategy) ─


def collect_visit_distribution(
    blueprint, encoder, game,
    n_trajectories: int,
    seed: int = 0,
) -> tuple[dict[int, int], dict[int, tuple]]:
    """Self-play the blueprint many times; count abstraction-key visits AND
    remember one representative history per key.

    Returns ``(counts, sample_history)`` where:
      * ``counts[key]``         = number of visits
      * ``sample_history[key]`` = one history tuple that maps to ``key``

    The cache builder picks the top-N keys by count, then solves a
    mini-CFR rooted at the sample history for each. The exact history
    matters because the abstraction key is lossy (e.g. board texture
    bucket loses the exact suits), so we need a concrete deal to feed
    the subgame solver.
    """
    from .action_slots import legal_actions_to_slots

    rng = np.random.default_rng(seed)
    counts:  dict[int, int]   = {}
    samples: dict[int, tuple] = {}

    for _ in range(n_trajectories):
        history = game.sample_deal(rng) if hasattr(game, "sample_deal") else None
        if history is None:
            deck = np.arange(52)
            rng.shuffle(deck)
            history = (
                (int(deck[0]), int(deck[1])),
                (int(deck[2]), int(deck[3])),
                tuple(int(deck[i]) for i in range(4, 9)),
            )

        while not game.is_terminal(history):
            try:
                # Derive key from the encoded state vector — same path the
                # C++ buffer pipeline uses at backfill time. Cache build +
                # backfill + Python-side lookup all go through key_from_
                # state_vector so keys agree by construction (no parity
                # gap between history-based and state-vector-based keys).
                _p = game.current_player(history)
                _sv = encoder.encode(history, _p)
                k = key_from_state_vector(_sv, encoder)
                counts[k] = counts.get(k, 0) + 1
                if k not in samples:
                    samples[k] = history
            except Exception:
                pass
            actions = game.legal_actions(history)
            player  = game.current_player(history)
            state   = encoder.encode(history, player)
            slots   = legal_actions_to_slots(actions, blueprint.metadata.action_size)
            probs   = np.asarray(blueprint.query_by_slots(state, slots),
                                 dtype=np.float64)
            probs   = np.clip(probs, 0, None)
            s = probs.sum()
            probs   = probs / s if s > 1e-9 else np.ones(len(actions)) / len(actions)
            idx     = int(rng.choice(len(actions), p=probs))
            history = game.apply_action(history, actions[idx])

    return counts, samples


def make_meta(
    game,
    n_spots_requested: int,
    iter_per_spot:     int,
) -> CFRCacheMeta:
    """Helper: build a CFRCacheMeta from a game object + run parameters."""
    return CFRCacheMeta(
        starting_stack    = float(game.starting_stack),
        raise_fractions   = tuple(float(x) for x in game.raise_fractions),
        max_actions       = 3 + len(game.raise_fractions),
        iter_per_spot     = int(iter_per_spot),
        n_spots_requested = int(n_spots_requested),
        timestamp         = time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
