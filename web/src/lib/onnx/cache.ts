/**
 * CFR advisor cache — browser side.
 *
 * Loads the binary cache emitted by Python's CFRCache.save_binary() and
 * exposes binary-search lookup. The encoder fills the 12 advisor dims
 * [37:49] of each state vector from this cache before ONNX inference.
 *
 * Binary format (little-endian, packed) — mirrors C++ CFRCacheLoader:
 *   uint32  magic   = 0x43464341 ('CFCA')
 *   uint32  version = 1
 *   uint32  n_entries
 *   uint32  prob_dim (=6)
 *   uint32  ev_dim   (=6)
 *   float32 ev_norm  (= 2*starting_stack)
 *   uint8   reserved[12]
 *   entries[]:
 *     uint64 key      (little-endian)
 *     float32 probs[6]
 *     float32 evs[6]
 *
 * Entries are sorted by key ascending so lookup is O(log N).
 */

const MAGIC   = 0x43464341
const VERSION = 1
const HEADER_BYTES = 36
const ENTRY_BYTES  = 56  // 8 (key) + 4*6 (probs) + 4*6 (evs)

export interface AdvisorCache {
  readonly nEntries: number
  readonly evNorm:   number
  // Parallel arrays. keysHi/Lo split the uint64 key into two uint32 halves
  // for fast binary search (JS bigint compares are slower than two int32
  // compares when there are tens of thousands of entries).
  readonly keysHi: Uint32Array
  readonly keysLo: Uint32Array
  readonly probs:  Float32Array  // [n * 6] row-major
  readonly evs:    Float32Array  // [n * 6] row-major
}

export interface CacheEntry {
  probs: Float32Array  // length 6
  evs:   Float32Array  // length 6
}

/** Fetch + parse the binary cache. Throws on bad magic/version. */
export async function loadCache(url: string): Promise<AdvisorCache> {
  const res = await fetch(url, { cache: 'force-cache' })
  if (!res.ok) throw new Error(`cache fetch failed: ${res.status} ${url}`)
  const buf = await res.arrayBuffer()
  return parseCache(buf)
}

export function parseCache(buf: ArrayBuffer): AdvisorCache {
  if (buf.byteLength < HEADER_BYTES) {
    throw new Error(`cache too small (${buf.byteLength} bytes)`)
  }
  const view = new DataView(buf)
  const magic    = view.getUint32(0,  true)
  const version  = view.getUint32(4,  true)
  const nEntries = view.getUint32(8,  true)
  const probDim  = view.getUint32(12, true)
  const evDim    = view.getUint32(16, true)
  const evNorm   = view.getFloat32(20, true)

  if (magic !== MAGIC)       throw new Error(`bad cache magic 0x${magic.toString(16)}`)
  if (version !== VERSION)   throw new Error(`unsupported cache version ${version}`)
  if (probDim !== 6 || evDim !== 6) {
    throw new Error(`unexpected dims prob=${probDim} ev=${evDim}`)
  }
  const expectedBytes = HEADER_BYTES + nEntries * ENTRY_BYTES
  if (buf.byteLength < expectedBytes) {
    throw new Error(`truncated cache (${buf.byteLength} < ${expectedBytes})`)
  }

  const keysHi = new Uint32Array(nEntries)
  const keysLo = new Uint32Array(nEntries)
  const probs  = new Float32Array(nEntries * 6)
  const evs    = new Float32Array(nEntries * 6)

  for (let i = 0; i < nEntries; i++) {
    const off = HEADER_BYTES + i * ENTRY_BYTES
    // little-endian uint64 = lo 32 bits at off, hi 32 bits at off+4
    keysLo[i] = view.getUint32(off,     true)
    keysHi[i] = view.getUint32(off + 4, true)
    for (let j = 0; j < 6; j++) {
      probs[i * 6 + j] = view.getFloat32(off + 8  + j * 4, true)
      evs  [i * 6 + j] = view.getFloat32(off + 32 + j * 4, true)
    }
  }

  return { nEntries, evNorm, keysHi, keysLo, probs, evs }
}

// Reusable views to avoid allocating Float32Array per lookup.
const _PROB_VIEW = { entry: { probs: new Float32Array(6), evs: new Float32Array(6) } as CacheEntry }

/** Binary-search lookup. Returns null on miss. Returned arrays are
 * REUSED across calls — copy if you need to retain the values. */
export function lookupCache(cache: AdvisorCache, key: bigint): CacheEntry | null {
  // Split incoming key into hi/lo halves for 32-bit comparison.
  const MASK32 = BigInt('0xFFFFFFFF')
  const SHIFT32 = BigInt(32)
  const keyLo = Number(key & MASK32)
  const keyHi = Number((key >> SHIFT32) & MASK32)

  let lo = 0
  let hi = cache.nEntries
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    const mHi = cache.keysHi[mid]
    const mLo = cache.keysLo[mid]
    if (mHi < keyHi || (mHi === keyHi && mLo < keyLo)) {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  if (lo >= cache.nEntries) return null
  if (cache.keysHi[lo] !== keyHi || cache.keysLo[lo] !== keyLo) return null

  const off = lo * 6
  for (let i = 0; i < 6; i++) {
    _PROB_VIEW.entry.probs[i] = cache.probs[off + i]
    _PROB_VIEW.entry.evs[i]   = cache.evs  [off + i]
  }
  return _PROB_VIEW.entry
}

// ── Key derivation ───────────────────────────────────────────────────────────
//
// Mirrors src/deep_cfr/cfr_cache.py:key_from_state_vector and the C++
// NLHEStateEncoder::key_from_state_vector. Must agree bit-for-bit with the
// cache build pipeline so lookups hit. The state-vector layout (read-only
// from slots 0..36) is documented in encoder.ts.

const _FIELD_OFFSETS = {
  street: 0,
  player: 2,
  raises: 3,
  last_aggressor: 6,
  pot_bucket: 8,
  spr_bucket: 11,
  board_bucket: 14,
  hand_bucket: 17,
}
const _FIELD_WIDTHS = {
  street: 2, player: 1, raises: 3, last_aggressor: 2,
  pot_bucket: 3, spr_bucket: 3, board_bucket: 3, hand_bucket: 3,
}

function _argmaxBlock(sv: Float32Array, lo: number, hi: number): number {
  let best = 0
  let bv = sv[lo]
  for (let i = lo + 1; i < hi; i++) {
    if (sv[i] > bv) { bv = sv[i]; best = i - lo }
  }
  return best
}

function _potBucket(potChips: number, sbChips: number): number {
  const potBb = potChips / (2.0 * sbChips)
  if (potBb < 5)   return 0
  if (potBb < 8)   return 1
  if (potBb < 14)  return 2
  if (potBb < 22)  return 3
  if (potBb < 35)  return 4
  if (potBb < 60)  return 5
  if (potBb < 100) return 6
  return 7
}

function _sprBucket(myStack: number, oppStack: number, potChips: number): number {
  const eff = Math.min(myStack, oppStack)
  if (potChips <= 1e-6) return 7
  const spr = eff / potChips
  if (spr < 0.5)  return 0
  if (spr < 1.5)  return 1
  if (spr < 3)    return 2
  if (spr < 5)    return 3
  if (spr < 9)    return 4
  if (spr < 15)   return 5
  if (spr < 25)   return 6
  return 7
}

export function keyFromStateVector(sv: Float32Array, startingStack: number): bigint {
  const handBucket = _argmaxBlock(sv, 0, 8)

  let boardBucket = 0
  let bsum = 0
  for (let i = 8; i < 16; i++) bsum += sv[i]
  if (bsum > 0) boardBucket = _argmaxBlock(sv, 8, 16)

  const street = _argmaxBlock(sv, 16, 20)
  const player = sv[36] > 0.5 ? 0 : 1

  // Raise count heuristic over the last-8 action history slots.
  let raises = 0
  for (let i = 24; i < 32; i++) {
    const v = sv[i]
    const raiseLike = (v > 0.35 && v < 0.65) || Math.abs(v - 0.6) < 0.02
    const allinLike = v >= 0.95
    if (raiseLike || allinLike) raises++
  }
  const lastAggressorRel = raises > 0 ? 2 : 0

  // Undo encoder normalisation, then re-bucket.
  const potNorm  = sv[20]
  const potChips = potNorm * (2.0 * startingStack)
  const myStack  = sv[22] * startingStack
  const oppStack = sv[23] * startingStack
  const sbChips  = 1.0     // matches PostflopNLHE default sb=1

  const pb = _potBucket(potChips, sbChips)
  const sb = _sprBucket(myStack, oppStack, Math.max(potChips, 1e-6))

  // Pack into 20-bit key (same field widths as Python encode_key()).
  const fields = {
    street, player, raises,
    last_aggressor: lastAggressorRel,
    pot_bucket:   pb,
    spr_bucket:   sb,
    board_bucket: boardBucket,
    hand_bucket:  handBucket,
  } as const
  let key = BigInt(0)
  for (const name of ['street', 'player', 'raises', 'last_aggressor',
                       'pot_bucket', 'spr_bucket', 'board_bucket',
                       'hand_bucket'] as const) {
    const w   = _FIELD_WIDTHS[name]
    const off = _FIELD_OFFSETS[name]
    const max = (1 << w) - 1
    const v   = Math.max(0, Math.min(fields[name], max))
    key |= BigInt(v) << BigInt(off)
  }
  return key
}
