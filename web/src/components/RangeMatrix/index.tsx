'use client'

import { useState, useTransition } from 'react'
import { RANKS, SUITS, makeCard, cardRank, cardSuit, cardLabel } from '@/lib/onnx/encoder'
import { queryStrategyBatch, type ActionProbs } from '@/lib/onnx/session'
import { STREET_NAMES, type Street } from '@/types'

// ── Helpers ───────────────────────────────────────────────────────────────────

function handLabel(r1: number, r2: number, suited: boolean): string {
  const hi = Math.max(r1, r2)
  const lo = Math.min(r1, r2)
  if (hi === lo) return RANKS[hi] + RANKS[lo]
  return RANKS[hi] + RANKS[lo] + (suited ? 's' : 'o')
}

function cellGradient(p: ActionProbs): string {
  const stops: string[] = []
  let pct = 0
  const add = (color: string, w: number) => {
    if (w < 0.5) return
    stops.push(`${color} ${pct.toFixed(1)}%`)
    pct += w
    stops.push(`${color} ${pct.toFixed(1)}%`)
  }
  add('#4a5a50', p.fold  * 100)
  add('#22c55e', p.call  * 100)
  add('#f59e0b', p.raise * 100)
  add('#a78bfa', p.allIn * 100)
  return stops.length ? `linear-gradient(to bottom, ${stops.join(', ')})` : 'var(--bg3)'
}

// ── Board card picker ─────────────────────────────────────────────────────────

function BoardPicker({ value, excluded, onChange, onClear }: {
  value: number | null
  excluded: number[]
  onChange: (c: number) => void
  onClear: () => void
}) {
  const [open, setOpen] = useState(false)
  const isRed = (si: number) => si === 1 || si === 2

  return (
    <div className="relative" style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <button
        onClick={() => setOpen(v => !v)}
        className={`card-btn ${value !== null ? 'selected' : ''}`}
        style={{ minWidth: 48, fontSize: 13 }}
      >
        {value !== null
          ? <span style={{ color: isRed(cardSuit(value)) ? '#ef4444' : undefined }}>{cardLabel(value)}</span>
          : '—'}
      </button>
      {value !== null && (
        <button onClick={onClear} className="clear-btn">×</button>
      )}
      {open && (
        <div className="card-grid">
          {RANKS.slice().reverse().map(r => {
            const ri = RANKS.indexOf(r)
            return SUITS.map((s, si) => {
              const card = makeCard(ri, si)
              const disabled = excluded.includes(card)
              return (
                <button
                  key={card}
                  disabled={disabled}
                  onClick={() => { onChange(card); setOpen(false) }}
                  className={`card-cell ${disabled ? 'opacity-20' : ''}`}
                >
                  <span style={{ color: isRed(si) ? '#ef4444' : undefined }}>{r}{s}</span>
                </button>
              )
            })
          })}
        </div>
      )}
    </div>
  )
}

// ── Action history presets ────────────────────────────────────────────────────
// Antaa mallille realistisen betting-historian sen sijaan että kaikki kyselyt
// tehdään tyhjällä historialla. Vastaa yleisimpiä pelattavia tilanteita.

type HistoryPreset = { label: string; desc: string; actions: number[] }

const HISTORY_PRESETS: HistoryPreset[] = [
  { label: 'No action',    desc: 'First to act, no prior betting',         actions: [] },
  { label: 'Facing bet',   desc: 'Opponent has bet / raised once',         actions: [2] },
  { label: 'Facing 3-bet', desc: 'You raised, opponent 3-bet',             actions: [2, 2] },
  { label: 'Check–check',  desc: 'Both players checked previous street',   actions: [0, 0] },
]

// ── Main ──────────────────────────────────────────────────────────────────────

const BOARD_COUNTS = [0, 3, 4, 5]

export default function RangeMatrix() {
  const [street, setStreet]         = useState<Street>(0)
  const [board, setBoard]           = useState<(number | null)[]>([null, null, null, null, null])
  const [pot, setPot]               = useState(6)
  const [toCall, setToCall]         = useState(2)
  const [historyIdx, setHistoryIdx] = useState(0)
  const [results, setResults]       = useState<Record<string, ActionProbs>>({})
  const [error, setError]           = useState<string | null>(null)
  const [pending, startTransition]  = useTransition()

  const nBoard      = BOARD_COUNTS[street]
  const visibleBoard = board.slice(0, nBoard).filter((c): c is number => c !== null)

  const setBoardCard = (i: number) => (c: number) =>
    setBoard(prev => { const next = [...prev]; next[i] = c; return next })

  const clearBoardCard = (i: number) => () =>
    setBoard(prev => { const next = [...prev]; next[i] = null; return next })

  const runMatrix = () => {
    setError(null)
    startTransition(async () => {
      try {
        const out: Record<string, ActionProbs> = {}
        const hands: { label: string; card0: number; card1: number }[] = []

        for (let ri = 12; ri >= 0; ri--) {
          for (let ci = 12; ci >= 0; ci--) {
            const suited = ri > ci
            const pair   = ri === ci
            const label  = pair ? RANKS[ri] + RANKS[ri] : handLabel(ri, ci, suited)
            const s1     = 0
            const s2     = suited ? 1 : 2
            const hi     = pair ? ri : Math.max(ri, ci)
            const lo     = pair ? ri : Math.min(ri, ci)
            const card0  = makeCard(hi, s1)
            const card1  = makeCard(lo, pair ? 1 : s2)
            if (visibleBoard.includes(card0) || visibleBoard.includes(card1)) continue
            hands.push({ label, card0, card1 })
          }
        }

        // Batch-encode all hands into a single [N, STATE_SIZE] tensor.
        // One session.run instead of N sequential calls — 10-50× faster.
        const actionHistory = HISTORY_PRESETS[historyIdx].actions
        const encodeInputs = hands.map(({ card0, card1 }) => ({
          holeCards:     [card0, card1] as [number, number],
          boardCards:    visibleBoard,
          street,
          pot,
          toCall,
          myStack:       200 - pot / 2,
          oppStack:      200 - pot / 2,
          actionHistory,
        }))

        const probs = await queryStrategyBatch(encodeInputs)

        for (let i = 0; i < hands.length; i++) {
          out[hands[i].label] = probs[i]
        }
        setResults({ ...out })
      } catch (e) {
        setError('Model failed to load — check /public/models/strategy_net.onnx')
        console.error(e)
      }
    })
  }

  const ranks = [...RANKS].reverse() // A K Q J T 9 8 7 6 5 4 3 2

  return (
    <div className="matrix-page">
      <div className="header">
        <h1 className="title">Range Matrix</h1>
        <p className="subtitle">GTO action frequencies across all 169 hand combinations</p>
      </div>

      <div className="section">
        <p className="section-label">Street</p>
        <div className="seg-row">
          {([0,1,2,3] as Street[]).map(s => (
            <button key={s} onClick={() => setStreet(s)} className={`seg-btn ${street === s ? 'active' : ''}`}>
              {STREET_NAMES[s]}
            </button>
          ))}
        </div>
      </div>

      <div className="section">
        <p className="section-label">Situation</p>
        <div className="seg-row">
          {HISTORY_PRESETS.map((p, i) => (
            <button
              key={i}
              onClick={() => setHistoryIdx(i)}
              className={`seg-btn ${historyIdx === i ? 'active' : ''}`}
              title={p.desc}
            >
              {p.label}
            </button>
          ))}
        </div>
        <p className="slider-desc" style={{ marginTop: 4 }}>{HISTORY_PRESETS[historyIdx].desc}</p>
      </div>

      {nBoard > 0 && (
        <div className="section">
          <p className="section-label">Board</p>
          <div className="flex gap-3 flex-wrap">
            {Array.from({ length: nBoard }).map((_, i) => (
              <BoardPicker
                key={i}
                value={board[i]}
                excluded={board.filter((c, j): c is number => c !== null && j !== i)}
                onChange={setBoardCard(i)}
                onClear={clearBoardCard(i)}
              />
            ))}
          </div>
        </div>
      )}

      <div className="matrix-controls">
        <div className="slider-grid" style={{ width: '100%' }}>
          <div className="slider-field">
            <div className="slider-header">
              <span className="slider-label">Pot</span>
              <span><span className="slider-value">{pot}</span><span className="slider-unit"> BB</span></span>
            </div>
            <input type="range" min={2} max={400} step={1} value={pot}
              style={{ '--pct': `${((pot-2)/398*100).toFixed(1)}%` } as React.CSSProperties}
              onChange={e => setPot(Number(e.target.value))} />
            <span className="slider-desc">Total chips currently in the middle.</span>
          </div>
          <div className="slider-field">
            <div className="slider-header">
              <span className="slider-label">To call</span>
              <span><span className="slider-value">{toCall}</span><span className="slider-unit"> BB</span></span>
            </div>
            <input type="range" min={0} max={200} step={1} value={toCall}
              style={{ '--pct': `${(toCall/200*100).toFixed(1)}%` } as React.CSSProperties}
              onChange={e => setToCall(Number(e.target.value))} />
            <span className="slider-desc">Amount to pay to stay in the hand.</span>
          </div>
        </div>
        <button onClick={runMatrix} disabled={pending} className="query-btn">
          {pending ? `[ Computing… ]` : '[ Generate matrix → ]'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="matrix-legend">
        {([
          ['#4a5a50', 'Fold / Check'],
          ['#22c55e', 'Call'],
          ['#f59e0b', 'Raise'],
          ['#a78bfa', 'All-in'],
        ] as [string, string][]).map(([color, label]) => (
          <div key={label} className="legend-item">
            <div className="legend-swatch" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
        <span className="legend-note">Cell height = action frequency</span>
      </div>

      {Object.keys(results).length > 0 ? (
        <div className="matrix-wrap">
          <div className="matrix-grid">
            <div className="matrix-corner" />
            {ranks.map(r => <div key={`h-${r}`} className="matrix-header">{r}</div>)}
            {ranks.map((rowRank, ri) => (
              <>
                <div key={`row-${rowRank}`} className="matrix-row-header">{rowRank}</div>
                {ranks.map((colRank, ci) => {
                  const r1    = RANKS.indexOf(rowRank)
                  const r2    = RANKS.indexOf(colRank)
                  const suited = ri < ci
                  const pair   = ri === ci
                  const label  = pair ? rowRank + colRank : handLabel(r1, r2, suited)
                  const probs  = results[label]
                  return (
                    <div
                      key={label}
                      className="matrix-cell"
                      title={probs
                        ? `${label}  fold ${(probs.fold*100).toFixed(0)}%  call ${(probs.call*100).toFixed(0)}%  raise ${(probs.raise*100).toFixed(0)}%  all-in ${(probs.allIn*100).toFixed(0)}%`
                        : label}
                      style={{ background: probs ? cellGradient(probs) : 'var(--bg3)' }}
                    >
                      <span className="cell-label">{label}</span>
                    </div>
                  )
                })}
              </>
            ))}
          </div>
        </div>
      ) : (
        !pending && (
          <div className="matrix-empty">
            Select a street and click Generate matrix to compute GTO frequencies for all 169 hands.
          </div>
        )
      )}
    </div>
  )
}
