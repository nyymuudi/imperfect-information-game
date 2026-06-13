'use client'

import { useState, useCallback, useTransition, useEffect, useMemo } from 'react'
import { RANKS, SUITS, makeCard, cardLabel, cardRank, cardSuit, STARTING_STACK as STACK_SIZE } from '@/lib/onnx/encoder'
import { queryStrategy, type ActionProbs } from '@/lib/onnx/session'
import { STREET_NAMES, type Street } from '@/types'
import { createSafeSubgameSolver, handKey, type SubgameStrategy } from '@/lib/cfr'
import type { NLHEHistory } from '@/lib/cfr'

// ── Constants ─────────────────────────────────────────────────────────────────

const STARTING_STACK = STACK_SIZE

const ACTION_META: Record<keyof ActionProbs, { label: string; color: string; desc: string }> = {
  fold:  { label: 'Fold / Check', color: '#7a8a80', desc: 'Surrender your hand or pass without betting' },
  call:  { label: 'Call',         color: '#22c55e', desc: "Match the opponent's bet to stay in the hand" },
  raise: { label: 'Raise',        color: '#f59e0b', desc: 'Increase the bet (75% of the pot)' },
  allIn: { label: 'All-in',       color: '#a78bfa', desc: 'Commit all remaining chips to the pot' },
}

const STREET_INFO: Record<Street, { cards: number; desc: string }> = {
  0: { cards: 0, desc: 'Two private cards dealt. No community cards yet.' },
  1: { cards: 3, desc: 'Three community cards on the board.' },
  2: { cards: 4, desc: 'Fourth community card (turn) revealed.' },
  3: { cards: 5, desc: 'Fifth and final card (river) on the board.' },
}

// ── Card picker ───────────────────────────────────────────────────────────────

function CardPicker({ label, value, excluded, onChange }: {
  label: string; value: number | null; excluded: number[]; onChange: (c: number) => void
}) {
  const [open, setOpen] = useState(false)
  const isRed = (si: number) => si === 1 || si === 2

  return (
    <div className="relative">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <button
        onClick={() => setOpen(v => !v)}
        className={`card-btn ${value !== null ? 'selected' : ''}`}
      >
        {value !== null ? (
          <span style={{ color: isRed(cardSuit(value)) ? '#ef4444' : undefined }}>
            {cardLabel(value)}
          </span>
        ) : '—'}
      </button>
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
                  <span style={{ color: isRed(si) ? '#ef4444' : undefined }}>
                    {r}{s}
                  </span>
                </button>
              )
            })
          })}
        </div>
      )}
    </div>
  )
}

// ── Slider ────────────────────────────────────────────────────────────────────

function Slider({ label, value, min, max, step = 1, unit = 'BB', desc, onChange }: {
  label: string; value: number; min: number; max: number
  step?: number; unit?: string; desc: string; onChange: (v: number) => void
}) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="slider-field">
      <div className="slider-header">
        <span className="slider-label">{label}</span>
        <span>
          <span className="slider-value">{value}</span>
          <span className="slider-unit">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--pct': `${pct}%` } as React.CSSProperties}
        onChange={e => onChange(Number(e.target.value))}
      />
      <span className="slider-desc">{desc}</span>
    </div>
  )
}

// ── Probability bar ───────────────────────────────────────────────────────────

function ProbBar({ action, value }: { action: keyof ActionProbs; value: number }) {
  const { label, color, desc } = ACTION_META[action]
  const pct = (value * 100).toFixed(1)

  return (
    <div className="prob-row">
      <div className="prob-header">
        <span className="prob-label" style={{ color }}>{label}</span>
        <span className="prob-pct" style={{ color }}>{pct}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="prob-desc">{desc}</span>
    </div>
  )
}

// ── Poker diagram SVG ─────────────────────────────────────────────────────────

function PokerDiagram({ pot, toCall, myStack, street }: {
  pot: number; toCall: number; myStack: number; street: Street
}) {
  return (
    <svg width="100%" viewBox="0 0 320 140" style={{ display: 'block', margin: '4px 0 8px' }}>
      {/* Table */}
      <ellipse cx="160" cy="70" rx="130" ry="55" fill="#1c2420" stroke="#2a3a30" strokeWidth="1"/>
      <ellipse cx="160" cy="70" rx="115" ry="42" fill="none" stroke="#2a3a30" strokeWidth="0.5" strokeDasharray="4 3"/>

      {/* Pot label */}
      <text x="160" y="66" textAnchor="middle" fill="#22c55e" fontSize="13" fontFamily="IBM Plex Mono" fontWeight="600">{pot} BB</text>
      <text x="160" y="80" textAnchor="middle" fill="#4a5a50" fontSize="9" fontFamily="IBM Plex Mono">POT</text>

      {/* Hero (bottom) */}
      <rect x="120" y="112" width="80" height="22" fill="#1c2420" stroke="#2a3a30" strokeWidth="1"/>
      <text x="160" y="127" textAnchor="middle" fill="#e2e8e4" fontSize="10" fontFamily="IBM Plex Mono">HERO · {myStack}BB</text>

      {/* Villain (top) */}
      <rect x="120" y="6" width="80" height="22" fill="#1c2420" stroke="#2a3a30" strokeWidth="1"/>
      <text x="160" y="21" textAnchor="middle" fill="#7a8a80" fontSize="10" fontFamily="IBM Plex Mono">VILLAIN</text>

      {/* To call arrow */}
      {toCall > 0 && (
        <>
          <line x1="160" y1="112" x2="160" y2="90" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 2"/>
          <text x="168" y="104" fill="#f59e0b" fontSize="9" fontFamily="IBM Plex Mono">{toCall}BB</text>
        </>
      )}

      {/* Street indicator */}
      <text x="8" y="78" fill="#4a5a50" fontSize="9" fontFamily="IBM Plex Mono" textAnchor="start">
        {STREET_NAMES[street].toUpperCase()}
      </text>
    </svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StrategyExplorer() {
  const [hole0, setHole0]   = useState<number | null>(null)
  const [hole1, setHole1]   = useState<number | null>(null)
  const [board, setBoard]   = useState<(number | null)[]>([null, null, null, null, null])
  const [street, setStreet] = useState<Street>(0)
  const [pot, setPot]       = useState(6)
  const [toCall, setToCall] = useState(2)
  const [myStack, setMyStack] = useState(STARTING_STACK - 2)
  const [probs, setProbs]   = useState<ActionProbs | null>(null)
  const [error, setError]   = useState<string | null>(null)
  const [pending, startTransition] = useTransition()

  // ── Subgame solver state ────────────────────────────────────────────────────
  const [refinedProbs, setRefinedProbs] = useState<ActionProbs | null>(null)
  const [refineStatus, setRefineStatus] = useState<string | null>(null)
  const [refining, setRefining] = useState(false)

  // Cached solver instance — built once per session.
  const solver = useMemo(() => createSafeSubgameSolver({
    startingStack: STARTING_STACK,
    maxRaisesPerStreet: 1,
    raiseFraction: 0.75,
  }), [])

  const visibleBoardCount = STREET_INFO[street].cards

  const setBoard0 = useCallback((i: number) => (c: number) => {
    setBoard(prev => { const next = [...prev]; next[i] = c; return next })
  }, [])

  /**
   * Build a uniform opponent range (all 1326 hole-card combos minus blockers).
   * MVP — future versions could let the user customise the range.
   */
  const buildUniformOppRange = useCallback((blockers: number[]): Map<string, number> => {
    const range = new Map<string, number>()
    const blockSet = new Set(blockers)
    let combos = 0
    for (let i = 0; i < 52; i++) {
      if (blockSet.has(i)) continue
      for (let j = i + 1; j < 52; j++) {
        if (blockSet.has(j)) continue
        range.set(handKey([i, j]), 1.0)
        combos++
      }
    }
    // Normalise to 1/combos so probabilities sum to 1.
    for (const k of range.keys()) range.set(k, 1.0 / combos)
    return range
  }, [])

  const handleRefine = async () => {
    if (hole0 === null || hole1 === null) {
      setError('Select both hole cards before refining')
      return
    }
    if (visibleBoardCount > 0 &&
        board.slice(0, visibleBoardCount).some(c => c === null)) {
      setError('Fill all visible board cards before refining')
      return
    }
    setError(null)
    setRefining(true)
    setRefineStatus('Setting up subgame…')

    try {
      // Build a full 5-card board: pad with deterministic unseen cards so the
      // subgame solver has something to enumerate. We don't actually use these
      // for showdown unless the spot reaches river.
      const visible = board
        .slice(0, visibleBoardCount)
        .filter((c): c is number => c !== null)
      const used = new Set<number>([hole0, hole1, ...visible])
      const fullBoard: number[] = [...visible]
      for (let c = 0; c < 52 && fullBoard.length < 5; c++) {
        if (!used.has(c)) {
          fullBoard.push(c)
          used.add(c)
        }
      }

      // Hero placeholder opp cards — substituted by SubgameGame per deal.
      const oppPlaceholder: [number, number] = [
        fullBoard[fullBoard.length - 2], fullBoard[fullBoard.length - 1],
      ]

      const rootHistory: NLHEHistory = [
        [hole0, hole1],
        oppPlaceholder,
        fullBoard,
      ] as const

      const heroRange = new Map([[handKey([hole0, hole1]), 1.0]])
      const oppRange  = buildUniformOppRange([hole0, hole1, ...visible])

      setRefineStatus(`Solving (${oppRange.size} opp combos)…`)
      const refined = await solver.solve({
        rootHistory,
        heroPlayer: 0,
        heroRange,
        opponentRange: oppRange,
        iterations: 50,    // browser-friendly default; raise for more precision
        maxDeals: 100,
        onProgress: (it) => setRefineStatus(`CFR iter ${it}/50…`),
      })

      if (refined === null) {
        setError('Subgame had no valid deals (card conflicts)')
        setRefinedProbs(null)
      } else {
        // Read hero's strategy at the root.
        const probs = refined.query(rootHistory, 0)
        // Map back to ActionProbs. Legal actions on first decision:
        // [check, raise?, all-in?] (no fold facing zero bet).
        const result: ActionProbs = {
          fold:  0, call:  0, raise: 0, allIn: 0,
        }
        // First legal action position depends on toCall>0 (face bet) or 0 (open).
        // For MVP assume open spot (toCall=0): probs ordering = [check, raise, allIn]
        const keys: Array<keyof ActionProbs> = ['fold', 'raise', 'allIn']
        for (let i = 0; i < probs.length && i < keys.length; i++) {
          result[keys[i]] = probs[i]
        }
        setRefinedProbs(result)
        setRefineStatus(`Solved — ${refined.size()} info sets`)
      }
    } catch (e) {
      setError(`Subgame solver failed: ${(e as Error).message}`)
      console.error(e)
      setRefinedProbs(null)
      setRefineStatus(null)
    } finally {
      setRefining(false)
    }
  }

  const handleQuery = () => {
    if (hole0 === null || hole1 === null) {
      setError('Select both hole cards before analysing')
      return
    }
    setError(null)
    startTransition(async () => {
      try {
        const result = await queryStrategy({
          holeCards: [hole0, hole1],
          boardCards: board.slice(0, visibleBoardCount).filter((c): c is number => c !== null),
          street,
          pot,
          toCall,
          myStack,
          oppStack: STARTING_STACK - (pot / 2),
          actionHistory: [],
        })
        setProbs(result)
      } catch (e) {
        setError('Model failed to load — check /public/models/strategy_net.onnx')
        console.error(e)
      }
    })
  }

  return (
    <div className="explorer">

      {/* Header */}
      <div className="header">
        <h1 className="title">CFR // Strategy Explorer</h1>
        <p className="subtitle">Deep CFR · HU NLHE · {STARTING_STACK}BB · 75% pot · Neural inference</p>
      </div>

      {/* Street */}
      <div className="section">
        <p className="section-label">Street</p>
        <div className="seg-row">
          {([0, 1, 2, 3] as Street[]).map(s => (
            <button key={s} onClick={() => setStreet(s)} className={`seg-btn ${street === s ? 'active' : ''}`}>
              {STREET_NAMES[s]}
            </button>
          ))}
        </div>
        <div className="info-box">
          <strong>{STREET_NAMES[street]}:</strong> {STREET_INFO[street].desc}
        </div>
      </div>

      {/* Hole cards */}
      <div className="section">
        <p className="section-label">Hole cards</p>
        <div className="info-box">
          <strong>Your private cards</strong> — only you see these. The network encodes them into a 52-dimensional one-hot vector and estimates your preflop equity against a random hand.
        </div>
        <div className="flex gap-3 flex-wrap">
          <CardPicker
            label="Card 1"
            value={hole0}
            excluded={[hole1, ...board].filter((c): c is number => c !== null)}
            onChange={setHole0}
          />
          <CardPicker
            label="Card 2"
            value={hole1}
            excluded={[hole0, ...board].filter((c): c is number => c !== null)}
            onChange={setHole1}
          />
        </div>
      </div>

      {/* Board */}
      {visibleBoardCount > 0 && (
        <div className="section">
          <p className="section-label">Board cards</p>
          <div className="info-box">
            <strong>Community cards</strong> — visible to all players. The network uses these to estimate board strength.
          </div>
          <div className="flex gap-3 flex-wrap">
            {Array.from({ length: visibleBoardCount }).map((_, i) => (
              <CardPicker
                key={i}
                label={`Board ${i + 1}`}
                value={board[i]}
                excluded={[hole0, hole1, ...board.filter((_, j) => j !== i)].filter((c): c is number => c !== null)}
                onChange={setBoard0(i)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Betting state */}
      <div className="section">
        <p className="section-label">Betting state</p>
        <PokerDiagram pot={pot} toCall={toCall} myStack={myStack} street={street} />
        <div className="slider-grid">
          <Slider
            label="Pot"
            value={pot}
            min={2}
            max={400}
            unit=" BB"
            desc="Total chips currently in the middle. Starts at 3BB (SB 1 + BB 2). The network normalises this by 2× the starting stack."
            onChange={setPot}
          />
          <Slider
            label="To call"
            value={toCall}
            min={0}
            max={STARTING_STACK}
            unit=" BB"
            desc="How much you need to pay to stay in the hand. 0 means no bet to face. Directly influences the fold vs. call decision."
            onChange={setToCall}
          />
          <Slider
            label="My stack"
            value={myStack}
            min={1}
            max={STARTING_STACK}
            unit=" BB"
            desc="Your remaining chips. A deeper stack increases the significance of the all-in decision on later streets."
            onChange={setMyStack}
          />
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <button onClick={handleQuery} disabled={pending} className="query-btn">
        {pending ? '[ Running… ]' : '[ Analyse hand → ]'}
      </button>

      {probs && (
        <div className="section">
          <p className="section-label">GTO strategy</p>
          <div className="info-box" style={{ marginBottom: 8 }}>
            Probabilities represent an approximation of the <strong>Nash equilibrium</strong> — a strategy your opponent cannot exploit regardless of how they play.
          </div>
          <div className="results">
            {(Object.entries(probs) as [keyof ActionProbs, number][]).map(([action, value]) => (
              <ProbBar key={action} action={action} value={value} />
            ))}
          </div>

          {/* Subgame solver refinement panel */}
          <div style={{ marginTop: 24 }}>
            <button
              onClick={handleRefine}
              disabled={refining}
              className="query-btn"
              style={{ background: '#5b21b6' }}
            >
              {refining ? `[ ${refineStatus ?? 'Refining…'} ]` : '[ Refine with subgame solver → ]'}
            </button>
            <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 8 }}>
              Runs in-browser CFR (50 iters) over a gadget game with safety
              guarantee. Computes a strategy at least as non-exploitable as the
              blueprint vs. a uniform opponent range. ~10-30 seconds.
            </p>

            {refinedProbs && (
              <div style={{ marginTop: 16 }}>
                <p className="section-label">Subgame-refined strategy</p>
                <div className="info-box" style={{ marginBottom: 8 }}>
                  <strong>Safe re-solve</strong> via Moravčík et al. (2016) gadget
                  game. The opponent gets an opt-out option whose payoff equals
                  their blueprint EV — guaranteeing the refined hero strategy is
                  at least as good as the blueprint (Burch et al. 2014, Thm. 1).
                </div>
                <div className="results">
                  {(Object.entries(refinedProbs) as [keyof ActionProbs, number][])
                    .map(([action, value]) => (
                      <ProbBar key={action} action={action} value={value} />
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* About */}
      <div className="section">
        <p className="section-label">About this project</p>
        <div className="about-intro">
          What does a mathematically optimal poker strategy look like — and how do you
          compute it without any human expertise? This project builds a self-learning
          algorithm that figures it out from scratch, starting from a simple 3-card
          game and scaling up to full No-Limit Hold&apos;em.
        </div>
        <div className="about-grid">
          <div className="about-card">
            <h3 className="about-title">Learning by regretting</h3>
            <p className="about-text">
              The algorithm plays against itself millions of times and keeps track
              of one thing: regret. After every hand it asks — <em>"what would have
              happened if I had bet instead of checked?"</em> Over time it plays the
              actions it regrets skipping the most. No poker rules were hard-coded;
              bluffing and value-betting emerge entirely on their own.
            </p>
          </div>
          <div className="about-card">
            <h3 className="about-title">Why a neural network?</h3>
            <p className="about-text">
              A simple lookup table would need a separate entry for every possible
              game situation. In real poker that&apos;s more entries than atoms in the
              observable universe. A neural network sidesteps this by learning
              patterns — it generalises from situations it has seen to ones it
              hasn&apos;t, the same way a human learns to play new hands from experience.
            </p>
          </div>
          <div className="about-card">
            <h3 className="about-title">What is a Nash equilibrium?</h3>
            <p className="about-text">
              A strategy where neither player can do better by changing their
              approach — even if they know exactly what the opponent is doing.
              Think of it as a perfectly balanced strategy: unpredictable enough
              that no one can exploit it, yet rational enough that it doesn&apos;t
              throw away value. This is what the algorithm converges towards.
            </p>
          </div>
          <div className="about-card">
            <h3 className="about-title">From toy games to real poker</h3>
            <p className="about-text">
              The project starts with Kuhn Poker — a 3-card game solvable in
              milliseconds — and scales up to full No-Limit Hold&apos;em. Each step
              adds cards, betting rounds, or players, revealing exactly what makes
              the problem harder. The same algorithm runs on all of them; only the
              computational cost changes.
            </p>
          </div>
          <div className="about-card">
            <h3 className="about-title">Making it fast</h3>
            <p className="about-text">
              Training requires hundreds of millions of simulated hands. The core
              engine is written in C++ and runs on the GPU, cutting each training
              round from hours to minutes on a standard laptop. The trained strategy
              is then compressed into an ONNX model that runs inference directly
              in your browser — no server involved.
            </p>
          </div>
          <div className="about-card">
            <h3 className="about-title">How to use this tool</h3>
            <p className="about-text">
              Select your two hole cards, set the street and pot size, then hit
              Analyse. The network returns the GTO probability for each action —
              how often a theoretically optimal player would fold, call, raise,
              or go all-in in that exact situation. Higher iterations of training
              produce sharper, more reliable recommendations.
            </p>
          </div>
        </div>
        <a
          href="https://github.com/nyymuudi/imperfect-information-game"
          target="_blank"
          rel="noopener noreferrer"
          className="github-link"
        >
          <span>⌥</span> View source on GitHub →
        </a>
      </div>

    </div>
  )
}
