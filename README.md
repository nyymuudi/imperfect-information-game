# Imperfect-Information Game Solving: From Kuhn to NLHE

A constructive complexity analysis demonstrating how Nash equilibrium computation scales across game complexity axes using Counterfactual Regret Minimization (CFR).

## Motivation

How does solving imperfect-information games become harder as you add cards, betting rounds, players, and information asymmetry? This project answers that question empirically by implementing the same domain-agnostic CFR solver across a progression of games — mirroring the historical trajectory of the academic literature from Kuhn (1950) through Libratus (Brown & Sandholm, 2017).

The key insight: **CFR is game-agnostic.** It doesn't know anything about poker. It operates on any extensive-form game with imperfect information through a generic interface, converging to a Nash equilibrium at a known rate. By scaling the game while holding the algorithm constant, we isolate exactly which complexity dimensions drive computational cost.

## Results

![Convergence comparison](convergence_comparison.png)

### Game Complexity Scaling

|  | Kuhn | Leduc | NLHE (8-bucket) |
|---|---|---|---|
| Cards | 3 | 6 (2 suits × 3 ranks) | 52 → 8 buckets |
| Betting rounds | 1 | 2 | 1 (preflop) |
| Information sets | 12 | 288 | 240 |
| Initial states | 6 | 120 | 64 |
| Terminal nodes | 30 | 5,880 | ~800 |
| CFR ms/iteration | 0.97 | 193 | 112 |
| Scaling factor | 1× | 199× | 116× |

### Solver Variant Comparison (Kuhn Poker)

|  | CFR | CFR+ | MCCFR |
|---|---|---|---|
| ms / iteration | 0.97 | 0.97 | 0.21 |
| Exploitability (10k iter) | 0.0029 | 0.0031 | 0.0096* |
| Convergence | Smooth | Smooth | Noisy |
| Tree traversal | Full | Full | Sampled |

*MCCFR at 50,000 iterations to match wall-clock time.

### Convergence Visualization

![Convergence comparison](convergence_comparison-1.png)

**Left — vs iterations:** CFR and CFR+ converge smoothly to ~0.003 exploitability in 10k iterations. MCCFR requires ~50k iterations for comparable quality, with visible sampling variance (shaded band = min/max across 5 random seeds).

**Center — vs wall-clock time:** With equivalent compute budget (~10s), CFR achieves lower exploitability than MCCFR on Kuhn Poker. This reverses on larger games where full tree traversal dominates.

**Right — cost per iteration:** MCCFR is 4.5× cheaper per iteration (0.21ms vs 0.97ms) because it samples one chance outcome and one opponent action per node instead of exhaustively expanding all branches.

### Kuhn Poker Nash Equilibrium

Kuhn Poker has a **family** of Nash equilibria parameterized by α ∈ [0, 1/3]:

| Player | Info Set | Strategy |
|---|---|---|
| P0 | J (initial) | Bet with probability α |
| P0 | Q (initial) | Always check |
| P0 | K (initial) | Bet with probability 3α |
| P1 | K (facing bet) | Always call |
| P1 | Q (facing bet) | Call with probability 1/3 |
| P1 | J (facing bet) | Always fold |
| P1 | K (after check) | Always bet |
| P1 | J (after check) | Bet with probability 1/3 |

**Key invariant:** β = 3α — King's value-bet frequency is always 3× Jack's bluff frequency. The solver discovers this ratio independently.

### Preflop NLHE — Abstracted Strategy

After 1,000 CFR iterations with 8 equity buckets (169 canonical hands → 8 clusters):

```
SB Opening Strategy:
  B0 (trash,  eq 0.32-0.38): fold 80%, raise 20%  ← bluff
  B1 (weak,   eq 0.38-0.45): fold 71%, raise 29%  ← bluff
  B2 (medium, eq 0.45-0.51): call 100%             ← limp
  B3 (medium, eq 0.51-0.57): call 99%              ← limp
  B4 (good,   eq 0.58-0.64): raise 91%             ← value
  B5 (strong, eq 0.65-0.70): raise 100%            ← value
  B6 (99/TT,  eq 0.73-0.76): call 62%, raise 38%  ← trap
  B7 (QQ+,    eq 0.80-0.84): call 45%, raise 54%  ← trap
```

The solver independently discovers the **polarized range structure** — the same pattern human professionals and GTO solvers use.

## Findings

### 1. Computational cost scales with tree size, not info set count

NLHE has fewer info sets (240) than Leduc (288) due to abstraction, yet costs 112ms/iter vs 193ms/iter. The difference comes from initial state count: Leduc pre-expands 120 chance combinations vs NLHE's 64 bucket pairs. **The bottleneck is tree traversal volume, not strategic complexity.**

### 2. Linear averaging is the single most impactful algorithmic improvement

Across all games and variants, weighting strategy accumulation by iteration number (Linear CFR, Brown & Sandholm 2019) improves convergence more than CFR+ regret clamping. The improvement is from O(1/√T) to O(1/T) — a theoretical guarantee, not an empirical artifact.

### 3. MCCFR's advantage is game-size-dependent

On Kuhn Poker (12 info sets, 30 terminals), MCCFR's 4.5× iteration speed advantage is overwhelmed by its variance penalty. Full tree traversal is so cheap that sampling noise hurts more than it helps. This tradeoff reverses on larger games where full traversal is prohibitively expensive — which is exactly why Libratus and Pluribus use MCCFR variants, not vanilla CFR.

### 4. Card abstraction enables tractability but introduces approximation error

NLHE's 169 canonical hands compressed to 8 equity buckets makes CFR feasible, but the solver can only distinguish 8 "hand strengths." A hand like ATs (bucket 5) plays identically to AKo despite different postflop potential. Finer abstractions (more buckets, equity distribution clustering) reduce this error at the cost of larger game trees.

### 5. The solver discovers known poker theory without domain knowledge

The CFR solver receives zero poker knowledge — no concept of "bluffing," "value betting," or "trapping." Yet it independently discovers:
- **Polarized ranges:** betting with the strongest AND weakest hands, checking the middle
- **β = 3α invariant:** value-to-bluff ratio of 3:1 (Kuhn)
- **Indifference principle:** mixing frequencies that make opponents indifferent between calling and folding

These emergent properties validate the solver's correctness and demonstrate that GTO play arises from mathematical structure, not human intuition.

## Architecture

```
src/
├── games/
│   ├── base.py              # Abstract ExtensiveFormGame interface
│   ├── kuhn.py              # Kuhn Poker (3 cards, 12 info sets)
│   ├── leduc.py             # Leduc Hold'em (6 cards, 288 info sets)
│   └── nlhe_preflop.py      # Preflop NLHE with abstraction (240 info sets)
├── solvers/
│   ├── cfr.py               # Vanilla CFR + Linear CFR + CFR+
│   └── mccfr.py             # External Sampling Monte Carlo CFR
├── abstraction/
│   ├── equity.py            # Monte Carlo equity calculator + hand evaluator
│   └── card_abstraction.py  # Equity-based hand clustering (169 → k buckets)
├── analysis/
│   ├── convergence.py       # Exploitability, Nash verification, tracking
│   └── convergence_benchmark.py  # Solver variant comparison & visualization
├── main.py                  # Kuhn CLI runner
└── nlhe_main.py             # NLHE CLI runner

tests/
├── test_kuhn_cfr.py         # 41 tests
├── test_leduc.py            # 30 tests
└── test_nlhe.py             # 43 tests
                               114 total
```

The `ExtensiveFormGame` abstract class ensures complete solver-game separation:

```python
class ExtensiveFormGame(ABC):
    def num_players(self) -> int: ...
    def initial_histories(self) -> list[tuple[History, float]]: ...
    def is_terminal(self, history) -> bool: ...
    def terminal_payoffs(self, history) -> tuple[float, ...]: ...
    def current_player(self, history) -> int: ...
    def info_set_key(self, history, player) -> InfoSetKey: ...
    def legal_actions(self, history) -> list[Action]: ...
    def apply_action(self, history, action) -> History: ...
```

Any game implementing this interface can be solved by any solver — the solver never accesses cards, ranks, or game-specific state.

## Theoretical Background

### Counterfactual Regret Minimization

CFR (Zinkevich et al., 2007) iteratively traverses the game tree, computes counterfactual values at each information set, and updates strategy proportional to accumulated positive regret. The average strategy converges to Nash at O(1/√T).

**CFR+** (Tammelin, 2014) clamps cumulative regrets to ≥ 0 after each iteration, preventing negative regret accumulation from polluting future strategies.

**Linear CFR** (Brown & Sandholm, 2019) weights strategy accumulation by iteration number, improving convergence to O(1/T).

**MCCFR** (Lanctot et al., 2009) samples chance outcomes and opponent actions instead of traversing the full tree. External Sampling MCCFR expands all traversing player actions but samples one opponent action per node.

### Abstraction

Full NLHE has ~10¹⁴ information sets. The Abstraction-Solving-Translation pipeline:
1. **Card abstraction:** 169 hands → k equity buckets via Monte Carlo equity clustering
2. **Action abstraction:** continuous sizing → discrete actions {fold, call, raise, all-in}
3. **Solve** the abstracted game
4. **Translate** bucket strategies back to specific hands

## Usage

```bash
pip install numpy pytest matplotlib

# Kuhn solver
python -m src.main --iterations 10000

# NLHE solver
python -m src.nlhe_main --buckets 8 --iterations 1000

# Convergence benchmark
python -m src.analysis.convergence_benchmark

# Tests (114 total)
pytest tests/ -v
```

## References

- Kuhn, H. W. (1950). "Simplified Two-Person Poker." *Contributions to the Theory of Games*.
- Southey, F. et al. (2005). "Bayes' Bluff: Opponent Modelling in Poker." *UAI*.
- Billings, D. et al. (2003). "Approximating Game-Theoretic Optimal Strategies for Full-scale Poker." *IJCAI*.
- Zinkevich, M. et al. (2007). "Regret Minimization in Games with Incomplete Information." *NIPS*.
- Lanctot, M. et al. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games." *NIPS*.
- Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+." *arXiv:1407.5042*.
- Brown, N. & Sandholm, T. (2017). "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals." *Science*.
- Brown, N. & Sandholm, T. (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization." *AAAI*.
- Brown, N. & Sandholm, T. (2019). "Superhuman AI for multiplayer poker." *Science*.

## License

MIT