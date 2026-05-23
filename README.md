# Imperfect-Information Game Solving: From Kuhn to NLHE

A constructive complexity analysis demonstrating how Nash equilibrium computation scales across game complexity axes using Counterfactual Regret Minimization (CFR).

## Motivation

How does solving imperfect-information games become harder as you add cards, betting rounds, players, and information asymmetry? This project answers that question empirically by implementing the same domain-agnostic CFR solver across a progression of games — mirroring the historical trajectory of the academic literature from Kuhn (1950) through Libratus (Brown & Sandholm, 2017).

The key insight: **CFR is game-agnostic.** It doesn't know anything about poker. It operates on any extensive-form game with imperfect information through a generic interface, converging to a Nash equilibrium at a known rate. By scaling the game while holding the algorithm constant, we isolate exactly which complexity dimensions drive computational cost.

## Empirical Results

### Complexity Scaling

|  | Kuhn | Leduc | NLHE (8-bucket) |
|---|---|---|---|
| Cards | 3 | 6 (2 suits × 3 ranks) | 52 → 8 buckets |
| Betting rounds | 1 | 2 | 1 (preflop) |
| Information sets | 12 | 288 | 240 |
| Initial states | 6 | 120 | 64 |
| Terminal nodes | 30 | 5,880 | ~800 |
| ms / CFR iteration | 0.74 | 193 | 112 |
| Scaling factor | 1× | 261× | 152× |

### Kuhn Poker — Analytical Nash Verification

After 10,000 Linear CFR iterations:

```
Exploitability:  0.003
Info sets:       12
α (inferred):    0.235
β = 3α check:    K bets 0.694 ≈ 3 × 0.235 = 0.704  ✓
All 11 structural Nash properties verified ✓
```

Kuhn Poker has a family of Nash equilibria parameterized by α ∈ [0, 1/3]:

| Player | Info Set | Strategy |
|---|---|---|
| P0 | J (initial) | Bet with probability α |
| P0 | Q (initial) | Always check |
| P0 | K (initial) | Bet with probability 3α |
| P0 | K (check-bet) | Always call |
| P0 | Q (check-bet) | Indifferent (call prob makes J indifferent) |
| P0 | J (check-bet) | Always fold |
| P1 | K (facing bet) | Always call |
| P1 | Q (facing bet) | Call with probability 1/3 |
| P1 | J (facing bet) | Always fold |
| P1 | K (after check) | Always bet |
| P1 | Q (after check) | Never bet |
| P1 | J (after check) | Bet with probability 1/3 |

**Key invariant:** β = 3α (King's bet frequency is always 3× Jack's bluff frequency).

**Game value:** P0 EV = −1/18 ≈ −0.0556 (first-mover disadvantage from information asymmetry).

### Preflop NLHE — Abstracted Strategy

After 1,000 iterations with 8 equity buckets:

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

The solver independently discovers the **polarized range structure** — bluffing with the weakest hands, value-raising with strong hands, limping with medium hands, and trapping with premiums.

## Theoretical Background

### Counterfactual Regret Minimization

CFR (Zinkevich et al., 2007) computes Nash equilibria for finite extensive-form games by iteratively:

1. Traversing the game tree for each player
2. Computing counterfactual values at each information set
3. Accumulating regret for actions not taken
4. Updating strategy proportional to positive regret (regret matching)

The **average strategy** over all iterations converges to Nash at rate O(1/√T) in exploitability.

This implementation uses **Linear CFR** (Brown & Sandholm, 2019), which weights strategy accumulation by iteration number, improving convergence to O(1/T).

### Abstraction (NLHE)

Full NLHE has ~10¹⁴ information sets — intractable for vanilla CFR. The standard Abstraction-Solving-Translation pipeline:

1. **Card abstraction:** 169 canonical preflop hands → k equity buckets via equal-width binning on Monte Carlo equity vs random opponent
2. **Action abstraction:** continuous bet sizing → discrete actions (fold, call, raise, all-in)
3. **Solve** the abstracted game with CFR
4. **Translate** back: bucket strategies map to specific hands

### Exploitability

Exploitability measures distance from Nash equilibrium:

```
exploitability(σ) = Σᵢ max_{σ'ᵢ} uᵢ(σ'ᵢ, σ₋ᵢ)
```

At Nash, exploitability = 0. Small games use exact enumeration of all pure strategies; larger games use tree-walk best response.

## Architecture

```
src/
├── games/
│   ├── base.py              # Abstract ExtensiveFormGame interface
│   ├── kuhn.py              # Kuhn Poker (3 cards, 12 info sets)
│   ├── leduc.py             # Leduc Hold'em (6 cards, 288 info sets)
│   └── nlhe_preflop.py      # Preflop NLHE with abstraction (240 info sets)
├── solvers/
│   └── cfr.py               # Vanilla CFR + Linear CFR averaging
├── abstraction/
│   ├── equity.py            # Monte Carlo equity calculator + hand evaluator
│   └── card_abstraction.py  # Equity-based hand clustering (169 → k buckets)
├── analysis/
│   └── convergence.py       # Exploitability, Nash verification, convergence tracking
├── main.py                  # Kuhn CLI runner
└── nlhe_main.py             # NLHE CLI runner

tests/
├── test_kuhn_cfr.py         # 41 tests: game mechanics → solver → Nash verification
├── test_leduc.py            # 30 tests: tree structure → mechanics → info sets → CFR
└── test_nlhe.py             # 43 tests: hand evaluator → equity → mechanics → CFR
```

The `ExtensiveFormGame` abstract class defines the full interface:

```python
class ExtensiveFormGame(ABC):
    def num_players(self) -> int: ...
    def initial_histories(self) -> list[tuple[History, float]]: ...
    def is_terminal(self, history: History) -> bool: ...
    def terminal_payoffs(self, history: History) -> tuple[float, ...]: ...
    def current_player(self, history: History) -> int: ...
    def info_set_key(self, history: History, player: int) -> InfoSetKey: ...
    def legal_actions(self, history: History) -> list[Action]: ...
    def apply_action(self, history: History, action: Action) -> History: ...
```

Any game implementing this interface can be solved by any solver — complete domain-agnosticism.

## Usage

```bash
# Install dependencies
pip install numpy pytest

# Run Kuhn solver
python -m src.main --iterations 10000

# Run NLHE solver
python -m src.nlhe_main --buckets 8 --iterations 1000

# Run all tests (114 total)
pytest tests/ -v
```

## References

- Kuhn, H. W. (1950). "Simplified Two-Person Poker." *Contributions to the Theory of Games*, 1, 97–103.
- Southey, F. et al. (2005). "Bayes' Bluff: Opponent Modelling in Poker." *UAI*.
- Zinkevich, M. et al. (2007). "Regret Minimization in Games with Incomplete Information." *NIPS*.
- Billings, D. et al. (2003). "Approximating Game-Theoretic Optimal Strategies for Full-scale Poker." *IJCAI*.
- Brown, N. & Sandholm, T. (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization." *AAAI*.
- Brown, N. & Sandholm, T. (2017). "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals." *Science*.
- Brown, N. & Sandholm, T. (2019). "Superhuman AI for multiplayer poker." *Science*.

## License

MIT
