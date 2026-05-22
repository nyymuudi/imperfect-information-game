# Imperfect-Information Game Solving: From Kuhn to NLHE

A constructive complexity analysis demonstrating how Nash equilibrium computation scales across game complexity axes using Counterfactual Regret Minimization (CFR).

## Motivation

How does solving imperfect-information games become harder as you add cards, betting rounds, players, and information asymmetry? This project answers that question empirically by implementing the same domain-agnostic CFR solver across a progression of games — mirroring the historical trajectory of the academic literature from Kuhn (1950) through Libratus (Brown & Sandholm, 2017).

The key insight: **CFR is game-agnostic.** It doesn't know anything about poker. It operates on any extensive-form game with imperfect information through a generic interface, converging to a Nash equilibrium at a known rate. By scaling the game while holding the algorithm constant, we isolate exactly which complexity dimensions drive computational cost.

## Complexity Axes

| Axis | Minimum (Kuhn) | Maximum (NLHE) |
|---|---|---|
| Cards | 3 | 52 |
| Betting rounds | 1 | 4 |
| Actions per node | 2 (check/bet) | Continuous sizing |
| Players | 2 | 6+ |
| Information structure | 1 private card | 2 private + 5 community |
| Information sets | 12 | ~10¹⁴ (unabstracted) |

## Implementation Roadmap

- [x] **Tier 1 — Kuhn Poker** (3 cards, 12 info sets): Analytical Nash verification
- [ ] **Tier 2 — Leduc Hold'em** (6 cards, 2 rounds, ~936 info sets): CFR+/MCCFR comparison
- [ ] **Tier 3 — Preflop NLHE** (52 cards, abstraction required): Range visualization

## Theoretical Background

### Counterfactual Regret Minimization

CFR (Zinkevich et al., 2007) computes Nash equilibria for finite extensive-form games by iteratively:

1. Traversing the game tree for each player
2. Computing counterfactual values at each information set
3. Accumulating regret for actions not taken
4. Updating strategy proportional to positive regret (regret matching)

The **average strategy** over all iterations converges to Nash at rate O(1/√T) in exploitability.

This implementation uses **Linear CFR** (Brown & Sandholm, 2019), which weights strategy accumulation by iteration number, improving convergence to O(1/T).

### Exploitability

Exploitability measures distance from Nash equilibrium:

```
exploitability(σ) = Σᵢ max_{σ'ᵢ} uᵢ(σ'ᵢ, σ₋ᵢ) 
```

At Nash, exploitability = 0. Our implementation computes exact exploitability via full pure-strategy enumeration (tractable for small games; will require optimization for larger ones).

### Kuhn Poker Nash Equilibrium

Kuhn Poker has a **family** of Nash equilibria parameterized by α ∈ [0, 1/3]:

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

## Empirical Results (Kuhn Poker)

After 10,000 Linear CFR iterations:

```
Exploitability:  0.003
Info sets:       12
α (inferred):    0.235
β = 3α check:    K bets 0.694 ≈ 3 × 0.235 = 0.704  ✓

All 11 structural Nash properties verified ✓
```

Convergence profile:
```
  iter    500: exploit = 0.0106
  iter   2500: exploit = 0.0058
  iter   5000: exploit = 0.0042
  iter  10000: exploit = 0.0029
```

## Architecture

```
src/
├── games/
│   ├── base.py          # Abstract ExtensiveFormGame interface
│   └── kuhn.py          # Kuhn Poker (3 cards, 2 players, 12 info sets)
├── solvers/
│   └── cfr.py           # Vanilla CFR + Linear CFR averaging
├── analysis/
│   └── convergence.py   # Exploitability, Nash verification, convergence tracking
└── main.py              # CLI entry point

tests/
└── test_kuhn_cfr.py     # 41 tests: game mechanics → solver → Nash verification
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

# Run solver
python -m src.main --iterations 10000

# Run tests
pytest tests/ -v
```

## References

- Kuhn, H. W. (1950). "Simplified Two-Person Poker." *Contributions to the Theory of Games*, 1, 97–103.
- Zinkevich, M. et al. (2007). "Regret Minimization in Games with Incomplete Information." *NIPS*.
- Brown, N. & Sandholm, T. (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization." *AAAI*.
- Brown, N. & Sandholm, T. (2017). "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals." *Science*.
- Brown, N. & Sandholm, T. (2019). "Superhuman AI for multiplayer poker." *Science*.

## License

MIT
