# Benchmarks

Kaikki mittaukset on tehty seuraavalla laitteistolla — lisää oma laitteistosi
kun ajat benchmarkit uudelleen.

## Laitteisto

| Kenttä         | Arvo                        |
|----------------|-----------------------------|
| Kone           | MacBook Air (M-sarja)       |
| CPU            | (täytä: `sysctl -n machdep.cpu.brand_string`) |
| RAM            | (täytä: `sysctl -n hw.memsize \| awk '{print $1/1024/1024/1024 " GB"}'`) |
| Python         | (täytä: `python3 --version`) |
| PyTorch        | (täytä: `python3 -c "import torch; print(torch.__version__)"`) |
| LibTorch       | sama kuin PyTorch            |
| Käyttöjärjestelmä | (täytä: `sw_vers`)       |

> **Huom:** MacBook Air -laitteissa terminen kuristus alkaa ~5 min jälkeen
> intensiivisessä laskennassa. Kaikki traversal-nopeudet mitattu ennen kuristusta.

---

## CFR-solverin nopeus (ms/iteraatio)

Ajot: `python3 src/analysis/convergence_benchmark.py`
Mediaani 5 toistosta, 95. prosenttipiste suluissa.

| Peli     | CFR   | CFR+  | MCCFR |
|----------|-------|-------|-------|
| Kuhn     | ?     | ?     | ?     |
| Leduc    | ?     | ?     | ?     |

Täytä ajamalla:
```bash
python3 -c "
import time, statistics
from src.games.kuhn import KuhnPoker
from src.solvers.cfr import CFRSolver

game = KuhnPoker()
times = []
for _ in range(5):
    solver = CFRSolver(game=game)
    t0 = time.perf_counter()
    solver.solve(iterations=1000)
    times.append((time.perf_counter() - t0) / 1000 * 1000)
print(f'CFR Kuhn: median={statistics.median(times):.3f}ms  '
      f'p95={sorted(times)[4]:.3f}ms  n=5')
"
```

---

## C++ MCCFR traversal-nopeus (trav/s)

Ajot: `python3 src/analysis/convergence_benchmark.py` tai manuaalisesti
`NLHEMCCFREngine.run_traversals_uniform()` + aikaistaminen.

| Backend                    | trav/s  | Muutos    |
|----------------------------|---------|-----------|
| Python baseline            | ?       | —         |
| C++ + Python callbacks     | ?       | ?×        |
| C++ + LibTorch             | ?       | ?×        |
| C++ + state-vector buffer  | ?       | ?×        |

> Aiemmat mittaukset (M1 MacBook Air, ennen termistä kuristusta):
> Python 219 trav/s → C++ + LibTorch 927 trav/s (4.2×)
> Nämä ovat yhden ajon arvoja — mittaa uudelleen omalla laitteistollasi.

---

## Konvergenssikäyrät

Ajotettu log-log-sovitus exploitability-datalle (kulmakerroin vastaa O()-rajaa).

| Solveri  | Peli  | Mitattu kulmakerroin | Teoreettinen |
|----------|-------|----------------------|--------------|
| CFR      | Kuhn  | ?                    | -0.5 (O(1/√T)) |
| CFR+     | Kuhn  | ?                    | -1.0 (O(1/T)) |
| Linear   | Kuhn  | ?                    | -1.0 (O(1/T)) |

Täytä ajamalla `python3 src/analysis/convergence_benchmark.py`
ja sovittamalla log-log-regressio `numpy.polyfit(log(iters), log(exploits), 1)`.

---

## Kuhn Pokerin peliarvon tarkkuus

Analyyttinen arvo: P0 EV = −1/18 ≈ −0.05556

| Iteraatiot | CFR EV  | Virhe   |
|------------|---------|---------|
| 1 000      | ?       | ?       |
| 5 000      | ?       | ?       |
| 10 000     | ?       | ?       |
| 20 000     | ?       | ?       |

Täytä `test_kuhn_cfr.py::TestNashVerification::test_game_value_converges_to_analytical`
-testillä eri iteraatiomäärillä.

---

*Päivitetty: täytä päivämäärä kun ajat benchmarkit*
