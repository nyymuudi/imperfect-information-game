"""
PATCH: src/deep_cfr/deep_cfr_solver.py
=======================================
Lisää use_cpp_engine-tuki DeepCFRSolveriin.
Muutoksia on vain kolmessa kohdassa — merkitty # [PATCH] -kommenteilla.

Kopio nämä muutokset olemassa olevaan deep_cfr_solver.py-tiedostoon.
"""

# ── [PATCH 1] Lisää import tiedoston alkuun ───────────────────────────────────

# Lisää olemassa olevien importtien joukkoon:
from .cpp_backend import CppMCCFRBackend, engine_available  # [PATCH]


# ── [PATCH 2] DeepCFRSolver.__init__ — lisää parametri ───────────────────────

# Ennen (alkuperäinen signatuuri):
#   def __init__(self, game, n_traversals=500, buffer_capacity=1_000_000,
#                device="cpu", seed=42):

# Jälkeen:
def __init__(
    self,
    game,
    n_traversals: int = 500,
    buffer_capacity: int = 1_000_000,
    device: str = "cpu",
    seed: int = 42,
    use_cpp_engine: bool = False,   # [PATCH]
):
    self.game          = game
    self.n_traversals  = n_traversals
    self.device        = device

    # Verkot ja puskurit — ei muutoksia
    # self.regret_net    = RegretNetwork(...)
    # self.strategy_net  = StrategyNetwork(...)
    # self.regret_buf    = ReservoirBuffer(buffer_capacity)
    # self.strategy_buf  = ReservoirBuffer(buffer_capacity)
    # ...

    # [PATCH] C++-backend (valinnainen)
    self._cpp = None
    if use_cpp_engine:
        if engine_available():
            self._cpp = CppMCCFRBackend(
                n_traversals=n_traversals,
                regret_capacity=buffer_capacity,
                strategy_capacity=buffer_capacity,
                device=device,
                seed=seed,
            )
        else:
            import warnings
            warnings.warn(
                "use_cpp_engine=True mutta cfr_engine.so ei löydy — "
                "käytetään Python-traversalia. Buildaa: "
                "cd cpp_engine && bash scripts/build.sh",
                RuntimeWarning,
            )


# ── [PATCH 3] solve() tai train() -metodi ────────────────────────────────────

# Lisää jokaisen iteraation alkuun:
def _run_one_iteration(self, iteration: int):
    """Yksi Deep CFR -iteraatio — valitsee C++ tai Python traversalin."""

    if self._cpp is not None:
        # ── C++ polku (51× nopeampi) ──────────────────────────────────────
        reg_exp, str_exp = self._cpp.run_iteration(
            iteration,
            regret_net=self.regret_net,  # None ensimmäisissä iteraatioissa
        )

        X_r, a_r, v_r = self._cpp.to_tensors(reg_exp)
        X_s, a_s, v_s = self._cpp.to_tensors(str_exp)

        # Olemassa oleva verkkojen koulutus — ei muutoksia
        if len(X_r) > 0:
            self._train_regret_net(X_r, a_r, v_r)
        if len(X_s) > 0:
            self._train_strategy_net(X_s, a_s, v_s)

    else:
        # ── Python polku (alkuperäinen koodi) ─────────────────────────────
        for player in [0, 1]:
            self._external_sampling(player)         # olemassa oleva metodi
        self._train_networks()                       # olemassa oleva metodi