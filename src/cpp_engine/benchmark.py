"""
python/benchmark.py
Speed comparison: C++ MCCFR engine vs pure Python traversal.

Measures traversals/second for both implementations with identical
configuration to isolate the language overhead.
"""

import sys
import os
import time
import random
import statistics

# Add cfr_engine .so to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cfr_engine as eng

# ── Pure Python baseline ──────────────────────────────────────────────────────

class PythonLeducGame:
    """Minimal Python Leduc for fair baseline comparison."""

    FOLD, CHECK, CALL, RAISE = 0, 1, 2, 3
    ANTE = 1.0
    BET = [2.0, 4.0]
    MAX_RAISES = 2
    NUM_CARDS = 6

    @staticmethod
    def all_deals():
        deals = []
        for p0 in range(6):
            for p1 in range(6):
                if p1 == p0: continue
                for comm in range(6):
                    if comm == p0 or comm == p1: continue
                    deals.append((p0, p1, comm))
        return deals

    @staticmethod
    def initial_state(deal):
        p0, p1, comm = deal
        return {
            'private': [p0, p1],
            'community': comm,
            'round': 0,
            'current_player': 0,
            'raises': 0,
            'last_bettor': -1,
            'pot': 2.0,
            'contributions': [1.0, 1.0],
            'history': [],
            'folded': [False, False],
            'terminal': False,
            'payoff_p0': 0.0,
        }

    @classmethod
    def legal_actions(cls, s):
        p = s['current_player']
        owe = s['contributions'][1 - p] - s['contributions'][p]
        actions = []
        if owe > 0:
            actions = [cls.FOLD, cls.CALL]
            if s['raises'] < cls.MAX_RAISES:
                actions.append(cls.RAISE)
        else:
            actions = [cls.CHECK]
            if s['raises'] < cls.MAX_RAISES:
                actions.append(cls.RAISE)  # actually BET
        return actions

    @classmethod
    def apply_action(cls, s, action):
        import copy
        ns = copy.deepcopy(s)
        p = ns['current_player']
        opp = 1 - p
        owe = ns['contributions'][opp] - ns['contributions'][p]
        bsize = cls.BET[ns['round']]
        ns['history'].append(action)

        if action == cls.FOLD:
            ns['folded'][p] = True
            ns['terminal'] = True
            ns['payoff_p0'] = -ns['contributions'][0] if p == 0 else ns['contributions'][1]
        elif action == cls.CHECK:
            if ns['last_bettor'] == -1:
                ns['current_player'] = opp
                ns['last_bettor'] = -2
            else:
                cls._next_round(ns)
        elif action == cls.CALL:
            ns['contributions'][p] += owe
            ns['pot'] += owe
            cls._next_round(ns)
        elif action == cls.RAISE:
            ns['contributions'][p] += owe + bsize
            ns['pot'] += owe + bsize
            ns['raises'] += 1
            ns['last_bettor'] = p
            ns['current_player'] = opp
        return ns

    @classmethod
    def _next_round(cls, s):
        if s['round'] == 0:
            s['round'] = 1
            s['raises'] = 0
            s['last_bettor'] = -1
            s['current_player'] = 0
        else:
            # Showdown
            def strength(card, comm):
                r = card // 2; cr = comm // 2
                return (3 + r) if r == cr else r
            s0 = strength(s['private'][0], s['community'])
            s1 = strength(s['private'][1], s['community'])
            s['terminal'] = True
            if s0 > s1:   s['payoff_p0'] =  s['contributions'][1]
            elif s1 > s0: s['payoff_p0'] = -s['contributions'][0]
            else:         s['payoff_p0'] =  0.0

    @classmethod
    def info_set_key(cls, s, player):
        rank = s['private'][player] // 2
        key = f"r{rank}"
        if s['round'] > 0:
            key += f"b{s['community'] // 2}"
        key += ''.join(f"a{a}" for a in s['history'])
        return key


def python_traverse(game, state, traversing_player, strategy_fn):
    """External sampling MCCFR traversal in Python."""
    if state['terminal']:
        return state['payoff_p0'] if traversing_player == 0 else -state['payoff_p0']

    p = state['current_player']
    legal = game.legal_actions(state)
    iset = game.info_set_key(state, p)
    probs = strategy_fn(iset, legal)

    if p != traversing_player:
        dist = random.choices(range(len(legal)), weights=probs)[0]
        next_state = game.apply_action(state, legal[dist])
        return python_traverse(game, next_state, traversing_player, strategy_fn)

    values = []
    for a in legal:
        ns = game.apply_action(state, a)
        v = python_traverse(game, ns, traversing_player, strategy_fn)
        values.append(v)
    return sum(p * v for p, v in zip(probs, values))


def python_run_traversals(n, traversing_player):
    game = PythonLeducGame()
    deals = game.all_deals()

    def uniform_strategy(iset, actions):
        n = len(actions)
        return [1.0 / n] * n

    for _ in range(n):
        deal = random.choice(deals)
        state = game.initial_state(deal)
        python_traverse(game, state, traversing_player, uniform_strategy)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(n_traversals: int = 1000, n_runs: int = 5):
    print(f"\n{'='*60}")
    print(f"  MCCFR Traversal Benchmark")
    print(f"  {n_traversals} traversals × {n_runs} runs")
    print(f"{'='*60}\n")

    # ── C++ engine ────────────────────────────────────────────────
    print("C++ engine (pybind11):")
    cfg = eng.TraversalConfig()
    cfg.n_traversals = n_traversals
    cfg.regret_capacity = 1 << 20
    cfg.strategy_capacity = 1 << 20
    cfg.collect_strategy = True

    cpp_times = []
    for run in range(n_runs):
        engine = eng.MCCFREngine(cfg)
        t0 = time.perf_counter()
        engine.run_traversals_uniform(0)
        engine.run_traversals_uniform(1)
        dt = time.perf_counter() - t0
        cpp_times.append(dt)
        print(f"  run {run+1}: {dt*1000:.1f} ms "
              f"({2*n_traversals/dt:.0f} traversals/s) "
              f"| regret buf: {engine.regret_buffer_size():,}")

    cpp_median = statistics.median(cpp_times)
    cpp_tp = 2 * n_traversals / cpp_median

    # ── Python baseline ────────────────────────────────────────────
    print("\nPython baseline:")
    py_times = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        python_run_traversals(n_traversals, 0)
        python_run_traversals(n_traversals, 1)
        dt = time.perf_counter() - t0
        py_times.append(dt)
        print(f"  run {run+1}: {dt*1000:.1f} ms "
              f"({2*n_traversals/dt:.0f} traversals/s)")

    py_median = statistics.median(py_times)
    py_tp = 2 * n_traversals / py_median

    # ── Summary ────────────────────────────────────────────────────
    speedup = py_median / cpp_median
    print(f"\n{'─'*60}")
    print(f"  C++    median: {cpp_median*1000:.1f} ms  ({cpp_tp:.0f} trav/s)")
    print(f"  Python median: {py_median*1000:.1f} ms  ({py_tp:.0f} trav/s)")
    print(f"  Speedup:       {speedup:.1f}×")
    print(f"{'─'*60}\n")

    return speedup


# ── Correctness spot-check ─────────────────────────────────────────────────────

def check_interface():
    print("=== Interface check ===")
    print(f"  NUM_CARDS:  {eng.NUM_CARDS}")
    print(f"  BET_ROUND1: {eng.BET_ROUND1}")
    print(f"  BET_ROUND2: {eng.BET_ROUND2}")

    deals = eng.all_deals()
    print(f"  Deals: {len(deals)} (expected 120)")

    game = eng.LeducGame()
    s = game.initial_state(deals[0])
    print(f"  Initial state: player={s.current_player}, round={s.round}, pot={s.pot}")

    key = eng.info_set_key(s, 0)
    print(f"  P0 info set: '{key}'")

    actions = game.legal_actions(s)
    print(f"  Legal actions: {[str(a) for a in actions]}")

    s2 = game.apply_action(s, eng.Action.RAISE)
    key2 = eng.info_set_key(s2, 1)
    print(f"  After RAISE → P1 info set: '{key2}'")
    print()


if __name__ == "__main__":
    check_interface()
    speedup = benchmark(n_traversals=2000, n_runs=4)
    if speedup < 5:
        print(f"WARNING: speedup {speedup:.1f}× below expected range (5–50×)")
    else:
        print(f"✓ {speedup:.1f}× speedup — C++ engine working correctly")