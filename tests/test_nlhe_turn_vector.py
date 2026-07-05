"""Turn depth-limited vector CFR: all-in equity identity + smoke."""

import numpy as np
import pytest
import torch

from src.games.postflop_nlhe import PostflopNLHE
from src.search.cfv_net import CFVNet
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS
from src.search.nlhe_river_vector import COMBOS, N_COMBOS
from src.search.nlhe_turn_vector import TurnVectorCFR

BOARD5 = (0, 5, 10, 15, 20)
NODE = ((48, 49), (24, 1), BOARD5, "k", "c", "c", "c")


@pytest.fixture
def game():
    return PostflopNLHE(starting_stack=50.0, max_raises_per_street=1,
                        raise_fractions=(0.5, 1.0))


@pytest.fixture
def zero_net():
    """Untrained-but-deterministic tiny net; boundary values are whatever
    the net emits — tests here exercise mechanics, not net quality."""
    torch.manual_seed(0)
    net = CFVNet(input_dim=RIVER_ENCODING_DIMS, output_dim=100, hidden=8)
    net.eval()
    return net


def _sparse_range(rng, n=10):
    bs = set(BOARD5[:4])
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    idx = rng.choice(live, size=n, replace=False)
    v = np.zeros(N_COMBOS)
    v[idx] = rng.dirichlet(np.ones(n))
    return v


class TestAllinEquity:

    def test_allin_showdown_matches_bruteforce(self, game, zero_net):
        """46-runout equity sweep vs direct pair enumeration."""
        from src.abstraction.equity import evaluate_7card
        rng = np.random.default_rng(2)
        x_opp = _sparse_range(rng, 8)
        solver = TurnVectorCFR(game, NODE, x_opp, x_opp, zero_net,
                               iterations=1)
        # all-in + call at the turn → terminal with runouts
        rep = solver._rep(("a", "k"))
        assert game._parse_state(rep)["all_in"]
        inv = float(game._parse_state(rep)["invested"][0])
        u = solver._allin_showdown(rep, x_opp, traverser=0)

        turn = BOARD5[:4]
        rivers = [c for c in range(52) if c not in turn]
        for i in rng.choice(np.where(solver.live_mask)[0], size=6,
                            replace=False):
            hero = COMBOS[i]
            brute = 0.0
            for rc in rivers:
                if rc in hero:
                    continue
                b5 = turn + (rc,)
                s_h = evaluate_7card(tuple(hero) + b5)
                for j in np.where(x_opp > 0)[0]:
                    opp = COMBOS[j]
                    if (set(opp) & set(hero)) or rc in opp:
                        continue
                    s_o = evaluate_7card(tuple(opp) + b5)
                    brute += x_opp[j] * (inv if s_h > s_o
                                         else (-inv if s_h < s_o else 0.0))
            brute /= len(rivers)
            np.testing.assert_allclose(u[i], brute, atol=1e-9)


class TestSolveSmoke:

    def test_solve_runs_and_strategies_normalised(self, game, zero_net):
        rng = np.random.default_rng(3)
        x0, x1 = _sparse_range(rng, 15), _sparse_range(rng, 15)
        solver = TurnVectorCFR(game, NODE, x0, x1, zero_net, iterations=5)
        solver.solve()
        hero = game.current_player(NODE)
        x_hero = x0 if hero == 0 else x1
        for i in np.where(x_hero > 0)[0][:5]:
            h = list(NODE)
            h[hero] = COMBOS[i]
            probs = solver.strategy_at(tuple(h), hero)
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-9)
            assert np.all(probs >= 0)
