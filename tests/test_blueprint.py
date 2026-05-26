"""
Tests for Blueprint — save, load, and query interface.

Run with:  pytest tests/test_blueprint.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Minimal stubs so tests run without the full project installed ─────────────

class _FakeStrategyNet(nn.Module):
    def __init__(self, state_size=122, action_size=4, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
    def forward(self, x, action_mask=None):
        import torch.nn.functional as F
        logits = self.net(x)
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e9)
        return F.softmax(logits, dim=-1)


class _FakeSolver:
    """Minimal solver stub for Blueprint.from_solver."""
    def __init__(self):
        self.strategy_net = _FakeStrategyNet()
        self.iterations = 42
        self.traversals_per_iter = 200
        self.strategy_buffer = list(range(1337))   # len() = 1337

        class _Game:
            starting_stack = 200.0
            raise_fractions = (0.75,)
            max_raises_per_street = 2
        self.game = _Game()


# ── Import Blueprint (adjust path if running from project root) ───────────────

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deep_cfr.blueprint import Blueprint, BlueprintMetadata, ScriptableStrategyNet


# ── BlueprintMetadata ─────────────────────────────────────────────────────────

class TestBlueprintMetadata:

    def test_roundtrip_json(self):
        meta = BlueprintMetadata(iterations=100, hidden_size=256)
        loaded = BlueprintMetadata.from_json(meta.to_json())
        assert loaded.iterations == 100
        assert loaded.hidden_size == 256
        assert loaded.state_size == 122

    def test_all_fields_serialised(self):
        meta = BlueprintMetadata(
            state_size=122, action_size=4, hidden_size=128,
            starting_stack=100.0, raise_fraction=0.5, max_raises=3,
            iterations=50, traversals_per_iter=500, strategy_samples=10_000,
        )
        d = json.loads(meta.to_json())
        assert d["starting_stack"] == 100.0
        assert d["raise_fraction"] == 0.5
        assert d["strategy_samples"] == 10_000


# ── ScriptableStrategyNet ─────────────────────────────────────────────────────

class TestScriptableStrategyNet:

    def test_output_is_probability_distribution(self):
        net = ScriptableStrategyNet(state_size=10, action_size=4, hidden_size=32)
        x    = torch.randn(3, 10)
        mask = torch.ones(3, 4)
        probs = net(x, mask)
        assert probs.shape == (3, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_illegal_actions_get_zero_probability(self):
        net = ScriptableStrategyNet(state_size=10, action_size=4, hidden_size=32)
        x    = torch.randn(1, 10)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])   # only 2 legal actions
        probs = net(x, mask)
        assert probs[0, 2].item() < 1e-6
        assert probs[0, 3].item() < 1e-6

    def test_from_strategy_network_copies_weights(self):
        orig = _FakeStrategyNet(state_size=10, action_size=4, hidden_size=32)
        wrapper = ScriptableStrategyNet.from_strategy_network(orig)
        # Forward pass should give same result
        x    = torch.randn(2, 10)
        mask = torch.ones(2, 4)
        with torch.no_grad():
            p1 = orig(x, mask)
            p2 = wrapper(x, mask)
        assert torch.allclose(p1, p2, atol=1e-6)


# ── Blueprint.from_solver ─────────────────────────────────────────────────────

class TestBlueprintFromSolver:

    def test_metadata_populated_from_solver(self):
        solver = _FakeSolver()
        bp = Blueprint.from_solver(solver)
        assert bp.metadata.iterations == 42
        assert bp.metadata.traversals_per_iter == 200
        assert bp.metadata.strategy_samples == 1337
        assert bp.metadata.starting_stack == 200.0
        assert bp.metadata.raise_fraction == 0.75

    def test_network_weights_not_shared(self):
        """Modifying solver weights after from_solver must not affect blueprint."""
        solver = _FakeSolver()
        bp = Blueprint.from_solver(solver)

        # Mutate solver net
        with torch.no_grad():
            for p in solver.strategy_net.parameters():
                p.fill_(99.0)

        x    = torch.randn(1, 122)
        mask = torch.ones(1, 4)
        probs = bp._net(x.to(bp._device), mask.to(bp._device))
        assert not torch.allclose(probs, torch.full_like(probs, 99.0))


# ── Blueprint save / load ─────────────────────────────────────────────────────

class TestBlueprintPersistence:

    def _make_blueprint(self) -> Blueprint:
        solver = _FakeSolver()
        return Blueprint.from_solver(solver)

    def test_save_creates_expected_files(self, tmp_path):
        bp = self._make_blueprint()
        bp.save(tmp_path / "bp")
        assert (tmp_path / "bp" / "strategy_weights.pt").exists()
        assert (tmp_path / "bp" / "metadata.json").exists()

    def test_load_restores_metadata(self, tmp_path):
        bp = self._make_blueprint()
        bp.save(tmp_path / "bp")
        loaded = Blueprint.load(tmp_path / "bp", device="cpu")
        assert loaded.metadata.iterations == bp.metadata.iterations
        assert loaded.metadata.hidden_size == bp.metadata.hidden_size
        assert loaded.metadata.raise_fraction == bp.metadata.raise_fraction

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Blueprint.load(tmp_path / "nonexistent")

    def test_save_load_roundtrip_weights(self, tmp_path):
        """Weights must survive save → load → inference."""
        bp = self._make_blueprint()
        state_vec = np.random.randn(122).astype(np.float32)
        probs_before = bp.query(state_vec, num_actions=3)

        bp.save(tmp_path / "bp")
        loaded = Blueprint.load(tmp_path / "bp", device="cpu")
        probs_after = loaded.query(state_vec, num_actions=3)

        np.testing.assert_allclose(probs_before, probs_after, atol=1e-5)


# ── Blueprint.query ───────────────────────────────────────────────────────────

class TestBlueprintQuery:

    @pytest.fixture
    def bp(self):
        return Blueprint.from_solver(_FakeSolver())

    def test_query_returns_valid_distribution(self, bp):
        state_vec = np.random.randn(122).astype(np.float32)
        for n in [1, 2, 3, 4]:
            probs = bp.query(state_vec, num_actions=n)
            assert probs.shape == (n,)
            assert all(p >= 0 for p in probs)
            assert abs(sum(probs) - 1.0) < 1e-5

    def test_query_illegal_action_count_raises(self, bp):
        state_vec = np.random.randn(122).astype(np.float32)
        with pytest.raises(AssertionError):
            bp.query(state_vec, num_actions=0)
        with pytest.raises(AssertionError):
            bp.query(state_vec, num_actions=5)

    def test_query_batch_shape(self, bp):
        states = np.random.randn(16, 122).astype(np.float32)
        counts = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2]
        probs = bp.query_batch(states, counts)
        assert probs.shape == (16, 4)

    def test_query_batch_valid_distributions(self, bp):
        states = np.random.randn(8, 122).astype(np.float32)
        counts = [3] * 8
        probs = bp.query_batch(states, counts)
        row_sums = probs[:, :3].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(8), atol=1e-5)

    def test_query_batch_padding_zeroed(self, bp):
        states = np.random.randn(4, 122).astype(np.float32)
        counts = [2, 2, 2, 2]
        probs = bp.query_batch(states, counts)
        # Columns 2 and 3 should be zero for num_actions=2
        assert np.all(probs[:, 2:] == 0.0)

    def test_query_matches_query_batch(self, bp):
        """Single query and batch query must agree."""
        state_vec = np.random.randn(122).astype(np.float32)
        single = bp.query(state_vec, num_actions=3)
        batch  = bp.query_batch(state_vec[None], [3])
        np.testing.assert_allclose(single, batch[0, :3], atol=1e-5)


# ── Repr ──────────────────────────────────────────────────────────────────────

class TestBlueprintRepr:

    def test_repr_contains_key_info(self):
        bp = Blueprint.from_solver(_FakeSolver())
        r = repr(bp)
        assert "200" in r     # starting_stack
        assert "75%" in r     # raise_fraction
        assert "42"  in r     # iterations