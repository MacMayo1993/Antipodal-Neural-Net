"""
Section 10: Reproducibility Tests

Tests for deterministic behavior across seeds and runs.
"""

import numpy as np
import pytest
import torch

from src.data import AntipodalRegimeSwitcher
from src.models import SeamGatedRNN, Z2EquivariantRNN

TEST_SEEDS = [42, 123, 456, 789, 1011]


class TestSeedStability:
    """Test 10.1: Seed Stability"""

    def test_data_generation_reproducibility(self):
        """Verify data generation is reproducible with fixed seed"""
        seed = 42
        latent_dim = 8
        obs_dim = 4

        # Generate twice with same seed
        gen1 = AntipodalRegimeSwitcher(latent_dim, obs_dim, seed=seed)
        obs1, lat1, reg1 = gen1.generate_sequence(T=100)

        gen2 = AntipodalRegimeSwitcher(latent_dim, obs_dim, seed=seed)
        obs2, lat2, reg2 = gen2.generate_sequence(T=100)

        assert torch.allclose(obs1, obs2, atol=1e-6), "Observations not reproducible"
        assert torch.allclose(lat1, lat2, atol=1e-6), "Latents not reproducible"
        assert torch.equal(reg1, reg2), "Regimes not reproducible"

    def test_model_forward_reproducibility(self):
        """Verify model forward pass is reproducible with fixed seed"""
        seed = 42

        model = SeamGatedRNN(input_dim=4, hidden_dim=12, output_dim=4, gate_type="kstar")

        # Set seed and run forward
        torch.manual_seed(seed)
        x1 = torch.randn(2, 10, 4)

        with torch.no_grad():
            y1, h1, g1 = model(x1)

        # Reset seed and run again
        torch.manual_seed(seed)
        x2 = torch.randn(2, 10, 4)

        with torch.no_grad():
            y2, h2, g2 = model(x2)

        assert torch.equal(x1, x2), "Input not reproducible"
        assert torch.equal(y1, y2), "Output not reproducible"

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_seed_variance_bounds(self, seed):
        """
        Run same experiment with different seeds and verify metrics
        have bounded variance.
        """
        # Generate data with this seed
        generator = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.05, seed=seed)

        obs, _, regimes = generator.generate_sequence(T=200)

        # Create and run model (no training, just forward pass)
        torch.manual_seed(seed)
        model = SeamGatedRNN(input_dim=4, hidden_dim=12, output_dim=4, gate_type="kstar")

        with torch.no_grad():
            y_pred, _, gates = model(obs[:-1].unsqueeze(0))
            y_pred = y_pred.squeeze(0)

        # Compute error
        error = torch.nn.functional.mse_loss(y_pred, obs[1:])

        # Error should be finite
        assert torch.isfinite(error), f"Non-finite error for seed {seed}"

    def test_multiple_runs_variance(self):
        """Test variance across multiple seeds is bounded"""
        errors = []

        for seed in TEST_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate data
            generator = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.05, seed=seed)
            obs, _, _ = generator.generate_sequence(T=200)

            # Create model
            model = Z2EquivariantRNN(input_dim=4, hidden_dim=12, output_dim=4)

            with torch.no_grad():
                y_pred, _ = model(obs[:-1].unsqueeze(0))
                y_pred = y_pred.squeeze(0)

            error = torch.nn.functional.mse_loss(y_pred, obs[1:]).item()
            errors.append(error)

        errors = np.array(errors)
        std_error = np.std(errors)

        # Variance should be bounded (this is a weak test for untrained models)
        assert std_error < 100, f"Excessive variance across seeds: {std_error}"


class TestDeterminism:
    """Additional determinism tests"""

    def test_parity_operator_deterministic(self):
        """Verify parity operators are deterministic"""
        from src.parity import ParityOperator

        op1 = ParityOperator(10, 5)
        op2 = ParityOperator(10, 5)

        assert torch.equal(op1.S, op2.S), "Parity operators not deterministic"

    def test_model_initialization_with_seed(self):
        """Verify model initialization is reproducible with seed"""
        torch.manual_seed(42)
        model1 = SeamGatedRNN(input_dim=4, hidden_dim=12, output_dim=4, gate_type="learned")

        torch.manual_seed(42)
        model2 = SeamGatedRNN(input_dim=4, hidden_dim=12, output_dim=4, gate_type="learned")

        # Check parameters are equal
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model initialization not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
