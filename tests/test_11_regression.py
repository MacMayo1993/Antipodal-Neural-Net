"""
Section 11: Regression Tests

Tests for numerical stability and absence of NaN/Inf values.
"""

import pytest
import torch
import torch.optim as optim
import numpy as np

from src.models import Z2EquivariantRNN, SeamGatedRNN, GRUBaseline
from src.data import AntipodalRegimeSwitcher
from src.losses import quotient_loss, rank1_projector_loss


class TestNumericalStability:
    """Test 11.1: Numerical Stability"""

    def test_no_nan_or_inf(self):
        """
        Verify no NaN/Inf during training and evaluation in:
        - Loss values
        - Hidden states
        - Gate values
        """
        # Generate data
        generator = AntipodalRegimeSwitcher(
            latent_dim=8, obs_dim=4, p_switch=0.05, seed=42
        )
        obs, _, _ = generator.generate_sequence(T=200)

        # Test all model types
        models = [
            Z2EquivariantRNN(4, 12, 4),
            SeamGatedRNN(4, 12, 4, gate_type='kstar'),
            SeamGatedRNN(4, 12, 4, gate_type='learned'),
            GRUBaseline(4, 12, 4)
        ]

        for model in models:
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            X = obs[:-1].unsqueeze(0)
            Y = obs[1:].unsqueeze(0)

            # Training loop
            for step in range(20):
                if hasattr(model, 'gru'):
                    y_pred, h, gates = model(X)
                elif hasattr(model, 'gate_type'):
                    y_pred, h, gates = model(X)
                else:
                    y_pred, h = model(X)
                    gates = torch.zeros(1)

                loss = torch.nn.functional.mse_loss(y_pred, Y)

                # Check for NaN/Inf
                assert torch.isfinite(loss).all(), \
                    f"{model.__class__.__name__}: Non-finite loss at step {step}"

                assert torch.isfinite(y_pred).all(), \
                    f"{model.__class__.__name__}: Non-finite predictions at step {step}"

                assert torch.isfinite(h).all(), \
                    f"{model.__class__.__name__}: Non-finite hidden states at step {step}"

                if gates.numel() > 1:
                    assert torch.isfinite(gates).all(), \
                        f"{model.__class__.__name__}: Non-finite gates at step {step}"

                optimizer.zero_grad()
                loss.backward()

                # Check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all(), \
                            f"{model.__class__.__name__}: Non-finite gradient in {name} at step {step}"

                optimizer.step()

    def test_extreme_input_handling(self):
        """Test models handle extreme input values gracefully"""
        model = SeamGatedRNN(4, 12, 4, gate_type='kstar')

        # Very large inputs
        x_large = torch.ones(1, 10, 4) * 100

        with torch.no_grad():
            y, h, g = model(x_large)

        assert torch.isfinite(y).all(), "Model outputs non-finite for large inputs"
        assert torch.isfinite(h).all(), "Hidden states non-finite for large inputs"

        # Very small inputs
        x_small = torch.ones(1, 10, 4) * 1e-6

        with torch.no_grad():
            y, h, g = model(x_small)

        assert torch.isfinite(y).all(), "Model outputs non-finite for small inputs"

    def test_zero_input_handling(self):
        """Test models handle zero inputs"""
        model = SeamGatedRNN(4, 12, 4, gate_type='learned')

        x_zero = torch.zeros(1, 10, 4)

        with torch.no_grad():
            y, h, g = model(x_zero)

        assert torch.isfinite(y).all(), "Model outputs non-finite for zero inputs"
        assert torch.isfinite(h).all(), "Hidden states non-finite for zero inputs"


class TestLossStability:
    """Test loss functions for numerical stability"""

    def test_quotient_loss_stability(self):
        """Verify quotient loss handles edge cases"""
        # Identical predictions
        y = torch.randn(5, 3)
        loss = quotient_loss(y, y)
        assert torch.isfinite(loss), "Quotient loss non-finite for perfect match"
        assert loss < 1e-5, "Quotient loss not near zero for perfect match"

        # Opposite predictions (should also give zero for quotient loss)
        loss_opp = quotient_loss(y, -y)
        assert torch.isfinite(loss_opp), "Quotient loss non-finite for opposite vectors"
        assert loss_opp < 1e-5, "Quotient loss not near zero for opposite vectors"

        # Zero vectors
        y_zero = torch.zeros(5, 3)
        loss_zero = quotient_loss(y_zero, y)
        assert torch.isfinite(loss_zero), "Quotient loss non-finite for zero targets"

    def test_rank1_loss_stability(self):
        """Verify rank-1 loss handles edge cases"""
        # Small norm vectors
        y_small = torch.randn(5, 3) * 1e-6
        y_pred_small = torch.randn(5, 3) * 1e-6

        loss = rank1_projector_loss(y_small, y_pred_small)
        assert torch.isfinite(loss), "Rank-1 loss non-finite for small vectors"

        # Zero vectors
        y_zero = torch.zeros(5, 3)
        loss_zero = rank1_projector_loss(y_zero, y_pred_small)

        # Should handle gracefully (may return NaN, but shouldn't crash)
        # In practice, we'd avoid this by adding epsilon in normalization


class TestGradientStability:
    """Test gradient computation stability"""

    def test_gradient_norm_bounded(self):
        """Verify gradients don't explode during training"""
        model = SeamGatedRNN(4, 12, 4, gate_type='kstar')
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Generate data
        x = torch.randn(1, 50, 4)
        y = torch.randn(1, 50, 4)

        for step in range(10):
            y_pred, _, _ = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()

            # Check gradient norms
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5

            assert total_norm < 1000, \
                f"Gradient explosion at step {step}: norm={total_norm}"

            optimizer.step()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_timestep(self):
        """Test models work with single timestep input"""
        model = SeamGatedRNN(4, 12, 4, gate_type='kstar')

        x = torch.randn(1, 4)  # Single timestep

        with torch.no_grad():
            y, h, g = model(x)

        assert y.shape == (1, 4), "Single timestep output shape incorrect"
        assert torch.isfinite(y).all(), "Single timestep produces non-finite output"

    def test_batch_size_one(self):
        """Test models work with batch size 1"""
        model = Z2EquivariantRNN(4, 12, 4)

        x = torch.randn(1, 10, 4)

        with torch.no_grad():
            y, h = model(x)

        assert y.shape == (1, 10, 4)
        assert torch.isfinite(y).all()

    def test_various_sequence_lengths(self):
        """Test models work with various sequence lengths"""
        model = SeamGatedRNN(4, 12, 4, gate_type='learned')

        for seq_len in [1, 5, 10, 50, 100]:
            x = torch.randn(2, seq_len, 4)

            with torch.no_grad():
                y, h, g = model(x)

            assert y.shape == (2, seq_len, 4), \
                f"Incorrect shape for seq_len={seq_len}"
            assert torch.isfinite(y).all(), \
                f"Non-finite output for seq_len={seq_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
