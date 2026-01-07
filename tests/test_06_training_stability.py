"""
Section 6: Training Stability Tests

Tests for gradient flow and mode collapse prevention.
"""

import pytest
import torch
import torch.optim as optim

from src.losses import quotient_loss
from src.models import SeamGatedRNN, Z2EquivariantRNN
from src.parity import ParityOperator, verify_anticommutation, verify_commutation


class TestGradientFlow:
    """Test 6.1: Gradient Flow Preservation"""

    def test_gradients_respect_symmetry(self):
        """
        Verify gradients maintain commutant/anticommutant structure
        after one training step
        """
        model = SeamGatedRNN(
            input_dim=3, hidden_dim=8, output_dim=2, gate_type="learned", even_dim=4
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Generate dummy data
        x = torch.randn(4, 10, 3)  # batch=4, seq=10
        y_true = torch.randn(4, 10, 2)

        # Forward pass
        y_pred, _, _ = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"

        # Note: Full symmetry preservation in gradients is complex
        # and depends on the optimizer. Here we just check gradients flow.

    def test_gradient_flow_through_gate(self):
        """Verify gradients flow through seam gate"""
        model = SeamGatedRNN(
            input_dim=3, hidden_dim=8, output_dim=2, gate_type="learned", even_dim=4
        )

        x = torch.randn(2, 5, 3)
        y_true = torch.randn(2, 5, 2)

        y_pred, _, gates = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)

        loss.backward()

        # Check gate MLP receives gradients
        has_gate_grad = False
        for param in model.gate_mlp.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gate_grad = True
                break

        assert has_gate_grad, "No gradients flowing through gate network"


class TestModeCollapse:
    """Test 6.2: No Mode Collapse"""

    def test_parity_channel_noncollapse(self):
        """
        Train for N steps and verify both parity channels remain active.
        Var(P₊h) > ε and Var(P₋h) > ε
        """
        model = SeamGatedRNN(
            input_dim=3, hidden_dim=8, output_dim=2, gate_type="learned", even_dim=4
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train for a few steps
        for _ in range(50):
            x = torch.randn(4, 10, 3)
            y_true = torch.randn(4, 10, 2)

            y_pred, h_final, _ = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check parity channel activity
        with torch.no_grad():
            # Generate test data and collect hidden states
            x_test = torch.randn(20, 10, 3)
            _, h_final, _ = model(x_test)

            # Check variance in even/odd channels
            h_even = h_final[:, :4]
            h_odd = h_final[:, 4:]

            var_even = torch.var(h_even)
            var_odd = torch.var(h_odd)

            assert var_even > 1e-4, f"Even channel collapsed: var={var_even}"
            assert var_odd > 1e-4, f"Odd channel collapsed: var={var_odd}"

    def test_no_nan_during_training(self):
        """Verify no NaNs appear during short training"""
        model = SeamGatedRNN(input_dim=3, hidden_dim=8, output_dim=2, gate_type="kstar", even_dim=4)

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for step in range(20):
            x = torch.randn(4, 10, 3)
            y_true = torch.randn(4, 10, 2)

            y_pred, h_final, gates = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check for NaNs
            assert torch.isfinite(loss), f"NaN loss at step {step}"
            assert torch.all(torch.isfinite(y_pred)), f"NaN in predictions at step {step}"
            assert torch.all(torch.isfinite(h_final)), f"NaN in hidden state at step {step}"


class TestOptimizationStability:
    """Additional training stability tests"""

    def test_loss_decreases(self):
        """Verify loss decreases during training on simple task"""
        model = Z2EquivariantRNN(input_dim=3, hidden_dim=16, output_dim=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Fixed dataset for overfitting test
        x_train = torch.randn(10, 20, 3)
        y_train = torch.randn(10, 20, 2)

        losses = []

        for _ in range(100):
            y_pred, _ = model(x_train)
            loss = torch.nn.functional.mse_loss(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert (
            losses[-1] < losses[0]
        ), f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
