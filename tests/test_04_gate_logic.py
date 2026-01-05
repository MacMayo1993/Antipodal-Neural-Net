"""
Section 4: Gate Logic Tests

Tests for seam gate computation and k* threshold behavior.
"""

import pytest
import torch
import numpy as np

from src.models import SeamGatedRNN
from src.parity import ParityOperator, ParityProjectors


class TestKStarThreshold:
    """Test 4.2: k* Threshold Behavior"""

    def test_kstar_gate_monotonicity(self):
        """
        Verify gate is monotone increasing in α₋ and crosses 0.5 at k*
        """
        kstar = 0.721
        tau = 0.1

        model = SeamGatedRNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=2,
            gate_type='kstar',
            kstar=kstar,
            tau=tau,
            even_dim=4
        )

        # Create states with varying α₋
        alpha_values = torch.linspace(0, 1, 50)
        gate_values = []

        for alpha in alpha_values:
            # Create state with specific α₋
            # α₋ = ||P₋h||² / ||h||²
            # Set h = [h₊, h₋] where h₊ has norm sqrt(1-alpha), h₋ has norm sqrt(alpha)

            h_plus_norm = torch.sqrt(1 - alpha + 1e-8)
            h_minus_norm = torch.sqrt(alpha)

            h = torch.zeros(1, 8)
            if h_plus_norm > 0:
                h[:, :4] = torch.randn(1, 4)
                h[:, :4] = h[:, :4] / torch.norm(h[:, :4]) * h_plus_norm

            if h_minus_norm > 0:
                h[:, 4:] = torch.randn(1, 4)
                h[:, 4:] = h[:, 4:] / torch.norm(h[:, 4:]) * h_minus_norm

            g = model.compute_gate(h.squeeze(0))
            gate_values.append(g.item())

        gate_values = torch.tensor(gate_values)

        # Check monotonicity
        diffs = gate_values[1:] - gate_values[:-1]
        assert torch.all(diffs >= -1e-5), "Gate not monotone increasing in α₋"

        # Check that g(k*) ≈ 0.5
        # Find closest alpha to k*
        idx_kstar = torch.argmin(torch.abs(alpha_values - kstar))
        g_at_kstar = gate_values[idx_kstar]

        assert abs(g_at_kstar - 0.5) < 0.1, \
            f"g(k*) = {g_at_kstar}, expected ≈ 0.5"

    def test_kstar_gate_endpoints(self):
        """Verify gate approaches 0 for α₋→0 and 1 for α₋→1"""
        model = SeamGatedRNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=2,
            gate_type='kstar',
            kstar=0.721,
            tau=0.1,
            even_dim=4
        )

        # State with α₋ ≈ 0 (all even)
        h_even = torch.zeros(1, 8)
        h_even[:, :4] = torch.randn(1, 4)
        g_even = model.compute_gate(h_even.squeeze(0))

        # State with α₋ ≈ 1 (all odd)
        h_odd = torch.zeros(1, 8)
        h_odd[:, 4:] = torch.randn(1, 4)
        g_odd = model.compute_gate(h_odd.squeeze(0))

        assert g_even < 0.1, f"g(α₋≈0) = {g_even}, expected ≈ 0"
        assert g_odd > 0.9, f"g(α₋≈1) = {g_odd}, expected ≈ 1"


class TestLearnedGate:
    """Test 4.3: Learned Gate Sanity"""

    def test_learned_gate_range(self):
        """Verify learned gate output ∈ (0,1)"""
        model = SeamGatedRNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=2,
            gate_type='learned',
            even_dim=4
        )

        torch.manual_seed(42)
        h_batch = torch.randn(10, 8)

        g = model.compute_gate(h_batch)

        assert torch.all(g >= 0) and torch.all(g <= 1), \
            f"Learned gate outside [0,1]: min={g.min()}, max={g.max()}"

    def test_learned_gate_gradients(self):
        """Verify gradients flow through learned gate"""
        model = SeamGatedRNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=2,
            gate_type='learned',
            even_dim=4
        )

        h = torch.randn(1, 8, requires_grad=True)
        x = torch.randn(1, 3)

        y, h_next, g = model(x, h)

        loss = y.sum()
        loss.backward()

        # Check gate MLP has gradients
        for param in model.gate_mlp.parameters():
            assert param.grad is not None, "Gate MLP not receiving gradients"
            assert torch.any(param.grad != 0), "Gate MLP gradients are zero"


class TestGateTypes:
    """Test different gate configurations"""

    @pytest.mark.parametrize("gate_type", ['fixed', 'learned', 'kstar'])
    def test_all_gate_types_work(self, gate_type):
        """Verify all gate types produce valid outputs"""
        model = SeamGatedRNN(
            input_dim=3,
            hidden_dim=8,
            output_dim=2,
            gate_type=gate_type,
            even_dim=4
        )

        x = torch.randn(2, 5, 3)
        y, h, g = model(x)

        assert y.shape == (2, 5, 2)
        assert torch.all(torch.isfinite(y))
        assert torch.all((g >= 0) & (g <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
