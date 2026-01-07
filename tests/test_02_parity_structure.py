"""
Section 2: ℤ₂ Structural Invariance Tests

Tests for parity operators, projectors, and symmetry properties.
"""

import numpy as np
import pytest
import torch

from src.parity import (
    ParityOperator,
    ParityProjectors,
    construct_anticommutant_weight,
    construct_commutant_weight,
    verify_anticommutation,
    verify_commutation,
    verify_eigenvalues,
    verify_involution,
    verify_projector_properties,
)


class TestParityOperator:
    """Test 2.1: Parity Operator Validity"""

    @pytest.mark.parametrize("dim,even_dim", [(10, 5), (20, 10), (16, 8), (9, 5)])
    def test_parity_operator_involution(self, dim, even_dim):
        """
        Verify S² = I (involution property)
        """
        parity_op = ParityOperator(dim, even_dim)

        is_involution = verify_involution(parity_op.S)

        assert is_involution, f"S² ≠ I for dim={dim}, even_dim={even_dim}"

    @pytest.mark.parametrize("dim,even_dim", [(10, 5), (20, 10), (16, 8)])
    def test_parity_eigenvalues(self, dim, even_dim):
        """
        Verify eigenvalues ∈ {+1, -1}
        """
        parity_op = ParityOperator(dim, even_dim)

        is_valid = verify_eigenvalues(parity_op.S)

        assert is_valid, f"Eigenvalues not in {{+1, -1}} for dim={dim}"

    def test_dimensional_split(self):
        """Verify correct dimensional split between even/odd subspaces"""
        dim = 10
        even_dim = 6
        odd_dim = 4

        parity_op = ParityOperator(dim, even_dim)

        assert parity_op.even_dim == even_dim
        assert parity_op.odd_dim == odd_dim
        assert parity_op.even_dim + parity_op.odd_dim == dim


class TestProjectors:
    """Test 2.2: Projector Properties"""

    @pytest.mark.parametrize("dim,even_dim", [(10, 5), (20, 10), (16, 8)])
    def test_projectors_form_partition(self, dim, even_dim):
        """
        Verify projector properties:
        - P₊² = P₊ (idempotent)
        - P₋² = P₋ (idempotent)
        - P₊P₋ = 0 (orthogonal)
        - P₊ + P₋ = I (partition)
        """
        parity_op = ParityOperator(dim, even_dim)
        projectors = ParityProjectors(parity_op)

        checks = verify_projector_properties(projectors.P_plus, projectors.P_minus)

        assert checks["idempotent_plus"], "P₊² ≠ P₊"
        assert checks["idempotent_minus"], "P₋² ≠ P₋"
        assert checks["orthogonal"], "P₊P₋ ≠ 0"
        assert checks["partition"], "P₊ + P₋ ≠ I"

    def test_parity_energy_bounds(self):
        """
        Test 4.1: α₋ Computation Correctness
        Verify 0 ≤ α₋ ≤ 1 and α₋ + α₊ = 1
        """
        parity_op = ParityOperator(10, 5)
        projectors = ParityProjectors(parity_op)

        # Test with random states
        torch.manual_seed(42)
        for _ in range(20):
            h = torch.randn(10)

            alpha_minus = projectors.parity_energy(h)

            # Check bounds
            assert 0 <= alpha_minus <= 1, f"α₋ = {alpha_minus} not in [0, 1]"

            # Verify α₊ + α₋ = 1
            h_plus = projectors.project_plus(h)
            h_minus = projectors.project_minus(h)

            norm_h_sq = (h**2).sum()
            norm_plus_sq = (h_plus**2).sum()
            norm_minus_sq = (h_minus**2).sum()

            alpha_plus = norm_plus_sq / norm_h_sq if norm_h_sq > 1e-8 else 0

            assert (
                abs(alpha_plus + alpha_minus - 1.0) < 1e-5
            ), f"α₊ + α₋ = {alpha_plus + alpha_minus} ≠ 1"

    def test_projection_preserves_norm_partition(self):
        """Verify ||P₊h||² + ||P₋h||² = ||h||²"""
        parity_op = ParityOperator(10, 5)
        projectors = ParityProjectors(parity_op)

        torch.manual_seed(42)
        h = torch.randn(10)

        h_plus = projectors.project_plus(h)
        h_minus = projectors.project_minus(h)

        norm_h = torch.norm(h) ** 2
        norm_plus = torch.norm(h_plus) ** 2
        norm_minus = torch.norm(h_minus) ** 2

        assert abs(norm_plus + norm_minus - norm_h) < 1e-6, "Norm partition violated"


class TestCommutantWeights:
    """Test 2.3: Weight Commutation (W_comm)"""

    def test_commutant_weights_commute(self):
        """
        Verify W_comm S = S W_comm for commutant weights
        """
        parity_op = ParityOperator(10, 5)

        # Create random commutant weight (block diagonal)
        A_plus = torch.randn(5, 5)
        A_minus = torch.randn(5, 5)

        W_comm = construct_commutant_weight(A_plus, A_minus, parity_op)

        is_commutant = verify_commutation(W_comm, parity_op.S)

        assert is_commutant, "W_comm does not commute with S"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_commutant_various_matrices(self, seed):
        """Test commutation with various random matrices"""
        torch.manual_seed(seed)

        parity_op = ParityOperator(12, 6)

        A_plus = torch.randn(6, 6)
        A_minus = torch.randn(6, 6)

        W_comm = construct_commutant_weight(A_plus, A_minus, parity_op)

        assert verify_commutation(W_comm, parity_op.S), f"Commutation failed for seed {seed}"


class TestAnticommutantWeights:
    """Test 2.4: Flip Operator Anticommutation (W_flip)"""

    def test_flip_weights_anticommute(self):
        """
        Verify W_flip S = -S W_flip for anticommutant weights
        """
        parity_op = ParityOperator(10, 5)

        # Create random anticommutant weight (off-block-diagonal)
        B_plus_minus = torch.randn(5, 5)
        B_minus_plus = torch.randn(5, 5)

        W_flip = construct_anticommutant_weight(B_plus_minus, B_minus_plus, parity_op)

        is_anticommutant = verify_anticommutation(W_flip, parity_op.S)

        assert is_anticommutant, "W_flip does not anticommute with S"

    def test_flip_parity_swap(self):
        """
        Verify parity swap: W_flip(V₊) ⊆ V₋ and W_flip(V₋) ⊆ V₊
        """
        parity_op = ParityOperator(10, 5)
        projectors = ParityProjectors(parity_op)

        # Create anticommutant weight
        B_plus_minus = torch.randn(5, 5)
        B_minus_plus = torch.randn(5, 5)

        W_flip = construct_anticommutant_weight(B_plus_minus, B_minus_plus, parity_op)

        # Create vectors in even/odd subspaces
        v_plus = torch.zeros(10)
        v_plus[:5] = torch.randn(5)  # Only even components

        v_minus = torch.zeros(10)
        v_minus[5:] = torch.randn(5)  # Only odd components

        # Apply W_flip
        w_plus = W_flip @ v_plus
        w_minus = W_flip @ v_minus

        # W_flip(V₊) should be in V₋ (only odd components nonzero)
        even_part_w_plus = projectors.project_plus(w_plus)
        assert torch.norm(even_part_w_plus) < 1e-6, "W_flip(V₊) not in V₋"

        # W_flip(V₋) should be in V₊ (only even components nonzero)
        odd_part_w_minus = projectors.project_minus(w_minus)
        assert torch.norm(odd_part_w_minus) < 1e-6, "W_flip(V₋) not in V₊"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_anticommutant_various_matrices(self, seed):
        """Test anticommutation with various random matrices"""
        torch.manual_seed(seed)

        parity_op = ParityOperator(12, 6)

        B_plus_minus = torch.randn(6, 6)
        B_minus_plus = torch.randn(6, 6)

        W_flip = construct_anticommutant_weight(B_plus_minus, B_minus_plus, parity_op)

        assert verify_anticommutation(
            W_flip, parity_op.S
        ), f"Anticommutation failed for seed {seed}"


class TestParityStructureConsistency:
    """Additional consistency tests"""

    def test_parity_double_application(self):
        """Verify S(Sh) = h"""
        parity_op = ParityOperator(10, 5)

        torch.manual_seed(42)
        h = torch.randn(10)

        h_flipped = parity_op(h)
        h_double_flipped = parity_op(h_flipped)

        assert torch.allclose(h, h_double_flipped, atol=1e-6), "S(Sh) ≠ h"

    def test_projector_sum_identity(self):
        """Verify (P₊ + P₋)h = h for any h"""
        parity_op = ParityOperator(10, 5)
        projectors = ParityProjectors(parity_op)

        torch.manual_seed(42)
        h = torch.randn(10)

        h_plus = projectors.project_plus(h)
        h_minus = projectors.project_minus(h)
        h_reconstructed = h_plus + h_minus

        assert torch.allclose(h, h_reconstructed, atol=1e-6), "(P₊ + P₋)h ≠ h"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
