"""
Test suite for dimension validation in parity operators and models.

Tests that odd hidden dimensions fail fast with clear error messages,
and that valid dimensions (even or with explicit even_dim) work correctly.
"""

import pytest
import torch

from src.parity import ParityOperator, ParityProjectors
from src.models import Z2EquivariantRNN, SeamGatedRNN


class TestParityOperatorDimensions:
    """Test dimension validation in ParityOperator"""

    def test_even_hidden_dim_succeeds(self):
        """Even hidden_dim should work without even_dim parameter"""
        parity_op = ParityOperator(dim=8)
        assert parity_op.dim == 8
        assert parity_op.even_dim == 4
        assert parity_op.odd_dim == 4

    def test_odd_hidden_dim_fails(self):
        """Odd hidden_dim should fail with clear error message"""
        with pytest.raises(AssertionError, match="hidden_dim must be even"):
            ParityOperator(dim=7)

    def test_explicit_even_dim_valid(self):
        """Explicit even_dim should work for any total dim"""
        parity_op = ParityOperator(dim=10, even_dim=3)
        assert parity_op.dim == 10
        assert parity_op.even_dim == 3
        assert parity_op.odd_dim == 7

    def test_explicit_even_dim_zero_fails(self):
        """even_dim=0 should fail"""
        with pytest.raises(AssertionError, match="even_dim must satisfy"):
            ParityOperator(dim=8, even_dim=0)

    def test_explicit_even_dim_equals_dim_fails(self):
        """even_dim=dim should fail (no odd subspace)"""
        with pytest.raises(AssertionError, match="even_dim must satisfy"):
            ParityOperator(dim=8, even_dim=8)

    def test_explicit_even_dim_exceeds_dim_fails(self):
        """even_dim > dim should fail"""
        with pytest.raises(AssertionError, match="even_dim must satisfy"):
            ParityOperator(dim=8, even_dim=10)


class TestParityProjectorsDimensions:
    """Test dimension validation in ParityProjectors"""

    def test_valid_parity_operator(self):
        """ParityProjectors should work with valid ParityOperator"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)
        assert projectors.dim == 8

    def test_invalid_input_type_fails(self):
        """ParityProjectors should reject non-ParityOperator input"""
        with pytest.raises(AssertionError, match="must be a ParityOperator"):
            ParityProjectors("not a parity operator")


class TestModelDimensions:
    """Test dimension validation in models"""

    def test_z2_equivariant_even_hidden_dim(self):
        """Z2EquivariantRNN should work with even hidden_dim"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        assert model.hidden_dim == 8
        assert model.even_dim == 4
        assert model.odd_dim == 4

    def test_z2_equivariant_odd_hidden_dim_fails(self):
        """Z2EquivariantRNN should fail with odd hidden_dim"""
        with pytest.raises(ValueError, match="requires even hidden_dim"):
            Z2EquivariantRNN(input_dim=4, hidden_dim=7, output_dim=4)

    def test_z2_equivariant_explicit_even_dim(self):
        """Z2EquivariantRNN should work with explicit even_dim"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=10, output_dim=4, even_dim=3)
        assert model.hidden_dim == 10
        assert model.even_dim == 3
        assert model.odd_dim == 7

    def test_seam_gated_even_hidden_dim(self):
        """SeamGatedRNN should work with even hidden_dim"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='fixed')
        assert model.hidden_dim == 8
        assert model.even_dim == 4
        assert model.odd_dim == 4

    def test_seam_gated_odd_hidden_dim_fails(self):
        """SeamGatedRNN should fail with odd hidden_dim"""
        with pytest.raises(ValueError, match="requires even hidden_dim"):
            SeamGatedRNN(input_dim=4, hidden_dim=7, output_dim=4, gate_type='fixed')

    def test_seam_gated_explicit_even_dim(self):
        """SeamGatedRNN should work with explicit even_dim"""
        model = SeamGatedRNN(
            input_dim=4, hidden_dim=10, output_dim=4,
            even_dim=3, gate_type='fixed'
        )
        assert model.hidden_dim == 10
        assert model.even_dim == 3
        assert model.odd_dim == 7


class TestDimensionConsistency:
    """Test that dimension validation is consistent across operations"""

    def test_parity_operator_constructs_valid_matrix(self):
        """Parity operator S should have correct dimensions"""
        parity_op = ParityOperator(dim=8)
        assert parity_op.S.shape == (8, 8)

        # Verify S^2 = I
        S_squared = parity_op.S @ parity_op.S
        I = torch.eye(8)
        assert torch.allclose(S_squared, I, atol=1e-6)

    def test_projectors_construct_valid_matrices(self):
        """Projectors should have correct dimensions"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)

        assert projectors.P_plus.shape == (8, 8)
        assert projectors.P_minus.shape == (8, 8)

        # Verify projector properties
        I = torch.eye(8)
        assert torch.allclose(projectors.P_plus + projectors.P_minus, I, atol=1e-6)

    def test_model_forward_with_valid_dimensions(self):
        """Models should handle forward pass with valid dimensions"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x = torch.randn(2, 10, 4)  # batch=2, seq=10, dim=4

        outputs, h = model(x)
        assert outputs.shape == (2, 10, 4)
        assert h.shape == (2, 8)


class TestErrorMessages:
    """Test that error messages are clear and actionable"""

    def test_parity_operator_odd_dim_message(self):
        """Error message should suggest solutions"""
        with pytest.raises(AssertionError) as exc_info:
            ParityOperator(dim=7)

        error_msg = str(exc_info.value)
        assert "hidden_dim must be even" in error_msg
        assert "7" in error_msg
        assert "even_dim explicitly" in error_msg

    def test_model_odd_dim_message(self):
        """Model error message should be clear"""
        with pytest.raises(ValueError) as exc_info:
            Z2EquivariantRNN(input_dim=4, hidden_dim=7, output_dim=4)

        error_msg = str(exc_info.value)
        assert "requires even hidden_dim" in error_msg
        assert "7" in error_msg
        assert "even_dim explicitly" in error_msg
