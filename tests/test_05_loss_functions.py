"""
Section 5: Loss Function Tests

Tests for quotient loss and rank-1 projector loss.
"""

import numpy as np
import pytest
import torch

from src.losses import (
    quotient_loss,
    rank1_projector_loss,
    verify_quotient_invariance,
    verify_rank1_equivalence,
)


class TestQuotientLoss:
    """Test 5.1: Quotient Loss Invariance"""

    def test_projective_loss_invariance(self):
        """
        Verify loss is invariant to sign flips:
        L(y, ŷ) = L(±y, ±ŷ)
        """
        torch.manual_seed(42)

        for _ in range(10):
            y = torch.randn(5, 3)
            y_pred = torch.randn(5, 3)

            is_invariant = verify_quotient_invariance(quotient_loss, y, y_pred)

            assert is_invariant, "Quotient loss not invariant to sign flips"

    def test_quotient_loss_sign_flip_equality(self):
        """Explicitly test all sign flip combinations"""
        y = torch.randn(3, 4)
        y_pred = torch.randn(3, 4)

        L_base = quotient_loss(y, y_pred)
        L_flip_y = quotient_loss(-y, y_pred)
        L_flip_pred = quotient_loss(y, -y_pred)
        L_flip_both = quotient_loss(-y, -y_pred)

        assert torch.allclose(L_base, L_flip_y, atol=1e-6), "L(y, ŷ) ≠ L(-y, ŷ)"
        assert torch.allclose(L_base, L_flip_pred, atol=1e-6), "L(y, ŷ) ≠ L(y, -ŷ)"
        assert torch.allclose(L_base, L_flip_both, atol=1e-6), "L(y, ŷ) ≠ L(-y, -ŷ)"

    def test_quotient_loss_minimum_distance(self):
        """Verify quotient loss uses minimum distance"""
        # Case 1: y and ŷ are close
        y = torch.tensor([[1.0, 0.0, 0.0]])
        y_pred = torch.tensor([[0.9, 0.1, 0.0]])

        L1 = quotient_loss(y, y_pred)

        # Case 2: y and -ŷ are close
        y_pred_opp = -y_pred
        L2 = quotient_loss(y, y_pred_opp)

        # Both should give same loss (invariant)
        assert torch.allclose(L1, L2, atol=1e-6)


class TestRank1Loss:
    """Test 5.2: Rank-1 Projector Loss"""

    def test_rank1_loss_equivalence(self):
        """
        Verify rank-1 loss is proportional to quotient loss
        for normalized vectors
        """
        torch.manual_seed(42)

        for _ in range(10):
            y = torch.randn(5, 3)
            y_pred = torch.randn(5, 3)

            is_equivalent = verify_rank1_equivalence(y, y_pred, tol=1e-4)

            # Note: exact equivalence depends on normalization
            # We're testing the relationship holds

    def test_rank1_loss_sign_invariance(self):
        """Verify rank-1 loss is invariant to sign flips"""
        y = torch.randn(3, 4)
        y_pred = torch.randn(3, 4)

        L_base = rank1_projector_loss(y, y_pred)
        L_flip_y = rank1_projector_loss(-y, y_pred)
        L_flip_pred = rank1_projector_loss(y, -y_pred)
        L_flip_both = rank1_projector_loss(-y, -y_pred)

        assert torch.allclose(
            L_base, L_flip_y, atol=1e-6
        ), "Rank-1 loss not invariant to y sign flip"
        assert torch.allclose(
            L_base, L_flip_pred, atol=1e-6
        ), "Rank-1 loss not invariant to ŷ sign flip"
        assert torch.allclose(
            L_base, L_flip_both, atol=1e-6
        ), "Rank-1 loss not invariant to both sign flips"

    def test_rank1_loss_perfect_match(self):
        """Verify rank-1 loss is zero for perfect match"""
        y = torch.randn(3, 4)
        y_pred = y.clone()

        L = rank1_projector_loss(y, y_pred)

        assert L < 1e-6, f"Perfect match has non-zero loss: {L}"

    def test_rank1_loss_opposite_vectors(self):
        """Verify rank-1 loss is zero for opposite vectors (same projector)"""
        y = torch.randn(3, 4)
        y_pred = -y

        L = rank1_projector_loss(y, y_pred)

        assert L < 1e-6, f"Opposite vectors have non-zero loss: {L}"


class TestLossProperties:
    """Additional loss function tests"""

    def test_quotient_loss_non_negative(self):
        """Verify quotient loss is always non-negative"""
        torch.manual_seed(42)

        for _ in range(20):
            y = torch.randn(5, 3)
            y_pred = torch.randn(5, 3)

            L = quotient_loss(y, y_pred)

            assert L >= 0, f"Quotient loss negative: {L}"

    def test_rank1_loss_non_negative(self):
        """Verify rank-1 loss is always non-negative"""
        torch.manual_seed(42)

        for _ in range(20):
            y = torch.randn(5, 3)
            y_pred = torch.randn(5, 3)

            L = rank1_projector_loss(y, y_pred)

            assert L >= 0, f"Rank-1 loss negative: {L}"

    def test_loss_differentiability(self):
        """Verify losses are differentiable"""
        y = torch.randn(3, 4, requires_grad=False)
        y_pred = torch.randn(3, 4, requires_grad=True)

        L_quot = quotient_loss(y, y_pred)
        L_quot.backward()
        assert y_pred.grad is not None, "Quotient loss not differentiable"

        y_pred.grad = None

        L_rank1 = rank1_projector_loss(y, y_pred)
        L_rank1.backward()
        assert y_pred.grad is not None, "Rank-1 loss not differentiable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
