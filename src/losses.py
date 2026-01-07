"""
Loss Functions for Non-Orientable Neural Networks

Implements:
- Quotient loss (projective/antipodal invariance)
- Rank-1 projector loss
- Standard MSE for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quotient_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Quotient loss invariant to sign flips: L(y, ŷ) = L(±y, ±ŷ)

    Uses: min(||y - ŷ||², ||y + ŷ||²)

    Args:
        y_true: (batch, dim) ground truth
        y_pred: (batch, dim) predictions

    Returns:
        Scalar loss
    """
    # Compute both distances
    dist_pos = ((y_true - y_pred) ** 2).sum(dim=-1)
    dist_neg = ((y_true + y_pred) ** 2).sum(dim=-1)

    # Take minimum
    min_dist = torch.min(dist_pos, dist_neg)

    return min_dist.mean()


def rank1_projector_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Rank-1 projector loss: ||yyᵀ - ŷŷᵀ||²_F

    This is invariant to sign flips and measures projective distance.

    Args:
        y_true: (batch, dim) ground truth (assumed normalized or will be normalized)
        y_pred: (batch, dim) predictions (assumed normalized or will be normalized)

    Returns:
        Scalar loss
    """
    # Normalize to unit vectors (add epsilon for numerical stability)
    eps = 1e-8
    y_true = F.normalize(y_true + eps, dim=-1)
    y_pred = F.normalize(y_pred + eps, dim=-1)

    # Compute outer products (batch, dim, dim)
    P_true = y_true.unsqueeze(-1) @ y_true.unsqueeze(-2)
    P_pred = y_pred.unsqueeze(-1) @ y_pred.unsqueeze(-2)

    # Frobenius norm squared
    diff = P_true - P_pred
    loss = (diff**2).sum(dim=(-2, -1))

    return loss.mean()


def verify_quotient_invariance(
    loss_fn, y: torch.Tensor, y_pred: torch.Tensor, tol: float = 1e-6
) -> bool:
    """
    Verify loss is invariant to sign flips.

    Checks:
        L(y, ŷ) = L(-y, ŷ)
        L(y, ŷ) = L(y, -ŷ)
        L(y, ŷ) = L(-y, -ŷ)

    Args:
        loss_fn: Loss function to test
        y: Ground truth
        y_pred: Predictions
        tol: Tolerance for equality

    Returns:
        True if all invariances hold
    """
    L_base = loss_fn(y, y_pred)
    L_flip_y = loss_fn(-y, y_pred)
    L_flip_pred = loss_fn(y, -y_pred)
    L_flip_both = loss_fn(-y, -y_pred)

    checks = [
        torch.abs(L_base - L_flip_y) < tol,
        torch.abs(L_base - L_flip_pred) < tol,
        torch.abs(L_base - L_flip_both) < tol,
    ]

    return all(checks)


def verify_rank1_equivalence(y: torch.Tensor, y_pred: torch.Tensor, tol: float = 1e-5) -> bool:
    """
    Verify rank-1 loss is equivalent to min-distance quotient loss.

    For normalized vectors:
        ||yyᵀ - ŷŷᵀ||²_F = 2(1 - |y·ŷ|²) = min(||y-ŷ||², ||y+ŷ||²) / 2

    Args:
        y: Ground truth (will be normalized)
        y_pred: Predictions (will be normalized)
        tol: Tolerance

    Returns:
        True if equivalence holds
    """
    # Compute both losses
    L_rank1 = rank1_projector_loss(y, y_pred)
    L_quot = quotient_loss(F.normalize(y, dim=-1), F.normalize(y_pred, dim=-1))

    # For normalized vectors, rank1 loss = quotient loss
    # (up to a constant factor for some formulations)

    # Check if they're proportional or equal (depending on normalization)
    # The exact relationship: ||yyᵀ - ŷŷᵀ||²_F = 2 * min(||y-ŷ||², ||y+ŷ||²) for unit vectors

    return torch.abs(L_rank1 - 2 * L_quot) < tol


class QuotientLoss(nn.Module):
    """PyTorch module wrapper for quotient loss"""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return quotient_loss(y_true, y_pred)


class Rank1ProjectorLoss(nn.Module):
    """PyTorch module wrapper for rank-1 projector loss"""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return rank1_projector_loss(y_true, y_pred)
