"""
ℤ₂ Parity Operators and Projectors

This module implements the fundamental algebraic structures for non-orientable
neural networks: the parity operator S, projectors P₊/P₋, and utilities for
verifying commutation/anticommutation properties.
"""

import torch
import torch.nn as nn
import numpy as np


class ParityOperator:
    """
    Parity operator S for ℤ₂ symmetry.

    Properties:
    - S² = I (involution)
    - Eigenvalues ∈ {+1, -1}
    - Defines even/odd subspaces V₊, V₋
    """

    def __init__(self, dim: int, even_dim: int = None):
        """
        Args:
            dim: Total dimension of state space
            even_dim: Dimension of even subspace (defaults to dim//2)
        """
        self.dim = dim
        self.even_dim = even_dim if even_dim is not None else dim // 2
        self.odd_dim = dim - self.even_dim

        # Construct S = diag(+1, ..., +1, -1, ..., -1)
        self.S = torch.diag(torch.cat([
            torch.ones(self.even_dim),
            -torch.ones(self.odd_dim)
        ]))

    def __call__(self, h):
        """Apply parity operator: S @ h"""
        return h @ self.S.T if h.dim() > 1 else self.S @ h

    def to(self, device):
        """Move operator to device"""
        self.S = self.S.to(device)
        return self


class ParityProjectors:
    """
    Projectors onto even/odd subspaces.

    Properties:
    - P₊² = P₊, P₋² = P₋ (idempotent)
    - P₊P₋ = 0 (orthogonal)
    - P₊ + P₋ = I (partition)
    - P₊ = ½(I + S), P₋ = ½(I - S)
    """

    def __init__(self, parity_op: ParityOperator):
        """
        Args:
            parity_op: ParityOperator instance
        """
        self.S = parity_op.S
        self.dim = parity_op.dim

        # Construct projectors
        I = torch.eye(self.dim)
        self.P_plus = 0.5 * (I + self.S)
        self.P_minus = 0.5 * (I - self.S)

    def project_plus(self, h):
        """Project onto even subspace: P₊ @ h"""
        return h @ self.P_plus.T if h.dim() > 1 else self.P_plus @ h

    def project_minus(self, h):
        """Project onto odd subspace: P₋ @ h"""
        return h @ self.P_minus.T if h.dim() > 1 else self.P_minus @ h

    def parity_energy(self, h):
        """
        Compute α₋(h) = ||P₋h||² / ||h||²

        Returns:
            Scalar in [0, 1] representing odd energy fraction
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)

        h_minus = self.project_minus(h)
        norm_h_sq = (h ** 2).sum(dim=-1)
        norm_h_minus_sq = (h_minus ** 2).sum(dim=-1)

        # Avoid division by zero
        alpha_minus = torch.where(
            norm_h_sq > 1e-8,
            norm_h_minus_sq / norm_h_sq,
            torch.zeros_like(norm_h_sq)
        )

        return alpha_minus

    def to(self, device):
        """Move projectors to device"""
        self.S = self.S.to(device)
        self.P_plus = self.P_plus.to(device)
        self.P_minus = self.P_minus.to(device)
        return self


def verify_involution(S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify S² = I"""
    S_squared = S @ S
    I = torch.eye(S.shape[0], device=S.device)
    return torch.allclose(S_squared, I, atol=tol)


def verify_eigenvalues(S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify eigenvalues ∈ {+1, -1}"""
    eigenvalues = torch.linalg.eigvalsh(S)
    valid = torch.all(torch.abs(torch.abs(eigenvalues) - 1.0) < tol)
    return valid.item()


def verify_projector_properties(P_plus: torch.Tensor, P_minus: torch.Tensor,
                                 tol: float = 1e-6) -> dict:
    """
    Verify all projector properties.

    Returns:
        dict with boolean flags for each property
    """
    I = torch.eye(P_plus.shape[0], device=P_plus.device)
    zero = torch.zeros_like(P_plus)

    checks = {
        'idempotent_plus': torch.allclose(P_plus @ P_plus, P_plus, atol=tol),
        'idempotent_minus': torch.allclose(P_minus @ P_minus, P_minus, atol=tol),
        'orthogonal': torch.allclose(P_plus @ P_minus, zero, atol=tol),
        'partition': torch.allclose(P_plus + P_minus, I, atol=tol)
    }

    return checks


def verify_commutation(W: torch.Tensor, S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify W commutes with S: WS = SW"""
    return torch.allclose(W @ S, S @ W, atol=tol)


def verify_anticommutation(W: torch.Tensor, S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify W anticommutes with S: WS = -SW"""
    return torch.allclose(W @ S, -S @ W, atol=tol)


def construct_commutant_weight(A_plus: torch.Tensor, A_minus: torch.Tensor,
                                 parity_op: ParityOperator) -> torch.Tensor:
    """
    Construct weight matrix that commutes with S.

    Block structure:
        W_comm = [A₊  0 ]
                 [0  A₋]

    Args:
        A_plus: even-to-even block
        A_minus: odd-to-odd block
        parity_op: ParityOperator instance

    Returns:
        Block-diagonal weight matrix
    """
    W = torch.zeros(parity_op.dim, parity_op.dim)
    W[:parity_op.even_dim, :parity_op.even_dim] = A_plus
    W[parity_op.even_dim:, parity_op.even_dim:] = A_minus
    return W


def construct_anticommutant_weight(B_plus_minus: torch.Tensor,
                                     B_minus_plus: torch.Tensor,
                                     parity_op: ParityOperator) -> torch.Tensor:
    """
    Construct weight matrix that anticommutes with S.

    Block structure:
        W_flip = [0      B₊₋]
                 [B₋₊    0 ]

    Args:
        B_plus_minus: even-to-odd block
        B_minus_plus: odd-to-even block
        parity_op: ParityOperator instance

    Returns:
        Off-block-diagonal weight matrix
    """
    W = torch.zeros(parity_op.dim, parity_op.dim)
    W[:parity_op.even_dim, parity_op.even_dim:] = B_plus_minus
    W[parity_op.even_dim:, :parity_op.even_dim] = B_minus_plus
    return W
