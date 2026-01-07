"""
ℤ₂ Parity Operators and Projectors

This module implements the fundamental algebraic structures for non-orientable
neural networks: the parity operator S, projectors P₊/P₋, and utilities for
verifying commutation/anticommutation properties.
"""

import numpy as np
import torch
import torch.nn as nn


class ParityOperator(nn.Module):
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

        Raises:
            AssertionError: If dimension constraints are violated
        """
        super().__init__()

        self.dim = dim

        # Enforce parity splitting constraints
        if even_dim is None:
            # Default split requires even dimension
            assert dim % 2 == 0, (
                f"hidden_dim must be even when even_dim is not specified. "
                f"Got hidden_dim={dim}. Either use even hidden_dim or provide even_dim explicitly."
            )
            self.even_dim = dim // 2
        else:
            # Custom split must be valid
            assert 0 < even_dim < dim, (
                f"even_dim must satisfy 0 < even_dim < dim. " f"Got even_dim={even_dim}, dim={dim}."
            )
            self.even_dim = even_dim

        self.odd_dim = dim - self.even_dim
        assert self.odd_dim > 0, f"odd_dim must be positive. Got odd_dim={self.odd_dim}"

        # Construct S = diag(+1, ..., +1, -1, ..., -1) and register as buffer
        S = torch.diag(torch.cat([torch.ones(self.even_dim), -torch.ones(self.odd_dim)]))
        self.register_buffer("S", S)

    def __call__(self, h):
        """Apply parity operator: S @ h"""
        return h @ self.S.T if h.dim() > 1 else self.S @ h


class ParityProjectors(nn.Module):
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

        Raises:
            AssertionError: If parity operator constraints are violated
        """
        super().__init__()

        assert isinstance(
            parity_op, ParityOperator
        ), f"parity_op must be a ParityOperator instance. Got {type(parity_op)}"
        self.dim = parity_op.dim

        # Verify even/odd dimensions are positive
        assert parity_op.even_dim > 0, f"even_dim must be positive. Got {parity_op.even_dim}"
        assert parity_op.odd_dim > 0, f"odd_dim must be positive. Got {parity_op.odd_dim}"

        # Construct projectors and register as buffers
        I = torch.eye(self.dim, device=parity_op.S.device)
        P_plus = 0.5 * (I + parity_op.S)
        P_minus = 0.5 * (I - parity_op.S)

        self.register_buffer("S", parity_op.S)
        self.register_buffer("P_plus", P_plus)
        self.register_buffer("P_minus", P_minus)

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
        norm_h_sq = (h**2).sum(dim=-1)
        norm_h_minus_sq = (h_minus**2).sum(dim=-1)

        # Avoid division by zero - use h.new_zeros for device consistency
        alpha_minus = torch.where(
            norm_h_sq > 1e-8, norm_h_minus_sq / norm_h_sq, h.new_zeros(norm_h_sq.shape)
        )

        return alpha_minus


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


def verify_projector_properties(
    P_plus: torch.Tensor, P_minus: torch.Tensor, tol: float = 1e-6
) -> dict:
    """
    Verify all projector properties.

    Returns:
        dict with boolean flags for each property
    """
    I = torch.eye(P_plus.shape[0], device=P_plus.device)
    zero = torch.zeros_like(P_plus)

    checks = {
        "idempotent_plus": torch.allclose(P_plus @ P_plus, P_plus, atol=tol),
        "idempotent_minus": torch.allclose(P_minus @ P_minus, P_minus, atol=tol),
        "orthogonal": torch.allclose(P_plus @ P_minus, zero, atol=tol),
        "partition": torch.allclose(P_plus + P_minus, I, atol=tol),
    }

    return checks


def verify_commutation(W: torch.Tensor, S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify W commutes with S: WS = SW"""
    return torch.allclose(W @ S, S @ W, atol=tol)


def verify_anticommutation(W: torch.Tensor, S: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify W anticommutes with S: WS = -SW"""
    return torch.allclose(W @ S, -S @ W, atol=tol)


def construct_commutant_weight(
    A_plus: torch.Tensor, A_minus: torch.Tensor, parity_op: ParityOperator
) -> torch.Tensor:
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
    W[: parity_op.even_dim, : parity_op.even_dim] = A_plus
    W[parity_op.even_dim :, parity_op.even_dim :] = A_minus
    return W


def construct_anticommutant_weight(
    B_plus_minus: torch.Tensor, B_minus_plus: torch.Tensor, parity_op: ParityOperator
) -> torch.Tensor:
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
    W[: parity_op.even_dim, parity_op.even_dim :] = B_plus_minus
    W[parity_op.even_dim :, : parity_op.even_dim] = B_minus_plus
    return W
