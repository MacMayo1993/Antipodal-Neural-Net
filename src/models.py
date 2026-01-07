"""
Non-Orientable Neural Network Models

Implements:
- ℤ₂-equivariant RNN (commutant only)
- Seam-gated models (with learned/fixed/k*-based gates)
- Standard baselines (GRU)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parity import ParityOperator, ParityProjectors


class Z2EquivariantRNN(nn.Module):
    """
    ℤ₂-equivariant RNN with commutant weights only (no seam coupling).

    Update: h_{t+1} = σ(W_comm [h_t, x_t] + b)

    where W_comm S = S W_comm (parity-preserving)
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, even_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Validate dimensions for parity splitting
        if even_dim is None and hidden_dim % 2 != 0:
            raise ValueError(
                f"Z2EquivariantRNN requires even hidden_dim when even_dim is not provided. "
                f"Got hidden_dim={hidden_dim}. Either use even hidden_dim or provide even_dim explicitly."
            )

        # Parity structure (will validate dimensions internally)
        self.parity_op = ParityOperator(hidden_dim, even_dim)
        self.projectors = ParityProjectors(self.parity_op)

        self.even_dim = self.parity_op.even_dim
        self.odd_dim = self.parity_op.odd_dim

        # Commutant weights (block diagonal)
        # W_comm = [A₊  0 ]
        #          [0  A₋]
        self.W_even = nn.Linear(input_dim + self.even_dim, self.even_dim, bias=False)
        self.W_odd = nn.Linear(input_dim + self.odd_dim, self.odd_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            h: Initial hidden state (batch, hidden_dim) or None

        Returns:
            outputs: (batch, seq_len, output_dim) or (batch, output_dim)
            h_final: (batch, hidden_dim)
        """
        if x.dim() == 2:
            # Single timestep
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        batch_size, seq_len, _ = x.shape

        if h is None:
            h = x.new_zeros(batch_size, self.hidden_dim)

        outputs = []

        for t in range(seq_len):
            h = self.step(x[:, t], h)
            y = self.output_layer(h)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)

        if single_step:
            outputs = outputs.squeeze(1) if outputs.dim() > 2 else outputs

        return outputs, h

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single RNN step with commutant weights only"""
        # Track if either input is unbatched (1D) - if so, return unbatched output
        x_was_1d = x.dim() == 1
        h_was_1d = h.dim() == 1
        return_1d = x_was_1d or h_was_1d

        # Ensure both inputs have batch dimension for consistent processing
        if x_was_1d:
            x = x.unsqueeze(0)
        if h_was_1d:
            h = h.unsqueeze(0)

        # Extract even and odd components directly (first even_dim, last odd_dim)
        h_even = h[..., : self.even_dim]
        h_odd = h[..., self.even_dim :]

        # Concatenate input with respective parity channels
        u_even = torch.cat([x, h_even], dim=-1)
        u_odd = torch.cat([x, h_odd], dim=-1)

        # Apply block-diagonal weights
        h_even_next = self.W_even(u_even)
        h_odd_next = self.W_odd(u_odd)

        # Combine and add bias
        h_next = torch.cat([h_even_next, h_odd_next], dim=-1) + self.bias

        # Activation
        h_next = torch.tanh(h_next)

        # Restore original dimensionality - squeeze if either input was 1D
        if return_1d and h_next.size(0) == 1:
            h_next = h_next.squeeze(0)

        return h_next

    def get_weight_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract commutant weight matrix"""
        # Reconstruct block-diagonal structure
        W_comm = torch.zeros(self.hidden_dim, self.hidden_dim + self.input_dim)

        W_even_full = self.W_even.weight
        W_odd_full = self.W_odd.weight

        # Place even block
        W_comm[: self.even_dim, : self.input_dim] = W_even_full[:, : self.input_dim]
        W_comm[: self.even_dim, self.input_dim : self.input_dim + self.even_dim] = W_even_full[
            :, self.input_dim :
        ]

        # Place odd block
        W_comm[self.even_dim :, : self.input_dim] = W_odd_full[:, : self.input_dim]
        W_comm[self.even_dim :, self.input_dim + self.even_dim :] = W_odd_full[:, self.input_dim :]

        return W_comm, torch.zeros_like(W_comm)  # No flip weights


class SeamGatedRNN(nn.Module):
    """
    ℤ₂-equivariant RNN with seam gate.

    Update:
        h_{t+1} = σ(W_comm u_t + g(h_t) W_flip S u_t + b)

    where:
        - W_comm commutes with S
        - W_flip anticommutes with S
        - g(h_t) ∈ [0,1] is the seam gate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gate_type: str = "learned",  # 'fixed', 'learned', 'kstar'
        fixed_gate_value: float = 0.5,
        kstar: float = 0.721,
        tau: float = 0.1,
        even_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gate_type = gate_type
        self.fixed_gate_value = fixed_gate_value
        self.kstar = kstar
        self.tau = tau

        # Validate dimensions for parity splitting
        if even_dim is None and hidden_dim % 2 != 0:
            raise ValueError(
                f"SeamGatedRNN requires even hidden_dim when even_dim is not provided. "
                f"Got hidden_dim={hidden_dim}. Either use even hidden_dim or provide even_dim explicitly."
            )

        # Parity structure (will validate dimensions internally)
        self.parity_op = ParityOperator(hidden_dim, even_dim)
        self.projectors = ParityProjectors(self.parity_op)

        self.even_dim = self.parity_op.even_dim
        self.odd_dim = self.parity_op.odd_dim

        # Commutant weights (block diagonal)
        self.W_even = nn.Linear(input_dim + self.even_dim, self.even_dim, bias=False)
        self.W_odd = nn.Linear(input_dim + self.odd_dim, self.odd_dim, bias=False)

        # Anticommutant weights (off-block-diagonal)
        # W_flip = [0      B₊₋]
        #          [B₋₊    0 ]
        self.W_even_to_odd = nn.Linear(input_dim + self.even_dim, self.odd_dim, bias=False)
        self.W_odd_to_even = nn.Linear(input_dim + self.odd_dim, self.even_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Gate network (if learned)
        if gate_type == "learned":
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
            )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Move parity structures to correct device
        self.register_buffer("S", self.parity_op.S)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            h: Initial hidden state (batch, hidden_dim) or None

        Returns:
            outputs: (batch, seq_len, output_dim) or (batch, output_dim)
            h_final: (batch, hidden_dim)
            gates: (batch, seq_len) gate values
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        batch_size, seq_len, _ = x.shape

        if h is None:
            h = x.new_zeros(batch_size, self.hidden_dim)

        outputs = []
        gates = []

        for t in range(seq_len):
            h, g = self.step(x[:, t], h)
            y = self.output_layer(h)
            outputs.append(y)
            gates.append(g)

        outputs = torch.stack(outputs, dim=1)
        gates = torch.stack(gates, dim=1)

        if single_step:
            outputs = outputs.squeeze(1) if outputs.dim() > 2 else outputs
            gates = gates.squeeze(1) if gates.dim() > 1 else gates

        return outputs, h, gates

    def step(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step with seam gate"""
        # Track if either input is unbatched (1D) - if so, return unbatched output
        x_was_1d = x.dim() == 1
        h_was_1d = h.dim() == 1
        return_1d = x_was_1d or h_was_1d

        # Ensure both inputs have batch dimension for consistent processing
        if x_was_1d:
            x = x.unsqueeze(0)
        if h_was_1d:
            h = h.unsqueeze(0)

        # Compute gate
        g = self.compute_gate(h)

        # Extract even and odd components directly
        h_even = h[..., : self.even_dim]
        h_odd = h[..., self.even_dim :]

        # Concatenate input
        u_even = torch.cat([x, h_even], dim=-1)
        u_odd = torch.cat([x, h_odd], dim=-1)

        # Commutant contribution (parity-preserving)
        h_comm_even = self.W_even(u_even)
        h_comm_odd = self.W_odd(u_odd)

        # Anticommutant contribution (parity-swapping) with S applied
        # Apply S to h and extract components
        h_flipped = h @ self.S.T
        h_flipped_even = h_flipped[..., : self.even_dim]
        h_flipped_odd = h_flipped[..., self.even_dim :]

        u_even_flipped = torch.cat([x, h_flipped_even], dim=-1)
        u_odd_flipped = torch.cat([x, h_flipped_odd], dim=-1)

        h_flip_odd = self.W_even_to_odd(u_even_flipped)
        h_flip_even = self.W_odd_to_even(u_odd_flipped)

        # Combine
        h_even_next = h_comm_even + g.unsqueeze(-1) * h_flip_even
        h_odd_next = h_comm_odd + g.unsqueeze(-1) * h_flip_odd

        h_next = torch.cat([h_even_next, h_odd_next], dim=-1) + self.bias
        h_next = torch.tanh(h_next)

        # Restore original dimensionality - squeeze if either input was 1D
        if return_1d and h_next.size(0) == 1:
            h_next = h_next.squeeze(0)
            g = g.squeeze(0)

        return h_next, g

    def compute_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Compute seam gate value g(h) ∈ [0, 1]"""
        if self.gate_type == "fixed":
            return h.new_full((h.shape[0],), self.fixed_gate_value)

        elif self.gate_type == "learned":
            return self.gate_mlp(h).squeeze(-1)

        elif self.gate_type == "kstar":
            alpha_minus = self.projectors.parity_energy(h)
            gate = torch.sigmoid((alpha_minus - self.kstar) / self.tau)
            return gate

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

    def _apply_parity_to_concat(self, u: torch.Tensor) -> torch.Tensor:
        """Apply parity operator to concatenated [x, h]"""
        # Extract h part and apply S
        h = u[:, self.input_dim :]
        x = u[:, : self.input_dim]

        h_flipped = h @ self.S.T
        return torch.cat([x, h_flipped], dim=-1)


class GRUBaseline(nn.Module):
    """Standard GRU baseline"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            h: Initial hidden state (batch, hidden_dim) or None

        Returns:
            outputs: (batch, seq_len, output_dim) or (batch, output_dim)
            h_final: (batch, hidden_dim)
            gates: Dummy gates for API compatibility
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        # Expand h to GRU's expected (num_layers, batch, hidden) format if needed
        if h is not None and h.dim() == 2:
            h = h.unsqueeze(0)

        h_out, h_final = self.gru(x, h)
        outputs = self.output_layer(h_out)

        if single_step:
            outputs = outputs.squeeze(1)

        # Squeeze h_final to (batch, hidden_dim) for consistency with other models
        h_final = h_final.squeeze(0)

        # Return dummy gates for compatibility
        gates = x.new_zeros(outputs.shape[0], outputs.shape[1] if outputs.dim() > 2 else 1)

        return outputs, h_final, gates
