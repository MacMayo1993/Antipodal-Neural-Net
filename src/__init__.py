"""
Antipodal Neural Networks with Z_2 Seam Gating

A library for non-orientable neural networks with parity structure,
designed for regime-switching time series with antipodal symmetry.
"""

__version__ = "0.1.0"

# Core models
from .models import Z2EquivariantRNN, SeamGatedRNN, GRUBaseline

# Parity operators
from .parity import ParityOperator, ParityProjectors

# Data generation
from .data import AntipodalRegimeSwitcher, find_regime_switches

# Loss functions
from .losses import quotient_loss, rank1_projector_loss

# Baselines
from .baselines import AR1Model, IMMFilter

__all__ = [
    # Models
    "Z2EquivariantRNN",
    "SeamGatedRNN",
    "GRUBaseline",
    # Parity
    "ParityOperator",
    "ParityProjectors",
    # Data
    "AntipodalRegimeSwitcher",
    "find_regime_switches",
    # Losses
    "quotient_loss",
    "rank1_projector_loss",
    # Baselines
    "AR1Model",
    "IMMFilter",
]
