"""
Unified random seed utilities for reproducibility across CPU/GPU.

Ensures deterministic behavior for PyTorch, NumPy, and Python's random module.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for all libraries and optionally enable deterministic operations.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (may impact performance)

    Example:
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Note: Some operations may not have deterministic implementations
        # This will warn rather than error
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def create_generator(seed: Optional[int] = None, device: str = "cpu") -> torch.Generator:
    """
    Create a torch.Generator with optional seed for reproducible random operations.

    Args:
        seed: Random seed value (if None, uses current RNG state)
        device: Device for generator ('cpu' or 'cuda')

    Returns:
        torch.Generator instance

    Example:
        >>> gen = create_generator(42, device='cpu')
        >>> x = torch.randn(10, generator=gen)  # Reproducible
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def get_rng_state() -> dict:
    """
    Capture current RNG state for all libraries.

    Returns:
        Dictionary containing RNG states

    Example:
        >>> state = get_rng_state()
        >>> # ... some random operations ...
        >>> set_rng_state(state)  # Restore state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """
    Restore RNG state from previously captured state.

    Args:
        state: Dictionary containing RNG states (from get_rng_state)

    Example:
        >>> state = get_rng_state()
        >>> # ... some random operations ...
        >>> set_rng_state(state)  # Restore state
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
