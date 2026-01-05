"""
Data Generator for Antipodal Regime Switching

Generates time series with:
- Latent dynamics that switch between A and -A
- Partial observability (rank-deficient observation matrix)
- Markovian regime switching
"""

import torch
import numpy as np
from typing import Tuple, Optional


class AntipodalRegimeSwitcher:
    """
    Generate antipodal regime-switching time series.

    Latent dynamics:
        Regime A: z_{t+1} = A z_t + ε_t
        Regime B: z_{t+1} = -A z_t + ε_t

    Observations:
        x_t = C z_t + η_t

    where C has rank < latent_dim (partial observability)
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        p_switch: float = 0.05,
        obs_noise_std: float = 0.1,
        latent_noise_std: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Args:
            latent_dim: Dimension of latent state z
            obs_dim: Dimension of observations x (must be < latent_dim for partial observability)
            p_switch: Probability of regime switch per timestep
            obs_noise_std: Standard deviation of observation noise η
            latent_noise_std: Standard deviation of latent noise ε
            seed: Random seed for reproducibility
        """
        assert obs_dim < latent_dim, "obs_dim must be < latent_dim for partial observability"

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.p_switch = p_switch
        self.obs_noise_std = obs_noise_std
        self.latent_noise_std = latent_noise_std

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate stable dynamics matrix A
        self.A = self._generate_stable_dynamics()

        # Generate rank-deficient observation matrix C
        self.C = self._generate_observation_matrix()

    def _generate_stable_dynamics(self) -> torch.Tensor:
        """Generate stable dynamics matrix with spectral radius < 1"""
        # Random matrix
        A = torch.randn(self.latent_dim, self.latent_dim) * 0.5

        # Ensure stability
        eigenvalues = torch.linalg.eigvals(A)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()

        if spectral_radius > 0.95:
            A = A * (0.95 / spectral_radius)

        return A

    def _generate_observation_matrix(self) -> torch.Tensor:
        """Generate rank-deficient observation matrix"""
        # Create matrix with guaranteed rank deficiency
        C = torch.randn(self.obs_dim, self.latent_dim)

        # Verify rank deficiency
        rank = torch.linalg.matrix_rank(C).item()
        assert rank < self.latent_dim, f"C has rank {rank}, expected < {self.latent_dim}"

        return C

    def generate_sequence(
        self,
        T: int,
        initial_regime: int = 0,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate antipodal regime-switching sequence.

        Args:
            T: Sequence length
            initial_regime: Initial regime (0 or 1)
            initial_state: Initial latent state (if None, sample from N(0, I))

        Returns:
            observations: (T, obs_dim) tensor
            latents: (T, latent_dim) tensor
            regimes: (T,) tensor of regime indicators (0 or 1)
        """
        # Initialize
        if initial_state is None:
            z = torch.randn(self.latent_dim)
        else:
            z = initial_state.clone()

        regime = initial_regime

        latents = []
        observations = []
        regimes = []

        for t in range(T):
            # Record current state
            latents.append(z.clone())
            regimes.append(regime)

            # Generate observation
            obs_noise = torch.randn(self.obs_dim) * self.obs_noise_std
            x = self.C @ z + obs_noise
            observations.append(x)

            # Evolve latent state
            latent_noise = torch.randn(self.latent_dim) * self.latent_noise_std

            if regime == 0:
                z_next = self.A @ z + latent_noise
            else:  # regime == 1
                z_next = -self.A @ z + latent_noise

            z = z_next

            # Regime switching (Markov)
            if np.random.rand() < self.p_switch:
                regime = 1 - regime

        return (
            torch.stack(observations),
            torch.stack(latents),
            torch.tensor(regimes, dtype=torch.long)
        )

    def verify_antipodal_symmetry(self, z: torch.Tensor, regime_a: bool = True) -> float:
        """
        Verify antipodal dynamics: A z + (-A)(-z) ≈ 0

        Args:
            z: Latent state
            regime_a: If True, compare regime A forward vs regime B with -z

        Returns:
            Symmetry error (should be near 0)
        """
        z_next_A = self.A @ z
        z_next_B = -self.A @ (-z)

        error = torch.norm(z_next_A - z_next_B).item()
        return error

    def verify_observation_rank(self) -> int:
        """Verify observation matrix C has rank < latent_dim"""
        return torch.linalg.matrix_rank(self.C).item()

    def estimate_switch_rate(self, T: int = 50000) -> float:
        """
        Estimate empirical switch rate from long sequence.

        Args:
            T: Sequence length

        Returns:
            Empirical switch probability
        """
        _, _, regimes = self.generate_sequence(T)

        # Count switches
        switches = (regimes[1:] != regimes[:-1]).sum().item()
        switch_rate = switches / (T - 1)

        return switch_rate


def create_train_test_split(
    generator: AntipodalRegimeSwitcher,
    train_length: int,
    test_length: int,
    seed: Optional[int] = None
) -> Tuple[dict, dict]:
    """
    Create train/test split with separate sequences.

    Args:
        generator: AntipodalRegimeSwitcher instance
        train_length: Training sequence length
        test_length: Test sequence length
        seed: Random seed

    Returns:
        train_data: dict with 'obs', 'latents', 'regimes'
        test_data: dict with 'obs', 'latents', 'regimes'
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate train sequence
    train_obs, train_latents, train_regimes = generator.generate_sequence(train_length)

    # Generate test sequence
    test_obs, test_latents, test_regimes = generator.generate_sequence(test_length)

    train_data = {
        'obs': train_obs,
        'latents': train_latents,
        'regimes': train_regimes
    }

    test_data = {
        'obs': test_obs,
        'latents': test_latents,
        'regimes': test_regimes
    }

    return train_data, test_data


def find_regime_switches(regimes: torch.Tensor, window: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find regime switch times and create transition windows.

    Args:
        regimes: (T,) tensor of regime indicators
        window: Number of timesteps before/after switch to mark as transition

    Returns:
        switch_times: (num_switches,) tensor of switch indices
        transition_mask: (T,) boolean tensor, True in ±window around switches
    """
    # Find switch points
    switches = (regimes[1:] != regimes[:-1])
    switch_times = torch.where(switches)[0] + 1  # +1 because we compared [1:] with [:-1]

    # Create transition mask
    T = len(regimes)
    transition_mask = torch.zeros(T, dtype=torch.bool)

    for switch_t in switch_times:
        start = max(0, switch_t - window)
        end = min(T, switch_t + window + 1)
        transition_mask[start:end] = True

    return switch_times, transition_mask
