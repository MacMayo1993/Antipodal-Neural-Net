"""
Section 1: Data Generator Tests

Tests for the antipodal regime switching data generator.
"""

import pytest
import torch
import numpy as np

from src.data import AntipodalRegimeSwitcher, find_regime_switches


# Test seeds for reproducibility
TEST_SEEDS = [42, 123, 456, 789, 1011]


class TestAntipodalDynamics:
    """Test 1.1: Antipodal Regime Generator Correctness"""

    @pytest.mark.parametrize("seed", TEST_SEEDS[:3])
    def test_antipodal_dynamics_symmetry(self, seed):
        """
        Verify that latent dynamics are exactly antipodal:
        A z + (-A)(-z) ≈ 0
        """
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, p_switch=0.05, seed=seed)

        # Test with random latent states
        torch.manual_seed(seed)
        for _ in range(10):
            z = torch.randn(generator.latent_dim)

            # Compute both regime updates
            z_next_A = generator.A @ z
            z_next_B = -generator.A @ (-z)

            # Should be identical
            symmetry_error = torch.norm(z_next_A - z_next_B).item()

            assert symmetry_error < 1e-6, f"Antipodal symmetry violated: error = {symmetry_error}"

    def test_dynamics_matrices_stable(self):
        """Verify dynamics matrix A has spectral radius < 1"""
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, seed=42)

        eigenvalues = torch.linalg.eigvals(generator.A)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()

        assert spectral_radius < 0.96, f"Dynamics unstable: spectral radius = {spectral_radius}"


class TestRegimeSwitching:
    """Test 1.2: Regime Switching Statistics"""

    @pytest.mark.parametrize("p_switch", [0.01, 0.05, 0.1])
    def test_switch_probability(self, p_switch):
        """
        Verify empirical switch rate matches theoretical probability.
        """
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, p_switch=p_switch, seed=42)

        # Generate long sequence
        T = 50000
        empirical_rate = generator.estimate_switch_rate(T=T)

        # Allow for statistical variation (3 sigma bound)
        # Variance of Bernoulli: p(1-p) / n
        std_error = np.sqrt(p_switch * (1 - p_switch) / T)
        tolerance = 3 * std_error

        error = abs(empirical_rate - p_switch)

        assert (
            error < tolerance
        ), f"Switch rate mismatch: expected {p_switch:.4f}, got {empirical_rate:.4f} (error {error:.4f} > tol {tolerance:.4f})"

    def test_regime_sequence_validity(self):
        """Verify regime sequence contains only 0 and 1"""
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, seed=42)

        _, _, regimes = generator.generate_sequence(T=1000)

        assert torch.all((regimes == 0) | (regimes == 1)), "Invalid regime values"


class TestPartialObservability:
    """Test 1.3: Partial Observability Check"""

    def test_observation_rank_deficiency(self):
        """
        Verify observation matrix C has rank < latent dimension.
        This ensures latent sign is not directly observable.
        """
        latent_dim = 10
        obs_dim = 6

        generator = AntipodalRegimeSwitcher(latent_dim=latent_dim, obs_dim=obs_dim, seed=42)

        rank = generator.verify_observation_rank()

        assert (
            rank < latent_dim
        ), f"Observation matrix not rank-deficient: rank={rank}, latent_dim={latent_dim}"

    @pytest.mark.parametrize("latent_dim,obs_dim", [(10, 5), (20, 10), (8, 4)])
    def test_rank_deficiency_various_dims(self, latent_dim, obs_dim):
        """Test rank deficiency for various dimensions"""
        generator = AntipodalRegimeSwitcher(latent_dim=latent_dim, obs_dim=obs_dim, seed=42)

        rank = generator.verify_observation_rank()

        assert (
            rank < latent_dim
        ), f"Rank deficiency failed for dim={latent_dim}, obs={obs_dim}: rank={rank}"


class TestSequenceGeneration:
    """Additional tests for sequence generation"""

    def test_sequence_shapes(self):
        """Verify generated sequences have correct shapes"""
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, seed=42)

        T = 100
        obs, latents, regimes = generator.generate_sequence(T)

        assert obs.shape == (T, 5), f"Observation shape incorrect: {obs.shape}"
        assert latents.shape == (T, 10), f"Latent shape incorrect: {latents.shape}"
        assert regimes.shape == (T,), f"Regime shape incorrect: {regimes.shape}"

    def test_initial_state_propagation(self):
        """Verify initial state is used correctly"""
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, seed=42)

        initial_state = torch.ones(10)
        _, latents, _ = generator.generate_sequence(T=10, initial_state=initial_state)

        # First latent should match initial state
        assert torch.allclose(latents[0], initial_state, atol=1e-6)

    def test_no_nans_in_sequence(self):
        """Verify generated sequences contain no NaN or Inf values"""
        generator = AntipodalRegimeSwitcher(latent_dim=10, obs_dim=5, seed=42)

        obs, latents, regimes = generator.generate_sequence(T=1000)

        assert torch.all(torch.isfinite(obs)), "NaN/Inf in observations"
        assert torch.all(torch.isfinite(latents)), "NaN/Inf in latents"


class TestRegimeSwitchDetection:
    """Tests for regime switch detection utilities"""

    def test_find_regime_switches(self):
        """Verify switch detection finds correct switch points"""
        regimes = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        switch_times, _ = find_regime_switches(regimes)

        expected_switches = torch.tensor([3, 6, 8])

        assert torch.equal(
            switch_times, expected_switches
        ), f"Switch detection incorrect: got {switch_times}, expected {expected_switches}"

    def test_transition_window_mask(self):
        """Verify transition window mask is created correctly"""
        regimes = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        window = 1

        switch_times, transition_mask = find_regime_switches(regimes, window=window)

        # Switch at t=3: window covers [2, 4]
        # Switch at t=6: window covers [5, 7]
        # Switch at t=8: window covers [7, 9]

        assert transition_mask[2] == True
        assert transition_mask[3] == True
        assert transition_mask[4] == True
        assert transition_mask[1] == False  # Before first switch window


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
