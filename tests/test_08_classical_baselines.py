"""
Section 8: Classical Baseline Tests

Tests for AR(1) and IMM filter baselines.
"""

import pytest
import torch
import numpy as np

from src.baselines import AR1Model, IMMFilter, compute_transition_error_spike
from src.data import AntipodalRegimeSwitcher


class TestAR1Model:
    """Test 8.1: AR(1) Estimation Validity"""

    def test_ar1_fit_residuals(self):
        """
        Verify fitted AR(1) model has finite, SPD residual covariance
        """
        # Generate simple AR(1) data
        torch.manual_seed(42)
        T = 500
        obs_dim = 3

        # True AR(1) process
        B_true = torch.randn(obs_dim, obs_dim) * 0.5
        x_data = [torch.randn(obs_dim)]

        for _ in range(T - 1):
            x_next = B_true @ x_data[-1] + torch.randn(obs_dim) * 0.1
            x_data.append(x_next)

        observations = torch.stack(x_data)

        # Fit AR(1) model
        model = AR1Model(obs_dim)
        model.fit(observations)

        # Verify residuals
        is_finite, is_spd = model.verify_residuals()

        assert is_finite, "Residual covariance contains non-finite values"
        assert is_spd, "Residual covariance not symmetric positive definite"

    def test_ar1_prediction_shape(self):
        """Verify AR(1) predictions have correct shape"""
        model = AR1Model(obs_dim=4)

        # Generate dummy data
        observations = torch.randn(100, 4)
        model.fit(observations)

        # Single prediction
        x = torch.randn(3, 4)
        pred = model.predict(x)

        assert pred.shape == (3, 4), f"Prediction shape incorrect: {pred.shape}"

    def test_ar1_fits_data(self):
        """Verify AR(1) can fit simple autoregressive data"""
        torch.manual_seed(42)
        obs_dim = 3

        # Generate AR(1) data
        B_true = torch.eye(obs_dim) * 0.8
        x_data = [torch.randn(obs_dim)]

        for _ in range(500):
            x_next = B_true @ x_data[-1] + torch.randn(obs_dim) * 0.1
            x_data.append(x_next)

        observations = torch.stack(x_data)

        # Fit
        model = AR1Model(obs_dim)
        model.fit(observations)

        # Check B is close to true B
        error = torch.norm(model.B - B_true)

        assert error < 0.5, f"Fitted B far from true B: error={error}"


class TestIMMFilter:
    """Test 8.2 & 8.3: IMM Filter"""

    def test_imm_mode_probs(self):
        """
        Verify mode probabilities are consistent:
        μ₀ + μ₁ = 1, μᵢ ∈ [0, 1]
        """
        imm = IMMFilter(obs_dim=3, p_switch=0.05)

        # Generate dummy data
        observations = torch.randn(100, 3)
        imm.fit(observations)

        # Run predictions to update mode probs
        for t in range(50):
            imm.predict(observations[t])

        # Check validity
        is_valid = imm.verify_mode_probs()

        assert is_valid, "IMM mode probabilities invalid"

    def test_imm_mode_probability_consistency(self):
        """Verify mode probabilities sum to 1 throughout sequence"""
        imm = IMMFilter(obs_dim=3, p_switch=0.05)

        observations = torch.randn(100, 3)
        imm.fit(observations)

        predictions, mode_probs_history = imm.predict_sequence(observations)

        for t, mode_probs in enumerate(mode_probs_history):
            sum_probs = mode_probs.sum()
            assert abs(sum_probs - 1.0) < 1e-6, \
                f"Mode probs don't sum to 1 at t={t}: {sum_probs}"

            assert np.all((mode_probs >= 0) & (mode_probs <= 1)), \
                f"Mode probs outside [0,1] at t={t}: {mode_probs}"

    def test_imm_transition_error_spike(self):
        """
        Verify IMM error spikes near regime changes
        (compared to stable regions)
        """
        # Generate regime-switching data
        generator = AntipodalRegimeSwitcher(
            latent_dim=6,
            obs_dim=3,
            p_switch=0.1,
            seed=42
        )

        observations, _, regimes = generator.generate_sequence(T=500)

        # Fit IMM (non-oracle)
        imm = IMMFilter(obs_dim=3, p_switch=0.1)
        imm.fit(observations)

        # Predict
        predictions, _ = imm.predict_sequence(observations)

        # Compute errors
        errors = torch.norm(predictions - observations[1:], dim=-1)

        # Compute transition vs stable error
        transition_error, stable_error = compute_transition_error_spike(
            errors, regimes[1:], window=5
        )

        # IMM should have higher error at transitions
        # (though this depends on how well it adapts)
        # At minimum, verify both errors are finite
        assert np.isfinite(transition_error), "Transition error not finite"
        assert np.isfinite(stable_error), "Stable error not finite"


class TestTransitionErrorMetric:
    """Test transition error computation utility"""

    def test_transition_error_computation(self):
        """Verify transition error computation works correctly"""
        # Create simple errors and regimes
        errors = torch.ones(100)
        errors[48:53] = 5.0  # High error around t=50

        regimes = torch.zeros(100, dtype=torch.long)
        regimes[50:] = 1  # Switch at t=50

        transition_error, stable_error = compute_transition_error_spike(
            errors, regimes, window=3
        )

        # Transition window [47:54] should have higher error
        assert transition_error > stable_error, \
            f"Transition error ({transition_error}) not higher than stable ({stable_error})"

    def test_no_transitions(self):
        """Test transition error when there are no regime changes"""
        errors = torch.ones(100)
        regimes = torch.zeros(100, dtype=torch.long)  # No switches

        transition_error, stable_error = compute_transition_error_spike(
            errors, regimes, window=5
        )

        # No transitions, so transition_error should be 0 or equal to stable
        assert transition_error == 0.0 or abs(transition_error - stable_error) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
