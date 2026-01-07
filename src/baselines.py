"""
Classical Baselines for Regime Switching

Implements:
- AR(1) model
- Interacting Multiple Model (IMM) filter for regime switching
"""

import torch
import numpy as np
from typing import Tuple, Optional


class AR1Model:
    """
    Autoregressive model of order 1: x_t = B x_{t-1} + ε_t

    Fitted via least squares on training data.
    """

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.B = None
        self.noise_cov = None

    def fit(self, observations: torch.Tensor):
        """
        Fit AR(1) model via least squares.

        Args:
            observations: (T, obs_dim) time series
        """
        X = observations[:-1]  # x_t
        Y = observations[1:]  # x_{t+1}

        # Least squares: B = (XᵀX)⁻¹XᵀY
        XtX = X.T @ X
        XtY = X.T @ Y

        # Add regularization for stability
        reg = 1e-6 * torch.eye(self.obs_dim, device=X.device)
        self.B = torch.linalg.solve(XtX + reg, XtY)

        # Estimate noise covariance
        residuals = Y - X @ self.B
        self.noise_cov = (residuals.T @ residuals) / (len(Y) - 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-step prediction: x_{t+1} = B x_t

        Args:
            x: (batch, obs_dim) current observations

        Returns:
            predictions: (batch, obs_dim)
        """
        if self.B is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return x @ self.B

    def predict_sequence(self, x_init: torch.Tensor, T: int) -> torch.Tensor:
        """
        Multi-step prediction.

        Args:
            x_init: (obs_dim,) initial observation
            T: Number of steps

        Returns:
            predictions: (T, obs_dim)
        """
        predictions = [x_init]
        x = x_init

        for _ in range(T - 1):
            x = self.predict(x.unsqueeze(0)).squeeze(0)
            predictions.append(x)

        return torch.stack(predictions)

    def verify_residuals(self) -> Tuple[bool, bool]:
        """
        Verify residual covariance is finite and SPD.

        Returns:
            is_finite: All entries finite
            is_spd: Symmetric positive definite
        """
        if self.noise_cov is None:
            return False, False

        is_finite = torch.all(torch.isfinite(self.noise_cov)).item()

        # Check SPD: all eigenvalues > 0
        try:
            eigenvalues = torch.linalg.eigvalsh(self.noise_cov)
            is_spd = torch.all(eigenvalues > 0).item()
        except:
            is_spd = False

        return is_finite, is_spd


class IMMFilter:
    """
    Interacting Multiple Model (IMM) filter for regime-switching AR models.

    Models two regimes with different dynamics and switching probabilities.
    Non-oracle version: fitted from training data without knowledge of true regimes.
    """

    def __init__(self, obs_dim: int, p_switch: float = 0.05):
        """
        Args:
            obs_dim: Observation dimension
            p_switch: Regime switching probability
        """
        self.obs_dim = obs_dim
        self.p_switch = p_switch

        # Regime models (will be fitted)
        self.model_0 = AR1Model(obs_dim)
        self.model_1 = AR1Model(obs_dim)

        # Mode probabilities
        self.mode_probs = np.array([0.5, 0.5])

        # Transition matrix
        self.transition_matrix = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])

    def fit(self, observations: torch.Tensor, regimes: Optional[torch.Tensor] = None):
        """
        Fit IMM filter.

        If regimes are provided, fit separate models per regime.
        Otherwise, use simple heuristic (alternating or clustering).

        Args:
            observations: (T, obs_dim) time series
            regimes: (T,) regime indicators (optional, for oracle version)
        """
        if regimes is not None:
            # Oracle: fit models on separate regimes
            mask_0 = regimes == 0
            mask_1 = regimes == 1

            obs_0 = observations[mask_0]
            obs_1 = observations[mask_1]

            if len(obs_0) > 1:
                self.model_0.fit(obs_0)
            if len(obs_1) > 1:
                self.model_1.fit(obs_1)
        else:
            # Non-oracle: use simple heuristic
            # Fit both models on full data (they'll learn average dynamics)
            self.model_0.fit(observations)
            self.model_1.fit(observations)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        One-step IMM prediction.

        Args:
            x: (obs_dim,) current observation

        Returns:
            prediction: (obs_dim,) weighted prediction
            mode_probs: (2,) updated mode probabilities
        """
        # Prediction from each model
        pred_0 = self.model_0.predict(x.unsqueeze(0)).squeeze(0)
        pred_1 = self.model_1.predict(x.unsqueeze(0)).squeeze(0)

        # Update mode probabilities (simplified, no measurement update)
        self.mode_probs = self.transition_matrix.T @ self.mode_probs

        # Weighted prediction
        prediction = self.mode_probs[0] * pred_0 + self.mode_probs[1] * pred_1

        return prediction, self.mode_probs.copy()

    def predict_sequence(self, observations: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Multi-step IMM prediction.

        Args:
            observations: (T, obs_dim) observations

        Returns:
            predictions: (T-1, obs_dim) one-step-ahead predictions
            mode_probs_history: (T-1, 2) mode probability evolution
        """
        T = len(observations)
        predictions = []
        mode_probs_history = []

        for t in range(T - 1):
            pred, mode_probs = self.predict(observations[t])
            predictions.append(pred)
            mode_probs_history.append(mode_probs)

        return torch.stack(predictions), np.array(mode_probs_history)

    def verify_mode_probs(self) -> bool:
        """
        Verify mode probabilities are valid.

        Checks:
        - μ₀ + μ₁ = 1
        - μᵢ ∈ [0, 1]

        Returns:
            True if valid
        """
        sum_valid = np.abs(self.mode_probs.sum() - 1.0) < 1e-6
        range_valid = np.all((self.mode_probs >= 0) & (self.mode_probs <= 1))

        return sum_valid and range_valid


def compute_transition_error_spike(
    errors: torch.Tensor, regimes: torch.Tensor, window: int = 5
) -> Tuple[float, float]:
    """
    Measure error spike around regime transitions.

    Args:
        errors: (T,) prediction errors
        regimes: (T,) regime indicators
        window: Window around transitions

    Returns:
        mean_transition_error: Mean error in transition windows
        mean_stable_error: Mean error outside transitions
    """
    # Find transitions
    switches = regimes[1:] != regimes[:-1]
    switch_indices = torch.where(switches)[0] + 1

    # Mark transition windows
    T = len(errors)
    transition_mask = torch.zeros(T, dtype=torch.bool)

    for idx in switch_indices:
        start = max(0, idx - window)
        end = min(T, idx + window + 1)
        transition_mask[start:end] = True

    # Compute errors
    if transition_mask.any():
        transition_error = errors[transition_mask].mean().item()
    else:
        transition_error = 0.0

    if (~transition_mask).any():
        stable_error = errors[~transition_mask].mean().item()
    else:
        stable_error = 0.0

    return transition_error, stable_error
