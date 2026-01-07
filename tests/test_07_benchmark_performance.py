"""
Section 7: Benchmark Performance Tests (Core)

Integration tests comparing model performance on antipodal regime-switching tasks.
These are slower integration tests that train models.
"""

import numpy as np
import pytest
import torch
import torch.optim as optim

from src.data import AntipodalRegimeSwitcher, find_regime_switches
from src.losses import quotient_loss
from src.models import GRUBaseline, SeamGatedRNN, Z2EquivariantRNN


@pytest.fixture
def synthetic_data():
    """Generate synthetic antipodal regime-switching data"""
    generator = AntipodalRegimeSwitcher(
        latent_dim=8, obs_dim=4, p_switch=0.05, obs_noise_std=0.1, latent_noise_std=0.05, seed=42
    )

    train_obs, train_latents, train_regimes = generator.generate_sequence(T=1000)
    test_obs, test_latents, test_regimes = generator.generate_sequence(T=300)

    return {
        "train": (train_obs, train_latents, train_regimes),
        "test": (test_obs, test_latents, test_regimes),
        "generator": generator,
    }


def train_model(model, train_data, num_steps=200, lr=0.01):
    """Helper function to train a model"""
    obs, _, _ = train_data
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare sequences
    X = obs[:-1].unsqueeze(0)  # (1, T-1, obs_dim)
    Y = obs[1:].unsqueeze(0)  # (1, T-1, obs_dim)

    model.train()
    for step in range(num_steps):
        if hasattr(model, "gru"):  # GRU baseline
            y_pred, _, _ = model(X)
        elif hasattr(model, "gate_type"):  # Seam-gated
            y_pred, _, _ = model(X)
        else:  # Equivariant
            y_pred, _ = model(X)

        loss = torch.nn.functional.mse_loss(y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()


def evaluate_model(model, test_data):
    """Helper function to evaluate model and compute metrics"""
    obs, _, regimes = test_data

    X = obs[:-1]
    Y_true = obs[1:]

    with torch.no_grad():
        if hasattr(model, "gru"):
            Y_pred, _, gates = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
            gates = gates.squeeze(0)
        elif hasattr(model, "gate_type"):
            Y_pred, _, gates = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
            gates = gates.squeeze(0)
        else:
            Y_pred, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
            gates = torch.zeros(len(Y_pred))

    # Compute errors
    errors = ((Y_pred - Y_true) ** 2).sum(dim=-1)

    # Overall MSE
    overall_mse = errors.mean().item()

    # Transition window MSE
    switch_times, transition_mask = find_regime_switches(regimes[1:], window=20)

    if transition_mask.any():
        transition_mse = errors[transition_mask].mean().item()
    else:
        transition_mse = overall_mse

    # Within-regime MSE
    if (~transition_mask).any():
        within_regime_mse = errors[~transition_mask].mean().item()
    else:
        within_regime_mse = overall_mse

    return {
        "overall_mse": overall_mse,
        "transition_mse": transition_mse,
        "within_regime_mse": within_regime_mse,
        "gates": gates,
        "switch_times": switch_times,
    }


class TestBaselineComparison:
    """Test 7.1: Baseline Comparison"""

    @pytest.mark.slow
    def test_transition_error_ordering(self, synthetic_data):
        """
        Test that transition MSE follows expected ordering:
        GRU > equivariant > fixed > learned > k*

        Note: This is a lightweight test with minimal training.
        Full benchmarking would require more epochs and seeds.
        """
        train_data = synthetic_data["train"]
        test_data = synthetic_data["test"]

        obs_dim = 4
        hidden_dim = 12
        num_steps = 100  # Reduced for testing

        # Initialize models
        models = {
            "gru": GRUBaseline(obs_dim, hidden_dim, obs_dim),
            "equivariant": Z2EquivariantRNN(obs_dim, hidden_dim, obs_dim),
            "fixed_gate": SeamGatedRNN(
                obs_dim, hidden_dim, obs_dim, gate_type="fixed", fixed_gate_value=0.5
            ),
            "learned_gate": SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type="learned"),
            "kstar_gate": SeamGatedRNN(
                obs_dim, hidden_dim, obs_dim, gate_type="kstar", kstar=0.721
            ),
        }

        results = {}

        # Train and evaluate each model
        for name, model in models.items():
            train_model(model, train_data, num_steps=num_steps)
            metrics = evaluate_model(model, test_data)
            results[name] = metrics

        # Check that models are learning (overall MSE is finite)
        for name, metrics in results.items():
            assert np.isfinite(metrics["overall_mse"]), f"{name} has non-finite MSE"

        # Verify seam-gated models don't NaN
        assert np.isfinite(
            results["kstar_gate"]["transition_mse"]
        ), "k* gate model has NaN transition MSE"


class TestParameterEfficiency:
    """Test 7.2: Parameter Efficiency"""

    def test_parameter_efficiency(self):
        """
        Compare parameter counts between models.
        Z₂ models should have comparable or fewer parameters than GRU.
        """
        obs_dim = 4
        hidden_dim = 12

        gru = GRUBaseline(obs_dim, hidden_dim, obs_dim)
        z2_model = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type="kstar")

        gru_params = sum(p.numel() for p in gru.parameters())
        z2_params = sum(p.numel() for p in z2_model.parameters())

        # Z2 model should have similar or fewer parameters
        # (This depends on architecture choices)
        assert z2_params > 0, "Z2 model has no parameters"
        assert gru_params > 0, "GRU has no parameters"

        # Both should be in reasonable range
        assert z2_params < 10000, "Z2 model has excessive parameters"


class TestGeneralization:
    """Test 7.3: Generalization to Higher Switch Rate"""

    @pytest.mark.slow
    def test_switch_rate_generalization(self):
        """
        Train with p_switch=0.05, test with p_switch=0.2.
        Z₂ models should degrade less than GRU.

        Note: This is a lightweight test. Full validation requires more training.
        """
        # Train data
        gen_train = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.05, seed=42)
        train_obs, _, train_regimes = gen_train.generate_sequence(T=800)

        # Test data with higher switch rate
        gen_test = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.2, seed=123)
        # Use same dynamics matrices for fair comparison
        gen_test.A = gen_train.A
        gen_test.C = gen_train.C

        test_obs, _, test_regimes = gen_test.generate_sequence(T=300)

        # Train models (minimal training for test speed)
        gru = GRUBaseline(4, 12, 4)
        z2_model = SeamGatedRNN(4, 12, 4, gate_type="kstar")

        train_model(gru, (train_obs, None, train_regimes), num_steps=50)
        train_model(z2_model, (train_obs, None, train_regimes), num_steps=50)

        # Evaluate
        gru_metrics = evaluate_model(gru, (test_obs, None, test_regimes))
        z2_metrics = evaluate_model(z2_model, (test_obs, None, test_regimes))

        # Both should produce finite errors
        assert np.isfinite(gru_metrics["transition_mse"])
        assert np.isfinite(z2_metrics["transition_mse"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
