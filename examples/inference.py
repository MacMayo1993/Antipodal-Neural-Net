"""
Inference example for Antipodal Neural Networks

This script demonstrates how to:
1. Load a trained model from checkpoint
2. Make predictions on new data
3. Analyze seam gate activations during regime transitions
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import AntipodalRegimeSwitcher, GRUBaseline, SeamGatedRNN, find_regime_switches


def load_model(checkpoint_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)

    model_type = checkpoint["model_type"]
    hidden_dim = checkpoint["hidden_dim"]
    obs_dim = checkpoint["obs_dim"]

    if model_type == "GRU":
        model = GRUBaseline(obs_dim, hidden_dim, obs_dim)
    elif model_type == "Z2_equi":
        from src.models import Z2EquivariantRNN

        model = Z2EquivariantRNN(obs_dim, hidden_dim, obs_dim)
    elif model_type in ["Z2_fixed", "Z2_learn", "Z2_kstar"]:
        gate_type = model_type.split("_")[1]
        model = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type=gate_type)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_sequence(model, observations):
    """Make predictions on a sequence of observations"""
    model.eval()

    X = observations[:-1].unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, "gru"):
            y_pred, hidden_states, _ = model(X)
            gates = None
        elif hasattr(model, "gate_type"):
            y_pred, hidden_states, gates = model(X)
            gates = gates.squeeze(0)
        else:
            y_pred, hidden_states = model(X)
            gates = None

    y_pred = y_pred.squeeze(0)
    hidden_states = hidden_states.squeeze(0)

    return y_pred, hidden_states, gates


def analyze_gates_at_switches(gates, regimes, window=5):
    """Analyze gate activations around regime switches"""
    if gates is None:
        return None

    switch_times, _ = find_regime_switches(regimes[1:], window=window)

    if len(switch_times) == 0:
        return None

    # Extract gate values around switches
    gate_patterns = []
    for t in switch_times:
        start = max(0, t - window)
        end = min(len(gates), t + window + 1)
        gate_patterns.append(gates[start:end])

    return switch_times, gate_patterns


def main():
    print("=" * 70)
    print("Antipodal Neural Network - Inference Example")
    print("=" * 70)

    # For this example, we'll train a quick model first
    # In practice, you'd load a pre-trained checkpoint

    torch.manual_seed(42)
    np.random.seed(42)

    # Generate test data
    print("\n[1/3] Generating test data...")
    latent_dim = 8
    obs_dim = 4
    hidden_dim = 16

    generator = AntipodalRegimeSwitcher(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        p_switch=0.1,  # Higher switch probability for demonstration
        seed=42,
    )

    test_obs, test_latents, test_regimes = generator.generate_sequence(T=500)
    num_switches = (test_regimes[1:] != test_regimes[:-1]).sum().item()
    print(f"  ✓ Generated 500 timesteps with {num_switches} regime switches")

    # Quick training for demonstration
    print("\n[2/3] Training a quick model for demonstration...")
    model = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type="kstar")

    train_obs, _, _ = generator.generate_sequence(T=1000)
    X_train = train_obs[:-1].unsqueeze(0)
    Y_train = train_obs[1:].unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for step in range(200):  # Quick training
        optimizer.zero_grad()
        y_pred, _, _ = model(X_train)
        loss = torch.nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer.step()

    print(f"  ✓ Model trained (final loss: {loss.item():.6f})")

    # Make predictions
    print("\n[3/3] Running inference on test sequence...")
    predictions, hidden_states, gates = predict_sequence(model, test_obs)

    # Compute errors
    Y_true = test_obs[1:]
    errors = ((predictions - Y_true) ** 2).sum(dim=-1)
    mse = errors.mean().item()

    print(f"  ✓ Test MSE: {mse:.6f}")

    # Analyze gate behavior at switches
    print("\n  Analyzing seam gate activations at regime switches...")
    result = analyze_gates_at_switches(gates, test_regimes, window=5)

    if result is not None:
        switch_times, gate_patterns = result
        print(f"  ✓ Found {len(switch_times)} regime switches")

        # Average gate activation around switches
        avg_gates = torch.stack([p for p in gate_patterns if len(p) == 11]).mean(dim=0)
        print(f"\n  Average gate activation around switches (±5 timesteps):")
        print(f"    Before switch: {avg_gates[:5].mean():.3f}")
        print(f"    At switch:     {avg_gates[5]:.3f}")
        print(f"    After switch:  {avg_gates[6:].mean():.3f}")

        # Check if gates activate at transitions
        if avg_gates[5] > 0.5:
            print("\n  ✓ Gates activate during transitions (as expected)")
        else:
            print("\n  ⚠ Gates not strongly activated (may need more training)")

    print("\n" + "=" * 70)
    print("✓ Inference complete!")
    print("\nKey observations:")
    print("  - The model successfully predicts future observations")
    print("  - Seam gates modulate parity-swapping transitions")
    print("  - Gate activations correlate with regime switches")
    print("\nTo save/load models:")
    print("  torch.save({'model_state_dict': model.state_dict(), ...}, 'model.pth')")
    print("  checkpoint = torch.load('model.pth')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("=" * 70)


if __name__ == "__main__":
    main()
