"""
Simple training example for Antipodal Neural Networks

This script demonstrates basic usage:
1. Generate synthetic antipodal regime-switching data
2. Train a ℤ₂-equivariant model with k*-based seam gating
3. Evaluate and compare against a GRU baseline
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src import AntipodalRegimeSwitcher, GRUBaseline, SeamGatedRNN, find_regime_switches


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Antipodal Neural Network - Simple Training Example")
    print("=" * 70)

    # Configuration
    latent_dim = 8
    obs_dim = 4
    hidden_dim = 16
    p_switch = 0.05
    train_length = 1000
    test_length = 500
    training_steps = 500
    lr = 0.01

    # Step 1: Generate training data
    print("\n[1/4] Generating training data...")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Observation dimension: {obs_dim}")
    print(f"  - Switch probability: {p_switch}")
    print(f"  - Sequence length: {train_length}")

    generator = AntipodalRegimeSwitcher(
        latent_dim=latent_dim, obs_dim=obs_dim, p_switch=p_switch, seed=42
    )

    train_obs, train_latents, train_regimes = generator.generate_sequence(T=train_length)
    test_obs, test_latents, test_regimes = generator.generate_sequence(T=test_length)

    num_switches = (train_regimes[1:] != train_regimes[:-1]).sum().item()
    print(f"  ✓ Generated {train_length} training timesteps with {num_switches} regime switches")

    # Verify observation rank
    rank = generator.verify_observation_rank()
    print(f"  ✓ Observation matrix rank: {rank} < {latent_dim} (partial observability)")

    # Step 2: Initialize models
    print("\n[2/4] Initializing models...")

    model_z2 = SeamGatedRNN(
        input_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=obs_dim,
        gate_type="kstar",  # Use k*-based seam gating
        kstar=0.721,
        tau=0.1,
    )

    model_gru = GRUBaseline(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=obs_dim)

    print(f"  - ℤ₂ + k* model: {sum(p.numel() for p in model_z2.parameters())} parameters")
    print(f"  - GRU baseline: {sum(p.numel() for p in model_gru.parameters())} parameters")

    # Step 3: Train models
    print(f"\n[3/4] Training models ({training_steps} steps)...")

    X_train = train_obs[:-1].unsqueeze(0)
    Y_train = train_obs[1:].unsqueeze(0)

    # Train ℤ₂ model
    print("  Training ℤ₂ + k* model...")
    optimizer_z2 = torch.optim.Adam(model_z2.parameters(), lr=lr)
    model_z2.train()

    for step in range(training_steps):
        optimizer_z2.zero_grad()
        y_pred, _, _ = model_z2(X_train)
        loss = nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer_z2.step()

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}: Loss = {loss.item():.6f}")

    final_loss_z2 = loss.item()

    # Train GRU
    print("  Training GRU baseline...")
    optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=lr)
    model_gru.train()

    for step in range(training_steps):
        optimizer_gru.zero_grad()
        y_pred, _, _ = model_gru(X_train)
        loss = nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer_gru.step()

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}: Loss = {loss.item():.6f}")

    final_loss_gru = loss.item()

    print(f"  ✓ Training complete")
    print(f"    ℤ₂ + k* final loss: {final_loss_z2:.6f}")
    print(f"    GRU final loss: {final_loss_gru:.6f}")

    # Step 4: Evaluate on test set
    print(f"\n[4/4] Evaluating on test set ({test_length} timesteps)...")

    X_test = test_obs[:-1].unsqueeze(0)
    Y_test = test_obs[1:].unsqueeze(0)

    model_z2.eval()
    model_gru.eval()

    with torch.no_grad():
        # ℤ₂ model
        y_pred_z2, _, gates_z2 = model_z2(X_test)
        y_pred_z2 = y_pred_z2.squeeze(0)
        gates_z2 = gates_z2.squeeze(0)

        # GRU
        y_pred_gru, _, _ = model_gru(X_test)
        y_pred_gru = y_pred_gru.squeeze(0)

    # Compute errors
    Y_test_squeezed = Y_test.squeeze(0)
    errors_z2 = ((y_pred_z2 - Y_test_squeezed) ** 2).sum(dim=-1)
    errors_gru = ((y_pred_gru - Y_test_squeezed) ** 2).sum(dim=-1)

    # Find regime switches
    switch_times, transition_mask = find_regime_switches(test_regimes[1:], window=20)

    # Compute metrics
    mse_z2_overall = errors_z2.mean().item()
    mse_gru_overall = errors_gru.mean().item()

    if transition_mask.any():
        mse_z2_transition = errors_z2[transition_mask].mean().item()
        mse_gru_transition = errors_gru[transition_mask].mean().item()
    else:
        mse_z2_transition = mse_z2_overall
        mse_gru_transition = mse_gru_overall

    if (~transition_mask).any():
        mse_z2_within = errors_z2[~transition_mask].mean().item()
        mse_gru_within = errors_gru[~transition_mask].mean().item()
    else:
        mse_z2_within = mse_z2_overall
        mse_gru_within = mse_gru_overall

    print("\n  Test Results:")
    print("  " + "-" * 50)
    print(f"  {'Metric':<20} {'ℤ₂ + k*':<15} {'GRU':<15}")
    print("  " + "-" * 50)
    print(f"  {'Overall MSE':<20} {mse_z2_overall:<15.6f} {mse_gru_overall:<15.6f}")
    print(f"  {'Within-regime MSE':<20} {mse_z2_within:<15.6f} {mse_gru_within:<15.6f}")
    print(f"  {'Transition MSE':<20} {mse_z2_transition:<15.6f} {mse_gru_transition:<15.6f}")
    print("  " + "-" * 50)

    # Compute improvement
    improvement = (mse_gru_transition - mse_z2_transition) / mse_gru_transition * 100
    print(f"\n  ℤ₂ model reduces transition error by {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("\nKey takeaway:")
    print("  The ℤ₂-equivariant model with k*-based seam gating typically shows")
    print("  improved performance at regime transitions compared to standard RNNs,")
    print("  demonstrating the value of non-orientable geometric structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
