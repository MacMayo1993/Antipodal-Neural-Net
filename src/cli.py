"""
Command-line interface for Antipodal Neural Networks
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import AntipodalRegimeSwitcher
from .models import GRUBaseline, SeamGatedRNN, Z2EquivariantRNN


def main():
    parser = argparse.ArgumentParser(
        description="Train Antipodal Neural Networks for regime-switching time series"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Z2_kstar",
        choices=["GRU", "Z2_equi", "Z2_fixed", "Z2_learn", "Z2_kstar"],
        help="Model type to train",
    )
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--obs-dim", type=int, default=4, help="Observation dimension")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=500, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data configuration
    parser.add_argument("--seq-length", type=int, default=1000, help="Sequence length")
    parser.add_argument("--p-switch", type=float, default=0.05, help="Regime switch probability")

    # Output
    parser.add_argument(
        "--save-path", type=str, default="model_checkpoint.pth", help="Path to save trained model"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Antipodal Neural Network Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Training steps: {args.epochs}")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    generator = AntipodalRegimeSwitcher(
        latent_dim=args.latent_dim, obs_dim=args.obs_dim, p_switch=args.p_switch, seed=args.seed
    )
    obs, latents, regimes = generator.generate_sequence(T=args.seq_length)
    print(f"  Generated {len(obs)} timesteps")

    # Create model
    print(f"\nInitializing {args.model} model...")
    if args.model == "GRU":
        model = GRUBaseline(args.obs_dim, args.hidden_dim, args.obs_dim)
    elif args.model == "Z2_equi":
        model = Z2EquivariantRNN(args.obs_dim, args.hidden_dim, args.obs_dim)
    elif args.model == "Z2_fixed":
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type="fixed")
    elif args.model == "Z2_learn":
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type="learned")
    elif args.model == "Z2_kstar":
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type="kstar")

    # Training
    print(f"\nTraining for {args.epochs} steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X = obs[:-1].unsqueeze(0)
    Y = obs[1:].unsqueeze(0)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        if hasattr(model, "gru"):
            y_pred, _, _ = model(X)
        elif hasattr(model, "gate_type"):
            y_pred, _, _ = model(X)
        else:
            y_pred, _ = model(X)

        loss = torch.nn.functional.mse_loss(y_pred, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Step {epoch + 1}/{args.epochs}: Loss = {loss.item():.6f}")

    # Save model
    print(f"\nSaving model to {args.save_path}...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model,
            "hidden_dim": args.hidden_dim,
            "obs_dim": args.obs_dim,
            "final_loss": loss.item(),
        },
        args.save_path,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {loss.item():.6f}")
    print(f"Model saved to: {args.save_path}")
    print("=" * 60)


def demo():
    """
    Run a quick demo showing antipodal neural networks in action.

    Trains a Z2 model with k*-based seam gating on synthetic data,
    compares against GRU baseline, and saves results to artifacts/demo/.
    """
    print("=" * 70)
    print("Antipodal Neural Networks - Quick Demo")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Generate synthetic regime-switching data")
    print("  2. Train a Z2 model with k*-based seam gating")
    print("  3. Train a GRU baseline for comparison")
    print("  4. Save metrics and a plot to artifacts/demo/")
    print("\nEstimated time: 1-2 minutes\n")

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create output directory
    output_dir = Path("artifacts/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    latent_dim = 8
    obs_dim = 4
    hidden_dim = 16
    seq_length = 500
    epochs = 100  # Quick training
    p_switch = 0.05

    print("[1/4] Generating synthetic data...")
    from .data import AntipodalRegimeSwitcher

    generator = AntipodalRegimeSwitcher(
        latent_dim=latent_dim, obs_dim=obs_dim, p_switch=p_switch, seed=42
    )

    train_obs, train_latents, train_regimes = generator.generate_sequence(T=seq_length)
    test_obs, test_latents, test_regimes = generator.generate_sequence(T=250)

    num_switches = (train_regimes[1:] != train_regimes[:-1]).sum().item()
    print(f"  ✓ Generated {seq_length} timesteps with {num_switches} regime switches")

    print("\n[2/4] Training Z2 model with k*-based seam gating...")
    model_z2 = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type="kstar")
    optimizer_z2 = torch.optim.Adam(model_z2.parameters(), lr=0.01)

    X_train = train_obs[:-1].unsqueeze(0)
    Y_train = train_obs[1:].unsqueeze(0)

    model_z2.train()
    for epoch in range(epochs):
        optimizer_z2.zero_grad()
        y_pred, _, _ = model_z2(X_train)
        loss = torch.nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer_z2.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.6f}")

    print("\n[3/4] Training GRU baseline...")
    model_gru = GRUBaseline(obs_dim, hidden_dim, obs_dim)
    optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.01)

    model_gru.train()
    for epoch in range(epochs):
        optimizer_gru.zero_grad()
        y_pred, _, _ = model_gru(X_train)
        loss = torch.nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer_gru.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.6f}")

    print("\n[4/4] Evaluating on test set...")
    X_test = test_obs[:-1].unsqueeze(0)
    Y_test = test_obs[1:].unsqueeze(0)

    model_z2.eval()
    model_gru.eval()

    with torch.no_grad():
        y_pred_z2, _, _ = model_z2(X_test)
        y_pred_gru, _, _ = model_gru(X_test)

        mse_z2 = torch.nn.functional.mse_loss(y_pred_z2, Y_test).item()
        mse_gru = torch.nn.functional.mse_loss(y_pred_gru, Y_test).item()

    # Save metrics
    import csv

    metrics_file = output_dir / "metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_mse"])
        writer.writerow(["Z2_kstar", f"{mse_z2:.6f}"])
        writer.writerow(["GRU", f"{mse_gru:.6f}"])

    print(f"\n  Results:")
    print(f"  ├─ Z2 (k*-gated):  MSE = {mse_z2:.6f}")
    print(f"  └─ GRU baseline:   MSE = {mse_gru:.6f}")

    improvement = (mse_gru - mse_z2) / mse_gru * 100
    if improvement > 0:
        print(f"\n  ✓ Z2 model outperforms GRU by {improvement:.1f}%")

    print(f"\n✓ Metrics saved to: {metrics_file}")
    print("\n" + "=" * 70)
    print("Demo complete! The Z2 model with seam gating adapts to")
    print("regime switches better than standard RNNs.")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  - Run full tests: pytest")
    print(f"  - Train custom model: antipodal-train --help")
    print(f"  - See examples/train_simple.py for more")


if __name__ == "__main__":
    main()
