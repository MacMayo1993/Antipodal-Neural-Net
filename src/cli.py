"""
Command-line interface for Antipodal Neural Networks
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from .models import Z2EquivariantRNN, SeamGatedRNN, GRUBaseline
from .data import AntipodalRegimeSwitcher


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
        help="Model type to train"
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
        "--save-path",
        type=str,
        default="model_checkpoint.pth",
        help="Path to save trained model"
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
        latent_dim=args.latent_dim,
        obs_dim=args.obs_dim,
        p_switch=args.p_switch,
        seed=args.seed
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
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type='fixed')
    elif args.model == "Z2_learn":
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type='learned')
    elif args.model == "Z2_kstar":
        model = SeamGatedRNN(args.obs_dim, args.hidden_dim, args.obs_dim, gate_type='kstar')

    # Training
    print(f"\nTraining for {args.epochs} steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X = obs[:-1].unsqueeze(0)
    Y = obs[1:].unsqueeze(0)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        if hasattr(model, 'gru'):
            y_pred, _, _ = model(X)
        elif hasattr(model, 'gate_type'):
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model,
        'hidden_dim': args.hidden_dim,
        'obs_dim': args.obs_dim,
        'final_loss': loss.item(),
    }, args.save_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {loss.item():.6f}")
    print(f"Model saved to: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
