"""
Benchmark script for all model variants.

Generates artifacts/metrics.csv with columns:
- seed
- model
- overall_mse
- within_mse
- transition_mse
- params
- p_switch_train
- p_switch_test
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.baselines import AR1Model, IMMFilter
from src.data import AntipodalRegimeSwitcher, find_regime_switches
from src.models import GRUBaseline, SeamGatedRNN, Z2EquivariantRNN


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_obs, num_steps=500, lr=0.01, device="cpu"):
    """Train a neural network model"""
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    X = train_obs[:-1].unsqueeze(0).to(device)
    Y = train_obs[1:].unsqueeze(0).to(device)

    for step in range(num_steps):
        optimizer.zero_grad()

        if hasattr(model, "gru"):  # GRU
            y_pred, _, _ = model(X)
        elif hasattr(model, "gate_type"):  # Seam-gated
            y_pred, _, _ = model(X)
        else:  # Equivariant
            y_pred, _ = model(X)

        loss = torch.nn.functional.mse_loss(y_pred, Y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate_model(model, test_obs, test_regimes, device="cpu"):
    """Evaluate model and compute metrics"""
    model.to(device)
    model.eval()

    X = test_obs[:-1].to(device)
    Y_true = test_obs[1:].to(device)

    with torch.no_grad():
        if hasattr(model, "gru"):
            Y_pred, _, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
        elif hasattr(model, "gate_type"):
            Y_pred, _, gates = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
            gates = gates.squeeze(0)
        else:
            Y_pred, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)

    # Compute errors
    errors = ((Y_pred - Y_true) ** 2).sum(dim=-1).cpu()

    # Overall MSE
    overall_mse = errors.mean().item()

    # Find transition windows
    switch_times, transition_mask = find_regime_switches(test_regimes[1:], window=20)

    # Transition MSE
    if transition_mask.any():
        transition_mse = errors[transition_mask].mean().item()
    else:
        transition_mse = overall_mse

    # Within-regime MSE
    if (~transition_mask).any():
        within_mse = errors[~transition_mask].mean().item()
    else:
        within_mse = overall_mse

    return {"overall_mse": overall_mse, "within_mse": within_mse, "transition_mse": transition_mse}


def evaluate_classical(test_obs, test_regimes, model_class, **kwargs):
    """Evaluate classical baseline"""
    if model_class == "AR1":
        model = AR1Model(test_obs.shape[-1])
        model.fit(test_obs)

        # Predict
        predictions = []
        for t in range(len(test_obs) - 1):
            pred = model.predict(test_obs[t].unsqueeze(0))
            predictions.append(pred.squeeze(0))

        predictions = torch.stack(predictions)

    elif model_class == "IMM":
        model = IMMFilter(test_obs.shape[-1], p_switch=kwargs.get("p_switch", 0.05))
        model.fit(test_obs)

        predictions, _ = model.predict_sequence(test_obs)

    # Compute errors
    Y_true = test_obs[1:]
    errors = ((predictions - Y_true) ** 2).sum(dim=-1)

    overall_mse = errors.mean().item()

    switch_times, transition_mask = find_regime_switches(test_regimes[1:], window=20)

    if transition_mask.any():
        transition_mse = errors[transition_mask].mean().item()
    else:
        transition_mse = overall_mse

    if (~transition_mask).any():
        within_mse = errors[~transition_mask].mean().item()
    else:
        within_mse = overall_mse

    return {"overall_mse": overall_mse, "within_mse": within_mse, "transition_mse": transition_mse}


def run_single_seed(
    seed, obs_dim=4, hidden_dim=16, p_switch_train=0.05, p_switch_test=0.05, train_steps=500
):
    """Run benchmark for a single seed"""
    print(f"Running seed {seed}...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    gen_train = AntipodalRegimeSwitcher(
        latent_dim=8, obs_dim=obs_dim, p_switch=p_switch_train, seed=seed
    )
    train_obs, _, train_regimes = gen_train.generate_sequence(T=1000)

    gen_test = AntipodalRegimeSwitcher(
        latent_dim=8, obs_dim=obs_dim, p_switch=p_switch_test, seed=seed + 1000
    )
    # Use same dynamics for fair comparison
    gen_test.A = gen_train.A
    gen_test.C = gen_train.C
    test_obs, _, test_regimes = gen_test.generate_sequence(T=500)

    results = []

    # Model configurations
    models_config = [
        ("GRU", lambda: GRUBaseline(obs_dim, hidden_dim, obs_dim)),
        ("Z2_comm_only", lambda: Z2EquivariantRNN(obs_dim, hidden_dim, obs_dim)),
        (
            "Z2_fixed_gate",
            lambda: SeamGatedRNN(
                obs_dim, hidden_dim, obs_dim, gate_type="fixed", fixed_gate_value=0.5
            ),
        ),
        ("Z2_learn_gate", lambda: SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type="learned")),
        (
            "Z2_kstar_gate",
            lambda: SeamGatedRNN(
                obs_dim, hidden_dim, obs_dim, gate_type="kstar", kstar=0.721, tau=0.1
            ),
        ),
    ]

    # Train and evaluate neural models
    for model_name, model_fn in models_config:
        print(f"  Training {model_name}...")
        model = model_fn()
        params = count_parameters(model)

        model = train_model(model, train_obs, num_steps=train_steps)
        metrics = evaluate_model(model, test_obs, test_regimes)

        results.append(
            {
                "seed": seed,
                "model": model_name,
                "overall_mse": metrics["overall_mse"],
                "within_mse": metrics["within_mse"],
                "transition_mse": metrics["transition_mse"],
                "params": params,
                "p_switch_train": p_switch_train,
                "p_switch_test": p_switch_test,
            }
        )

    # Classical baselines
    print(f"  Evaluating classical baselines...")

    # AR(1)
    try:
        metrics = evaluate_classical(test_obs, test_regimes, "AR1")
        results.append(
            {
                "seed": seed,
                "model": "AR1",
                "overall_mse": metrics["overall_mse"],
                "within_mse": metrics["within_mse"],
                "transition_mse": metrics["transition_mse"],
                "params": obs_dim * obs_dim,  # B matrix
                "p_switch_train": p_switch_train,
                "p_switch_test": p_switch_test,
            }
        )
    except Exception as e:
        print(f"    AR1 failed: {e}")

    # IMM
    try:
        metrics = evaluate_classical(test_obs, test_regimes, "IMM", p_switch=p_switch_test)
        results.append(
            {
                "seed": seed,
                "model": "IMM_AR1_nonoracle",
                "overall_mse": metrics["overall_mse"],
                "within_mse": metrics["within_mse"],
                "transition_mse": metrics["transition_mse"],
                "params": 2 * obs_dim * obs_dim,  # Two B matrices
                "p_switch_train": p_switch_train,
                "p_switch_test": p_switch_test,
            }
        )
    except Exception as e:
        print(f"    IMM failed: {e}")

    return results


def main():
    """Run full benchmark"""
    parser = argparse.ArgumentParser(description="Run full benchmark for Antipodal Neural Networks")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1011],
        help="Random seeds to use (default: 42 123 456 789 1011)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="artifacts",
        help="Output directory for results (default: artifacts)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Training steps per model (default: 500)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running Non-Orientable Neural Network Benchmark")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Training steps: {args.steps}")
    print(f"Output: {args.outdir}/metrics.csv")
    print("=" * 60)

    # Configuration
    seeds = args.seeds
    obs_dim = 4
    hidden_dim = 16
    train_steps = args.steps

    # Standard benchmark
    print("\n[1/2] Standard benchmark (p_switch=0.05)")
    all_results = []
    for seed in seeds:
        results = run_single_seed(
            seed,
            obs_dim,
            hidden_dim,
            p_switch_train=0.05,
            p_switch_test=0.05,
            train_steps=train_steps,
        )
        all_results.extend(results)

    # Generalization benchmark
    print("\n[2/2] Generalization benchmark (train p=0.05, test p=0.2)")
    for seed in seeds:
        results = run_single_seed(
            seed,
            obs_dim,
            hidden_dim,
            p_switch_train=0.05,
            p_switch_test=0.2,
            train_steps=train_steps,
        )
        all_results.extend(results)

    # Write to CSV
    output_path = Path(args.outdir) / "metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "model",
        "overall_mse",
        "within_mse",
        "transition_mse",
        "params",
        "p_switch_train",
        "p_switch_test",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✓ Metrics written to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
