"""
Generate publication-ready figures from benchmark runs.

Generates:
- fig1_switch_aligned_error.png: Error curves aligned on regime switches
- fig2_kstar_gate_alignment.png: Gate activation aligned on switches
- fig3_alpha_phase_transition.png: α₋ phase transition around switches
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import Z2EquivariantRNN, SeamGatedRNN, GRUBaseline
from src.data import AntipodalRegimeSwitcher, find_regime_switches
from src.parity import ParityProjectors


# Publication styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def collect_switch_aligned_errors(model, test_obs, test_regimes, window=30, device='cpu'):
    """Collect errors aligned around regime switches"""
    model.to(device)
    model.eval()

    X = test_obs[:-1].to(device)
    Y_true = test_obs[1:].to(device)

    # Forward pass
    with torch.no_grad():
        if hasattr(model, 'gru'):
            Y_pred, _, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
        elif hasattr(model, 'gate_type'):
            Y_pred, _, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)
        else:
            Y_pred, _ = model(X.unsqueeze(0))
            Y_pred = Y_pred.squeeze(0)

    # Compute errors
    errors = ((Y_pred - Y_true) ** 2).sum(dim=-1).cpu().numpy()

    # Find switches
    switch_times, _ = find_regime_switches(test_regimes[1:])

    # Align errors around switches
    aligned_errors = []
    for switch_t in switch_times:
        switch_t = switch_t.item()
        start = max(0, switch_t - window)
        end = min(len(errors), switch_t + window + 1)

        # Create aligned window
        window_errors = np.full(2 * window + 1, np.nan)
        offset_start = window - (switch_t - start)
        offset_end = offset_start + (end - start)

        window_errors[offset_start:offset_end] = errors[start:end]
        aligned_errors.append(window_errors)

    if len(aligned_errors) > 0:
        aligned_errors = np.array(aligned_errors)
        return aligned_errors
    else:
        return None


def collect_gate_aligned(model, test_obs, test_regimes, window=30, device='cpu'):
    """Collect gate values aligned around regime switches"""
    if not hasattr(model, 'gate_type'):
        return None

    model.to(device)
    model.eval()

    X = test_obs[:-1].to(device)

    with torch.no_grad():
        _, _, gates = model(X.unsqueeze(0))
        gates = gates.squeeze(0).cpu().numpy()

    # Find switches
    switch_times, _ = find_regime_switches(test_regimes[1:])

    # Align gates
    aligned_gates = []
    for switch_t in switch_times:
        switch_t = switch_t.item()
        start = max(0, switch_t - window)
        end = min(len(gates), switch_t + window + 1)

        window_gates = np.full(2 * window + 1, np.nan)
        offset_start = window - (switch_t - start)
        offset_end = offset_start + (end - start)

        window_gates[offset_start:offset_end] = gates[start:end]
        aligned_gates.append(window_gates)

    if len(aligned_gates) > 0:
        return np.array(aligned_gates)
    else:
        return None


def collect_alpha_aligned(model, test_obs, test_regimes, window=30, device='cpu'):
    """Collect α₋ values aligned around regime switches"""
    if not hasattr(model, 'projectors'):
        return None

    model.to(device)
    model.eval()

    X = test_obs[:-1].to(device)

    # Collect hidden states
    with torch.no_grad():
        h_sequence = []
        h = torch.zeros(1, model.hidden_dim, device=device)

        for t in range(len(X)):
            h, _ = model.step(X[t].unsqueeze(0), h.squeeze(0))
            h = h.unsqueeze(0)
            h_sequence.append(h.squeeze(0))

        h_sequence = torch.stack(h_sequence)

    # Compute α₋
    projectors = model.projectors
    alpha_sequence = []
    for h in h_sequence:
        alpha = projectors.parity_energy(h.unsqueeze(0))
        alpha_sequence.append(alpha.item())

    alpha_sequence = np.array(alpha_sequence)

    # Find switches
    switch_times, _ = find_regime_switches(test_regimes[1:])

    # Align α₋
    aligned_alpha = []
    for switch_t in switch_times:
        switch_t = switch_t.item()
        start = max(0, switch_t - window)
        end = min(len(alpha_sequence), switch_t + window + 1)

        window_alpha = np.full(2 * window + 1, np.nan)
        offset_start = window - (switch_t - start)
        offset_end = offset_start + (end - start)

        window_alpha[offset_start:offset_end] = alpha_sequence[start:end]
        aligned_alpha.append(window_alpha)

    if len(aligned_alpha) > 0:
        return np.array(aligned_alpha)
    else:
        return None


def generate_figure1(seeds, obs_dim=4, hidden_dim=16, output_path=None):
    """Figure 1: Switch-aligned error curves"""
    print("Generating Figure 1: Switch-aligned error curves...")

    window = 30
    model_configs = [
        ('GRU', lambda: GRUBaseline(obs_dim, hidden_dim, obs_dim), 'C0'),
        ('ℤ₂ Comm Only', lambda: Z2EquivariantRNN(obs_dim, hidden_dim, obs_dim), 'C1'),
        ('ℤ₂ + k* Gate', lambda: SeamGatedRNN(obs_dim, hidden_dim, obs_dim,
                                               gate_type='kstar', kstar=0.721), 'C2'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for model_name, model_fn, color in model_configs:
        all_aligned = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate data
            gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=obs_dim, p_switch=0.05, seed=seed)
            test_obs, _, test_regimes = gen.generate_sequence(T=500)

            # Train model (minimal for figure)
            model = model_fn()
            # In real usage, load pre-trained model or train here

            # Collect aligned errors
            aligned = collect_switch_aligned_errors(model, test_obs, test_regimes, window=window)
            if aligned is not None:
                all_aligned.append(aligned)

        if len(all_aligned) > 0:
            # Concatenate across seeds and switches
            all_aligned = np.vstack(all_aligned)

            # Compute mean and std
            mean_error = np.nanmean(all_aligned, axis=0)
            std_error = np.nanstd(all_aligned, axis=0)

            # Plot
            x = np.arange(-window, window + 1)
            ax.plot(x, mean_error, label=model_name, color=color, linewidth=1.5)
            ax.fill_between(x, mean_error - std_error, mean_error + std_error,
                           alpha=0.2, color=color)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Switch')
    ax.set_xlabel('Time offset from switch')
    ax.set_ylabel('MSE')
    ax.set_title('Error Aligned on Regime Switches')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_figure2(seeds, obs_dim=4, hidden_dim=16, output_path=None):
    """Figure 2: k*-gate activation aligned on switches"""
    print("Generating Figure 2: k*-gate activation alignment...")

    window = 30
    all_aligned = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data
        gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=obs_dim, p_switch=0.05, seed=seed)
        test_obs, _, test_regimes = gen.generate_sequence(T=500)

        # k*-gated model
        model = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type='kstar', kstar=0.721)
        # In real usage, load pre-trained model

        aligned = collect_gate_aligned(model, test_obs, test_regimes, window=window)
        if aligned is not None:
            all_aligned.append(aligned)

    if len(all_aligned) == 0:
        print("  ⚠ No gate data collected")
        return

    all_aligned = np.vstack(all_aligned)
    mean_gate = np.nanmean(all_aligned, axis=0)
    std_gate = np.nanstd(all_aligned, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = np.arange(-window, window + 1)
    ax.plot(x, mean_gate, color='C3', linewidth=1.5)
    ax.fill_between(x, mean_gate - std_gate, mean_gate + std_gate,
                    alpha=0.2, color='C3')
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Switch')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Time offset from switch')
    ax.set_ylabel('Gate value g(t)')
    ax.set_title('k*-Gate Activation Aligned on Regime Switches')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_figure3(seeds, obs_dim=4, hidden_dim=16, output_path=None):
    """Figure 3: α₋ phase transition around switches"""
    print("Generating Figure 3: α₋ phase transition...")

    window = 30
    kstar = 0.721
    all_aligned = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data
        gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=obs_dim, p_switch=0.05, seed=seed)
        test_obs, _, test_regimes = gen.generate_sequence(T=500)

        # k*-gated model
        model = SeamGatedRNN(obs_dim, hidden_dim, obs_dim, gate_type='kstar', kstar=kstar)
        # In real usage, load pre-trained model

        aligned = collect_alpha_aligned(model, test_obs, test_regimes, window=window)
        if aligned is not None:
            all_aligned.append(aligned)

    if len(all_aligned) == 0:
        print("  ⚠ No α₋ data collected")
        return

    all_aligned = np.vstack(all_aligned)
    mean_alpha = np.nanmean(all_aligned, axis=0)
    std_alpha = np.nanstd(all_aligned, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = np.arange(-window, window + 1)
    ax.plot(x, mean_alpha, color='C4', linewidth=1.5)
    ax.fill_between(x, mean_alpha - std_alpha, mean_alpha + std_alpha,
                    alpha=0.2, color='C4')
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Switch')
    ax.axhline(kstar, color='C2', linestyle=':', linewidth=1.5, alpha=0.7, label=f'k* = {kstar}')

    ax.set_xlabel('Time offset from switch')
    ax.set_ylabel('Parity energy α₋(t)')
    ax.set_title('α₋ Phase Transition at Regime Switches')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Generate all canonical figures"""
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)

    # Configuration
    seeds = [42, 123, 456]  # Subset for quick figure generation
    obs_dim = 4
    hidden_dim = 16

    # Output paths
    base_path = Path(__file__).parent.parent
    fig_dir = base_path / 'artifacts' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\n[1/3] Figure 1: Switch-aligned error curves")
    generate_figure1(seeds, obs_dim, hidden_dim,
                     output_path=fig_dir / 'fig1_switch_aligned_error.png')

    print("\n[2/3] Figure 2: k*-gate activation")
    generate_figure2(seeds, obs_dim, hidden_dim,
                     output_path=fig_dir / 'fig2_kstar_gate_alignment.png')

    print("\n[3/3] Figure 3: α₋ phase transition")
    generate_figure3(seeds, obs_dim, hidden_dim,
                     output_path=fig_dir / 'fig3_alpha_phase_transition.png')

    print("\n" + "=" * 60)
    print("✓ All figures generated successfully")
    print("=" * 60)


if __name__ == '__main__':
    main()
