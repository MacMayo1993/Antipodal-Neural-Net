"""
Generate publication-ready LaTeX tables from benchmark metrics.

Reads artifacts/metrics.csv and generates:
- artifacts/table_main.tex
- artifacts/table_ablation.tex
- artifacts/table_generalization.tex
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import numpy as np
import pandas as pd


def format_mean_std(values):
    """Format as mean ± std"""
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.4f} ± {std:.4f}"


def format_mean_std_bold(values, is_best=False):
    """Format with optional bold for best value"""
    formatted = format_mean_std(values)
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def load_metrics(metrics_path):
    """Load metrics CSV"""
    df = pd.read_csv(metrics_path)
    return df


def generate_main_table(df, output_path):
    """Generate main results table"""

    # Filter for standard benchmark (p_switch_test = 0.05)
    df_standard = df[df["p_switch_test"] == 0.05]

    # Model order
    model_order = [
        "GRU",
        "Z2_comm_only",
        "Z2_fixed_gate",
        "Z2_learn_gate",
        "Z2_kstar_gate",
        "AR1",
        "IMM_AR1_nonoracle",
    ]

    # Model display names
    model_names = {
        "GRU": "GRU",
        "Z2_comm_only": "ℤ₂ Equivariant",
        "Z2_fixed_gate": "ℤ₂ + Fixed Gate",
        "Z2_learn_gate": "ℤ₂ + Learned Gate",
        "Z2_kstar_gate": "ℤ₂ + k* Gate",
        "AR1": "AR(1)",
        "IMM_AR1_nonoracle": "IMM-AR(1)",
    }

    # Compute statistics
    stats = []
    for model in model_order:
        model_data = df_standard[df_standard["model"] == model]
        if len(model_data) == 0:
            continue

        stats.append(
            {
                "model": model,
                "display_name": model_names.get(model, model),
                "overall_mean": model_data["overall_mse"].mean(),
                "overall_std": model_data["overall_mse"].std(),
                "within_mean": model_data["within_mse"].mean(),
                "within_std": model_data["within_mse"].std(),
                "transition_mean": model_data["transition_mse"].mean(),
                "transition_std": model_data["transition_mse"].std(),
                "params": int(model_data["params"].iloc[0]),
            }
        )

    # Find best (minimum) for each metric (neural models only)
    neural_models = ["GRU", "Z2_comm_only", "Z2_fixed_gate", "Z2_learn_gate", "Z2_kstar_gate"]
    neural_stats = [s for s in stats if s["model"] in neural_models]

    best_overall = min(neural_stats, key=lambda x: x["overall_mean"])["model"]
    best_within = min(neural_stats, key=lambda x: x["within_mean"])["model"]
    best_transition = min(neural_stats, key=lambda x: x["transition_mean"])["model"]

    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Benchmark Results on Antipodal Regime-Switching Task (p = 0.05)}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Model & Overall MSE & Within MSE & Transition MSE & Params \\\\")
    latex.append("\\midrule")

    for stat in stats:
        model = stat["model"]
        name = stat["display_name"]

        overall_str = format_mean_std_bold(
            df_standard[df_standard["model"] == model]["overall_mse"].values,
            is_best=(model == best_overall),
        )

        within_str = format_mean_std_bold(
            df_standard[df_standard["model"] == model]["within_mse"].values,
            is_best=(model == best_within),
        )

        transition_str = format_mean_std_bold(
            df_standard[df_standard["model"] == model]["transition_mse"].values,
            is_best=(model == best_transition),
        )

        params_str = f"{stat['params']:,}"

        # Add separator before classical baselines
        if model == "AR1":
            latex.append("\\midrule")

        latex.append(
            f"{name} & {overall_str} & {within_str} & {transition_str} & {params_str} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"✓ Main table written to {output_path}")


def generate_ablation_table(df, output_path):
    """Generate ablation study table (transition MSE only)"""

    df_standard = df[df["p_switch_test"] == 0.05]

    # Ablation order (this is the key result)
    ablation_order = ["GRU", "Z2_comm_only", "Z2_fixed_gate", "Z2_learn_gate", "Z2_kstar_gate"]

    model_names = {
        "GRU": "GRU Baseline",
        "Z2_comm_only": "+ ℤ₂ Equivariance",
        "Z2_fixed_gate": "+ Seam (g=0.5)",
        "Z2_learn_gate": "+ Learned Gate",
        "Z2_kstar_gate": "+ k* Gate",
    }

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study: Transition Error Ordering}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\begin{tabular}{lc}")
    latex.append("\\toprule")
    latex.append("Model Variant & Transition MSE \\\\")
    latex.append("\\midrule")

    transition_values = []
    for model in ablation_order:
        model_data = df_standard[df_standard["model"] == model]
        if len(model_data) == 0:
            continue

        trans_mse = model_data["transition_mse"].values
        transition_values.append(trans_mse.mean())

        name = model_names.get(model, model)
        trans_str = format_mean_std(trans_mse)

        latex.append(f"{name} & {trans_str} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"✓ Ablation table written to {output_path}")

    # Verify ordering
    print("\n  Transition MSE ordering verification:")
    for i, (model, val) in enumerate(zip(ablation_order, transition_values)):
        status = "✓" if i == 0 or val < transition_values[i - 1] else "✗"
        print(f"    {status} {model_names[model]}: {val:.4f}")


def generate_generalization_table(df, output_path):
    """Generate generalization table (train p=0.05, test p=0.2)"""

    df_gen = df[df["p_switch_test"] == 0.2]

    if len(df_gen) == 0:
        print("⚠ No generalization data found (p_switch_test=0.2)")
        return

    model_order = ["GRU", "Z2_comm_only", "Z2_kstar_gate"]

    model_names = {"GRU": "GRU", "Z2_comm_only": "ℤ₂ Equivariant", "Z2_kstar_gate": "ℤ₂ + k* Gate"}

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Generalization: Train p=0.05, Test p=0.2}")
    latex.append("\\label{tab:generalization}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Model & Transition MSE (p=0.05) & Transition MSE (p=0.2) \\\\")
    latex.append("\\midrule")

    for model in model_order:
        # Standard
        model_data_std = df[(df["model"] == model) & (df["p_switch_test"] == 0.05)]
        # Generalization
        model_data_gen = df[(df["model"] == model) & (df["p_switch_test"] == 0.2)]

        if len(model_data_std) == 0 or len(model_data_gen) == 0:
            continue

        name = model_names.get(model, model)
        std_str = format_mean_std(model_data_std["transition_mse"].values)
        gen_str = format_mean_std(model_data_gen["transition_mse"].values)

        latex.append(f"{name} & {std_str} & {gen_str} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"✓ Generalization table written to {output_path}")


def main():
    """Generate all tables"""
    print("=" * 60)
    print("Generating LaTeX Tables")
    print("=" * 60)

    # Paths
    base_path = Path(__file__).parent.parent
    metrics_path = base_path / "artifacts" / "metrics.csv"
    output_dir = base_path / "artifacts"

    # Check if metrics exist
    if not metrics_path.exists():
        print(f"✗ Metrics file not found: {metrics_path}")
        print("  Run 'python scripts/run_benchmark.py' first.")
        return

    # Load data
    print(f"\nLoading metrics from {metrics_path}...")
    df = load_metrics(metrics_path)
    print(f"  Loaded {len(df)} rows")

    # Generate tables
    print("\nGenerating tables...")
    generate_main_table(df, output_dir / "table_main.tex")
    generate_ablation_table(df, output_dir / "table_ablation.tex")
    generate_generalization_table(df, output_dir / "table_generalization.tex")

    print("\n" + "=" * 60)
    print("✓ All tables generated successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
