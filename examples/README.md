# Examples

This directory contains usage examples for Antipodal Neural Networks.

## Training Example

**File:** [`train_simple.py`](train_simple.py)

A complete end-to-end training example demonstrating:
- Synthetic antipodal regime-switching data generation
- Model initialization (ℤ₂-equivariant with k*-based seam gating)
- Training loop with PyTorch
- Evaluation on test sequences
- Performance comparison against GRU baseline

**Run it:**
```bash
python examples/train_simple.py
```

**Expected output:**
- Training progress for both ℤ₂ and GRU models
- Test metrics (overall MSE, within-regime MSE, transition MSE)
- Improvement percentage at regime transitions

**Runtime:** ~2-3 minutes

---

## Inference Example

**File:** [`inference.py`](inference.py)

Demonstrates model inference and analysis:
- Loading trained model checkpoints
- Making predictions on new sequences
- Analyzing seam gate activations during regime transitions
- Correlating gate behavior with true regime switches

**Run it:**
```bash
python examples/inference.py
```

**Expected output:**
- Model training progress (quick demonstration)
- Test set predictions and MSE
- Gate activation patterns around regime switches
- Analysis of gate-switch alignment

**Runtime:** ~1-2 minutes

---

## Using the CLI

The package also provides a command-line interface:

```bash
# Install package first
pip install -e .

# Train a model via CLI
antipodal-train --model Z2_kstar --hidden-dim 16 --epochs 500 --save-path my_model.pth

# Available models: GRU, Z2_equi, Z2_fixed, Z2_learn, Z2_kstar
# See help for all options
antipodal-train --help
```

---

## Custom Training Loops

For custom training, import from the `src` package:

```python
from src import (
    # Models
    SeamGatedRNN,
    Z2EquivariantRNN,
    GRUBaseline,

    # Data
    AntipodalRegimeSwitcher,
    find_regime_switches,

    # Losses
    quotient_loss,
    rank1_projector_loss,

    # Parity operators
    ParityOperator,
    ParityProjectors
)

# Your custom training code here...
```

---

## Key Configuration Options

### Model Types

| Model | Description | When to use |
|-------|-------------|-------------|
| `Z2_kstar` | k*-based seam gating | **Recommended**: Principled phase boundary |
| `Z2_learn` | Learned seam gate | When k* is unknown |
| `Z2_fixed` | Fixed gate value | Ablation studies |
| `Z2_equi` | Commutant-only (no seam) | Baseline comparison |
| `GRU` | Standard GRU | Classical baseline |

### Data Generation

```python
generator = AntipodalRegimeSwitcher(
    latent_dim=8,           # Latent state dimension
    obs_dim=4,              # Observation dimension (< latent_dim)
    p_switch=0.05,          # Regime switch probability per timestep
    obs_noise_std=0.1,      # Observation noise level
    seed=42                 # Random seed for reproducibility
)
```

### Model Hyperparameters

```python
model = SeamGatedRNN(
    input_dim=4,
    hidden_dim=16,
    output_dim=4,
    gate_type='kstar',      # 'kstar', 'learned', or 'fixed'
    kstar=0.721,            # Phase boundary (for gate_type='kstar')
    tau=0.1,                # Temperature for sigmoid (smaller = sharper)
    fixed_gate_value=0.5    # Gate value (for gate_type='fixed')
)
```

---

## Next Steps

- See [`../scripts/README.md`](../scripts/README.md) for full benchmark pipeline
- See [`../docs/API_CONTRACT.md`](../docs/API_CONTRACT.md) for API guarantees
- See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for development guidelines
