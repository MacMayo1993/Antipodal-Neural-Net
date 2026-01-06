# Antipodal Neural Networks

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/MacMayo1993/Antipodal-Neural-Net/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Non-Orientable Neural Networks with ℤ₂ Seam Gating for Regime-Switching Dynamics**

This repository implements neural network architectures with **non-orientable internal geometry** designed for time series with antipodal regime switching—where dynamics flip by sign (x ↦ -x) but preserve structure.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MacMayo1993/Antipodal-Neural-Net.git
cd Antipodal-Neural-Net

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Basic Usage

```python
from src import SeamGatedRNN, AntipodalRegimeSwitcher

# Generate synthetic data
generator = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.05)
obs, latents, regimes = generator.generate_sequence(T=1000)

# Create model with k*-based seam gating
model = SeamGatedRNN(input_dim=4, hidden_dim=16, output_dim=4, gate_type='kstar')

# Train your model...
```

See [`examples/train_simple.py`](examples/train_simple.py) for a complete training example and [`examples/inference.py`](examples/inference.py) for inference.

### Running Tests

```bash
# Run all tests (excluding slow integration tests)
pytest

# Run all tests including slow benchmarks
pytest -m ""

# Run specific test sections
pytest tests/test_01_data_generator.py
pytest tests/test_02_parity_structure.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Running Benchmarks

```bash
# Full benchmark (all models, all seeds)
python scripts/run_benchmark.py

# Generate LaTeX tables
python scripts/make_tables.py

# Generate publication figures
python scripts/make_figures.py
```

**Expected runtime:** Benchmark ~20-30 min, tables ~1 sec, figures ~5-10 min

See `scripts/README.md` for detailed documentation.

## Project Structure

```
Antipodal-Neural-Net/
├── src/                   # Core implementation
│   ├── __init__.py        # Public API exports
│   ├── cli.py             # Command-line interface
│   ├── parity.py          # ℤ₂ parity operators and projectors
│   ├── data.py            # Antipodal regime-switching data generator
│   ├── models.py          # Neural network models (equivariant, seam-gated, GRU)
│   ├── losses.py          # Quotient and rank-1 projector losses
│   └── baselines.py       # Classical baselines (AR(1), IMM filter)
├── examples/              # Usage examples
│   ├── train_simple.py    # Complete training example
│   └── inference.py       # Inference and analysis example
├── tests/                 # Comprehensive test suite (90+ tests)
│   ├── test_01_data_generator.py        # Data generation tests
│   ├── test_02_parity_structure.py      # ℤ₂ structural invariance tests
│   ├── test_03_forward_dynamics.py      # Forward pass tests
│   ├── test_04_gate_logic.py            # Gate computation tests
│   ├── test_05_loss_functions.py        # Loss function tests
│   ├── test_06_training_stability.py    # Gradient flow and stability
│   ├── test_07_benchmark_performance.py # Integration benchmarks
│   ├── test_08_classical_baselines.py   # Classical method tests
│   ├── test_09_visualization.py         # Visualization consistency
│   ├── test_10_reproducibility.py       # Reproducibility tests
│   └── test_11_regression.py            # Numerical stability tests
├── scripts/               # Benchmark and figure generation
│   ├── run_benchmark.py   # Full benchmark runner
│   ├── make_tables.py     # LaTeX table generation
│   ├── make_figures.py    # Publication figure generation
│   └── README.md          # Script documentation
├── paper/                 # Paper draft
│   ├── draft.tex          # Main paper (LaTeX)
│   └── references.bib     # Bibliography
├── docs/                  # Documentation
│   └── API_CONTRACT.md    # Stable API specification
├── .github/workflows/     # CI/CD
│   ├── tests.yml          # Test suite automation
│   └── benchmark-smoke.yml # Quick smoke tests
├── artifacts/             # Generated outputs (created by scripts)
│   ├── metrics.csv        # Benchmark results
│   ├── table_*.tex        # LaTeX tables
│   └── figures/           # Publication figures
├── PROJECT_INTENT.md      # Detailed project goals and design
├── CONTRIBUTING.md        # Development guidelines
├── CHANGELOG.md           # Version history
├── pyproject.toml         # Package configuration
├── requirements.txt       # Python dependencies
└── LICENSE                # MIT License
```

## Core Concepts

### ℤ₂ Parity Structure

Hidden states are decomposed into **even** and **odd** parity channels:
- **Parity operator**: S with S² = I
- **Projectors**: P₊ = ½(I + S), P₋ = ½(I - S)
- **Parity energy**: α₋(h) = ||P₋h||² / ||h||²

### Seam Gate Mechanism

The network uses two types of weights:
- **Commutant weights** (W_comm): Preserve parity structure (WS = SW)
- **Anticommutant weights** (W_flip): Swap parity channels (WS = -SW)

A learned gate g(h) ∈ [0,1] controls when to activate seam crossing:

```
h_{t+1} = σ(W_comm u_t + g(h_t) W_flip S u_t + b)
```

### k*-Gated Version

Uses principled phase boundary at k* ≈ 0.721:

```
g(h) = sigmoid((α₋(h) - k*) / τ)
```

This automatically activates seam transitions when odd parity energy crosses the critical threshold.

## Test Suite

The comprehensive test suite validates:

1. **Data Generator** (Section 1)
   - Antipodal dynamics symmetry
   - Regime switching statistics
   - Partial observability

2. **ℤ₂ Structure** (Section 2)
   - Parity operator involution (S² = I)
   - Projector properties (idempotent, orthogonal, partition)
   - Commutation/anticommutation

3. **Forward Dynamics** (Section 3)
   - Parity preservation in equivariant model
   - Seam coupling effects

4. **Gate Logic** (Section 4)
   - k* threshold behavior
   - Monotonicity in α₋
   - Gradient flow through gates

5. **Loss Functions** (Section 5)
   - Quotient loss sign invariance
   - Rank-1 projector equivalence

6. **Training Stability** (Section 6)
   - Gradient flow preservation
   - No mode collapse

7. **Benchmark Performance** (Section 7)
   - Model comparison on regime-switching tasks
   - Parameter efficiency
   - Generalization to higher switch rates

8. **Classical Baselines** (Section 8)
   - AR(1) model validation
   - IMM filter consistency

9. **Visualization** (Section 9)
   - Gate-switch alignment
   - α₋ phase transitions

10. **Reproducibility** (Section 10)
    - Seed stability
    - Variance bounds across runs

11. **Regression** (Section 11)
    - No NaN/Inf during training
    - Numerical stability

## Model Variants

### Z2EquivariantRNN
Commutant-only model (no seam coupling):
```python
from src.models import Z2EquivariantRNN

model = Z2EquivariantRNN(
    input_dim=4,
    hidden_dim=16,
    output_dim=4
)
```

### SeamGatedRNN
Full model with seam gate:
```python
from src.models import SeamGatedRNN

# k*-gated version
model = SeamGatedRNN(
    input_dim=4,
    hidden_dim=16,
    output_dim=4,
    gate_type='kstar',
    kstar=0.721,
    tau=0.1
)

# Learned gate
model = SeamGatedRNN(
    input_dim=4,
    hidden_dim=16,
    output_dim=4,
    gate_type='learned'
)

# Fixed gate
model = SeamGatedRNN(
    input_dim=4,
    hidden_dim=16,
    output_dim=4,
    gate_type='fixed',
    fixed_gate_value=0.5
)
```

### GRUBaseline
Standard GRU for comparison:
```python
from src.models import GRUBaseline

model = GRUBaseline(
    input_dim=4,
    hidden_dim=16,
    output_dim=4
)
```

## Data Generation

Generate synthetic antipodal regime-switching sequences:

```python
from src.data import AntipodalRegimeSwitcher

generator = AntipodalRegimeSwitcher(
    latent_dim=8,
    obs_dim=4,
    p_switch=0.05,
    obs_noise_std=0.1,
    seed=42
)

observations, latents, regimes = generator.generate_sequence(T=1000)
```

## Loss Functions

### Quotient Loss
Sign-invariant loss for quotient space:
```python
from src.losses import quotient_loss

loss = quotient_loss(y_true, y_pred)
```

### Rank-1 Projector Loss
Projective distance via outer products:
```python
from src.losses import rank1_projector_loss

loss = rank1_projector_loss(y_true, y_pred)
```

## Citation

If you use this code, please cite:

```bibtex
@software{antipodal_neural_nets,
  title={Non-Orientable Neural Networks with ℤ₂ Seam Gating},
  author={[Authors]},
  year={2025},
  url={https://github.com/MacMayo1993/Antipodal-Neural-Net}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Project Intent

For detailed project goals, design philosophy, and success criteria, see [PROJECT_INTENT.md](PROJECT_INTENT.md).
