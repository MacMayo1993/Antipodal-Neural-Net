# Benchmark and Figure Generation Scripts

This directory contains scripts for running benchmarks and generating publication-ready artifacts.

## Scripts

### `run_benchmark.py`
Runs full benchmark across all model variants and seeds.

**Usage:**
```bash
python scripts/run_benchmark.py
```

**Output:**
- `artifacts/metrics.csv`: Complete benchmark results

**Configuration** (edit in script):
- `seeds`: List of random seeds (default: [42, 123, 456, 789, 1011])
- `obs_dim`: Observation dimension (default: 4)
- `hidden_dim`: Hidden dimension (default: 16)
- `train_steps`: Training iterations per model (default: 500)

**Runtime:** ~15-30 minutes (5 seeds × 7 models × 2 conditions)

---

### `make_tables.py`
Generates LaTeX tables from benchmark metrics.

**Usage:**
```bash
python scripts/make_tables.py
```

**Input:**
- `artifacts/metrics.csv`

**Output:**
- `artifacts/table_main.tex`: Main results table
- `artifacts/table_ablation.tex`: Ablation study table
- `artifacts/table_generalization.tex`: Generalization table

**Features:**
- Automatic mean ± std formatting
- Bold highlighting for best results
- Ablation ordering verification

---

### `make_figures.py`
Generates publication-ready figures.

**Usage:**
```bash
python scripts/make_figures.py
```

**Output:**
- `artifacts/figures/fig1_switch_aligned_error.png`
- `artifacts/figures/fig2_kstar_gate_alignment.png`
- `artifacts/figures/fig3_alpha_phase_transition.png`

**Configuration** (edit in script):
- `seeds`: Subset of seeds for figure generation (default: [42, 123, 456])
- `window`: Time window around switches (default: 30 steps)

**Runtime:** ~5-10 minutes

---

## Workflow

### Full Pipeline

```bash
# 1. Run benchmark
python scripts/run_benchmark.py

# 2. Generate tables
python scripts/make_tables.py

# 3. Generate figures
python scripts/make_figures.py
```

### Quick Test (Small Scale)

Edit `run_benchmark.py` to use:
```python
seeds = [42]
train_steps = 100
```

Then run the full pipeline.

---

## Output Structure

```
artifacts/
├── metrics.csv                          # Raw benchmark data
├── table_main.tex                       # Main results (LaTeX)
├── table_ablation.tex                   # Ablation study (LaTeX)
├── table_generalization.tex             # Generalization (LaTeX)
└── figures/
    ├── fig1_switch_aligned_error.png    # Switch-aligned error curves
    ├── fig2_kstar_gate_alignment.png    # Gate activation alignment
    └── fig3_alpha_phase_transition.png  # α₋ phase transition
```

---

## Dependencies

All scripts require:
- `torch >= 2.0`
- `numpy >= 1.24`
- `matplotlib` (for figures)
- `pandas` (for tables)

Install via:
```bash
pip install matplotlib pandas
```

---

## Reproducibility

All scripts use **fixed random seeds** for deterministic results. Running the same script twice produces identical outputs (bit-for-bit).

To verify:
```bash
python scripts/run_benchmark.py
cp artifacts/metrics.csv metrics_run1.csv

python scripts/run_benchmark.py
diff artifacts/metrics.csv metrics_run1.csv  # Should be identical
```

---

## Customization

### Adding a New Model

1. Edit `run_benchmark.py`:
   ```python
   models_config.append(
       ('MyModel', lambda: MyModel(obs_dim, hidden_dim, obs_dim))
   )
   ```

2. Update `make_tables.py`:
   ```python
   model_order.append('MyModel')
   model_names['MyModel'] = 'My Model Name'
   ```

### Changing Benchmark Parameters

Edit `run_benchmark.py`:
```python
# Different switch probabilities
p_switch_train = 0.1
p_switch_test = 0.3

# Longer sequences
T_train = 2000
T_test = 1000

# More training
train_steps = 1000
```

---

## Troubleshooting

### "No module named 'src'"

Ensure you're running from the repository root:
```bash
cd /path/to/Antipodal-Neural-Net
python scripts/run_benchmark.py
```

### "Metrics file not found"

Run `run_benchmark.py` before `make_tables.py` or `make_figures.py`.

### Memory Issues

Reduce batch size or sequence length in `run_benchmark.py`:
```python
# Smaller test sequences
test_length = 300  # instead of 500
```

---

## Contact

For issues with scripts, open an issue on GitHub or check the main README.
