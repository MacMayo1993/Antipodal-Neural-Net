# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1-topology-locked] - 2026-01-05

### Added

#### Core Implementation
- **ℤ₂ parity operators and projectors** (`src/parity.py`)
  - `ParityOperator` class with verified S² = I invariant
  - `ParityProjectors` class with verified idempotent, orthogonal, partition properties
  - Construction utilities for commutant/anticommutant weights
  - Verification functions for all algebraic properties

- **Antipodal regime-switching data generator** (`src/data.py`)
  - `AntipodalRegimeSwitcher` with guaranteed partial observability
  - Latent dynamics: z_{t+1} = ±A z_t with Markov switching
  - Regime switch detection utilities
  - Train/test split generation

- **Neural network models** (`src/models.py`)
  - `Z2EquivariantRNN`: Commutant-only model (no seam coupling)
  - `SeamGatedRNN`: Full model with adaptive seam gate
    - Fixed gate mode
    - Learned gate mode (MLP)
    - k*-gated mode (k ≈ 0.721 phase threshold)
  - `GRUBaseline`: Standard GRU for comparison

- **Loss functions** (`src/losses.py`)
  - Quotient loss (sign-invariant via min-distance)
  - Rank-1 projector loss (outer product Frobenius norm)
  - Invariance verification utilities

- **Classical baselines** (`src/baselines.py`)
  - AR(1) model with least-squares fitting
  - IMM (Interacting Multiple Model) filter for regime switching
  - Transition error spike computation

#### Test Suite (90+ tests across 11 sections)
- **Section 1**: Data generator tests (antipodal symmetry, switch statistics, rank deficiency)
- **Section 2**: ℤ₂ structural invariance (S² = I, projector properties, commutation)
- **Section 3**: Forward dynamics (equivariance, seam coupling)
- **Section 4**: Gate logic (k* threshold, monotonicity, gradient flow)
- **Section 5**: Loss functions (quotient invariance, rank-1 equivalence)
- **Section 6**: Training stability (gradient flow, mode collapse prevention)
- **Section 7**: Benchmark performance (model comparison, parameter efficiency, generalization)
- **Section 8**: Classical baselines (AR(1) validation, IMM consistency)
- **Section 9**: Visualization (gate-switch alignment, α₋ phase transitions)
- **Section 10**: Reproducibility (seed stability, variance bounds)
- **Section 11**: Regression (numerical stability, NaN/Inf handling)

#### Documentation
- Comprehensive README with quickstart, examples, API reference
- PROJECT_INTENT.md with design philosophy and success criteria
- API_CONTRACT.md defining stable interfaces and invariants
- CONTRIBUTING.md with development guidelines
- pytest configuration with slow test markers

### Guarantees (Enforced by Tests)

- **Mathematical correctness**: S² = I verified exactly
- **Structural properties**: Commutant/anticommutant separation enforced
- **Loss invariance**: Sign-flip invariance verified numerically
- **Numerical stability**: No NaN/Inf tolerance during training
- **Reproducibility**: Seed-deterministic behavior verified

### Breaking Changes

None (initial release)

---

## Future Releases

### Planned for v0.2
- Benchmark automation scripts
- Paper-ready figure generation
- LaTeX table export utilities
- CI/CD pipeline

### Under Consideration
- ℤ₄ and dihedral group extensions
- Real-world dataset benchmarks
- Visualization dashboard
- Model compression and quantization-aware training
