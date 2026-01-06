# Contributing to Antipodal Neural Networks

Thank you for your interest in contributing to this project!

## Code of Conduct

This project maintains a focus on **rigorous verification** of mathematical properties. All contributions should maintain or strengthen the invariants documented in `docs/API_CONTRACT.md`.

## Development Workflow

### 1. Setup

```bash
git clone https://github.com/MacMayo1993/Antipodal-Neural-Net.git
cd Antipodal-Neural-Net
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Fast tests only
pytest -m "not slow"

# Full test suite
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### 3. Make Changes

All changes must:
- Pass all existing tests
- Add new tests for new functionality
- Maintain API contract invariants
- Update documentation as needed

### 4. API Contract Rules

**Before modifying any public interface:**

1. Check if it violates guarantees in `docs/API_CONTRACT.md`
2. If yes: **Don't break it**. Extend instead.
3. If extending:
   - Update `docs/API_CONTRACT.md`
   - Add tests for new behavior
   - Update `CHANGELOG.md`

**Protected invariants** (cannot be weakened):
- Parity operator involution (S² = I)
- Projector properties (idempotent, orthogonal, partition)
- Commutation/anticommutation structure
- Loss function sign invariance

### 5. Commit Messages

Use conventional commits format:

```
feat: add dihedral group extension
fix: correct gate gradient flow
test: add coverage for edge cases
docs: update API contract for new projector
```

### 6. Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Run full test suite: `pytest`
4. Update documentation
5. Push and create PR
6. Ensure CI passes

## What We're Looking For

**High priority:**
- Bug fixes with regression tests
- Performance improvements that don't break invariants
- Additional benchmarks (real-world data with antipodal structure)
- Documentation improvements

**Medium priority:**
- Extensions to ℤ₄ or dihedral groups
- Visualization utilities
- Additional classical baselines

**Lower priority:**
- Architectural variations without theoretical justification
- Hyperparameter tuning without ablation studies

## Testing Requirements

All PRs must include tests. Specifically:

### For new models:
- Forward pass shape tests
- Gradient flow tests
- Numerical stability tests (NaN/Inf)

### For new symmetry structures:
- Algebraic property verification (group axioms)
- Commutation/anticommutation tests
- Projector/operator construction tests

### For new benchmarks:
- Reproducibility tests (seed stability)
- Metric export in standard format
- Comparison with existing baselines

## Running Benchmarks

Benchmarks are compute-intensive and marked with `@pytest.mark.slow`:

```bash
# Skip slow tests (default)
pytest

# Run benchmarks
pytest -m slow

# Run everything
pytest -m ""
```

## Questions?

Open an issue with:
- Clear description of what you want to contribute
- Whether it affects the API contract
- Expected behavior and tests

We'll respond within a few days.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).
