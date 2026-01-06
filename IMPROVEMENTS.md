# Future Improvements and Enhancements

This document tracks suggested improvements for the Antipodal Neural Networks repository based on comprehensive code review feedback.

## High Priority (Next Release)

### Code Quality
- [ ] **Add comprehensive type hints** - Extend type annotations to all functions, use `mypy` for static checking
- [ ] **Device handling** - Add consistent `device` parameter to all models and data generators
- [ ] **Enum for gate types** - Replace string literals ('kstar', 'learned', etc.) with `enum.Enum`
- [ ] **Logging instead of print** - Use Python `logging` module in scripts and examples
- [ ] **Refactor parity splitting** - Create `split_parity()` utility in `parity.py` to reduce duplication

### Testing
- [ ] **Coverage target** - Aim for >95% test coverage (run `pytest --cov=src`)
- [ ] **Integration test** - Add `test_benchmark.py` with mini-benchmark asserting metric ordering
- [ ] **Property-based tests** - Use `hypothesis` for testing parity properties on random inputs
- [ ] **Generalization tests** - Validate p_switch mismatch scenarios (train 0.05, test 0.2)

### Documentation
- [ ] **API documentation** - Generate with Sphinx from docstrings, link to `API_CONTRACT.md`
- [ ] **Jupyter tutorials** - Add interactive notebooks in `examples/`
- [ ] **Benchmark results section** - Add sample tables/plots to README from `metrics.csv`
- [ ] **Common issues** - Document troubleshooting (e.g., gate not activating → lower tau)

## Medium Priority

### Features
- [ ] **Configuration system** - Create `configs/` directory with YAML/JSON for experiments
- [ ] **Model factory** - Add `create_model(model_type, **kwargs)` to centralize instantiation
- [ ] **Visualization utilities** - Create `src/viz.py` with functions like `plot_gate_activations()`
- [ ] **Pre-trained models** - Add `checkpoints/` with sample trained models
- [ ] **Batch inference** - Optimize forward pass for batch_size > 1

### Performance
- [ ] **GPU benchmarking** - Add `--device cuda` to `run_benchmark.py` and profile
- [ ] **Parameter efficiency** - Explore low-rank adapters for W_flip
- [ ] **Early stopping** - Add validation split and early stopping in training loops
- [ ] **Parallel benchmarks** - Use `multiprocessing` to run seeds in parallel

### Infrastructure
- [ ] **Docker setup** - Add `Dockerfile` and `docker-compose.yml` for reproducibility
- [ ] **Results versioning** - Save artifacts with timestamps (e.g., `results/2026-01-06/`)
- [ ] **Pin dependencies** - Use exact versions in `requirements.txt` (e.g., `torch==2.0.0`)

## Lower Priority (Future Work)

### Research Extensions
- [ ] **Real-world datasets** - Add experiments on financial/physics data with antipodal structure
- [ ] **Hyperparameter tuning** - Integrate `optuna` or `ray.tune` for sweeping parameters
- [ ] **Extend to other symmetries** - Generalize to SO(3), dihedral groups in `symmetries.py`
- [ ] **Transformer adaptation** - Implement parity-aware attention mechanisms
- [ ] **Ablation studies** - Test different k* values, tau sweeps; visualize in figures
- [ ] **Regime detection metric** - Add gate peak alignment vs. true switches as benchmark metric

### Community & Publishing
- [ ] **arXiv submission** - Write paper with benchmark results
- [ ] **Social promotion** - Share on r/MachineLearning, Twitter/X with demo notebook
- [ ] **Issue templates** - Add GitHub templates for feature requests and bug reports
- [ ] **Contribution encouragement** - Highlight in CONTRIBUTING.md how to contribute

### Usability
- [ ] **Loss combinations** - Allow mixing losses (quotient + MSE) via config
- [ ] **CLI enhancements** - Add more flags to `antipodal-train` (e.g., `--checkpoint-dir`)
- [ ] **Progress bars** - Use `tqdm` in training loops for better UX

## Code Fixes Applied (2026-01-06)

### ✅ Completed
- [x] **Numerical stability in losses** - Added epsilon to `rank1_projector_loss` normalization
- [x] **Improved dynamics stability** - Enhanced spectral radius verification in `data.py`
- [x] **Gitignore artifacts** - Added artifacts/, logs, and temp files to `.gitignore`

### Already Correct
- [x] **Missing import** - `Tuple` already imported in `baselines.py`
- [x] **Error handling** - IMM filter and other edge cases handled appropriately

## Notes
- Repository is **production-ready** as of v0.1.0
- Test suite (90+ tests) validates all core functionality
- Benchmark scripts exist but are optional (test suite is primary validation)
- Focus on high-priority items for v0.2.0 release

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on implementing these improvements.
