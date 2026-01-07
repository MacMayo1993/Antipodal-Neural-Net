.PHONY: help demo test test-fast test-slow install clean benchmark tables figures lint format verify smoke-benchmark

help:
	@echo "Antipodal Neural Networks - Makefile"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make demo       - Run quick demo (1-2 min)"
	@echo "  make verify     - Run tests + smoke benchmark (green bar)"
	@echo "  make test       - Run all tests"
	@echo "  make test-fast  - Run fast tests only"
	@echo "  make install    - Install package in dev mode"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark  - Run full benchmark (~30 min)"
	@echo "  make smoke-benchmark - Run quick benchmark (1 seed, 50 steps)"
	@echo "  make tables     - Generate LaTeX tables"
	@echo "  make figures    - Generate publication figures"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove artifacts and cache"

demo:
	@echo "Running quick demo..."
	antipodal-demo

install:
	pip install -e .

test:
	pytest

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m slow

benchmark:
	python scripts/run_benchmark.py

tables:
	python scripts/make_tables.py

figures:
	python scripts/make_figures.py

lint:
	@echo "Checking code style..."
	@black --check src/ tests/ examples/ scripts/ || true
	@flake8 src/ tests/ examples/ scripts/ --max-line-length=100 --ignore=E203,W503 || true

format:
	black src/ tests/ examples/ scripts/

clean:
	@echo "Cleaning artifacts and cache..."
	rm -rf artifacts/metrics.csv artifacts/table_*.tex artifacts/figures/*.png
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .coverage htmlcov
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

verify:
	@echo "============================================"
	@echo "Running verification suite (green bar)..."
	@echo "============================================"
	@echo ""
	@echo "[1/3] Running pytest..."
	pytest -q
	@echo ""
	@echo "[2/3] Running quick smoke test..."
	python quick_smoke_test.py
	@echo ""
	@echo "[3/3] Running smoke benchmark (1 seed, 50 steps)..."
	python scripts/run_benchmark.py --seeds 42 --steps 50
	@echo ""
	@echo "============================================"
	@echo "✅ ALL VERIFICATION CHECKS PASSED"
	@echo "============================================"

smoke-benchmark:
	@echo "Running quick benchmark (1 seed, 50 steps)..."
	python scripts/run_benchmark.py --seeds 42 --steps 50
