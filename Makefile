.PHONY: help fixture run test lint format-check build verify-wheel verify-sdist verify-dist clean-generated clean

help:
	@printf '%s\n' \
	  'fixture       Build the deterministic smoke fixture under outputs/fixtures/' \
	  'run           Execute the complete CPU smoke pipeline' \
	  'test          Run unit and integration tests' \
	  'lint          Run Ruff linting' \
	  'format-check  Check Ruff formatting' \
	  'build         Build wheel and source distribution' \
	  'verify-dist   Install and smoke-test both distributions outside the source tree' \
	  'clean         Remove generated local build, cache, and smoke artifacts'

fixture:
	python -m evalforge build-fixture --output outputs/fixtures/synthetic_eval_pairs.csv

run:
	python -m evalforge run --config configs/smoke.yaml

test:
	python -m pytest

lint:
	ruff check .

format-check:
	ruff format --check .

build:
	python -m build --sdist --wheel --outdir dist
	python -m twine check dist/*

verify-wheel:
	python scripts/verify_distribution_install.py --dist-dir dist --distribution wheel --packaged-smoke-config --work-dir .distcheck/wheel

verify-sdist:
	python scripts/verify_distribution_install.py --dist-dir dist --distribution sdist --packaged-smoke-config --work-dir .distcheck/sdist

verify-dist: build
	$(MAKE) verify-wheel
	$(MAKE) verify-sdist

clean-generated:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (Path('build'), Path('dist'), Path('.distcheck'), Path('.pytest_cache'), Path('.ruff_cache'), Path('wheel-validation'), Path('distribution-validation-ubuntu-latest-wheel'), Path('distribution-validation-windows-latest-wheel'), Path('distribution-validation-ubuntu-latest-sdist'), Path('outputs/fixtures'), Path('outputs/runs'), *Path('outputs').glob('controlled_validation*'))]"

clean: clean-generated
