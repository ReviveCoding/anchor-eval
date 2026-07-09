# CI and Local Verification Runbook

## Local CPU verification

Install the checkout first. A `src/` package layout intentionally requires installation before normal CLI execution.

```bash
python -m pip install -e ".[dev]"
python -m evalforge doctor --json
ruff check .
ruff format --check .
python -m pytest -q
python -m evalforge run --config configs/smoke.yaml
```

The smoke configuration writes all generated fixture and run evidence under ignored `outputs/` paths. It does not require a checked-in fixture file.

Use `--set SECTION.KEY=VALUE` for a one-off output location or fixture path without editing the tracked YAML:

```bash
python -m evalforge run \
  --config configs/smoke.yaml \
  --set run.output_root=outputs/local_check \
  --set run.run_name=local_check
```

## Distribution verification

```bash
python -m pip install --upgrade build twine
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*

python scripts/verify_distribution_install.py \
  --dist-dir dist \
  --distribution wheel \
  --config configs/smoke.yaml \
  --work-dir .distcheck/wheel

python scripts/verify_distribution_install.py \
  --dist-dir dist \
  --distribution sdist \
  --config configs/smoke.yaml \
  --work-dir .distcheck/sdist
```

The verifier creates a clean virtual environment, installs exactly one selected distribution, runs `pip check`, confirms that EvalForge imports from `site-packages`, and executes a full smoke pipeline from outside the source tree. It resolves Python and CLI paths correctly for both POSIX and Windows virtual environments.

## What GitHub CI validates

- Ubuntu CPU paths under Python 3.11 and 3.12.
- Windows CPU path under Python 3.11.
- Dependency consistency with `pip check`.
- Runtime-origin diagnostics using `evalforge doctor`.
- Ruff lint and formatting status on the canonical Ubuntu/Python 3.11 job.
- Unit and integration tests with JUnit output on each OS/Python matrix entry.
- End-to-end synthetic CPU smoke pipeline on each OS/Python matrix entry.
- Built wheel and source-distribution metadata using `twine check`.
- Cross-job distribution artifact transfer.
- Clean wheel installation and outside-source smoke on Ubuntu and Windows.
- Clean source-distribution installation and outside-source smoke on Ubuntu.
- Short-lived JUnit, smoke, and package validation artifacts.

CI does not validate CUDA, external benchmark provenance, local open-weight judge inference, human-preference alignment, or DPO transfer. Those require the local RTX environment and locked external artifacts.

Interrupted local runs remain in hidden `.partial` folders. Inspect first, then remove only explicitly approved stale runs:

```bash
python -m evalforge clean-staging --output-root outputs/runs
python -m evalforge clean-staging --output-root outputs/runs --older-than-hours 24 --apply
```
