# Changelog

## v1.5.0 — Windows BF16 local-judge fallback

- Changed the Windows local scoring default from on-the-fly bitsandbytes NF4 to a Qwen model-card-compatible BF16 direct-load path using `device_map="auto"`, `low_cpu_mem_usage=True`, SafeTensors, and PyTorch SDPA.
- Added an isolated exact-model load probe before any FinQA/JudgeBench scoring. A native loader crash now fails at a small, named preflight stage instead of after artifact generation begins.
- Kept 4-bit mode as an explicit opt-in CLI path after independently validating that environment.
- Raised the BF16 scorer preflight requirement to 12 GiB free GPU memory and removed bitsandbytes execution from the default scoring runner.

## v1.5.0 — Runtime artifact lint isolation

- Fixed the actual Windows v1.3 runner failure where generated inline Python helper files under `outputs/` were recursively linted by `ruff check .`.
- Added Ruff `extend-exclude = ["outputs"]` and `force-exclude = true`; runtime reports, checkpoints, and generated helpers are never treated as maintained source.
- Added a regression test that creates deliberately non-Ruff-clean runtime helpers and verifies both recursive and explicit Ruff invocations skip them.
- Renamed the canonical Windows runner to `scripts/run_anchor_eval_v15.ps1` and aligned preflight/CI/script contracts to package version `0.15.0`.

## v1.5.0 — Windows native Python argument-passing repair

- Removed every `python -c` invocation from the Windows runner after a real Windows PowerShell run showed that embedded quotes in inline Python were lost during native argument reconstruction.
- Added UTF-8-without-BOM inline Python file materialization under the run directory and a PowerShell-native CSV row-count path.
- Added a standalone Python-file invocation preflight and Windows CI execution for it.
- Extended the Windows CI parser gate to parse every checked-in PowerShell script.
- Added regression tests requiring that the main runner contains no `"-c"` argument and that all Python preflight code is file-backed.

## v1.5.0 — PowerShell parser repair and release preflight

- Fixed the `InvalidVariableReferenceWithDrive` parser failure in the Windows runner by bracing expandable-string variables when followed by a colon.
- Added a repository-wide regression test that rejects unsafe unbraced `$variable:` interpolation in PowerShell scripts.
- Added `scripts/verify_powershell_syntax.ps1`, a standalone native `Parser.ParseFile` preflight that runs before the main runner.
- Updated the Windows GitHub Actions parser gate and the static CI contract to call the standalone parser verifier.
- Re-ran source, CPU, FinQA-adapter, DPO-config, wheel, and source-distribution validation after the repair.

## v1.1.0 — Fail-closed Windows execution and held-out reuse validation

- Added checksum-bound validation of completed held-out evaluation artifacts before reuse.
- Added native-command failure propagation to the Windows execution runner and fixed the boolean Torch-runtime probe.
- Added persistent/cache-safe clean-distribution verification and isolated headless Matplotlib configuration.
- Added a Windows GitHub Actions PowerShell parser gate for the local execution script.
- Added regression tests for held-out artifact tampering, native-runner contracts, distribution-cache handling, and lazy CLI command imports.

## v1.0.0 pre-execution final

- Rebuild FinQA-derived pairs at each run and content-address probe, scored, and checkpoint paths by the input SHA-256 prefix.
- Re-stage and re-normalize JudgeBench at each run, then content-address normalized, scored, checkpoint, and held-out artifacts by the canonical raw snapshot hash.
- Make the eight-row GPU probe use the same checkpoint, resume, checksum, model-ID, and row-count contract as the full scoring job.
- Enforce expected input and output CSV row counts in every checkpointed scoring call.
- Write FinQA evaluator runs below the invocation-specific execution directory to prevent stale candidate-bundle discovery across runs.
- No claim is made for unexecuted Windows GPU, full local-Qwen, or GitHub-hosted CI paths.

# v0.9.0 — Final pre-execution integrity and reuse hardening

- Added completed-score validation that binds any reused local-judge CSV to its input fingerprint, model ID, 4-bit mode, token limit, chunk plan, item order, row count, and output SHA-256.
- Added `evalforge validate-local-score` so PowerShell can fail closed instead of silently reusing a stale scored artifact.
- Added GPU free-memory preflight before model download and full 4-bit scoring.
- Bound JudgeBench held-out output folders to candidate-bundle and scored-input SHA-256 prefixes, preventing stale external-comparison reuse across reruns.
- Added regression coverage for tampered completed-score output rejection.

# v0.8.0 — Verified execution and resumable local scoring

- Fixed the repository formatting gate: `ruff format --check .` now passes.
- Added checkpointed, manifest-bound chunk scoring for the local judge with safe `--resume` behavior.
- Added input fingerprint, model ID, quantization mode, token limit, chunk plan, exact row-order, and local-score-status validation before cached chunks can be reused.
- Added atomic chunk and final-output publication plus a completion manifest.
- Added a CPU-only regression test that verifies resume reuses completed chunks without constructing a new GPU judge instance.
- Switched A/B likelihood continuations to the exact `A` and `B` output tokens after the chat-template assistant turn.

# v0.7.0 — Pre-execution reliability patch

- Added a conservative VCC one-class fallback: incomplete calibration diversity now yields `p_correct=0.5` and safe escalation instead of a fatal logistic-regression error or unsupported confidence claim.
- Replaced silent right-side local-judge prompt truncation with chat-template rendering and balanced task/response compaction that preserves the final A/B decision cue.
- Raised the minimum bitsandbytes version to 0.45.3 for CUDA 12.8 binary availability.

## 0.6.0

- Made the clean distribution verifier non-interactive, binary-preferring, and self-caching to avoid environment-specific pip prompts or unusable cache directories.

- Forced the non-interactive `Agg` Matplotlib backend before `pyplot` import so local terminal and headless CI report generation do not depend on a display server.

# Changelog

## v0.6.0 — Distribution Independence and Evidence-Contract Hardening

- Packaged the canonical smoke YAML as a wheel/sdist resource and added `evalforge write-smoke-config`, so clean installed distributions can run the CPU smoke path without borrowing `configs/smoke.yaml` from a source checkout.
- Added a pre-publication evidence-contract validator and checksum inventory at `numeric/artifact_manifest.json`; missing, symlinked, or malformed required artifacts now fail before atomic publication.
- Added pipeline duration telemetry to `operational_metrics.json`.
- Reduced CI artifact transfer substantially: clean-install jobs now upload only their validation JSON and published outputs, never the disposable virtual environment.
- Extended hosted CI coverage to Ubuntu Python 3.13, retained Ubuntu 3.11/3.12 and Windows 3.11, and added `PIP_NO_INPUT=1` to avoid interactive installer stalls.
- Fixed `make verify-dist` so it builds distributions once before both wheel and sdist verification.

## 0.5.0, 2026-06-26

- Removed legacy validation exports, superseded improvement-loop documents, a stale release manifest, and a redundant FinQA wrapper so the source bundle contains only current runnable code, current documentation, and configuration contracts.
- Moved generated smoke fixtures under ignored `outputs/fixtures/`; the canonical smoke pipeline now builds the fixture automatically and no longer mutates source-like `data/` paths.
- Replaced wheel-only verification with a single cross-platform distribution verifier that validates both a wheel and an sdist in clean virtual environments outside the source tree.
- Added source-distribution building and metadata checks with `python -m build --sdist --wheel` and `python -m twine check dist/*`.
- Restructured CI to build distributions once, transfer them as GitHub artifacts, verify Windows and Ubuntu wheel installs, and verify Ubuntu source-distribution installation.
- Kept tests and end-to-end smoke paths across the OS matrix while running lint/format only once on the canonical Ubuntu/Python 3.11 job to reduce redundant CI work.
- Added CI timeouts, PIP version-check suppression, and a stronger static workflow contract validator.

## 0.4.0, 2026-06-26

- Added a complete 2x2 order/style transformation orbit, including the style-and-swapped judge channel, raw orbit spread, and conservative raw nuisance routing.
- Corrected VCC so calibration and inference use the same full-orbit canonical score.
- Added controlled recovery for a stale **synthetic** fixture that is missing only the current schema columns; external and held-out data remain fail-closed.
- Added `evalforge doctor` and a `python -m evalforge` module entry point to diagnose package/runtime origin issues in `src/`-layout installations.
- Added a clean-wheel verifier that installs the built wheel in an isolated virtual environment and executes a smoke pipeline outside the source tree.
- Added explicit trust gating for joblib candidate bundles in held-out evaluation.

## 0.3.0, 2026-06-26

- Migrated to a `src/` package layout and aligned package version metadata.
- Added strict boolean parsing, config overrides, exact-file provenance manifests, and local-judge status enforcement.
- Added candidate-bundle checksums, protected source-group overlap checks, atomic publication, resolved config snapshots, and stale-staging recovery.
- Added core `tabulate` dependency required for Markdown reports.

## 0.2.0

- Introduced AAL, VCC, controlled multi-seed validation, and risk-aware routing.