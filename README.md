# AnchorEval: Reference-Conditioned LLM Evaluator Reliability

EvalForge is a local-first research and product-quality framework for deciding whether a response pair can enter preference post-training. It evaluates the **evaluator** rather than only the model being evaluated.

## Core decision

Each response pair is routed to one of three actions:

- `AUTO_ACCEPT`: eligible for the isolated DPO label pool.
- `ESCALATE`: requires a stronger evaluator or human review.
- `QUARANTINE`: blocked because of deterministic-verifier conflict or a protected failure condition.

A run also emits `PASS`, `CONDITIONAL_PASS`, or `BLOCK`. Controlled synthetic evidence validates software behavior but can never qualify a release.

## v1.5 pre-execution final verification and reliability highlights

On Windows, local judge scoring defaults to **BF16 direct loading** of Qwen rather than bitsandbytes 4-bit conversion. The runner probes the exact Qwen model in an isolated child process before creating scoring artifacts. Use 4-bit only via an explicit CLI opt-in after it has separately passed this probe on the local machine.

The final CPU and package-path checks are recorded in [`VALIDATION.md`](VALIDATION.md). The Windows runner is also parsed by a native PowerShell parser in the Windows GitHub Actions job before its normal CPU test and smoke stages.

For the Windows 11 / RTX 4090 FinQA + JudgeBench execution path, run `scripts/run_anchor_eval_v15.ps1`. Before the full runner, execute `scripts/verify_powershell_syntax.ps1` and `scripts/verify_python_file_invocation.ps1`.

- Manifest-bound checkpoint/resume support for local judge scoring; completed chunks are reused only when the input fingerprint, model ID, quantization mode, token limit, chunk plan, and exact item order match.
- Completed score reuse is fail-closed: the PowerShell runner validates the checkpoint request, completion status, final CSV SHA-256, row count, item order, local-score status, and judge model before it accepts an existing scored CSV.
- Atomic per-chunk and final CSV publication with a completion manifest.
- The Windows runner requires at least 12 GiB free GPU memory before it begins its BF16 Qwen load probe and writes each external held-out evaluation into a directory keyed by candidate-bundle and scored-input hashes.

- Generated smoke fixtures now live under ignored `outputs/fixtures/`; the canonical smoke pipeline creates them automatically, so a clean checkout does not mutate source-like `data/` paths.
- The complete `outputs/` runtime-artifact tree is excluded from Ruff linting and formatting. This includes generated inline Python helpers created by the Windows runner; `force-exclude = true` ensures that stale artifacts cannot break source quality gates on rerun.
- Historical validation exports and redundant implementation-loop documents were removed from the source bundle. The repository keeps a concise changelog, current validation record, configuration examples, and the runtime documentation needed to reproduce the system.
- Wheel-only installation checking was generalized into a single clean-distribution verifier that validates both wheels and source distributions outside the source tree.
- CI builds distributions once, validates package metadata with `twine check`, transfers artifacts between jobs, verifies wheel installation on Ubuntu and Windows, and verifies source-distribution installation on Ubuntu.
- Linting and formatting run once on the canonical Ubuntu/Python 3.11 job rather than redundantly across every matrix entry; tests and end-to-end smoke runs remain cross-platform.
- Every CI job has a timeout, least-privilege token permissions, dependency caching, and a static workflow contract test.
- The canonical smoke YAML is packaged with both wheel and source distribution. `evalforge write-smoke-config` exports it into an isolated directory, allowing installed-package smoke validation with no borrowed source-tree config.
- Before atomic publication, the pipeline validates its required evidence contract and writes a SHA-256 artifact inventory. A missing, malformed, or symlinked required file leaves the run unpublished.
- Clean-install CI artifacts exclude the temporary virtual environment; only the validation summary and atomically published outputs are retained.

## Candidate hierarchy

| Candidate | Role |
|---|---|
| C0 | deterministic verifier-only partial-coverage sanity baseline |
| C1 | precomputed raw local pairwise judge channel |
| C2 | CPU-runnable TF-IDF/logistic text-pair evaluator with swap augmentation |
| C3 | stratified static-anchor calibration |
| C4 | Adaptive Anchor Linking plus Verification-Conditioned Confidence |
| C5 | C4 with policy-fit risk-aware routing and release gates |

## Research-derived specialty modules

- **Adaptive Anchor Linking, AAL** adapts the PHT reference-state principle. A group-disjoint, diverse, verifier-supported anchor set calibrates score comparability and quantifies state coverage.
- **Verification-Conditioned Confidence, VCC** adapts residual bias correction. It estimates evaluator correctness using prediction margin, verifier availability, original/swap disagreement, style sensitivity, rubric uncertainty, answer-length asymmetry, and anchor coverage.

These are project-specific research-derived adaptations, not published-method or generalized-superiority claims.

## Protected data roles

| Split | Permitted use |
|---|---|
| `train` | text-pair evaluator and hybrid blend fitting |
| `calibration` | source-group-disjoint anchor-fit and VCC-fit subsets |
| `policy` | routing-threshold selection only |
| `dpo` | equal-budget preference-pool construction only |
| `final` | immutable evaluation only |

## Runtime notes

The clean distribution verifier uses a non-interactive, binary-preferring pip install with a writable persistent cache while preserving a configured CI cache when present.


Report generation forces Matplotlib's non-interactive `Agg` backend before importing `pyplot`, so the canonical CLI works in local terminals and headless CI without a display server.

## Quick start

Install the checkout first because EvalForge uses a `src/` package layout.

```bash
python -m pip install -e ".[dev]"
python -m evalforge doctor --json
ruff check .
ruff format --check .
python -m pytest -q
python -m evalforge run --config configs/smoke.yaml
```

The smoke pipeline creates `outputs/fixtures/synthetic_eval_pairs.csv` when it does not already exist, then writes an atomically published evidence run under `outputs/runs/`.
Every published run includes `numeric/artifact_manifest.json`, a pre-publication SHA-256 inventory of the required numerical files, model bundle, DPO pools, figures, and reports.

A one-off run can override only existing configuration keys without editing the tracked YAML:

```bash
python -m evalforge run \
  --config configs/smoke.yaml \
  --set run.output_root=outputs/local_check \
  --set run.run_name=local_check
```

## Full local package verification

```bash
python -m pip install --upgrade build twine
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*

python scripts/verify_distribution_install.py \
  --dist-dir dist \
  --distribution wheel \
  --packaged-smoke-config \
  --work-dir .distcheck/wheel

python scripts/verify_distribution_install.py \
  --dist-dir dist \
  --distribution sdist \
  --packaged-smoke-config \
  --work-dir .distcheck/sdist
```

Each verifier creates a clean virtual environment, performs `pip check`, confirms that EvalForge imports from `site-packages`, exports the **packaged** smoke config, and runs the full synthetic CPU pipeline outside the source tree. It does not borrow `configs/smoke.yaml` from the checkout.

## Controlled multi-seed validation

```bash
python -m evalforge controlled-validation \
  --config configs/smoke.yaml \
  --seeds 1729,1730,1731,1732,1733 \
  --output outputs/controlled_validation \
  --groups-per-domain 24
```

C4 may be described as a calibration and tested-transformation-stability challenger only where the saved evidence supports that claim. EvalForge does **not** claim general pairwise-accuracy superiority from controlled synthetic fixtures.

## Interrupted-run recovery

Runs are built in a hidden `.partial` directory and only atomically published on success. Inspect old unpublished runs before deleting them:

```bash
python -m evalforge clean-staging --output-root outputs/runs
python -m evalforge clean-staging --output-root outputs/runs --older-than-hours 24 --apply
```

The default is inspection only.

## External JudgeBench path

JudgeBench or another external final-only benchmark must be downloaded locally, normalized, scored by a real local judge, and bound to a provenance manifest before held-out evaluation.

```bash
# 1. Normalize a locally downloaded JSONL artifact.
python -m evalforge normalize-judgebench \
  --input path/to/judgebench.jsonl \
  --output data/external/judgebench_final_unscored.csv

# 2. GPU-only. Populate original, display-swapped, and style-perturbed score channels.
python -m evalforge score-local-judge \
  --input data/external/judgebench_final_unscored.csv \
  --output data/external/judgebench_final_scored.csv \
  --model-id <local-open-weight-model>

# 3. Create a dataset-bound provenance manifest. See docs/PROVENANCE_MANIFEST_SCHEMA.md.
# 4. Evaluate saved candidates without threshold or anchor tuning.
python -m evalforge evaluate-heldout \
  --bundle outputs/runs/<run>/artifacts/candidate_bundle.joblib \
  --input data/external/judgebench_final_scored.csv \
  --output outputs/external/judgebench_eval \
  --source-type external_benchmark \
  --provenance-verified \
  --provenance-manifest data/external/judgebench_provenance.json \
  --trust-candidate-bundle
```

JudgeBench is immutable final-only data in this workflow. EvalForge refuses unscored rows, source-group overlap, corrupted candidate bundles, missing provenance records, fingerprint mismatches, or non-final held-out data. Candidate bundles use joblib, so `--trust-candidate-bundle` must be supplied only for artifacts generated and stored by a trusted local workflow.

A pre-existing held-out output is never accepted by path alone. Before reuse, run:

```bash
python -m evalforge validate-heldout \
  --bundle outputs/runs/<run>/artifacts/candidate_bundle.joblib \
  --input data/external/judgebench_final_scored.csv \
  --output outputs/external/judgebench_eval \
  --source-type external_benchmark \
  --provenance-verified \
  --provenance-manifest data/external/judgebench_provenance.json
```

The validator checks the bundle checksum, benchmark fingerprint, provenance binding, candidate inventory, per-candidate prediction counts, and SHA-256/byte counts of all published held-out artifacts.

## DPO transfer boundary

`dpo_pools/raw_judge_equal_budget.jsonl` and `dpo_pools/routed_equal_budget.jsonl` contain the same isolated DPO item IDs and pair count. They differ only in label source. This creates the required comparison boundary but does not train a model or prove downstream improvement.

Actual GPU DPO must keep base model, prompt pool, pair budget, sequence limits, QLoRA configuration, epochs, and seed set fixed. Evaluate the arms with an independent external evaluator, not the label-generating judge alone.

## Claim boundary

Before external GPU evidence:

> Implemented a local-first evaluator-validity framework with source-group split protection, reference-conditioned calibration, counterfactual bias diagnostics, risk-aware routing, provenance-bound held-out evaluation, equal-budget DPO pool construction, and clean wheel/source-distribution installation verification.

Do not claim production readiness, human alignment, external superiority, DPO transfer improvement, or Apple deployment until immutable external artifacts, local GPU execution, multi-seed comparisons, and release gates support those claims.
