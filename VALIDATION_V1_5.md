# AnchorEval v1.5 Validation Record

## Root cause addressed

The v1.4 Windows log completed Qwen download but exited with Windows code `0xC0000005` at `Loading weights: 0/398` while `score-local-judge` was using on-the-fly bitsandbytes 4-bit loading. The exact crashing native component cannot be identified from that log alone. v1.5 therefore removes that path from the default Windows runner instead of silently retrying it.

## v1.5 execution policy

- Windows local judge scoring defaults to **BF16 direct loading**.
- The runner calls an isolated `probe-local-judge --precision bf16` process before any FinQA or JudgeBench scoring output is created.
- The previous 4-bit bitsandbytes path remains explicit CLI opt-in only, not the runner default.
- The BF16 preflight requires 12 GiB free VRAM. The reported target-machine log had 14.72 GiB free before the v1.4 crash.

## Re-executed validation

| Check | Result |
|---|---|
| Ruff lint | PASS |
| Ruff format check | PASS |
| Python byte-compilation | PASS |
| Test suite | PASS, 48 tests |
| CI workflow contract | PASS |
| CPU smoke pipeline | PASS |
| Controlled validation | PASS, 5 seeds / 30 candidate-seed rows |
| Actual FinQA archive adapter | PASS, 491 derived pairs from 500 requested examples |
| Wheel and sdist build | PASS |
| `twine check` | PASS |
| Release ZIP extraction, source tests, CI contract | PASS |
| Release ZIP CPU smoke | PASS |
| BF16 loader contract tests | PASS, mocked direct-loader and CLI/runner contract |

## Boundary

This environment did not execute native Windows PowerShell, an RTX 4090 BF16 Qwen load, full FinQA/JudgeBench scoring, or hosted GitHub Actions. v1.5 must first pass the local **Probe Qwen BF16 model load in isolated child process** stage on the target machine. A successful source/CPU validation does not by itself establish a successful GPU runtime.
