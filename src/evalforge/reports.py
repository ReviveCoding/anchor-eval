from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_markdown_reports(
    report_dir: str | Path, comparison: pd.DataFrame, release: dict, manifest: dict
) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table = comparison.to_markdown(index=False, floatfmt=".4f")
    evidence_tier = str(release.get("evidence_tier", "unknown"))
    (report_dir / "benchmark_card.md").write_text(
        "# Benchmark Card\n\n"
        "The packaged smoke fixture is a controlled synthetic small-data dataset. It validates data contracts, split "
        "protection, routing, artifact lineage, counterfactual plumbing, and equal-budget DPO-pool boundaries. It must "
        "not be cited as FinQA, JudgeBench, IFEval, human-preference, or production-performance evidence.\n",
        encoding="utf-8",
    )
    (report_dir / "evaluator_card.md").write_text(
        "# Evaluator Card\n\n"
        "C2 is a CPU-runnable text-pair evaluator trained only on the train split with A/B swap augmentation. C4 adds "
        "group-disjoint Adaptive Anchor Linking and a separately fit Verification-Conditioned Confidence layer. C5 "
        "tunes its route threshold on the isolated policy split and blocks deterministic-verifier conflicts.\n\n"
        f"## Candidate comparison\n\n{table}\n",
        encoding="utf-8",
    )
    (report_dir / "judge_bias_report.md").write_text(
        "# Judge Bias Report\n\n"
        "Measured stress dimensions are candidate-specific: A/B display-order flips and presentation-style flips. The "
        "report also tracks magnitude deltas rather than only a global raw-judge proxy. Synthetic diagnostics cannot "
        "establish robustness on external models or benchmarks.\n",
        encoding="utf-8",
    )
    (report_dir / "benchmark_leakage_analysis.md").write_text(
        "# Benchmark Leakage Analysis\n\n"
        "Source-group disjointness is validated before model fitting. Calibration is further divided into group-disjoint "
        "anchor-fit and VCC-fit subsets. The policy split alone fits the routing threshold. The dpo split alone creates "
        "preference pools. Final split rows cannot be used for anchors, calibration, routing policy selection, or DPO training.\n",
        encoding="utf-8",
    )
    (report_dir / "evaluator_induced_training_drift_report.md").write_text(
        "# Evaluator-Induced Training Drift Report\n\n"
        "The run materializes identical-item, equal-budget raw-judge and routed-evaluator DPO pools from an isolated dpo "
        "split. This establishes the comparison boundary only. A downstream gain claim requires actual equal-budget GPU "
        "training and independent external evaluation over multiple seeds.\n",
        encoding="utf-8",
    )
    (report_dir / "incident_replay_report.md").write_text(
        "# Incident Replay Report\n\n"
        "Deterministic verifier conflicts route to QUARANTINE. Candidate-specific display-order or presentation-style "
        "instability routes to ESCALATE. Missing lineage, source-group leakage, and non-dpo DPO pool use fail validation.\n",
        encoding="utf-8",
    )
    (report_dir / "release_decision.md").write_text(
        "# Release Decision\n\n"
        f"Evidence tier: `{evidence_tier}`\n\n"
        "```json\n" + json.dumps(release, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (report_dir / "run_manifest.md").write_text(
        "# Run Manifest\n\n```json\n" + json.dumps(manifest, indent=2) + "\n```\n",
        encoding="utf-8",
    )
