from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .anchors import select_adaptive_anchors, select_static_anchors
from .candidates import (
    AdaptiveAnchorVCC,
    DeterministicVerifier,
    LearnedEvaluator,
    ProductionRouter,
    RawJudge,
    StaticAnchorEvaluator,
)
from .config import validate_config
from .data import (
    build_synthetic_fixture,
    dataset_fingerprint,
    load_pairs,
    split_calibration_groups,
    split_frame,
    validate_provenance_manifest,
    write_manifest,
)
from .dpo import build_equal_budget_label_pools
from .evaluation import (
    evaluate_prediction,
    grouped_bootstrap_delta,
    operating_profile_scenarios,
    select_release_decision,
    threshold_scenarios,
    write_json,
)
from .heldout import save_candidate_bundle
from .reports import write_markdown_reports
from .visualizations import (
    plot_calibration,
    plot_candidate_comparison,
    plot_data_quality,
    plot_failure_taxonomy,
    plot_risk_coverage,
    write_html_summary,
)


def _config_hash(config: dict[str, Any]) -> str:
    return sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_dir(config: dict[str, Any]) -> Path:
    name = str(config["run"]["run_name"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    token = uuid4().hex[:8]
    return Path(config["run"]["output_root"]) / f"{name}_{stamp}_s{config['run']['seed']}_{token}"


def _metadata_row(prediction_metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in prediction_metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _required_evidence_paths(config: dict[str, Any]) -> list[Path]:
    """Return the publish-time artifact contract relative to a completed run directory."""
    numeric = [
        "numeric/config_resolved.yaml",
        "numeric/data_manifest.json",
        "numeric/static_anchor_diagnostics.csv",
        "numeric/adaptive_anchor_diagnostics.csv",
        "numeric/candidate_comparison.csv",
        "numeric/slice_metrics.csv",
        "numeric/predictions.csv",
        "numeric/decision_scenarios.csv",
        "numeric/operating_point_recommendations.csv",
        "numeric/dpo_pool_manifest.json",
        "numeric/metrics.json",
        "numeric/operational_metrics.json",
        "numeric/project_creation_handoff.json",
        "numeric/run_manifest.json",
    ]
    artifacts = [
        "artifacts/candidate_bundle.joblib",
        "artifacts/candidate_bundle.joblib.manifest.json",
        "dpo_pools/raw_judge_equal_budget.jsonl",
        "dpo_pools/routed_equal_budget.jsonl",
    ]
    figures = [
        "figures/data_composition.png",
        "figures/candidate_comparison.png",
        "figures/calibration.png",
        "figures/risk_coverage.png",
        "figures/failure_taxonomy.png",
    ]
    if bool(config["report"].get("save_svg", True)):
        figures.extend(path.replace(".png", ".svg") for path in figures.copy())
    reports = [
        "reports/benchmark_card.md",
        "reports/evaluator_card.md",
        "reports/judge_bias_report.md",
        "reports/benchmark_leakage_analysis.md",
        "reports/evaluator_induced_training_drift_report.md",
        "reports/incident_replay_report.md",
        "reports/release_decision.md",
        "reports/run_manifest.md",
    ]
    if bool(config["report"].get("save_html", True)):
        reports.append("reports/summary.html")
    return [Path(path) for path in numeric + artifacts + figures + reports]


def validate_evidence_package(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when a completed run is missing, corrupt, or staging-linked evidence.

    The manifest is written before atomic publication. This protects the public evidence contract
    from accidental refactors that silently omit an artifact while leaving a seemingly successful run.
    """
    expected = _required_evidence_paths(config)
    records: list[dict[str, object]] = []
    for relative in expected:
        path = run_dir / relative
        if not path.exists() or not path.is_file():
            raise RuntimeError(
                f"Evidence contract missing required artifact: {relative.as_posix()}"
            )
        if path.is_symlink():
            raise RuntimeError(f"Evidence artifact must not be a symlink: {relative.as_posix()}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    for relative in (
        "numeric/metrics.json",
        "numeric/operational_metrics.json",
        "numeric/run_manifest.json",
        "artifacts/candidate_bundle.joblib.manifest.json",
    ):
        try:
            json.loads((run_dir / relative).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Evidence contract contains invalid JSON: {relative}") from exc
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact_count": len(records),
        "artifacts": records,
        "claim_boundary": "Artifact inventory validates completeness and integrity only; it does not upgrade synthetic evidence.",
    }
    write_json(run_dir / "numeric" / "artifact_manifest.json", manifest)
    return manifest


def _run_into(config: dict[str, Any], run_dir: Path, public_run_dir: Path) -> None:
    started = time.perf_counter()
    seed = int(config["run"]["seed"])
    data_config = config["data"]
    fixture_path = Path(data_config["fixture_path"])
    if not fixture_path.exists():
        if str(data_config["source_type"]) != "synthetic_fixture":
            raise FileNotFoundError(
                f"Dataset does not exist: {fixture_path}. Only synthetic_fixture may be auto-generated."
            )
        build_synthetic_fixture(fixture_path, seed=seed)
    try:
        frame = load_pairs(fixture_path)
    except ValueError as exc:
        # Tracked smoke fixtures are reproducible generated assets.  A schema expansion
        # must not leave a fresh clone unable to run merely because an older fixture file
        # was retained.  Never repair non-synthetic or semantically invalid data silently.
        can_rebuild = (
            str(data_config["source_type"]) == "synthetic_fixture"
            and bool(data_config.get("rebuild_synthetic_fixture_on_schema_error", True))
            and "Missing required columns" in str(exc)
        )
        if not can_rebuild:
            raise
        build_synthetic_fixture(fixture_path, seed=seed)
        frame = load_pairs(fixture_path)
    provenance_record: dict[str, object] | None = None
    if str(data_config["source_type"]) == "external_benchmark":
        provenance_record = validate_provenance_manifest(
            str(data_config["provenance_manifest_path"]),
            fixture_path,
            set(frame["split"].astype(str).unique()),
        )
    if frame["raw_judge_status"].astype(str).eq("unscored").any():
        raise ValueError(
            "Pipeline input contains unscored raw judge channels. Run `evalforge score-local-judge` before evaluation."
        )

    numeric_dir = run_dir / "numeric"
    figure_dir = run_dir / "figures"
    report_dir = run_dir / "reports"
    dpo_dir = run_dir / "dpo_pools"
    artifact_dir = run_dir / "artifacts"
    numeric_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (numeric_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    data_manifest = write_manifest(
        numeric_dir / "data_manifest.json",
        fixture_path,
        frame,
        source_type=str(data_config["source_type"]),
        provenance_verified=bool(data_config["provenance_verified"]),
    )
    if provenance_record is not None:
        data_manifest["provenance_manifest"] = provenance_record
        write_json(numeric_dir / "data_manifest.json", data_manifest)
    train = split_frame(frame, str(data_config["train_split"]))
    calibration = split_frame(frame, str(data_config["calibration_split"]))
    policy = split_frame(frame, str(data_config["policy_split"]))
    dpo = split_frame(frame, str(data_config["dpo_split"]))
    final = split_frame(frame, str(data_config["final_split"]))
    anchor_fit, vcc_fit = split_calibration_groups(calibration, seed=seed)

    static = select_static_anchors(anchor_fit, int(config["anchors"]["static_budget"]))
    adaptive = select_adaptive_anchors(anchor_fit, int(config["anchors"]["adaptive_budget"]))
    static.diagnostics.to_csv(numeric_dir / "static_anchor_diagnostics.csv", index=False)
    adaptive.diagnostics.to_csv(numeric_dir / "adaptive_anchor_diagnostics.csv", index=False)

    raw = RawJudge().fit(train, calibration)
    verifier = DeterministicVerifier().fit(train, calibration)
    learned = LearnedEvaluator(random_state=seed).fit(train, calibration)
    static_eval = StaticAnchorEvaluator(static).fit(anchor_fit)
    aal_vcc = AdaptiveAnchorVCC(adaptive, learned, random_state=seed).fit(
        train, anchor_fit, vcc_fit
    )
    routed = ProductionRouter(
        aal_vcc,
        threshold=float(config["routing"]["auto_accept_min_p_correct"]),
        max_wrong_auto_label_rate=float(config["routing"]["max_wrong_auto_label_rate"]),
        max_escalation_rate=float(config["routing"]["max_escalation_rate"]),
        order_gap_limit=float(config["routing"]["max_order_delta"]),
        style_gap_limit=float(config["routing"]["max_style_delta"]),
    ).fit(policy)
    candidates = [verifier, raw, learned, static_eval, aal_vcc, routed]

    candidate_bundle_path = artifact_dir / "candidate_bundle.joblib"
    bundle_manifest = save_candidate_bundle(
        candidate_bundle_path,
        {
            "C0_verifier": verifier,
            "C1_raw_judge": raw,
            "C2_text_pair_evaluator": learned,
            "C3_static_anchor": static_eval,
            "C4_aal_vcc": aal_vcc,
            "C5_routed_production": routed,
        },
        metadata={
            "bundle_schema_version": "2.0",
            "evalforge_version": __version__,
            "dataset_fingerprint_sha256": dataset_fingerprint(fixture_path),
            "config_hash": _config_hash(config),
            "protected_source_groups": sorted(frame["source_group"].astype(str).unique()),
            "source_type": str(data_config["source_type"]),
            "provenance_verified": bool(data_config["provenance_verified"]),
            "raw_judge_statuses": sorted(frame["raw_judge_status"].astype(str).unique()),
        },
        public_path="artifacts/candidate_bundle.joblib",
    )

    metric_rows: list[dict[str, Any]] = []
    slices: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    prediction_map: dict[str, pd.DataFrame] = {}
    routed_final_prediction = None
    for candidate in candidates:
        prediction = candidate.predict(final)
        metrics, slice_df, enriched = evaluate_prediction(final, prediction)
        metric_rows.append(
            {
                "candidate_id": prediction.candidate_id,
                **metrics,
                **_metadata_row(prediction.metadata),
            }
        )
        slices.append(slice_df)
        predictions.append(enriched)
        prediction_map[prediction.candidate_id] = enriched
        if prediction.candidate_id == "C5_routed_production":
            routed_final_prediction = prediction
    comparison = pd.DataFrame(metric_rows).sort_values("candidate_id")
    all_slices = pd.concat(slices, ignore_index=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    comparison.to_csv(numeric_dir / "candidate_comparison.csv", index=False)
    all_slices.to_csv(numeric_dir / "slice_metrics.csv", index=False)
    all_predictions.to_csv(numeric_dir / "predictions.csv", index=False)

    assert routed_final_prediction is not None
    order_gap_limit = float(config["routing"]["max_order_delta"])
    style_gap_limit = float(config["routing"]["max_style_delta"])
    scenarios = threshold_scenarios(
        final,
        routed_final_prediction,
        thresholds=np.linspace(0.50, 0.99, 11),
        order_gap_limit=order_gap_limit,
        style_gap_limit=style_gap_limit,
    )
    scenarios.to_csv(numeric_dir / "decision_scenarios.csv", index=False)
    profile_scenarios = operating_profile_scenarios(
        final,
        routed_final_prediction,
        max_wrong_auto_label_rate=float(config["routing"]["max_wrong_auto_label_rate"]),
        order_gap_limit=order_gap_limit,
        style_gap_limit=style_gap_limit,
    )
    profile_scenarios.to_csv(numeric_dir / "operating_point_recommendations.csv", index=False)

    c5_metrics = (
        comparison.loc[comparison["candidate_id"] == "C5_routed_production"].iloc[0].to_dict()
    )
    raw_judge_statuses = set(final["raw_judge_status"].astype(str).unique())
    release = select_release_decision(
        c5_metrics,
        max_wrong_auto_label_rate=float(config["routing"]["max_wrong_auto_label_rate"]),
        max_order_flip_rate=float(config["routing"]["max_order_flip_rate"]),
        max_style_flip_rate=float(config["routing"]["max_style_flip_rate"]),
        max_ece=float(config["routing"]["max_ece"]),
        min_auto_accept_coverage=float(config["routing"]["min_auto_accept_coverage"]),
        source_type=str(data_config["source_type"]),
        provenance_verified=bool(data_config["provenance_verified"]),
        raw_judge_statuses=raw_judge_statuses,
        policy_feasible=bool(routed.policy_diagnostics.get("policy_feasible", False)),
    )
    release_payload = {
        "status": release.status,
        "reasons": list(release.reasons),
        "evidence_tier": (
            "synthetic_controlled_validation"
            if bool(data_manifest["synthetic_or_controlled_evidence"])
            else "external_or_user_supplied_pending_validation"
        ),
        "policy_diagnostics": routed.policy_diagnostics,
    }

    raw_dpo_prediction = raw.predict(dpo)
    routed_dpo_prediction = routed.predict(dpo)
    _, _, raw_dpo_enriched = evaluate_prediction(dpo, raw_dpo_prediction)
    _, _, routed_dpo_enriched = evaluate_prediction(dpo, routed_dpo_prediction)
    dpo_pools = build_equal_budget_label_pools(
        raw_dpo_enriched, routed_dpo_enriched, frame, dpo_dir, seed=seed
    )
    write_json(numeric_dir / "dpo_pool_manifest.json", dpo_pools)

    c4 = prediction_map["C4_aal_vcc"]
    c1 = prediction_map["C1_raw_judge"]
    bootstrap = grouped_bootstrap_delta(
        final, c4["correct"].to_numpy(), c1["correct"].to_numpy(), seed=seed
    )
    operational = {
        "cpu_python": sys.version,
        "platform": platform.platform(),
        "candidate_count": len(candidates),
        "final_rows": int(len(final)),
        "calibration_anchor_rows": int(len(anchor_fit)),
        "calibration_vcc_rows": int(len(vcc_fit)),
        "aal_minus_raw_grouped_bootstrap": bootstrap,
        "dpo_equal_budget": dpo_pools,
        "candidate_bundle_sha256": _sha256_file(candidate_bundle_path),
        "pipeline_duration_seconds_pre_report": round(time.perf_counter() - started, 6),
    }
    write_json(
        numeric_dir / "metrics.json", {"candidates": metric_rows, "release": release_payload}
    )
    write_json(numeric_dir / "operational_metrics.json", operational)

    project_creation_handoff = {
        "schema_version": "1.2",
        "project_id": "evalforge",
        "run_id": public_run_dir.name,
        "claim_boundary": (
            "Controlled synthetic small-data validation only. No external benchmark, human-alignment, local-LLM, "
            "DPO-transfer, or production-performance claim is validated."
        ),
        "project_depth": {
            "foundation_complete": True,
            "industry_baseline_complete": True,
            "advanced_technique_complete": True,
            "specialty_module_complete": True,
            "product_layer_complete": True,
        },
        "candidate_portfolio": [
            {
                "candidate_id": row["candidate_id"],
                "type": {
                    "C0_verifier": "sanity",
                    "C1_raw_judge": "industry",
                    "C2_text_pair_evaluator": "advanced",
                    "C3_static_anchor": "specialty",
                    "C4_aal_vcc": "hybrid",
                    "C5_routed_production": "production",
                }.get(row["candidate_id"], "unknown"),
                "metrics": {key: value for key, value in row.items() if key != "candidate_id"},
                "status": "controlled_synthetic_validated",
            }
            for row in metric_rows
        ],
        "specialty_transfer": [
            {
                "source_method": "Physical Health Timestep reference-state alignment",
                "adapted_name": "Adaptive Anchor Linking",
                "adaptation_type": "derived",
                "implementation_path": "src/evalforge/anchors.py",
                "test_path": "tests/test_anchors.py",
                "limitations": ["Controlled synthetic validation only."],
            },
            {
                "source_method": "Domain-informed residual bias correction",
                "adapted_name": "Verification-Conditioned Confidence",
                "adaptation_type": "inspired",
                "implementation_path": "src/evalforge/candidates.py",
                "test_path": "tests/test_routing.py",
                "limitations": ["External calibration remains required."],
            },
        ],
        "champion_selection": {
            "predictive_champion": "unselected_pending_external_evaluation",
            "production_champion": "unselected_pending_external_evaluation",
            "selection_criteria": [
                "wrong-auto-label risk",
                "raw counterfactual flip rate",
                "coverage",
                "calibration",
                "policy capacity",
            ],
            "tradeoffs": [
                "Synthetic results cannot establish a real champion or release qualification."
            ],
        },
        "outputs": {
            "numeric": [
                "metrics.json",
                "candidate_comparison.csv",
                "slice_metrics.csv",
                "decision_scenarios.csv",
                "operating_point_recommendations.csv",
                "dpo_pool_manifest.json",
                "config_resolved.yaml",
                "run_manifest.json",
                "artifact_manifest.json",
            ],
            "model_artifacts": ["artifacts/candidate_bundle.joblib"],
            "visualizations": [
                "data_composition",
                "candidate_comparison",
                "calibration",
                "risk_coverage",
                "failure_taxonomy",
            ],
            "reports": [
                "benchmark_card.md",
                "evaluator_card.md",
                "release_decision.md",
                "incident_replay_report.md",
            ],
            "cards": ["benchmark_card.md", "evaluator_card.md"],
            "api_or_cli_examples": ["evalforge build-fixture", "evalforge run"],
        },
        "experiment_lineage": {
            "dataset_fingerprints": [data_manifest["dataset_fingerprint_sha256"]],
            "config_hashes": [_config_hash(config)],
            "resolved_config": "numeric/config_resolved.yaml",
            "artifact_checksums": [bundle_manifest["bundle_sha256"]],
            "candidate_bundle": "artifacts/candidate_bundle.joblib",
            "dpo_pool_manifest": dpo_pools,
        },
    }
    write_json(numeric_dir / "project_creation_handoff.json", project_creation_handoff)
    run_manifest = {
        "run_id": public_run_dir.name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_hash": _config_hash(config),
        "seed": seed,
        "source_type": data_manifest["source_type"],
        "provenance_verified": data_manifest["provenance_verified"],
        "raw_judge_statuses": data_manifest["raw_judge_status_counts"],
        "claim_boundary": "Controlled synthetic small-data validation only; no external benchmark or production-performance claim.",
        "protected_split_roles": {
            "train": "evaluator learning",
            "calibration": "anchor/VCC split",
            "policy": "threshold selection",
            "dpo": "isolated label pool",
            "final": "immutable evaluation",
        },
        "artifact_manifest": {
            "path": "numeric/artifact_manifest.json",
            "artifact_count": len(_required_evidence_paths(config)),
        },
    }
    write_json(numeric_dir / "run_manifest.json", run_manifest)

    save_svg = bool(config["report"].get("save_svg", True))
    plot_data_quality(frame, figure_dir / "data_composition", save_svg)
    plot_candidate_comparison(comparison, figure_dir / "candidate_comparison", save_svg)
    plot_calibration(all_predictions, figure_dir / "calibration", save_svg)
    plot_risk_coverage(scenarios, figure_dir / "risk_coverage", save_svg)
    plot_failure_taxonomy(
        prediction_map["C5_routed_production"], figure_dir / "failure_taxonomy", save_svg
    )
    if bool(config["report"].get("save_html", True)):
        write_html_summary(report_dir / "summary.html", comparison, release_payload)
    write_markdown_reports(report_dir, comparison, release_payload, run_manifest)
    operational["pipeline_duration_seconds"] = round(time.perf_counter() - started, 6)
    write_json(numeric_dir / "operational_metrics.json", operational)
    validate_evidence_package(run_dir, config)


def run(config: dict[str, Any]) -> Path:
    """Run into an unpublished staging directory, then atomically publish complete evidence."""
    validate_config(config)
    final_dir = _run_dir(config)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.with_name(f".{final_dir.name}.partial")
    if staging_dir.exists() or final_dir.exists():
        raise FileExistsError(f"Run directory collision: {final_dir}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        _run_into(config, staging_dir, final_dir)
        staging_dir.replace(final_dir)
    except Exception as exc:
        failure = {
            "status": "FAILED_BEFORE_PUBLICATION",
            "staging_dir": str(staging_dir),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        (staging_dir / "failure_manifest.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise
    return final_dir
