import hashlib
import json
from pathlib import Path

from evalforge.data import build_synthetic_fixture
from evalforge.pipeline import run


def test_smoke_pipeline_creates_required_artifacts_and_is_not_release_qualified(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "pairs.csv"
    build_synthetic_fixture(fixture, groups_per_domain=48)
    config = {
        "run": {"run_name": "test", "seed": 1729, "output_root": str(tmp_path / "runs")},
        "data": {
            "fixture_path": str(fixture),
            "source_type": "synthetic_fixture",
            "provenance_verified": False,
            "final_split": "final",
            "calibration_split": "calibration",
            "train_split": "train",
            "policy_split": "policy",
            "dpo_split": "dpo",
        },
        "anchors": {"static_budget": 8, "adaptive_budget": 12},
        "routing": {
            "auto_accept_min_p_correct": 0.9,
            "max_wrong_auto_label_rate": 0.05,
            "max_escalation_rate": 0.95,
            "max_order_delta": 0.20,
            "max_style_delta": 0.15,
            "max_order_flip_rate": 0.10,
            "max_style_flip_rate": 0.10,
            "max_ece": 0.20,
            "min_auto_accept_coverage": 0.10,
        },
        "report": {"save_svg": False, "save_html": True},
    }
    result = run(config)
    assert (result / "numeric" / "metrics.json").exists()
    assert (result / "numeric" / "candidate_comparison.csv").exists()
    assert (result / "numeric" / "dpo_pool_manifest.json").exists()
    assert (result / "dpo_pools" / "raw_judge_equal_budget.jsonl").exists()
    assert (result / "reports" / "release_decision.md").exists()
    assert (result / "figures" / "risk_coverage.png").exists()
    artifact_manifest = json.loads((result / "numeric" / "artifact_manifest.json").read_text())
    assert artifact_manifest["artifact_count"] >= 25
    recorded = {row["path"]: row for row in artifact_manifest["artifacts"]}
    metrics = result / "numeric" / "metrics.json"
    assert (
        recorded["numeric/metrics.json"]["sha256"]
        == hashlib.sha256(metrics.read_bytes()).hexdigest()
    )


def test_missing_synthetic_fixture_is_created_inside_output_space(tmp_path: Path) -> None:
    fixture = tmp_path / "outputs" / "fixtures" / "synthetic_eval_pairs.csv"
    config = {
        "run": {
            "run_name": "auto_fixture",
            "seed": 1729,
            "output_root": str(tmp_path / "outputs" / "runs"),
        },
        "data": {
            "fixture_path": str(fixture),
            "source_type": "synthetic_fixture",
            "provenance_verified": False,
            "final_split": "final",
            "calibration_split": "calibration",
            "train_split": "train",
            "policy_split": "policy",
            "dpo_split": "dpo",
        },
        "anchors": {"static_budget": 8, "adaptive_budget": 12},
        "routing": {
            "auto_accept_min_p_correct": 0.9,
            "max_wrong_auto_label_rate": 0.05,
            "max_escalation_rate": 0.95,
            "max_order_delta": 0.20,
            "max_style_delta": 0.15,
            "max_order_flip_rate": 0.10,
            "max_style_flip_rate": 0.10,
            "max_ece": 0.20,
            "min_auto_accept_coverage": 0.10,
        },
        "report": {"save_svg": False, "save_html": False},
    }
    result = run(config)
    assert fixture.exists()
    assert (result / "numeric" / "data_manifest.json").exists()
