import json
from pathlib import Path

import numpy as np

from evalforge import __version__
from evalforge.anchors import select_adaptive_anchors
from evalforge.candidates import AdaptiveAnchorVCC, LearnedEvaluator
from evalforge.data import build_synthetic_fixture, load_pairs, split_calibration_groups
from evalforge.features import raw_orbit_spread, raw_style_flip, style_gap
from evalforge.pipeline import run


def test_fixture_contains_complete_order_style_orbit_and_flags_swapped_style_flip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    assert "raw_judge_p_a_style_swapped" in frame.columns
    row = frame.iloc[[0]].copy()
    row["raw_judge_p_a"] = 0.8
    row["raw_judge_p_a_swapped"] = 0.2  # swapped-back P(A)=0.8
    row["raw_judge_p_a_style"] = 0.8
    row["raw_judge_p_a_style_swapped"] = 0.8  # style-swapped-back P(A)=0.2, a flip only on that arm
    assert bool(raw_style_flip(row)[0]) is True
    assert style_gap(row)[0] > 0.5
    assert raw_orbit_spread(row)[0] > 0.5


def test_c4_uses_full_orbit_and_vcc_is_fit_on_public_orbit_score(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    build_synthetic_fixture(path, groups_per_domain=48)
    frame = load_pairs(path)
    train = frame.loc[frame["split"] == "train"]
    calibration = frame.loc[frame["split"] == "calibration"]
    final = frame.loc[frame["split"] == "final"]
    anchor_fit, vcc_fit = split_calibration_groups(calibration)
    learned = LearnedEvaluator().fit(train, calibration)
    anchors = select_adaptive_anchors(anchor_fit, budget=min(12, len(anchor_fit)))
    candidate = AdaptiveAnchorVCC(anchors, learned).fit(train, anchor_fit, vcc_fit)
    prediction = candidate.predict(final)
    assert prediction.metadata["transform_aggregation"] == "mean_full_order_style_orbit"
    assert prediction.metadata["mean_raw_orbit_spread"] >= 0.0
    assert np.allclose(prediction.p_a, prediction.p_a_swap_back)
    assert np.allclose(prediction.p_a, prediction.p_a_style)


def test_pipeline_repairs_only_stale_synthetic_schema_fixture(tmp_path: Path) -> None:
    stale = tmp_path / "stale.csv"
    frame = build_synthetic_fixture(stale, groups_per_domain=48)
    frame = frame.drop(columns=["raw_judge_p_a_style_swapped"])
    frame.to_csv(stale, index=False)
    config = {
        "run": {"run_name": "stale", "seed": 1729, "output_root": str(tmp_path / "runs")},
        "data": {
            "fixture_path": str(stale),
            "source_type": "synthetic_fixture",
            "provenance_verified": False,
            "rebuild_synthetic_fixture_on_schema_error": True,
            "train_split": "train",
            "calibration_split": "calibration",
            "policy_split": "policy",
            "dpo_split": "dpo",
            "final_split": "final",
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
    assert (result / "numeric" / "metrics.json").exists()
    manifest = json.loads(
        (result / "artifacts" / "candidate_bundle.joblib.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evalforge_version"] == __version__
    assert "raw_judge_p_a_style_swapped" in load_pairs(stale).columns
