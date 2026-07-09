from __future__ import annotations

from pathlib import Path

import pandas as pd

from evalforge.data import build_finqa_pair_rows
from evalforge.dpo import build_equal_budget_label_pools
from evalforge.pipeline import run


def _finqa_like_records(count: int = 42) -> list[dict]:
    return [
        {
            "id": f"source-{index}",
            "qa": {"question": f"Question {index}?", "answer": str(index + 1), "program": []},
        }
        for index in range(count)
    ]


def test_one_class_vcc_fallback_publishes_safe_evidence(tmp_path: Path) -> None:
    pairs = build_finqa_pair_rows(_finqa_like_records(), max_examples=42)
    pairs["raw_judge_status"] = "local_scored"
    pairs["raw_judge_model_id"] = "local-test-model"
    fixture = tmp_path / "finqa_like.csv"
    pairs.to_csv(fixture, index=False)
    config = {
        "run": {"run_name": "one_class", "seed": 1729, "output_root": str(tmp_path / "runs")},
        "data": {
            "fixture_path": str(fixture),
            "source_type": "public_derived_controlled",
            "provenance_verified": False,
            "train_split": "train",
            "calibration_split": "calibration",
            "policy_split": "policy",
            "dpo_split": "dpo",
            "final_split": "final",
        },
        "anchors": {"static_budget": 2, "adaptive_budget": 3},
        "routing": {
            "auto_accept_min_p_correct": 0.9,
            "max_wrong_auto_label_rate": 0.05,
            "max_escalation_rate": 0.2,
            "max_order_delta": 0.2,
            "max_style_delta": 0.15,
            "max_order_flip_rate": 0.1,
            "max_style_flip_rate": 0.1,
            "max_ece": 0.1,
            "min_auto_accept_coverage": 0.6,
        },
        "report": {"save_svg": False, "save_html": False},
    }
    result = run(config)
    predictions = pd.read_csv(result / "numeric" / "predictions.csv")
    routed = predictions.loc[predictions["candidate_id"] == "C5_routed_production"]
    assert set(routed["route"]) == {"ESCALATE"}
    assert set(routed["reason_code"]) == {"vcc_unidentifiable"}
    manifest = (result / "numeric" / "dpo_pool_manifest.json").read_text(encoding="utf-8")
    assert "unavailable_zero_routed_labels" in manifest


def test_zero_routed_dpo_pool_is_explicit_not_a_pipeline_failure(tmp_path: Path) -> None:
    full = pd.DataFrame(
        [
            {
                "item_id": "dpo-1",
                "source_group": "group-1",
                "split": "dpo",
                "domain": "x",
                "prompt": "p",
                "response_a": "a",
                "response_b": "b",
            }
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "item_id": "dpo-1",
                "candidate_id": "C1_raw_judge",
                "route": "UNROUTED",
                "prediction": "A",
                "p_correct": 0.9,
            }
        ]
    )
    routed = pd.DataFrame(
        [
            {
                "item_id": "dpo-1",
                "candidate_id": "C5_routed_production",
                "route": "ESCALATE",
                "prediction": "A",
                "p_correct": 0.5,
            }
        ]
    )
    result = build_equal_budget_label_pools(raw, routed, full, tmp_path / "pools")
    assert result["status"] == "unavailable_zero_routed_labels"
    assert (tmp_path / "pools" / "raw_judge_equal_budget.jsonl").read_text() == ""
    assert (tmp_path / "pools" / "routed_equal_budget.jsonl").read_text() == ""
