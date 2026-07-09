import pytest

from evalforge.config import apply_overrides, validate_config


def _config() -> dict:
    return {
        "run": {"run_name": "x", "seed": 1, "output_root": "outputs"},
        "data": {
            "fixture_path": "outputs/fixtures/x.csv",
            "source_type": "synthetic_fixture",
            "provenance_verified": False,
            "train_split": "train",
            "calibration_split": "calibration",
            "policy_split": "policy",
            "dpo_split": "dpo",
            "final_split": "final",
        },
        "anchors": {"static_budget": 2, "adaptive_budget": 2},
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


def test_config_rejects_duplicate_split_roles() -> None:
    config = _config()
    config["data"]["policy_split"] = "calibration"
    with pytest.raises(ValueError, match="must be distinct"):
        validate_config(config)


def test_config_requires_explicit_source_type() -> None:
    config = _config()
    del config["data"]["source_type"]
    with pytest.raises(ValueError, match="data section is missing"):
        validate_config(config)


def test_external_config_requires_manifest_and_strict_override() -> None:
    config = _config()
    config["data"]["source_type"] = "external_benchmark"
    config["data"]["provenance_verified"] = True
    with pytest.raises(ValueError, match="provenance_manifest_path"):
        validate_config(config)
    config["data"]["provenance_manifest_path"] = "data/provenance.json"
    updated = apply_overrides(config, ["run.seed=7", "report.save_html=false"])
    assert updated["run"]["seed"] == 7
    assert updated["report"]["save_html"] is False
    with pytest.raises(KeyError, match="does not exist"):
        apply_overrides(config, ["run.no_such_key=1"])
