from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .contracts import SOURCE_TYPES, SPLITS

REQUIRED_TOP_LEVEL = {"run", "data", "anchors", "routing", "report"}
REQUIRED_DATA = {
    "fixture_path",
    "source_type",
    "provenance_verified",
    "train_split",
    "calibration_split",
    "policy_split",
    "dpo_split",
    "final_split",
}
REQUIRED_ROUTING = {
    "auto_accept_min_p_correct",
    "max_wrong_auto_label_rate",
    "max_escalation_rate",
    "max_order_delta",
    "max_style_delta",
    "max_order_flip_rate",
    "max_style_flip_rate",
    "max_ece",
    "min_auto_accept_coverage",
}


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP_LEVEL.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing top-level sections: {sorted(missing)}")
    run = _require_mapping(config, "run")
    data = _require_mapping(config, "data")
    anchors = _require_mapping(config, "anchors")
    routing = _require_mapping(config, "routing")
    report = _require_mapping(config, "report")
    missing_data = REQUIRED_DATA.difference(data)
    if missing_data:
        raise ValueError(f"Configuration data section is missing: {sorted(missing_data)}")
    missing_routing = REQUIRED_ROUTING.difference(routing)
    if missing_routing:
        raise ValueError(f"Configuration routing section is missing: {sorted(missing_routing)}")
    for key in ("run_name", "seed", "output_root"):
        if key not in run:
            raise ValueError(f"Configuration run section is missing '{key}'.")
    for key in ("static_budget", "adaptive_budget"):
        if key not in anchors or int(anchors[key]) <= 0:
            raise ValueError(f"Configuration anchors.{key} must be a positive integer.")
    split_values = [
        str(data["train_split"]),
        str(data["calibration_split"]),
        str(data["policy_split"]),
        str(data["dpo_split"]),
        str(data["final_split"]),
    ]
    if any(value not in SPLITS for value in split_values):
        raise ValueError(f"Configured split roles must be members of {sorted(SPLITS)}.")
    if len(set(split_values)) != len(split_values):
        raise ValueError("train, calibration, policy, dpo, and final split roles must be distinct.")
    source_type = str(data["source_type"])
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"data.source_type must be one of {sorted(SOURCE_TYPES)}.")
    if not isinstance(data["provenance_verified"], bool):
        raise ValueError("data.provenance_verified must be a boolean.")
    if "rebuild_synthetic_fixture_on_schema_error" in data and not isinstance(
        data["rebuild_synthetic_fixture_on_schema_error"], bool
    ):
        raise ValueError("data.rebuild_synthetic_fixture_on_schema_error must be a boolean.")
    if source_type == "external_benchmark":
        manifest_path = data.get("provenance_manifest_path")
        if (
            not data["provenance_verified"]
            or not isinstance(manifest_path, str)
            or not manifest_path.strip()
        ):
            raise ValueError(
                "external_benchmark requires provenance_verified=true and data.provenance_manifest_path."
            )
    for key in REQUIRED_ROUTING:
        value = float(routing[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"routing.{key} must be in [0, 1].")
    if int(run["seed"]) < 0:
        raise ValueError("run.seed must be non-negative.")
    if not isinstance(report.get("save_svg", True), bool) or not isinstance(
        report.get("save_html", True), bool
    ):
        raise ValueError("report.save_svg and report.save_html must be booleans.")
    return config


def apply_overrides(config: dict[str, Any], assignments: list[str] | None = None) -> dict[str, Any]:
    """Apply strict ``section.key=value`` scalar overrides without changing the config schema."""
    if not assignments:
        return validate_config(config)
    updated = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    assert isinstance(updated, dict)
    for assignment in assignments:
        if "=" not in assignment or "." not in assignment.split("=", 1)[0]:
            raise ValueError(
                "Overrides must use existing dotted keys, for example run.output_root=outputs/run."
            )
        dotted_key, raw_value = assignment.split("=", 1)
        section, key = dotted_key.split(".", 1)
        if "." in key:
            raise ValueError("Only one-level section.key overrides are supported.")
        if (
            section not in updated
            or not isinstance(updated[section], dict)
            or key not in updated[section]
        ):
            raise KeyError(f"Override key does not exist in the schema: {dotted_key}")
        parsed_value = yaml.safe_load(raw_value)
        updated[section][key] = parsed_value
    return validate_config(updated)


def write_packaged_smoke_config(output: str | Path, overwrite: bool = False) -> Path:
    """Export the canonical packaged smoke configuration for installed-package validation.

    The resource is packaged with the wheel and source distribution so a clean installation can
    run the CPU smoke path without borrowing a YAML file from the source checkout.
    """
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing configuration: {destination}. Pass overwrite=True explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    resource = resources.files("evalforge.resources").joinpath("smoke.yaml")
    destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def packaged_smoke_config_text() -> str:
    """Return the packaged canonical smoke config for parity checks and diagnostics."""
    return resources.files("evalforge.resources").joinpath("smoke.yaml").read_text(encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return validate_config(config)
