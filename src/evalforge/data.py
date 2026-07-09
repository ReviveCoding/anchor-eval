from __future__ import annotations

import json
import math
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    PREFERENCE_LABELS,
    RAW_JUDGE_STATUSES,
    REQUIRED_COLUMNS,
    SOURCE_TYPES,
    SPLITS,
)

TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f", ""}


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _coerce_bool(series: pd.Series, column: str) -> pd.Series:
    """Parse bool-like values safely instead of treating non-empty strings as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = series.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{column} contains unsupported boolean values: {examples}")
    return normalized.isin(TRUE_VALUES)


def _controlled_response(item_id: str, side: str, quality: float, domain: str) -> str:
    """Synthetic text for deterministic smoke validation, never benchmark evidence."""
    evidence_terms = [
        "cites a relevant fact",
        "shows a checkable step",
        "respects the requested format",
    ]
    risk_terms = [
        "uses an unsupported leap",
        "omits a required condition",
        "contains a plausible but wrong step",
    ]
    if quality >= 0:
        main = evidence_terms[int(abs(quality) * 10) % len(evidence_terms)]
        extra = (
            risk_terms[int(abs(quality) * 7) % len(risk_terms)]
            if abs(quality) < 0.35
            else "keeps the conclusion scoped"
        )
    else:
        main = risk_terms[int(abs(quality) * 10) % len(risk_terms)]
        extra = (
            evidence_terms[int(abs(quality) * 7) % len(evidence_terms)]
            if abs(quality) < 0.35
            else "sounds confident"
        )
    return f"{domain} response {side} for {item_id}: {main}; {extra}."


def build_synthetic_fixture(
    output_path: str | Path,
    seed: int = 1729,
    groups_per_domain: int = 48,
) -> pd.DataFrame:
    """Create a deterministic group-disjoint fixture for controlled small-data validation."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    domains = ("financial_grounded_qa", "tool_trace", "instruction_following")
    difficulties = ("easy", "medium", "hard")
    split_cycle = ["train"] * 5 + ["calibration"] * 2 + ["policy"] * 2 + ["dpo"] * 2 + ["final"] * 2
    for domain in domains:
        for local_group in range(groups_per_domain):
            split = split_cycle[local_group % len(split_cycle)]
            source_group = f"{domain}_source_{local_group:03d}"
            for item_offset in range(3):
                difficulty = difficulties[(local_group + item_offset) % len(difficulties)]
                difficulty_value = {"easy": 0.35, "medium": 0.65, "hard": 1.00}[difficulty]
                latent_a = rng.normal(0.0, 1.0)
                latent_b = rng.normal(0.0, 1.0)
                gold_a = int(latent_a > latent_b)
                gold = "A" if gold_a else "B"
                length_ratio = float(np.exp(rng.normal(0.0, 0.65)))
                style_sensitivity = float(rng.uniform(0.0, 1.0))
                verifier_prob = 0.90 if domain != "instruction_following" else 0.76
                verifiable = bool(rng.random() < verifier_prob)
                base_margin = (latent_a - latent_b) / (0.72 + 0.60 * difficulty_value)
                verbosity_bias = 0.36 * math.log(max(length_ratio, 1e-6))
                style_bias = 0.20 * (style_sensitivity - 0.5)
                position_bias = 0.24
                raw_p_a = float(sigmoid(base_margin + verbosity_bias + style_bias + position_bias))
                # Probability assigned to B after presenting B in the A slot. Downstream converts it back to P(A).
                raw_p_a_swapped = float(
                    sigmoid(-base_margin - verbosity_bias + style_bias + position_bias)
                )
                raw_p_a_style = float(
                    sigmoid(base_margin + verbosity_bias + 0.42 * style_sensitivity + position_bias)
                )
                # Style applied after swapping original B into the displayed A slot.  This completes
                # the two-by-two order/style orbit used by C4 and C5 diagnostics.
                raw_p_a_style_swapped = float(
                    sigmoid(
                        -base_margin - verbosity_bias + 0.42 * style_sensitivity + position_bias
                    )
                )
                rubric_entropy = float(
                    np.clip(
                        0.16
                        + 0.43 * difficulty_value
                        + 0.28 * abs(length_ratio - 1.0) / (length_ratio + 1.0)
                        + rng.normal(0.0, 0.06),
                        0.01,
                        0.99,
                    )
                )
                family_a = "model_x" if rng.random() < 0.5 else "model_y"
                family_b = "model_y" if family_a == "model_x" else "model_x"
                item_id = f"{source_group}_pair_{item_offset:02d}"
                rows.append(
                    {
                        "item_id": item_id,
                        "source_group": source_group,
                        "split": split,
                        "domain": domain,
                        "difficulty": difficulty,
                        "prompt": (
                            f"Controlled {domain} prompt {item_id}. Choose the response with better evidence "
                            "and instruction adherence."
                        ),
                        "response_a": _controlled_response(item_id, "A", latent_a, domain),
                        "response_b": _controlled_response(item_id, "B", latent_b, domain),
                        "gold_preference": gold,
                        "deterministic_verifiable": verifiable,
                        "verifier_preference": gold if verifiable else "",
                        "raw_judge_p_a": raw_p_a,
                        "raw_judge_p_a_swapped": raw_p_a_swapped,
                        "raw_judge_p_a_style": raw_p_a_style,
                        "raw_judge_p_a_style_swapped": raw_p_a_style_swapped,
                        "raw_judge_status": "synthetic_fixture",
                        "raw_judge_model_id": "controlled_generator_v1",
                        "rubric_entropy": rubric_entropy,
                        "response_length_ratio": length_ratio,
                        "style_sensitivity": style_sensitivity,
                        "model_family_a": family_a,
                        "model_family_b": family_b,
                    }
                )
    frame = pd.DataFrame(rows)
    validate_pairs(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_pairs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame["deterministic_verifiable"] = _coerce_bool(
        frame["deterministic_verifiable"], "deterministic_verifiable"
    )
    validate_pairs(frame)
    return frame


def validate_pairs(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    normalized_bool = _coerce_bool(frame["deterministic_verifiable"], "deterministic_verifiable")
    if frame["item_id"].astype(str).str.strip().eq("").any():
        raise ValueError("item_id must be non-empty.")
    if frame["source_group"].astype(str).str.strip().eq("").any():
        raise ValueError("source_group must be non-empty.")
    if frame["item_id"].duplicated().any():
        raise ValueError("item_id must be unique.")
    if not set(frame["gold_preference"].astype(str)).issubset(PREFERENCE_LABELS):
        raise ValueError("gold_preference must be A or B.")
    if not set(frame["split"].astype(str)).issubset(SPLITS):
        raise ValueError("split contains unsupported values.")
    if not set(frame["raw_judge_status"].astype(str)).issubset(RAW_JUDGE_STATUSES):
        raise ValueError(f"raw_judge_status must be one of {sorted(RAW_JUDGE_STATUSES)}.")
    split_per_source = frame.groupby("source_group")["split"].nunique()
    if (split_per_source > 1).any():
        leaked = split_per_source[split_per_source > 1].index.tolist()[:5]
        raise ValueError(f"Source-group leakage across splits: {leaked}")
    verifier_rows = frame.loc[normalized_bool]
    if not bool(verifier_rows["verifier_preference"].astype(str).isin(PREFERENCE_LABELS).all()):
        raise ValueError("Verifiable rows require verifier_preference A or B.")
    local_scored = frame["raw_judge_status"].astype(str).eq("local_scored")
    if (
        local_scored.any()
        and frame.loc[local_scored, "raw_judge_model_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("local_scored rows require raw_judge_model_id.")
    for column in ("prompt", "response_a", "response_b"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be non-empty.")
    probability_columns = (
        "raw_judge_p_a",
        "raw_judge_p_a_swapped",
        "raw_judge_p_a_style",
        "raw_judge_p_a_style_swapped",
    )
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values <= 0.0) | (values >= 1.0)).any():
            raise ValueError(f"{column} must contain finite values strictly between 0 and 1.")
    bounded_columns = {"rubric_entropy": (0.0, 1.0), "style_sensitivity": (0.0, 1.0)}
    for column, (lower, upper) in bounded_columns.items():
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < lower) | (values > upper)).any():
            raise ValueError(f"{column} must contain finite values in [{lower}, {upper}].")
    length_ratio = pd.to_numeric(frame["response_length_ratio"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(length_ratio).all() or (length_ratio <= 0.0).any():
        raise ValueError("response_length_ratio must contain finite positive values.")


def dataset_fingerprint(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def source_type_is_synthetic(source_type: str, frame: pd.DataFrame) -> bool:
    return (
        source_type in {"synthetic_fixture", "public_derived_controlled"}
        or frame["raw_judge_status"]
        .astype(str)
        .isin({"synthetic_fixture", "simulated_controlled"})
        .all()
    )


def write_manifest(
    output_path: str | Path,
    data_path: str | Path,
    frame: pd.DataFrame,
    source_type: str,
    provenance_verified: bool,
) -> dict[str, object]:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}.")
    synthetic = source_type_is_synthetic(source_type, frame)
    payload: dict[str, object] = {
        "dataset_path": str(data_path),
        "dataset_fingerprint_sha256": dataset_fingerprint(data_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "domain_counts": frame["domain"].value_counts().sort_index().to_dict(),
        "source_group_count": int(frame["source_group"].nunique()),
        "source_type": source_type,
        "provenance_verified": bool(provenance_verified),
        "raw_judge_status_counts": frame["raw_judge_status"].value_counts().sort_index().to_dict(),
        "synthetic_or_controlled_evidence": bool(synthetic),
        "fixture_notice": (
            "Synthetic or controlled derived data. It can validate interfaces and controlled behavior but never "
            "qualifies a release or supports external-quality claims."
            if synthetic
            else "External or user-supplied data. Provenance must be verified before a release claim."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = frame.loc[frame["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Split '{split}' is empty.")
    return selected


def split_calibration_groups(
    calibration: pd.DataFrame, seed: int = 1729, anchor_fraction: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create source-group-disjoint anchor-fit and VCC-fit subsets from calibration."""
    if not 0.25 <= anchor_fraction <= 0.75:
        raise ValueError("anchor_fraction must be between 0.25 and 0.75.")
    groups = np.asarray(sorted(calibration["source_group"].astype(str).unique()))
    if len(groups) < 4:
        raise ValueError(
            "Calibration requires at least four source groups for anchor/VCC separation."
        )
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    cut = max(2, min(len(shuffled) - 2, int(round(len(shuffled) * anchor_fraction))))
    anchor_groups = set(shuffled[:cut])
    anchor_fit = calibration.loc[calibration["source_group"].astype(str).isin(anchor_groups)].copy()
    vcc_fit = calibration.loc[~calibration["source_group"].astype(str).isin(anchor_groups)].copy()
    if set(anchor_fit["source_group"]).intersection(set(vcc_fit["source_group"])):
        raise AssertionError("Anchor/VCC calibration group leakage.")
    return anchor_fit, vcc_fit


def build_finqa_pair_rows(
    records: Iterable[dict], max_examples: int = 250, seed: int = 1729
) -> pd.DataFrame:
    """Turn local FinQA-style records into controlled verifier-backed derived pairs.

    The derived corpus exercises deterministic verifier and pipeline paths only. Its simulated raw judge channels
    cannot support claims about local LLM judge quality or benchmark performance.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    split_cycle = ["train", "train", "train", "calibration", "policy", "dpo", "final"]
    for index, record in enumerate(records):
        if index >= max_examples:
            break
        question = str(record.get("qa", {}).get("question", record.get("question", "")))
        answer = str(record.get("qa", {}).get("answer", record.get("answer", ""))).strip()
        program = record.get("qa", {}).get("program", record.get("program", []))
        source_id = str(record.get("id", f"finqa_{index:05d}"))
        if not answer or not question:
            continue
        wrong_answer = f"{answer}0" if answer not in {"0", "0.0"} else "1"
        correct_a = bool(rng.integers(0, 2))
        correct_response = f"Answer: {answer}. Program: {program}"
        corrupt_response = f"Answer: {wrong_answer}. Program: corrupted"
        response_a = correct_response if correct_a else corrupt_response
        response_b = corrupt_response if correct_a else correct_response
        raw_p_a = 0.63 if correct_a else 0.43
        raw_p_a_swapped = 0.47 if correct_a else 0.58
        rows.append(
            {
                "item_id": f"{source_id}_derived_pair",
                "source_group": source_id,
                "split": split_cycle[index % len(split_cycle)],
                "domain": "financial_grounded_qa",
                "difficulty": "medium",
                "prompt": question,
                "response_a": response_a,
                "response_b": response_b,
                "gold_preference": "A" if correct_a else "B",
                "deterministic_verifiable": True,
                "verifier_preference": "A" if correct_a else "B",
                "raw_judge_p_a": raw_p_a,
                "raw_judge_p_a_swapped": raw_p_a_swapped,
                "raw_judge_p_a_style": raw_p_a,
                "raw_judge_p_a_style_swapped": raw_p_a_swapped,
                "raw_judge_status": "simulated_controlled",
                "raw_judge_model_id": "controlled_finqa_derived_v1",
                "rubric_entropy": 0.35,
                "response_length_ratio": 1.1,
                "style_sensitivity": 0.1,
                "model_family_a": "derived_a",
                "model_family_b": "derived_b",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        validate_pairs(result)
    return result


PROVENANCE_REQUIRED_FIELDS = {
    "dataset_fingerprint_sha256",
    "source_name",
    "source_uri",
    "license",
    "revision_or_commit",
    "retrieved_at_utc",
    "permitted_splits",
    "evidence_class",
}


def validate_provenance_manifest(
    manifest_path: str | Path,
    data_path: str | Path,
    expected_splits: set[str] | None = None,
) -> dict[str, object]:
    """Validate an auditable provenance record bound to one exact dataset file.

    This does not establish legal provenance by itself. It prevents an external-evidence
    claim from relying on a bare boolean or a manifest belonging to another dataset revision.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Provenance manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Provenance manifest must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Provenance manifest root must be a JSON object.")
    missing = PROVENANCE_REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"Provenance manifest is missing: {sorted(missing)}")
    actual_fingerprint = dataset_fingerprint(data_path)
    if str(payload["dataset_fingerprint_sha256"]) != actual_fingerprint:
        raise ValueError("Provenance manifest fingerprint does not match the supplied dataset.")
    if not isinstance(payload["permitted_splits"], list) or not all(
        isinstance(value, str) and value in SPLITS for value in payload["permitted_splits"]
    ):
        raise ValueError("provenance permitted_splits must be a list of supported split names.")
    if str(payload["evidence_class"]) not in {
        "external_benchmark",
        "public_derived_controlled",
        "synthetic_fixture",
        "test_only",
    }:
        raise ValueError("provenance evidence_class is not recognized.")
    if expected_splits is not None and not expected_splits.issubset(
        set(payload["permitted_splits"])
    ):
        raise ValueError(
            "Provenance manifest does not permit all requested splits: "
            f"expected={sorted(expected_splits)}, permitted={sorted(payload['permitted_splits'])}."
        )
    for field in ("source_name", "source_uri", "license", "revision_or_commit", "retrieved_at_utc"):
        if not str(payload[field]).strip():
            raise ValueError(f"Provenance manifest field '{field}' must be non-empty.")
    return payload
