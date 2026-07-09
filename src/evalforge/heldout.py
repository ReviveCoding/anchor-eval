from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import (
    dataset_fingerprint,
    load_pairs,
    validate_provenance_manifest,
    write_manifest,
)
from .evaluation import evaluate_prediction, write_json

BUNDLE_SCHEMA_VERSION = "2.0"
HELDOUT_RESULT_FILES = (
    "heldout_candidate_comparison.csv",
    "heldout_predictions.csv",
    "heldout_data_manifest.json",
)


def _hash_group(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _bundle_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _file_integrity(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required held-out artifact does not exist: {path}")
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "bytes": int(path.stat().st_size),
    }


def _load_bundle_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _bundle_manifest_path(path)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Candidate bundle and its integrity manifest must both exist.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Candidate bundle manifest must be valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Candidate bundle manifest root must be a JSON object.")
    actual_sha = sha256(path.read_bytes()).hexdigest()
    if actual_sha != manifest.get("bundle_sha256"):
        raise ValueError(
            "Candidate bundle checksum mismatch; refuse to load potentially stale or modified artifact."
        )
    candidate_ids = manifest.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or not all(isinstance(value, str) and value for value in candidate_ids)
    ):
        raise ValueError("Candidate bundle manifest must contain a non-empty candidate_ids list.")
    if candidate_ids != sorted(set(candidate_ids)):
        raise ValueError("Candidate bundle manifest candidate_ids must be sorted and unique.")
    return manifest


def save_candidate_bundle(
    path: str | Path,
    candidates: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    public_path: str | None = None,
) -> dict[str, Any]:
    """Save candidates with an integrity-checked metadata sidecar.

    The sidecar records protected source-group hashes so final-only evaluation can block accidental reuse of
    a group that was visible to model fitting, calibration, routing, or prior internal evaluation.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": BUNDLE_SCHEMA_VERSION, "candidates": candidates}, destination)
    metadata_copy = dict(metadata or {})
    protected = metadata_copy.pop("protected_source_groups", [])
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_path": public_path or str(destination),
        "bundle_sha256": sha256(destination.read_bytes()).hexdigest(),
        "candidate_ids": sorted(candidates),
        "protected_source_group_hashes": sorted({_hash_group(str(group)) for group in protected}),
        **metadata_copy,
    }
    _bundle_manifest_path(destination).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_candidate_bundle(
    path: str | Path, *, trust_candidate_bundle: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a local candidate bundle only after the caller explicitly trusts its origin.

    Joblib deserializes Python objects and must never be used for bundles obtained from
    an untrusted source. The checksum sidecar detects accidental modification but does
    not make a malicious bundle safe.
    """
    if not trust_candidate_bundle:
        raise PermissionError(
            "Refusing to deserialize a candidate bundle without explicit local trust. "
            "Pass trust_candidate_bundle=True only for a bundle you created or audited."
        )
    source = Path(path)
    manifest = _load_bundle_manifest(source)
    payload = joblib.load(source)
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate bundle schema version.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("Candidate bundle does not contain a candidate mapping.")
    if sorted(candidates) != manifest["candidate_ids"]:
        raise ValueError("Candidate bundle payload IDs do not match its manifest.")
    return candidates, manifest


def _validate_heldout_input_contract(
    input_path: str | Path,
    source_type: str,
    provenance_verified: bool,
    provenance_manifest_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_pairs(input_path)
    if set(frame["split"].astype(str)) != {"final"}:
        raise ValueError("Held-out evaluation input must contain only final split rows.")
    if frame["raw_judge_status"].astype(str).ne("local_scored").any():
        raise ValueError(
            "Held-out evaluator comparison requires raw_judge_status=local_scored for every row."
        )
    if source_type != "external_benchmark" or not provenance_verified:
        raise ValueError("Held-out evaluation requires verified external benchmark provenance.")
    if provenance_manifest_path is None:
        raise ValueError("Held-out evaluation requires an auditable provenance manifest path.")
    provenance_record = validate_provenance_manifest(
        provenance_manifest_path,
        input_path,
        {"final"},
    )
    if provenance_record["evidence_class"] != "external_benchmark":
        raise ValueError(
            "Held-out release-path evaluation requires provenance evidence_class=external_benchmark."
        )
    return frame, provenance_record


def _validate_source_group_overlap(frame: pd.DataFrame, bundle_manifest: dict[str, Any]) -> None:
    protected_hashes = set(bundle_manifest.get("protected_source_group_hashes", []))
    overlaps = sorted(
        {
            group
            for group in frame["source_group"].astype(str).unique()
            if _hash_group(group) in protected_hashes
        }
    )
    if overlaps:
        raise ValueError(f"Held-out source_group overlap with candidate bundle: {overlaps[:5]}")


def evaluate_heldout(
    bundle_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    source_type: str = "external_benchmark",
    provenance_verified: bool = True,
    provenance_manifest_path: str | Path | None = None,
    trust_candidate_bundle: bool = False,
) -> Path:
    """Score a separately scored final-only artifact without tuning or source-group reuse."""
    candidates, bundle_manifest = load_candidate_bundle(
        bundle_path, trust_candidate_bundle=trust_candidate_bundle
    )
    frame, provenance_record = _validate_heldout_input_contract(
        input_path,
        source_type,
        provenance_verified,
        provenance_manifest_path,
    )
    _validate_source_group_overlap(frame, bundle_manifest)

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Held-out output directory must be absent or empty.")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for candidate_id, candidate in candidates.items():
        prediction = candidate.predict(frame)
        metrics, _, enriched = evaluate_prediction(frame, prediction)
        rows.append({"candidate_id": candidate_id, **metrics, **prediction.metadata})
        predictions.append(enriched)

    comparison_path = destination / "heldout_candidate_comparison.csv"
    predictions_path = destination / "heldout_predictions.csv"
    data_manifest_path = destination / "heldout_data_manifest.json"
    comparison = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    comparison.to_csv(comparison_path, index=False)
    pd.concat(predictions, ignore_index=True).to_csv(predictions_path, index=False)
    data_manifest = write_manifest(
        data_manifest_path,
        input_path,
        frame,
        source_type=source_type,
        provenance_verified=provenance_verified,
    )
    output_files = {name: _file_integrity(destination / name) for name in HELDOUT_RESULT_FILES}
    write_json(
        destination / "heldout_manifest.json",
        {
            "schema_version": "1.0",
            "bundle_path": str(bundle_path),
            "bundle_sha256": bundle_manifest["bundle_sha256"],
            "candidate_ids": bundle_manifest["candidate_ids"],
            "input_path": str(input_path),
            "input_fingerprint_sha256": dataset_fingerprint(input_path),
            "rows": int(len(frame)),
            "source_group_overlap_count": 0,
            "data_manifest": data_manifest,
            "provenance_manifest": provenance_record,
            "output_files": output_files,
            "claim_boundary": (
                "Held-out metrics only. A superiority claim still requires immutable benchmark documentation, "
                "multi-seed local-judge evidence, and all release gates."
            ),
        },
    )
    return destination


def validate_completed_heldout_evaluation(
    bundle_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    source_type: str = "external_benchmark",
    provenance_verified: bool = True,
    provenance_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Fail closed before reusing a previously published held-out evaluation.

    Validation does not deserialize the bundle. It verifies the current bundle sidecar,
    scored input, provenance record, output artifact checksums, and expected prediction
    inventory. A changed bundle, benchmark CSV, or output CSV therefore cannot be reused
    merely because the output directory still exists.
    """
    bundle = Path(bundle_path)
    destination = Path(output_dir)
    bundle_manifest = _load_bundle_manifest(bundle)
    frame, provenance_record = _validate_heldout_input_contract(
        input_path,
        source_type,
        provenance_verified,
        provenance_manifest_path,
    )
    _validate_source_group_overlap(frame, bundle_manifest)

    manifest_path = destination / "heldout_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Held-out result reuse requires heldout_manifest.json.")
    try:
        result_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Held-out result manifest must be valid JSON.") from exc
    if not isinstance(result_manifest, dict):
        raise ValueError("Held-out result manifest root must be a JSON object.")

    if result_manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported held-out result manifest schema version.")
    if result_manifest.get("bundle_sha256") != bundle_manifest["bundle_sha256"]:
        raise ValueError(
            "Held-out result bundle checksum does not match the current candidate bundle."
        )
    if result_manifest.get("input_fingerprint_sha256") != dataset_fingerprint(input_path):
        raise ValueError(
            "Held-out result input fingerprint does not match the current scored input."
        )
    if result_manifest.get("rows") != int(len(frame)):
        raise ValueError("Held-out result row count does not match the current scored input.")
    if result_manifest.get("candidate_ids") != bundle_manifest["candidate_ids"]:
        raise ValueError("Held-out result candidate IDs do not match the current candidate bundle.")
    if result_manifest.get("source_group_overlap_count") != 0:
        raise ValueError("Held-out result manifest reports source-group overlap.")

    expected_provenance_fingerprint = provenance_record["dataset_fingerprint_sha256"]
    recorded_provenance = result_manifest.get("provenance_manifest")
    if not isinstance(recorded_provenance, dict) or (
        recorded_provenance.get("dataset_fingerprint_sha256") != expected_provenance_fingerprint
    ):
        raise ValueError(
            "Held-out result provenance does not match the current benchmark manifest."
        )

    artifacts = result_manifest.get("output_files")
    if not isinstance(artifacts, dict) or set(artifacts) != set(HELDOUT_RESULT_FILES):
        raise ValueError("Held-out result manifest has an invalid output artifact inventory.")
    for name in HELDOUT_RESULT_FILES:
        expected_integrity = artifacts[name]
        if not isinstance(expected_integrity, dict):
            raise ValueError(f"Held-out artifact metadata is invalid for {name}.")
        observed_integrity = _file_integrity(destination / name)
        if observed_integrity != expected_integrity:
            raise ValueError(f"Held-out artifact checksum or size mismatch: {name}")

    comparison = pd.read_csv(destination / "heldout_candidate_comparison.csv")
    predictions = pd.read_csv(destination / "heldout_predictions.csv")
    expected_candidates = bundle_manifest["candidate_ids"]
    comparison_candidates = comparison.get("candidate_id")
    prediction_candidates = predictions.get("candidate_id")
    if comparison_candidates is None or prediction_candidates is None:
        raise ValueError("Held-out output is missing candidate_id columns.")
    if sorted(comparison_candidates.astype(str).tolist()) != expected_candidates:
        raise ValueError("Held-out comparison candidate IDs do not match the current bundle.")
    prediction_counts = predictions["candidate_id"].astype(str).value_counts().to_dict()
    if set(prediction_counts) != set(expected_candidates) or any(
        int(prediction_counts[candidate_id]) != len(frame) for candidate_id in expected_candidates
    ):
        raise ValueError(
            "Held-out prediction inventory does not contain one row per item and candidate."
        )

    return {
        "status": "VALID",
        "rows": int(len(frame)),
        "candidate_ids": expected_candidates,
        "bundle_sha256": bundle_manifest["bundle_sha256"],
        "input_fingerprint_sha256": dataset_fingerprint(input_path),
        "output_dir": str(destination),
    }
