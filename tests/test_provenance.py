from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalforge.data import (
    build_synthetic_fixture,
    dataset_fingerprint,
    validate_provenance_manifest,
)


def _manifest(data_path: Path, permitted_splits: list[str]) -> dict[str, object]:
    return {
        "dataset_fingerprint_sha256": dataset_fingerprint(data_path),
        "source_name": "test-only provenance fixture",
        "source_uri": "https://example.invalid/test-fixture",
        "license": "test-only",
        "revision_or_commit": "unit-test",
        "retrieved_at_utc": "2026-06-26T00:00:00Z",
        "permitted_splits": permitted_splits,
        "evidence_class": "external_benchmark",
    }


def test_provenance_manifest_is_bound_to_exact_dataset_and_splits(tmp_path: Path) -> None:
    data_path = tmp_path / "fixture.csv"
    build_synthetic_fixture(data_path, groups_per_domain=16)
    manifest_path = tmp_path / "provenance.json"
    manifest_path.write_text(json.dumps(_manifest(data_path, ["final"])), encoding="utf-8")
    validated = validate_provenance_manifest(manifest_path, data_path, {"final"})
    assert validated["source_name"] == "test-only provenance fixture"
    with pytest.raises(ValueError, match="does not permit"):
        validate_provenance_manifest(manifest_path, data_path, {"train"})
    payload = _manifest(data_path, ["final"])
    payload["dataset_fingerprint_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_provenance_manifest(manifest_path, data_path, {"final"})
