from pathlib import Path

import pytest

from evalforge.candidates import RawJudge
from evalforge.data import build_synthetic_fixture, dataset_fingerprint, load_pairs
from evalforge.heldout import evaluate_heldout, save_candidate_bundle


def _provenance_manifest(path: Path, dataset_path: Path) -> Path:
    payload = {
        "dataset_fingerprint_sha256": dataset_fingerprint(dataset_path),
        "source_name": "unit-test heldout fixture",
        "source_uri": "https://example.invalid/unit-test",
        "license": "test-only",
        "revision_or_commit": "unit-test",
        "retrieved_at_utc": "2026-06-26T00:00:00Z",
        "permitted_splits": ["final"],
        "evidence_class": "external_benchmark",
    }
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    return path


def test_heldout_rejects_unscored_and_source_overlap(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.csv"
    build_synthetic_fixture(fixture, groups_per_domain=16)
    frame = load_pairs(fixture)
    bundle = tmp_path / "bundle.joblib"
    save_candidate_bundle(
        bundle,
        {"C1_raw_judge": RawJudge()},
        metadata={"protected_source_groups": frame["source_group"].unique().tolist()},
    )
    final = frame.loc[frame["split"] == "final"].copy()
    final_path = tmp_path / "final.csv"
    final.to_csv(final_path, index=False)
    with pytest.raises(ValueError, match="local_scored"):
        evaluate_heldout(
            bundle,
            final_path,
            tmp_path / "heldout",
            provenance_verified=True,
            provenance_manifest_path=_provenance_manifest(tmp_path / "prov.json", final_path),
            trust_candidate_bundle=True,
        )
    final["raw_judge_status"] = "local_scored"
    final["raw_judge_model_id"] = "test-model"
    final.to_csv(final_path, index=False)
    with pytest.raises(ValueError, match="source_group overlap"):
        evaluate_heldout(
            bundle,
            final_path,
            tmp_path / "heldout",
            provenance_verified=True,
            provenance_manifest_path=_provenance_manifest(tmp_path / "prov.json", final_path),
            trust_candidate_bundle=True,
        )


def test_heldout_rejects_checksum_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.joblib"
    save_candidate_bundle(
        bundle, {"C1_raw_judge": RawJudge()}, metadata={"protected_source_groups": []}
    )
    bundle.write_bytes(bundle.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        from evalforge.heldout import load_candidate_bundle

        load_candidate_bundle(bundle, trust_candidate_bundle=True)


def test_bundle_loading_requires_explicit_local_trust_and_does_not_mutate_metadata(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.joblib"
    metadata = {"protected_source_groups": ["g1"], "note": "keep"}
    save_candidate_bundle(bundle, {"C1_raw_judge": RawJudge()}, metadata=metadata)
    assert metadata == {"protected_source_groups": ["g1"], "note": "keep"}
    from evalforge.heldout import load_candidate_bundle

    with pytest.raises(PermissionError, match="explicit local trust"):
        load_candidate_bundle(bundle)


def test_completed_heldout_reuse_rejects_tampered_artifacts(tmp_path: Path) -> None:
    from evalforge.heldout import validate_completed_heldout_evaluation

    fixture = tmp_path / "fixture.csv"
    build_synthetic_fixture(fixture, groups_per_domain=16)
    frame = load_pairs(fixture)
    final = frame.loc[frame["split"] == "final"].copy()
    final["raw_judge_status"] = "local_scored"
    final["raw_judge_model_id"] = "test-model"
    final_path = tmp_path / "external_final.csv"
    final.to_csv(final_path, index=False)

    bundle = tmp_path / "bundle.joblib"
    save_candidate_bundle(
        bundle, {"C1_raw_judge": RawJudge()}, metadata={"protected_source_groups": []}
    )
    provenance = _provenance_manifest(tmp_path / "provenance.json", final_path)
    output = tmp_path / "heldout"
    evaluate_heldout(
        bundle,
        final_path,
        output,
        provenance_verified=True,
        provenance_manifest_path=provenance,
        trust_candidate_bundle=True,
    )

    validated = validate_completed_heldout_evaluation(
        bundle,
        final_path,
        output,
        provenance_verified=True,
        provenance_manifest_path=provenance,
    )
    assert validated["status"] == "VALID"
    assert validated["candidate_ids"] == ["C1_raw_judge"]

    comparison = output / "heldout_candidate_comparison.csv"
    comparison.write_bytes(comparison.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="checksum or size mismatch"):
        validate_completed_heldout_evaluation(
            bundle,
            final_path,
            output,
            provenance_verified=True,
            provenance_manifest_path=provenance,
        )
