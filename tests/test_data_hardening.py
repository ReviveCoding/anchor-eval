from pathlib import Path

import pytest

from evalforge.data import build_synthetic_fixture, load_pairs, validate_pairs, write_manifest


def test_bool_parser_treats_false_string_as_false(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_pairs.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    frame["deterministic_verifiable"] = frame["deterministic_verifiable"].astype(object)
    frame.loc[0, "deterministic_verifiable"] = "False"
    frame.loc[0, "verifier_preference"] = ""
    frame.to_csv(path, index=False)
    loaded = load_pairs(path)
    assert bool(loaded.loc[0, "deterministic_verifiable"]) is False


def test_validator_rejects_invalid_boolean_and_missing_local_model_id(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_pairs.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    bad_bool = frame.copy()
    bad_bool["deterministic_verifiable"] = bad_bool["deterministic_verifiable"].astype(object)
    bad_bool.loc[0, "deterministic_verifiable"] = "sometimes"
    with pytest.raises(ValueError, match="unsupported boolean"):
        validate_pairs(bad_bool)
    missing_model = frame.copy()
    missing_model.loc[0, "raw_judge_status"] = "local_scored"
    missing_model.loc[0, "raw_judge_model_id"] = ""
    with pytest.raises(ValueError, match="raw_judge_model_id"):
        validate_pairs(missing_model)


def test_manifest_uses_explicit_source_type_not_filename_heuristic(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_named_like_synthetic.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    manifest = write_manifest(
        tmp_path / "manifest.json",
        path,
        frame,
        source_type="external_benchmark",
        provenance_verified=True,
    )
    assert manifest["source_type"] == "external_benchmark"
    assert manifest["synthetic_or_controlled_evidence"] is True
