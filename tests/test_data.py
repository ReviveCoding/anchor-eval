from pathlib import Path

import pytest

from evalforge.data import (
    build_synthetic_fixture,
    load_pairs,
    split_calibration_groups,
    validate_pairs,
)


def test_fixture_has_source_group_disjoint_splits_and_dpo_pool(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    loaded = load_pairs(path)
    assert len(loaded) == len(frame)
    assert loaded.groupby("source_group")["split"].nunique().max() == 1
    assert "dpo" in set(loaded["split"])
    anchor, vcc = split_calibration_groups(loaded.loc[loaded["split"] == "calibration"], seed=1729)
    assert not set(anchor["source_group"]).intersection(set(vcc["source_group"]))


def test_validator_rejects_source_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    frame = build_synthetic_fixture(path, groups_per_domain=16)
    first_group = frame.loc[0, "source_group"]
    idx = frame.index[frame["source_group"] == first_group].tolist()
    frame.loc[idx[0], "split"] = "train"
    frame.loc[idx[1], "split"] = "final"
    with pytest.raises(ValueError, match="leakage"):
        validate_pairs(frame)
