from pathlib import Path

from evalforge.anchors import anchor_coverage, select_adaptive_anchors
from evalforge.data import build_synthetic_fixture, load_pairs, split_calibration_groups


def test_adaptive_anchors_are_source_group_unique_and_have_continuous_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.csv"
    build_synthetic_fixture(path, groups_per_domain=32)
    frame = load_pairs(path)
    calibration = frame.loc[frame["split"] == "calibration"]
    anchor_fit, _ = split_calibration_groups(calibration)
    selected = select_adaptive_anchors(anchor_fit, budget=min(12, len(anchor_fit)))
    assert len(selected.item_ids) > 0
    assert selected.diagnostics["source_group"].nunique() == len(selected.diagnostics)
    coverage = anchor_coverage(frame.loc[frame["split"] == "final"], selected.diagnostics)
    assert ((coverage >= 0.0) & (coverage <= 1.0)).all()
    assert len(set(coverage.round(6))) > 1
