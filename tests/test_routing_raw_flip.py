from pathlib import Path

from evalforge.anchors import select_adaptive_anchors
from evalforge.candidates import AdaptiveAnchorVCC, LearnedEvaluator, ProductionRouter
from evalforge.data import build_synthetic_fixture, load_pairs, split_calibration_groups


def test_raw_transform_flip_routes_to_escalate_even_if_gap_is_small(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    build_synthetic_fixture(path, groups_per_domain=48)
    frame = load_pairs(path)
    train = frame.loc[frame["split"] == "train"]
    calibration = frame.loc[frame["split"] == "calibration"]
    policy = frame.loc[frame["split"] == "policy"]
    final = frame.loc[frame["split"] == "final"].copy()
    anchor_fit, vcc_fit = split_calibration_groups(calibration)
    learned = LearnedEvaluator().fit(train, calibration)
    adaptive = select_adaptive_anchors(anchor_fit, budget=min(12, len(anchor_fit)))
    model = AdaptiveAnchorVCC(adaptive, learned).fit(train, anchor_fit, vcc_fit)
    router = ProductionRouter(model, threshold=0.5, max_escalation_rate=0.95).fit(policy)
    final.loc[final.index[0], "raw_judge_p_a"] = 0.49
    final.loc[final.index[0], "raw_judge_p_a_swapped"] = (
        0.49  # swapped-back score is 0.51: a raw flip, tiny gap.
    )
    prediction = router.predict(final)
    assert prediction.route[0] == "ESCALATE"
    assert prediction.reason_code[0] == "order_sensitive"
