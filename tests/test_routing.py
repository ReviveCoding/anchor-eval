from pathlib import Path

from evalforge.anchors import select_adaptive_anchors
from evalforge.candidates import AdaptiveAnchorVCC, LearnedEvaluator, ProductionRouter
from evalforge.data import build_synthetic_fixture, load_pairs, split_calibration_groups


def test_verifier_conflict_is_not_auto_accepted_and_policy_threshold_is_fitted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.csv"
    build_synthetic_fixture(path, groups_per_domain=48)
    frame = load_pairs(path)
    train = frame.loc[frame["split"] == "train"]
    calibration = frame.loc[frame["split"] == "calibration"]
    anchor_fit, vcc_fit = split_calibration_groups(calibration)
    policy = frame.loc[frame["split"] == "policy"]
    final = frame.loc[frame["split"] == "final"]
    learned = LearnedEvaluator().fit(train, calibration)
    adaptive = select_adaptive_anchors(anchor_fit, budget=min(12, len(anchor_fit)))
    model = AdaptiveAnchorVCC(adaptive, learned).fit(train, anchor_fit, vcc_fit)
    routed = ProductionRouter(model, threshold=0.51, max_escalation_rate=0.95).fit(policy)
    pred = routed.predict(final)
    labels = pred.p_a >= 0.5
    verifier_a = final["verifier_preference"].to_numpy() == "A"
    conflict = final["deterministic_verifiable"].to_numpy() & (labels != verifier_a)
    assert not ((pred.route == "AUTO_ACCEPT") & conflict).any()
    assert routed.policy_diagnostics["policy_threshold_fitted"] is True
