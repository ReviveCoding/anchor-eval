import numpy as np
import pandas as pd
import pytest

from evalforge.candidates import CandidatePrediction
from evalforge.evaluation import operating_profile_scenarios


@pytest.mark.parametrize("capacities", [(), (0.0, 0.25, 0.5, 1.0), (1.0, 0.25, 0.25)])
def test_operating_profiles_preserve_capacity_order_and_guardrails(capacities):
    frame = pd.DataFrame(
        {
            "gold_preference": ["A"] * 4,
            "deterministic_verifiable": [False, True, False, False],
            "verifier_preference": ["A", "B", "A", "A"],
            "raw_judge_p_a": [0.8, 0.8, 0.8, 0.8],
            "raw_judge_p_a_swapped": [0.2, 0.2, 0.2, 0.2],
            "raw_judge_p_a_style": [0.8, 0.8, 0.2, 0.8],
            "raw_judge_p_a_style_swapped": [0.2, 0.2, 0.8, 0.2],
        }
    )
    prediction = CandidatePrediction(
        candidate_id="fixture",
        p_a=np.array([0.8, 0.8, 0.8, 0.2]),
        p_correct=np.array([0.9, 0.95, 0.99, 0.7]),
        route=np.array(["UNROUTED"] * 4),
        reason_code=np.array([""] * 4),
        latency_ms=0.0,
        metadata={},
    )
    profiles = operating_profile_scenarios(frame, prediction, 0.0, 0.2, 0.2, capacities)
    if not capacities:
        assert profiles.empty
        return
    assert profiles["profile"].tolist() == [
        f"review_capacity_{int(round(c * 100))}pct" for c in capacities
    ]
    for capacity, row in zip(capacities, profiles.to_dict("records"), strict=True):
        assert row["feasible"] == (capacity >= 0.75)
        if capacity >= 0.75:
            assert row["coverage"] == 0.25
            assert row["threshold"] == 0.9
            assert row["wrong_auto_label_rate"] == 0.0
