from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PREFERENCE_LABELS = {"A", "B"}
SPLITS = {"train", "calibration", "policy", "dpo", "final"}
ROUTES = {"AUTO_ACCEPT", "ESCALATE", "QUARANTINE", "UNROUTED"}
RAW_JUDGE_STATUSES = {"synthetic_fixture", "simulated_controlled", "local_scored", "unscored"}
SOURCE_TYPES = {
    "synthetic_fixture",
    "public_derived_controlled",
    "external_benchmark",
    "user_supplied",
}


@dataclass(frozen=True)
class ReleaseDecision:
    status: Literal["PASS", "CONDITIONAL_PASS", "BLOCK"]
    reasons: tuple[str, ...]


REQUIRED_COLUMNS = {
    "item_id",
    "source_group",
    "split",
    "domain",
    "difficulty",
    "prompt",
    "response_a",
    "response_b",
    "gold_preference",
    "deterministic_verifiable",
    "verifier_preference",
    "raw_judge_p_a",
    "raw_judge_p_a_swapped",
    "raw_judge_p_a_style",
    "raw_judge_p_a_style_swapped",
    "raw_judge_status",
    "raw_judge_model_id",
    "rubric_entropy",
    "response_length_ratio",
    "style_sensitivity",
    "model_family_a",
    "model_family_b",
}
