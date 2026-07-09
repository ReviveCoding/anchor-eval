from __future__ import annotations

import numpy as np
import pandas as pd

DOMAIN_CODE = {
    "financial_grounded_qa": 0,
    "tool_trace": 1,
    "instruction_following": 2,
}
DIFFICULTY_CODE = {"easy": 0, "medium": 1, "hard": 2}


def swapped_back_score(frame: pd.DataFrame) -> np.ndarray:
    """Convert the probability of original B under a swapped display into P(original A)."""
    return 1.0 - frame["raw_judge_p_a_swapped"].to_numpy(dtype=float)


def style_swapped_back_score(frame: pd.DataFrame) -> np.ndarray:
    """Convert P(original B) under swapped + neutral-style presentation into P(original A)."""
    return 1.0 - frame["raw_judge_p_a_style_swapped"].to_numpy(dtype=float)


def symmetric_raw_score(frame: pd.DataFrame) -> np.ndarray:
    """Return P(A) averaged over original and swapped display orders."""
    raw = frame["raw_judge_p_a"].to_numpy(dtype=float)
    return 0.5 * (raw + swapped_back_score(frame))


def style_symmetric_raw_score(frame: pd.DataFrame) -> np.ndarray:
    """Return P(A) averaged over neutral-style original and neutral-style swapped displays."""
    style = frame["raw_judge_p_a_style"].to_numpy(dtype=float)
    return 0.5 * (style + style_swapped_back_score(frame))


def full_orbit_raw_score(frame: pd.DataFrame) -> np.ndarray:
    """Average all four order/style presentations on a common P(original A) scale."""
    return 0.5 * (symmetric_raw_score(frame) + style_symmetric_raw_score(frame))


def raw_orbit_spread(frame: pd.DataFrame) -> np.ndarray:
    """Maximum probability spread across the complete order/style presentation orbit."""
    values = np.column_stack(
        [
            frame["raw_judge_p_a"].to_numpy(dtype=float),
            swapped_back_score(frame),
            frame["raw_judge_p_a_style"].to_numpy(dtype=float),
            style_swapped_back_score(frame),
        ]
    )
    return values.max(axis=1) - values.min(axis=1)


def order_gap(frame: pd.DataFrame) -> np.ndarray:
    return np.abs(frame["raw_judge_p_a"].to_numpy(dtype=float) - swapped_back_score(frame))


def style_gap(frame: pd.DataFrame) -> np.ndarray:
    """Conservative style gap across both original and swapped displays."""
    original_gap = np.abs(
        frame["raw_judge_p_a"].to_numpy(dtype=float)
        - frame["raw_judge_p_a_style"].to_numpy(dtype=float)
    )
    swapped_gap = np.abs(swapped_back_score(frame) - style_swapped_back_score(frame))
    return np.maximum(original_gap, swapped_gap)


def raw_order_flip(frame: pd.DataFrame) -> np.ndarray:
    return (frame["raw_judge_p_a"].to_numpy(dtype=float) >= 0.5) != (
        swapped_back_score(frame) >= 0.5
    )


def raw_style_flip(frame: pd.DataFrame) -> np.ndarray:
    """Flag either original-side or swapped-side style-induced preference reversal."""
    original_flip = (frame["raw_judge_p_a"].to_numpy(dtype=float) >= 0.5) != (
        frame["raw_judge_p_a_style"].to_numpy(dtype=float) >= 0.5
    )
    swapped_flip = (swapped_back_score(frame) >= 0.5) != (style_swapped_back_score(frame) >= 0.5)
    return original_flip | swapped_flip


def feature_frame(frame: pd.DataFrame, include_raw: bool = True) -> pd.DataFrame:
    values = {
        "raw_p_a": frame["raw_judge_p_a"].astype(float),
        "symmetric_raw_p_a": symmetric_raw_score(frame),
        "full_orbit_raw_p_a": full_orbit_raw_score(frame),
        "order_gap": order_gap(frame),
        "style_gap": style_gap(frame),
        "raw_orbit_spread": raw_orbit_spread(frame),
        "raw_order_flip": raw_order_flip(frame).astype(int),
        "raw_style_flip": raw_style_flip(frame).astype(int),
        "rubric_entropy": frame["rubric_entropy"].astype(float),
        "log_length_ratio": np.log(
            np.clip(frame["response_length_ratio"].astype(float), 1e-6, None)
        ),
        "style_sensitivity": frame["style_sensitivity"].astype(float),
        "verifier_available": frame["deterministic_verifiable"].astype(int),
        "domain_code": frame["domain"].map(DOMAIN_CODE).fillna(-1).astype(int),
        "difficulty_code": frame["difficulty"].map(DIFFICULTY_CODE).fillna(-1).astype(int),
        "model_family_a_x": (frame["model_family_a"] == "model_x").astype(int),
    }
    result = pd.DataFrame(values, index=frame.index)
    if not include_raw:
        result = result.drop(columns=["raw_p_a"])
    return result


def anchor_state_features(frame: pd.DataFrame) -> np.ndarray:
    """Small interpretable state representation for anchor-coverage distance, not model training."""
    return np.column_stack(
        [
            frame["domain"].map(DOMAIN_CODE).fillna(-1).to_numpy(dtype=float),
            frame["difficulty"].map(DIFFICULTY_CODE).fillna(-1).to_numpy(dtype=float),
            frame["deterministic_verifiable"].astype(int).to_numpy(dtype=float),
            frame["rubric_entropy"].to_numpy(dtype=float),
            np.abs(
                np.log(np.clip(frame["response_length_ratio"].to_numpy(dtype=float), 1e-6, None))
            ),
            order_gap(frame),
            style_gap(frame),
            raw_orbit_spread(frame),
        ]
    )
