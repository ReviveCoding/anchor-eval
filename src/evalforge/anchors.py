from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import anchor_state_features, order_gap, style_gap


@dataclass(frozen=True)
class AnchorSelection:
    method: str
    item_ids: tuple[str, ...]
    diagnostics: pd.DataFrame


def _unique_source_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep at most one candidate per source group, preserving deterministic reproducibility."""
    return (
        frame.sort_values(["deterministic_verifiable", "item_id"], ascending=[False, True])
        .drop_duplicates("source_group", keep="first")
        .reset_index(drop=True)
    )


def select_static_anchors(calibration: pd.DataFrame, budget: int) -> AnchorSelection:
    work = _unique_source_candidates(calibration.copy())
    if work.empty:
        raise ValueError("No calibration candidates available for static anchors.")
    work["bucket"] = work["domain"].astype(str) + "::" + work["difficulty"].astype(str)
    buckets = sorted(work["bucket"].unique())
    selected: list[int] = []
    for bucket in buckets:
        matches = work.index[work["bucket"] == bucket].tolist()
        if matches and len(selected) < budget:
            selected.append(matches[0])
    for idx in work.index.tolist():
        if len(selected) >= min(budget, len(work)):
            break
        if idx not in selected:
            selected.append(idx)
    chosen = work.loc[selected].copy()
    diagnostics = chosen[
        [
            "item_id",
            "source_group",
            "domain",
            "difficulty",
            "deterministic_verifiable",
            "rubric_entropy",
            "response_length_ratio",
            "raw_judge_p_a",
            "raw_judge_p_a_swapped",
            "raw_judge_p_a_style",
            "raw_judge_p_a_style_swapped",
        ]
    ].copy()
    diagnostics["selection_score"] = 1.0
    return AnchorSelection("static_stratified", tuple(chosen["item_id"]), diagnostics)


def select_adaptive_anchors(calibration: pd.DataFrame, budget: int) -> AnchorSelection:
    """Group-disjoint greedy anchor selection with coverage, verifier support, and redundancy control."""
    work = _unique_source_candidates(calibration.copy())
    if work.empty:
        raise ValueError("No calibration candidates available for adaptive anchors.")
    work["order_gap"] = order_gap(work)
    work["style_gap"] = style_gap(work)
    work["verifier_score"] = work["deterministic_verifiable"].astype(float)
    work["uncertainty_information"] = 1.0 - np.abs(work["rubric_entropy"].astype(float) - 0.5) * 2.0
    work["base_score"] = (
        1.5 * work["verifier_score"]
        + 0.65 * work["uncertainty_information"]
        - 0.35 * work["order_gap"]
        - 0.20 * work["style_gap"]
    )
    selected_indices: list[int] = []
    covered_buckets: set[tuple[str, str]] = set()
    selected_families: set[tuple[str, str]] = set()
    remaining = set(work.index.tolist())
    while remaining and len(selected_indices) < min(budget, len(work)):
        best_index = None
        best_value = -float("inf")
        for idx in remaining:
            row = work.loc[idx]
            bucket = (str(row["domain"]), str(row["difficulty"]))
            family = (str(row["model_family_a"]), str(row["model_family_b"]))
            coverage_bonus = 1.50 if bucket not in covered_buckets else 0.0
            family_bonus = 0.20 if family not in selected_families else 0.0
            score = float(row["base_score"] + coverage_bonus + family_bonus)
            if score > best_value:
                best_value = score
                best_index = idx
        assert best_index is not None
        selected_indices.append(best_index)
        row = work.loc[best_index]
        covered_buckets.add((str(row["domain"]), str(row["difficulty"])))
        selected_families.add((str(row["model_family_a"]), str(row["model_family_b"])))
        remaining.remove(best_index)
    chosen = work.loc[selected_indices].copy()
    chosen["selection_score"] = chosen["base_score"]
    return AnchorSelection(
        "adaptive_group_disjoint",
        tuple(chosen["item_id"].tolist()),
        chosen[
            [
                "item_id",
                "source_group",
                "domain",
                "difficulty",
                "deterministic_verifiable",
                "rubric_entropy",
                "response_length_ratio",
                "raw_judge_p_a",
                "raw_judge_p_a_swapped",
                "raw_judge_p_a_style",
                "raw_judge_p_a_style_swapped",
                "order_gap",
                "style_gap",
                "selection_score",
            ]
        ],
    )


def anchor_coverage(frame: pd.DataFrame, anchors: pd.DataFrame) -> np.ndarray:
    """Continuous state coverage in [0, 1], unlike a trivial binary domain/difficulty membership flag."""
    if anchors.empty:
        return np.zeros(len(frame), dtype=float)
    x = anchor_state_features(frame)
    a = anchor_state_features(anchors)
    scale = np.maximum(np.std(a, axis=0, ddof=0), 0.25)
    distances = np.sqrt(((x[:, None, :] - a[None, :, :]) / scale) ** 2).mean(axis=2)
    return np.exp(-np.min(distances, axis=1))
