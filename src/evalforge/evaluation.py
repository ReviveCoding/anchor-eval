from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import CandidatePrediction
from .contracts import ReleaseDecision
from .features import order_gap, raw_orbit_spread, raw_order_flip, raw_style_flip, style_gap


def _truth(frame: pd.DataFrame) -> np.ndarray:
    return (frame["gold_preference"].to_numpy() == "A").astype(int)


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(probabilities) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - outcomes[mask].mean())
    return float(ece)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _slice_summary(
    enriched: pd.DataFrame, candidate_id: str, has_router: bool, min_slice_n: int
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    eligible_agreements: list[float] = []
    for column in ("domain", "difficulty", "deterministic_verifiable"):
        for value, group in enriched.loc[enriched["evaluable"]].groupby(column):
            n = int(len(group))
            agreement = float(group["correct"].mean())
            if n >= min_slice_n:
                eligible_agreements.append(agreement)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "slice_type": column,
                    "slice_value": str(value),
                    "n": n,
                    "agreement": agreement,
                    "brier": float(np.mean((group["p_correct"] - group["correct"]) ** 2)),
                    "auto_accept_coverage": (
                        float((group["route"] == "AUTO_ACCEPT").mean())
                        if has_router
                        else float("nan")
                    ),
                    "candidate_order_flip_rate": float(
                        ((group["p_a"] >= 0.5) != (group["p_a_swap_back"] >= 0.5)).mean()
                    ),
                    "candidate_style_flip_rate": float(
                        ((group["p_a"] >= 0.5) != (group["p_a_style"] >= 0.5)).mean()
                    ),
                    "raw_order_flip_rate": float(group["raw_order_flip"].mean()),
                    "raw_style_flip_rate": float(group["raw_style_flip"].mean()),
                }
            )
    return pd.DataFrame(rows), (
        float(min(eligible_agreements)) if eligible_agreements else float("nan")
    )


def evaluate_prediction(
    frame: pd.DataFrame, prediction: CandidatePrediction, min_slice_n: int = 3
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    truth = _truth(frame)
    predicted = (prediction.p_a >= 0.5).astype(int)
    correct = (truth == predicted).astype(int)
    p_correct = np.clip(prediction.p_correct, 1e-4, 1.0 - 1e-4)
    routes = prediction.route
    evaluable = (
        prediction.evaluable_mask
        if prediction.evaluable_mask is not None
        else np.ones(len(frame), dtype=bool)
    )
    auto = routes == "AUTO_ACCEPT"
    has_router = bool(np.any(routes != "UNROUTED"))
    candidate_order_delta = np.abs(prediction.p_a - prediction.p_a_swap_back)
    candidate_style_delta = np.abs(prediction.p_a - prediction.p_a_style)
    raw_order_flip_values = raw_order_flip(frame)
    raw_style_flip_values = raw_style_flip(frame)
    valid_correct = correct[evaluable]
    valid_conf = p_correct[evaluable]
    verifiable = frame["deterministic_verifiable"].to_numpy(dtype=bool) & evaluable
    nonverifiable = ~frame["deterministic_verifiable"].to_numpy(dtype=bool) & evaluable

    enriched = frame[
        [
            "item_id",
            "source_group",
            "split",
            "domain",
            "difficulty",
            "gold_preference",
            "deterministic_verifiable",
            "raw_judge_status",
            "raw_judge_model_id",
        ]
    ].copy()
    enriched["candidate_id"] = prediction.candidate_id
    enriched["p_a"] = prediction.p_a
    enriched["p_correct"] = prediction.p_correct
    enriched["prediction"] = np.where(predicted == 1, "A", "B")
    enriched["correct"] = correct
    enriched["evaluable"] = evaluable
    enriched["route"] = routes
    enriched["reason_code"] = prediction.reason_code
    enriched["p_a_swap_back"] = prediction.p_a_swap_back
    enriched["p_a_style"] = prediction.p_a_style
    enriched["candidate_order_delta"] = candidate_order_delta
    enriched["candidate_style_delta"] = candidate_style_delta
    enriched["raw_order_delta"] = order_gap(frame)
    enriched["raw_style_delta"] = style_gap(frame)
    enriched["raw_orbit_spread"] = raw_orbit_spread(frame)
    enriched["raw_order_flip"] = raw_order_flip_values
    enriched["raw_style_flip"] = raw_style_flip_values

    slice_df, worst_slice = _slice_summary(
        enriched, prediction.candidate_id, has_router, min_slice_n
    )
    domain_scores = [
        float(group["correct"].mean())
        for _, group in enriched.loc[enriched["evaluable"]].groupby("domain")
    ]
    metrics = {
        "evaluator_coverage": float(evaluable.mean()),
        "pairwise_agreement": _safe_mean(valid_correct),
        "macro_domain_pairwise_agreement": _safe_mean(np.asarray(domain_scores, dtype=float)),
        "brier_correctness": (
            float(np.mean((valid_conf - valid_correct) ** 2))
            if len(valid_correct)
            else float("nan")
        ),
        "ece_correctness": expected_calibration_error(valid_conf, valid_correct),
        "verifiable_agreement": _safe_mean(correct[verifiable]),
        "nonverifiable_agreement": _safe_mean(correct[nonverifiable]),
        "auto_accept_coverage": float(auto.mean()) if has_router else float("nan"),
        "wrong_auto_label_rate": (
            float(1.0 - correct[auto].mean())
            if auto.any()
            else (float("nan") if not has_router else 0.0)
        ),
        "quarantine_rate": float((routes == "QUARANTINE").mean()) if has_router else float("nan"),
        "candidate_order_flip_rate": (
            float(((prediction.p_a >= 0.5) != (prediction.p_a_swap_back >= 0.5))[evaluable].mean())
            if evaluable.any()
            else float("nan")
        ),
        "candidate_style_flip_rate": (
            float(((prediction.p_a >= 0.5) != (prediction.p_a_style >= 0.5))[evaluable].mean())
            if evaluable.any()
            else float("nan")
        ),
        "raw_order_flip_rate": float(raw_order_flip_values[evaluable].mean())
        if evaluable.any()
        else float("nan"),
        "raw_style_flip_rate": float(raw_style_flip_values[evaluable].mean())
        if evaluable.any()
        else float("nan"),
        "mean_raw_orbit_spread": _safe_mean(raw_orbit_spread(frame)[evaluable]),
        "mean_candidate_order_delta": _safe_mean(candidate_order_delta[evaluable]),
        "mean_candidate_style_delta": _safe_mean(candidate_style_delta[evaluable]),
        "mean_raw_order_delta": _safe_mean(order_gap(frame)[evaluable]),
        "mean_raw_style_delta": _safe_mean(style_gap(frame)[evaluable]),
        "mean_p_correct": _safe_mean(valid_conf),
        "auto_accept_raw_order_flip_rate": (
            float(raw_order_flip_values[auto].mean())
            if auto.any()
            else (float("nan") if not has_router else 0.0)
        ),
        "auto_accept_raw_style_flip_rate": (
            float(raw_style_flip_values[auto].mean())
            if auto.any()
            else (float("nan") if not has_router else 0.0)
        ),
        "auto_accept_mean_raw_orbit_spread": (
            _safe_mean(raw_orbit_spread(frame)[auto])
            if auto.any()
            else (float("nan") if not has_router else 0.0)
        ),
        "worst_slice_agreement": worst_slice,
        "worst_slice_min_n": float(min_slice_n),
        "latency_total_ms": float(prediction.latency_ms),
        "latency_per_item_ms": float(prediction.latency_ms / max(len(frame), 1)),
    }
    return metrics, slice_df, enriched


def _guardrail_masks(
    frame: pd.DataFrame,
    prediction: CandidatePrediction,
    order_gap_limit: float,
    style_gap_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_label = np.where(prediction.p_a >= 0.5, "A", "B")
    conflict = frame["deterministic_verifiable"].to_numpy(dtype=bool) & (
        truth_label != frame["verifier_preference"].to_numpy(dtype=object)
    )
    order_unstable = (order_gap(frame) > order_gap_limit) | raw_order_flip(frame)
    style_unstable = (style_gap(frame) > style_gap_limit) | raw_style_flip(frame)
    return conflict, order_unstable, style_unstable


def threshold_scenarios(
    frame: pd.DataFrame,
    prediction: CandidatePrediction,
    thresholds: Iterable[float],
    order_gap_limit: float,
    style_gap_limit: float,
) -> pd.DataFrame:
    truth = _truth(frame)
    pred = (prediction.p_a >= 0.5).astype(int)
    correct = (truth == pred).astype(int)
    conflict, order_unstable, style_unstable = _guardrail_masks(
        frame, prediction, order_gap_limit, style_gap_limit
    )
    unstable = order_unstable | style_unstable
    rows = []
    for threshold in thresholds:
        accept = (prediction.p_correct >= threshold) & ~unstable & ~conflict
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(accept.mean()),
                "wrong_auto_label_rate": (
                    float(1.0 - correct[accept].mean()) if accept.any() else 0.0
                ),
                "accepted_pairs": int(accept.sum()),
                "order_or_style_guardrail_blocks": int(unstable.sum()),
                "verifier_conflict_blocks": int(conflict.sum()),
            }
        )
    return pd.DataFrame(rows)


def operating_profile_scenarios(
    frame: pd.DataFrame,
    prediction: CandidatePrediction,
    max_wrong_auto_label_rate: float,
    order_gap_limit: float,
    style_gap_limit: float,
    escalation_capacities: tuple[float, ...] = (0.20, 0.30, 0.40),
) -> pd.DataFrame:
    """Evaluate review-capacity operating points with the same routing guardrails."""
    truth = _truth(frame)
    pred = (prediction.p_a >= 0.5).astype(int)
    correct = pred == truth
    conflict, order_unstable, style_unstable = _guardrail_masks(
        frame, prediction, order_gap_limit, style_gap_limit
    )
    eligible = ~(conflict | order_unstable | style_unstable)
    thresholds = np.unique(np.concatenate(([0.50], prediction.p_correct, [0.9999])))
    rows: list[dict[str, object]] = []
    for capacity in escalation_capacities:
        feasible: list[dict[str, float]] = []
        for threshold in thresholds:
            accept = eligible & (prediction.p_correct >= threshold)
            coverage = float(accept.mean())
            wrong = float(1.0 - correct[accept].mean()) if accept.any() else 0.0
            escalation = 1.0 - coverage
            if wrong <= max_wrong_auto_label_rate and escalation <= capacity:
                feasible.append(
                    {
                        "threshold": float(threshold),
                        "coverage": coverage,
                        "wrong_auto_label_rate": wrong,
                        "escalation_rate": escalation,
                    }
                )
        profile = f"review_capacity_{int(round(capacity * 100))}pct"
        if feasible:
            selected = max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))
            rows.append({"profile": profile, "feasible": True, **selected})
        else:
            rows.append(
                {
                    "profile": profile,
                    "feasible": False,
                    "threshold": float("nan"),
                    "coverage": float("nan"),
                    "wrong_auto_label_rate": float("nan"),
                    "escalation_rate": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def select_release_decision(
    metrics: dict[str, float],
    max_wrong_auto_label_rate: float,
    max_order_flip_rate: float,
    max_style_flip_rate: float,
    max_ece: float,
    min_auto_accept_coverage: float,
    source_type: str,
    provenance_verified: bool,
    raw_judge_statuses: set[str],
    policy_feasible: bool,
) -> ReleaseDecision:
    reasons: list[str] = []
    if source_type in {"synthetic_fixture", "public_derived_controlled"}:
        reasons.append("synthetic_or_controlled_evidence_not_release_qualified")
    if source_type != "external_benchmark" or not provenance_verified:
        reasons.append("external_provenance_not_verified")
    if raw_judge_statuses != {"local_scored"}:
        reasons.append("local_judge_scoring_not_verified")
    if not policy_feasible:
        reasons.append("policy_capacity_or_risk_constraint_infeasible")
    if (
        not np.isfinite(metrics["wrong_auto_label_rate"])
        or metrics["wrong_auto_label_rate"] > max_wrong_auto_label_rate
    ):
        reasons.append("wrong_auto_label_rate_exceeds_gate")
    if metrics["auto_accept_raw_order_flip_rate"] > max_order_flip_rate:
        reasons.append("auto_accept_raw_order_flip_exceeds_gate")
    if metrics["auto_accept_raw_style_flip_rate"] > max_style_flip_rate:
        reasons.append("auto_accept_raw_style_flip_exceeds_gate")
    if metrics["ece_correctness"] > max_ece:
        reasons.append("calibration_error_exceeds_gate")
    if (
        not np.isfinite(metrics["auto_accept_coverage"])
        or metrics["auto_accept_coverage"] < min_auto_accept_coverage
    ):
        reasons.append("auto_accept_coverage_below_gate")
    if not reasons:
        return ReleaseDecision("PASS", tuple())
    release_blockers = [
        reason
        for reason in reasons
        if reason
        not in {
            "synthetic_or_controlled_evidence_not_release_qualified",
            "external_provenance_not_verified",
            "local_judge_scoring_not_verified",
        }
    ]
    if not release_blockers:
        return ReleaseDecision("CONDITIONAL_PASS", tuple(reasons))
    return ReleaseDecision("BLOCK", tuple(reasons))


def grouped_bootstrap_delta(
    frame: pd.DataFrame,
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    seed: int = 1729,
    draws: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = frame["source_group"].astype(str).unique()
    grouped = {
        group: np.where(frame["source_group"].astype(str).to_numpy() == group)[0]
        for group in groups
    }
    deltas: list[float] = []
    for _ in range(draws):
        selected = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([grouped[group] for group in selected])
        deltas.append(float(left_correct[indices].mean() - right_correct[indices].mean()))
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "draws": float(draws),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
