from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .anchors import AnchorSelection, anchor_coverage
from .features import (
    order_gap,
    raw_orbit_spread,
    raw_order_flip,
    raw_style_flip,
    style_gap,
    style_symmetric_raw_score,
    symmetric_raw_score,
)


@dataclass
class CandidatePrediction:
    candidate_id: str
    p_a: np.ndarray
    p_correct: np.ndarray
    route: np.ndarray
    reason_code: np.ndarray
    latency_ms: float
    metadata: dict[str, Any]
    p_a_swap_back: np.ndarray | None = None
    p_a_style: np.ndarray | None = None
    evaluable_mask: np.ndarray | None = None


def _safe_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-4, 1.0 - 1e-4)


def _gold_binary(frame: pd.DataFrame) -> np.ndarray:
    return (frame["gold_preference"].to_numpy() == "A").astype(int)


def _swap_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Swap A/B evidence so a predictor can be tested for display-order sensitivity."""
    swapped = frame.copy()
    swapped[["response_a", "response_b"]] = frame[["response_b", "response_a"]].to_numpy()
    swapped[["model_family_a", "model_family_b"]] = frame[
        ["model_family_b", "model_family_a"]
    ].to_numpy()
    swapped["response_length_ratio"] = 1.0 / np.clip(
        frame["response_length_ratio"].to_numpy(dtype=float), 1e-6, None
    )
    return swapped


def _style_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a semantically neutral presentation layer to A for style-sensitivity testing."""
    styled = frame.copy()
    styled["response_a"] = (
        "## Executive Summary\n" + frame["response_a"].astype(str) + "\n**Conclusion:** see above."
    )
    return styled


class RawJudge:
    candidate_id = "C1_raw_judge"

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> RawJudge:
        return self

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        p_a = _safe_probabilities(frame["raw_judge_p_a"].to_numpy(dtype=float))
        swap_back = _safe_probabilities(1.0 - frame["raw_judge_p_a_swapped"].to_numpy(dtype=float))
        style = _safe_probabilities(frame["raw_judge_p_a_style"].to_numpy(dtype=float))
        confidence = np.maximum(p_a, 1.0 - p_a)
        return CandidatePrediction(
            self.candidate_id,
            p_a,
            confidence,
            np.full(len(frame), "UNROUTED", dtype=object),
            np.full(len(frame), "raw_judge", dtype=object),
            (perf_counter() - start) * 1000.0,
            {},
            p_a_swap_back=swap_back,
            p_a_style=style,
            evaluable_mask=np.ones(len(frame), dtype=bool),
        )


class DeterministicVerifier:
    candidate_id = "C0_verifier"

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> DeterministicVerifier:
        return self

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        verifier = frame["verifier_preference"].to_numpy(dtype=object)
        available = frame["deterministic_verifiable"].to_numpy(dtype=bool)
        p_a = np.where(available, np.where(verifier == "A", 0.999, 0.001), 0.5)
        p_correct = np.where(available, 0.999, 0.5)
        route = np.where(available, "AUTO_ACCEPT", "ESCALATE")
        reason = np.where(available, "deterministic_verifier", "verifier_unavailable")
        return CandidatePrediction(
            self.candidate_id,
            p_a,
            p_correct,
            route.astype(object),
            reason.astype(object),
            (perf_counter() - start) * 1000.0,
            {"coverage_type": "deterministic_only"},
            p_a_swap_back=p_a.copy(),
            p_a_style=p_a.copy(),
            evaluable_mask=available,
        )


class LearnedEvaluator:
    """A compact text-pair evaluator trained only on the train split.

    It is an advanced, CPU-runnable challenger, not a proxy that consumes the raw judge score as a feature.
    Training-time A/B augmentation enforces a basic pairwise anti-symmetry constraint.
    """

    candidate_id = "C2_text_pair_evaluator"

    def __init__(self, random_state: int = 1729) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=6000, sublinear_tf=True
        )
        self.model = LogisticRegression(max_iter=600, C=1.2, random_state=random_state)
        self.correctness_calibrator: IsotonicRegression | None = None

    @staticmethod
    def _texts(frame: pd.DataFrame) -> list[str]:
        return [
            f"TASK: {prompt}\n[A]\n{a}\n[B]\n{b}"
            for prompt, a, b in zip(
                frame["prompt"].astype(str),
                frame["response_a"].astype(str),
                frame["response_b"].astype(str),
                strict=False,
            )
        ]

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> LearnedEvaluator:
        swapped = _swap_pair_frame(train)
        train_texts = self._texts(train) + self._texts(swapped)
        y = np.concatenate([_gold_binary(train), 1 - _gold_binary(train)])
        x = self.vectorizer.fit_transform(train_texts)
        self.model.fit(x, y)
        cal_p = self._predict_p_a(calibration)
        cal_correct = ((cal_p >= 0.5).astype(int) == _gold_binary(calibration)).astype(int)
        confidence = np.maximum(cal_p, 1.0 - cal_p)
        if len(np.unique(cal_correct)) > 1:
            self.correctness_calibrator = IsotonicRegression(out_of_bounds="clip")
            self.correctness_calibrator.fit(confidence, cal_correct)
        return self

    def _predict_p_a(self, frame: pd.DataFrame) -> np.ndarray:
        return _safe_probabilities(
            self.model.predict_proba(self.vectorizer.transform(self._texts(frame)))[:, 1]
        )

    def _p_correct(self, p_a: np.ndarray) -> np.ndarray:
        confidence = np.maximum(p_a, 1.0 - p_a)
        if self.correctness_calibrator is None:
            return _safe_probabilities(confidence)
        return _safe_probabilities(self.correctness_calibrator.predict(confidence))

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        p_a = self._predict_p_a(frame)
        swap_back = _safe_probabilities(1.0 - self._predict_p_a(_swap_pair_frame(frame)))
        style = self._predict_p_a(_style_pair_frame(frame))
        return CandidatePrediction(
            self.candidate_id,
            p_a,
            self._p_correct(p_a),
            np.full(len(frame), "UNROUTED", dtype=object),
            np.full(len(frame), "text_pair_evaluator", dtype=object),
            (perf_counter() - start) * 1000.0,
            {"representation": "tfidf_word_1_2gram", "pairwise_swap_augmentation": True},
            p_a_swap_back=swap_back,
            p_a_style=style,
            evaluable_mask=np.ones(len(frame), dtype=bool),
        )


class StaticAnchorEvaluator:
    candidate_id = "C3_static_anchor"

    def __init__(self, anchors: AnchorSelection) -> None:
        self.anchors = anchors
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, anchor_fit: pd.DataFrame) -> StaticAnchorEvaluator:
        lookup = anchor_fit.set_index("item_id")
        anchor_frame = lookup.loc[list(self.anchors.item_ids)].reset_index()
        self.calibrator.fit(symmetric_raw_score(anchor_frame), _gold_binary(anchor_frame))
        return self

    def _predict_score(self, frame: pd.DataFrame, use_style: bool = False) -> np.ndarray:
        score = style_symmetric_raw_score(frame) if use_style else symmetric_raw_score(frame)
        return _safe_probabilities(self.calibrator.predict(score))

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        p_a = self._predict_score(frame)
        p_a_style = self._predict_score(frame, use_style=True)
        return CandidatePrediction(
            self.candidate_id,
            p_a,
            np.maximum(p_a, 1.0 - p_a),
            np.full(len(frame), "UNROUTED", dtype=object),
            np.full(len(frame), "static_anchor", dtype=object),
            (perf_counter() - start) * 1000.0,
            {
                "anchor_count": len(self.anchors.item_ids),
                "anchor_method": self.anchors.method,
                "presentation_canonicalization": "order_pair_mean",
            },
            p_a_swap_back=p_a.copy(),
            p_a_style=p_a_style,
            evaluable_mask=np.ones(len(frame), dtype=bool),
        )


class AdaptiveAnchorVCC:
    candidate_id = "C4_aal_vcc"

    def __init__(
        self,
        anchors: AnchorSelection,
        learned_evaluator: LearnedEvaluator,
        random_state: int = 1729,
    ) -> None:
        self.anchors = anchors
        self.learned_evaluator = learned_evaluator
        self.domain_calibrators: dict[str, IsotonicRegression] = {}
        self.global_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.blender = LogisticRegression(max_iter=600, random_state=random_state)
        self.vcc: LogisticRegression | None = LogisticRegression(
            max_iter=600, random_state=random_state
        )
        self.vcc_calibration_mode = "logistic"
        self.anchor_frame: pd.DataFrame | None = None

    def fit(
        self, train: pd.DataFrame, anchor_fit: pd.DataFrame, vcc_fit: pd.DataFrame
    ) -> AdaptiveAnchorVCC:
        lookup = anchor_fit.set_index("item_id")
        self.anchor_frame = lookup.loc[list(self.anchors.item_ids)].reset_index()
        self.global_calibrator.fit(
            symmetric_raw_score(self.anchor_frame), _gold_binary(self.anchor_frame)
        )
        for domain, group in self.anchor_frame.groupby("domain"):
            if len(group) >= 4 and len(np.unique(_gold_binary(group))) == 2:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(symmetric_raw_score(group), _gold_binary(group))
                self.domain_calibrators[str(domain)] = calibrator
        train_anchor = self._domain_calibrated_score(train)
        train_text = self.learned_evaluator.predict(train).p_a
        self.blender.fit(self._blend_features(train, train_anchor, train_text), _gold_binary(train))
        vcc_p, _ = self._orbit_blend(vcc_fit)
        vcc_pred = (vcc_p >= 0.5).astype(int)
        vcc_correct = (vcc_pred == _gold_binary(vcc_fit)).astype(int)
        if len(np.unique(vcc_correct)) < 2:
            # Small, easy, or heavily filtered calibration sets can contain only correct
            # (or only incorrect) outcomes. Fitting a logistic calibrator would fail and
            # turning the empirical one-class rate into a confident routing signal would
            # be unsafe. Fall back to an explicit 0.5 confidence so C5 escalates instead
            # of silently auto-accepting labels on unidentifiable calibration evidence.
            self.vcc = None
            self.vcc_calibration_mode = "conservative_one_class_fallback"
            return self
        assert self.vcc is not None
        self.vcc.fit(self._vcc_features(vcc_fit, vcc_p), vcc_correct)
        self.vcc_calibration_mode = "logistic"
        return self

    def _domain_calibrated_score(
        self, frame: pd.DataFrame, score: np.ndarray | None = None
    ) -> np.ndarray:
        raw = symmetric_raw_score(frame) if score is None else np.asarray(score, dtype=float)
        output = self.global_calibrator.predict(raw)
        for domain, calibrator in self.domain_calibrators.items():
            mask = frame["domain"].astype(str).to_numpy() == domain
            if mask.any():
                output[mask] = calibrator.predict(raw[mask])
        return _safe_probabilities(output)

    @staticmethod
    def _blend_features(
        frame: pd.DataFrame, anchor_p: np.ndarray, text_p: np.ndarray
    ) -> np.ndarray:
        return np.column_stack(
            [
                anchor_p,
                text_p,
                np.abs(anchor_p - text_p),
                order_gap(frame),
                style_gap(frame),
                raw_orbit_spread(frame),
                frame["rubric_entropy"].to_numpy(dtype=float),
            ]
        )

    def _blend_from_scores(
        self, frame: pd.DataFrame, anchor_p: np.ndarray, text_p: np.ndarray
    ) -> np.ndarray:
        return _safe_probabilities(
            self.blender.predict_proba(self._blend_features(frame, anchor_p, text_p))[:, 1]
        )

    def _blend(self, frame: pd.DataFrame) -> np.ndarray:
        """Single-presentation score used only to fit the blender on train data."""
        anchor_p = self._domain_calibrated_score(frame)
        text_p = self.learned_evaluator.predict(frame).p_a
        return self._blend_from_scores(frame, anchor_p, text_p)

    def _orbit_blend(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Produce the canonical four-presentation score and retain component scores for diagnostics."""
        order_anchor = self._domain_calibrated_score(frame)
        style_anchor = self._domain_calibrated_score(frame, style_symmetric_raw_score(frame))
        text_original = self.learned_evaluator._predict_p_a(frame)
        text_swap_back = _safe_probabilities(
            1.0 - self.learned_evaluator._predict_p_a(_swap_pair_frame(frame))
        )
        text_style = self.learned_evaluator._predict_p_a(_style_pair_frame(frame))
        text_style_swap_back = _safe_probabilities(
            1.0 - self.learned_evaluator._predict_p_a(_style_pair_frame(_swap_pair_frame(frame)))
        )
        orbit = np.column_stack(
            [
                self._blend_from_scores(frame, order_anchor, text_original),
                self._blend_from_scores(frame, order_anchor, text_swap_back),
                self._blend_from_scores(frame, style_anchor, text_style),
                self._blend_from_scores(frame, style_anchor, text_style_swap_back),
            ]
        )
        return _safe_probabilities(orbit.mean(axis=1)), orbit

    def _vcc_features(self, frame: pd.DataFrame, p_a: np.ndarray) -> np.ndarray:
        assert self.anchor_frame is not None
        coverage = anchor_coverage(frame, self.anchor_frame)
        return np.column_stack(
            [
                np.abs(p_a - 0.5),
                frame["deterministic_verifiable"].astype(int).to_numpy(),
                order_gap(frame),
                style_gap(frame),
                raw_orbit_spread(frame),
                frame["rubric_entropy"].to_numpy(dtype=float),
                np.abs(
                    np.log(
                        np.clip(frame["response_length_ratio"].to_numpy(dtype=float), 1e-6, None)
                    )
                ),
                frame["style_sensitivity"].to_numpy(dtype=float),
                coverage,
            ]
        )

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        # The public C4 score is the mean over the complete order/style orbit.  Each
        # component is converted to P(original A) first, so this is a canonicalized
        # score, not a claim that the underlying raw judge lacks presentation fragility.
        p_a, orbit = self._orbit_blend(frame)
        if self.vcc is None:
            # Preserve a safe route boundary when one-class calibration made
            # correctness probability unidentifiable.
            p_correct = np.full(len(frame), 0.5, dtype=float)
        else:
            p_correct = _safe_probabilities(
                self.vcc.predict_proba(self._vcc_features(frame, p_a))[:, 1]
            )
        return CandidatePrediction(
            self.candidate_id,
            p_a,
            p_correct,
            np.full(len(frame), "UNROUTED", dtype=object),
            np.full(len(frame), "aal_vcc", dtype=object),
            (perf_counter() - start) * 1000.0,
            {
                "anchor_count": len(self.anchors.item_ids),
                "anchor_method": self.anchors.method,
                "transform_aggregation": "mean_full_order_style_orbit",
                "presentation_canonicalization": "full_orbit_mean",
                "vcc_calibration_mode": self.vcc_calibration_mode,
                "raw_order_flip_rate": float(raw_order_flip(frame).mean()),
                "raw_style_flip_rate": float(raw_style_flip(frame).mean()),
                "mean_raw_orbit_spread": float(raw_orbit_spread(frame).mean()),
                "mean_blended_orbit_spread": float((orbit.max(axis=1) - orbit.min(axis=1)).mean()),
            },
            # The canonical score is identical under the tested group action by construction.
            # Routing still consumes raw orbit instability as an independent guardrail.
            p_a_swap_back=p_a.copy(),
            p_a_style=p_a.copy(),
            evaluable_mask=np.ones(len(frame), dtype=bool),
        )


class ProductionRouter:
    candidate_id = "C5_routed_production"

    def __init__(
        self,
        base: AdaptiveAnchorVCC,
        threshold: float = 0.90,
        max_wrong_auto_label_rate: float = 0.05,
        max_escalation_rate: float = 0.20,
        order_gap_limit: float = 0.20,
        style_gap_limit: float = 0.15,
    ) -> None:
        self.base = base
        self.threshold = threshold
        self.max_wrong_auto_label_rate = max_wrong_auto_label_rate
        self.max_escalation_rate = max_escalation_rate
        self.order_gap_limit = order_gap_limit
        self.style_gap_limit = style_gap_limit
        self.policy_diagnostics: dict[str, Any] = {"policy_threshold_fitted": False}

    def _guardrail_mask(
        self, frame: pd.DataFrame, prediction: CandidatePrediction
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        label = np.where(prediction.p_a >= 0.5, "A", "B")
        verifier_available = frame["deterministic_verifiable"].to_numpy(dtype=bool)
        verifier = frame["verifier_preference"].to_numpy(dtype=object)
        deterministic_conflict = verifier_available & (label != verifier)
        # Do not let transform aggregation hide a fragile underlying judge signal.
        # Routing consumes raw transform gaps even when C4's public decision has been
        # symmetrized across the equivalent presentation orbit.
        order_unstable = (order_gap(frame) > self.order_gap_limit) | raw_order_flip(frame)
        style_unstable = (style_gap(frame) > self.style_gap_limit) | raw_style_flip(frame)
        return deterministic_conflict, order_unstable, style_unstable

    def fit(self, policy: pd.DataFrame) -> ProductionRouter:
        prediction = self.base.predict(policy)
        if prediction.metadata.get("vcc_calibration_mode") != "logistic":
            # A one-class VCC calibration split cannot identify correctness probability.
            # Never fit a threshold on that split and accidentally convert a conservative
            # fallback score into automatic labeling.
            self.threshold = 1.0
            self.policy_diagnostics = {
                "policy_threshold_fitted": False,
                "policy_feasible": False,
                "policy_disabled_reason": "vcc_calibration_unidentifiable",
                "selected_threshold": self.threshold,
                "max_wrong_auto_label_rate": self.max_wrong_auto_label_rate,
                "max_escalation_rate": self.max_escalation_rate,
            }
            return self
        truth = _gold_binary(policy)
        pred_label = (prediction.p_a >= 0.5).astype(int)
        correct = pred_label == truth
        conflict, order_unstable, style_unstable = self._guardrail_mask(policy, prediction)
        eligible = ~(conflict | order_unstable | style_unstable)
        thresholds = np.unique(np.concatenate(([0.50], prediction.p_correct, [0.9999])))
        feasible_rows: list[dict[str, float]] = []
        all_rows: list[dict[str, float]] = []
        for threshold in thresholds:
            accept = eligible & (prediction.p_correct >= threshold)
            coverage = float(accept.mean())
            wrong_rate = float(1.0 - correct[accept].mean()) if accept.any() else 0.0
            escalation = 1.0 - coverage
            row = {
                "threshold": float(threshold),
                "coverage": coverage,
                "wrong_rate": wrong_rate,
                "escalation_rate": escalation,
            }
            all_rows.append(row)
            if (
                wrong_rate <= self.max_wrong_auto_label_rate
                and escalation <= self.max_escalation_rate
            ):
                feasible_rows.append(row)
        if feasible_rows:
            chosen = max(feasible_rows, key=lambda row: (row["coverage"], -row["threshold"]))
            policy_feasible = True
        else:
            chosen = min(
                all_rows, key=lambda row: (row["wrong_rate"], -row["coverage"], row["threshold"])
            )
            policy_feasible = False
        self.threshold = float(chosen["threshold"])
        self.policy_diagnostics = {
            "policy_threshold_fitted": True,
            "policy_feasible": policy_feasible,
            "selected_threshold": self.threshold,
            "policy_coverage": chosen["coverage"],
            "policy_wrong_auto_label_rate": chosen["wrong_rate"],
            "policy_escalation_rate": chosen["escalation_rate"],
            "max_wrong_auto_label_rate": self.max_wrong_auto_label_rate,
            "max_escalation_rate": self.max_escalation_rate,
            "candidate_threshold_count": len(all_rows),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> CandidatePrediction:
        start = perf_counter()
        base = self.base.predict(frame)
        conflict, order_unstable, style_unstable = self._guardrail_mask(frame, base)
        policy_disabled = bool(self.policy_diagnostics.get("policy_disabled_reason"))
        route = np.where(
            conflict,
            "QUARANTINE",
            np.where(
                policy_disabled,
                "ESCALATE",
                np.where(base.p_correct >= self.threshold, "AUTO_ACCEPT", "ESCALATE"),
            ),
        ).astype(object)
        route[(route == "AUTO_ACCEPT") & (order_unstable | style_unstable)] = "ESCALATE"
        reason = np.where(
            conflict,
            "verifier_conflict",
            np.where(
                policy_disabled,
                "vcc_unidentifiable",
                np.where(
                    order_unstable,
                    "order_sensitive",
                    np.where(
                        style_unstable,
                        "style_sensitive",
                        np.where(route == "AUTO_ACCEPT", "policy_risk_bound_met", "low_confidence"),
                    ),
                ),
            ),
        ).astype(object)
        return CandidatePrediction(
            self.candidate_id,
            base.p_a,
            base.p_correct,
            route,
            reason,
            base.latency_ms + (perf_counter() - start) * 1000.0,
            {**base.metadata, **self.policy_diagnostics, "auto_accept_threshold": self.threshold},
            p_a_swap_back=base.p_a_swap_back,
            p_a_style=base.p_a_style,
            evaluable_mask=base.evaluable_mask,
        )
