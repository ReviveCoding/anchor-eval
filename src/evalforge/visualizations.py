from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# EvalForge runs in local terminals and headless CI workers. Select a non-interactive
# backend before pyplot import so visualization generation cannot depend on a display.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, stem: Path, save_svg: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    if save_svg:
        fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def plot_data_quality(frame: pd.DataFrame, path: str | Path, save_svg: bool = True) -> None:
    counts = frame.groupby(["split", "domain"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Data composition by split and domain")
    ax.set_xlabel("Split")
    ax.set_ylabel("Response pairs")
    _save(fig, Path(path), save_svg)


def plot_candidate_comparison(
    comparison: pd.DataFrame, path: str | Path, save_svg: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ordered = comparison.sort_values("pairwise_agreement", ascending=False)
    ax.bar(ordered["candidate_id"], ordered["pairwise_agreement"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Held-out pairwise agreement")
    ax.set_ylabel("Agreement")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, Path(path), save_svg)


def plot_calibration(predictions: pd.DataFrame, path: str | Path, save_svg: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for candidate_id, group in predictions.groupby("candidate_id"):
        bins = pd.cut(group["p_correct"], bins=np.linspace(0, 1, 8), include_lowest=True)
        curve = group.groupby(bins, observed=True).agg(
            confidence=("p_correct", "mean"), correctness=("correct", "mean")
        )
        if not curve.empty:
            ax.plot(curve["confidence"], curve["correctness"], marker="o", label=candidate_id)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Ideal")
    ax.set_title("Evaluator correctness calibration")
    ax.set_xlabel("Predicted correctness probability")
    ax.set_ylabel("Observed correctness")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    _save(fig, Path(path), save_svg)


def plot_risk_coverage(scenarios: pd.DataFrame, path: str | Path, save_svg: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(scenarios["coverage"], scenarios["wrong_auto_label_rate"], marker="o")
    for _, row in scenarios.iterrows():
        ax.annotate(
            f"{row['threshold']:.2f}", (row["coverage"], row["wrong_auto_label_rate"]), fontsize=7
        )
    ax.set_title("Automation risk-coverage frontier")
    ax.set_xlabel("Auto-accept coverage")
    ax.set_ylabel("Wrong-auto-label rate")
    _save(fig, Path(path), save_svg)


def plot_failure_taxonomy(
    predictions: pd.DataFrame, path: str | Path, save_svg: bool = True
) -> None:
    failures = predictions.loc[predictions["correct"] == 0, "reason_code"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if failures.empty:
        ax.text(0.5, 0.5, "No held-out failures", ha="center", va="center")
        ax.axis("off")
    else:
        failures.plot(kind="bar", ax=ax)
        ax.set_title("Failure taxonomy")
        ax.set_xlabel("Reason code")
        ax.set_ylabel("Incorrect predictions")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, Path(path), save_svg)


def write_html_summary(path: str | Path, comparison: pd.DataFrame, release: dict) -> None:
    table = comparison.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>EvalForge report</title></head>
<body><h1>EvalForge smoke report</h1>
<p>This report is generated from a synthetic CPU fixture. It is not external benchmark evidence.</p>
<h2>Release decision</h2><pre>{json.dumps(release, indent=2)}</pre>
<h2>Candidate comparison</h2>{table}
</body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
