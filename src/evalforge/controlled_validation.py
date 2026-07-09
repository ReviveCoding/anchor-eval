from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .data import build_synthetic_fixture
from .pipeline import run


def run_controlled_validation(
    config: dict, seeds: Iterable[int], output_root: str | Path, groups_per_domain: int = 48
) -> Path:
    """Run several controlled synthetic seeds and aggregate results without cherry-picking one fixture.

    This is a system-validation artifact, not external model evidence. It is intentionally separate from release
    qualification and reports whether the proposed method improves, regresses, or is inconclusive under the
    controlled generator.
    """
    output_root = Path(output_root)
    fixture_dir = output_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    release_rows: list[dict] = []
    for seed in seeds:
        run_config = copy.deepcopy(config)
        fixture = fixture_dir / f"synthetic_seed_{seed}.csv"
        build_synthetic_fixture(fixture, seed=int(seed), groups_per_domain=groups_per_domain)
        run_config["run"]["seed"] = int(seed)
        run_config["run"]["run_name"] = f"controlled_seed_{seed}"
        run_config["run"]["output_root"] = str(output_root / "runs")
        run_config["data"]["fixture_path"] = str(fixture)
        result = run(run_config)
        comparison = pd.read_csv(result / "numeric" / "candidate_comparison.csv")
        comparison["seed"] = int(seed)
        rows.append(comparison)
        metrics = json.loads((result / "numeric" / "metrics.json").read_text(encoding="utf-8"))
        release_rows.append({"seed": int(seed), **metrics["release"]})
    all_rows = pd.concat(rows, ignore_index=True)
    metric_cols = [
        "pairwise_agreement",
        "macro_domain_pairwise_agreement",
        "brier_correctness",
        "ece_correctness",
        "worst_slice_agreement",
        "raw_order_flip_rate",
        "raw_style_flip_rate",
        "auto_accept_coverage",
        "wrong_auto_label_rate",
    ]
    summary = (
        all_rows.groupby("candidate_id")[metric_cols].agg(["mean", "std", "count"]).reset_index()
    )
    summary.columns = [
        "candidate_id" if col[0] == "candidate_id" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    c1 = all_rows.loc[all_rows["candidate_id"] == "C1_raw_judge"].set_index("seed")
    c4 = all_rows.loc[all_rows["candidate_id"] == "C4_aal_vcc"].set_index("seed")
    joined = c4.join(c1, lsuffix="_c4", rsuffix="_c1", how="inner")
    deltas = pd.DataFrame(
        {
            "seed": joined.index,
            "agreement_delta_c4_minus_c1": joined["pairwise_agreement_c4"]
            - joined["pairwise_agreement_c1"],
            "ece_delta_c4_minus_c1": joined["ece_correctness_c4"] - joined["ece_correctness_c1"],
            "order_flip_delta_c4_minus_c1": joined["raw_order_flip_rate_c4"]
            - joined["raw_order_flip_rate_c1"],
            "style_flip_delta_c4_minus_c1": joined["raw_style_flip_rate_c4"]
            - joined["raw_style_flip_rate_c1"],
        }
    )
    summary_path = output_root / "controlled_validation_summary.csv"
    seed_path = output_root / "controlled_validation_seed_results.csv"
    delta_path = output_root / "controlled_validation_c4_vs_c1.csv"
    release_path = output_root / "controlled_validation_release_status.csv"
    summary.to_csv(summary_path, index=False)
    all_rows.to_csv(seed_path, index=False)
    deltas.to_csv(delta_path, index=False)
    pd.DataFrame(release_rows).to_csv(release_path, index=False)
    manifest = {
        "seeds": [int(seed) for seed in seeds],
        "groups_per_domain": groups_per_domain,
        "artifact_paths": [str(summary_path), str(seed_path), str(delta_path), str(release_path)],
        "claim_boundary": "Controlled synthetic multi-seed validation only. Results do not establish external evaluator superiority.",
    }
    (output_root / "controlled_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_root
