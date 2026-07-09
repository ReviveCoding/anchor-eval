from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import validate_pairs


def _infer_domain(source: str) -> str:
    source_lower = source.lower()
    if any(token in source_lower for token in ("math", "gsm", "aime", "mathqa")):
        return "objective_math"
    if any(token in source_lower for token in ("code", "human_eval", "mbpp", "coding")):
        return "objective_coding"
    if any(token in source_lower for token in ("mmlu", "knowledge", "science")):
        return "objective_knowledge"
    return "objective_reasoning"


def normalize_judgebench_jsonl(
    input_path: str | Path, output_path: str | Path, split: str = "final"
) -> pd.DataFrame:
    """Normalize a locally downloaded JudgeBench JSONL artifact into EvalForge's raw-score-ready schema.

    JudgeBench's public schema includes `pair_id`, `original_id`, `source`, `question`, `response_A`,
    `response_B`, and an objective pair label such as `B>A`. Raw judge probabilities are neutral placeholders
    until `evalforge score-local-judge` has run. The normalized file is intended for immutable evaluation only.
    """
    records: list[dict] = []
    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        label = str(item.get("label", "")).replace(" ", "")
        if label not in {"A>B", "B>A"}:
            raise ValueError(f"Unsupported JudgeBench label: {label!r}")
        pair_id = str(item.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("JudgeBench pair_id must be non-empty.")
        raw_original_id = item.get("original_id")
        original_id = (
            str(raw_original_id).strip()
            if raw_original_id is not None
            and str(raw_original_id).strip().lower() not in {"", "null", "none"}
            else pair_id
        )
        source = str(item.get("source", "judgebench"))
        records.append(
            {
                "item_id": f"judgebench::{pair_id}",
                "source_group": f"judgebench::{original_id}",
                "split": split,
                "domain": _infer_domain(source),
                "difficulty": "hard",
                "prompt": str(item.get("question", "")),
                "response_a": str(item.get("response_A", "")),
                "response_b": str(item.get("response_B", "")),
                "gold_preference": "A" if label == "A>B" else "B",
                "deterministic_verifiable": False,
                "verifier_preference": "",
                "raw_judge_p_a": 0.5,
                "raw_judge_p_a_swapped": 0.5,
                "raw_judge_p_a_style": 0.5,
                "raw_judge_p_a_style_swapped": 0.5,
                "raw_judge_status": "unscored",
                "raw_judge_model_id": "",
                "rubric_entropy": 0.5,
                "response_length_ratio": 1.0,
                "style_sensitivity": 0.0,
                "model_family_a": str(item.get("response_model", "unknown")),
                "model_family_b": str(item.get("response_model", "unknown")),
            }
        )
    frame = pd.DataFrame(records)
    validate_pairs(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
