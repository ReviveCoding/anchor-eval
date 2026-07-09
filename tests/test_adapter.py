import json
from pathlib import Path

from evalforge.adapters import normalize_judgebench_jsonl
from evalforge.data import load_pairs


def test_normalize_judgebench_schema(tmp_path: Path) -> None:
    raw = tmp_path / "judgebench.jsonl"
    raw.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "original_id": "question-1",
                "source": "mmlu-pro-computer science",
                "question": "What is correct?",
                "response_model": "example-model",
                "response_A": "A response",
                "response_B": "B response",
                "label": "B>A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.csv"
    normalize_judgebench_jsonl(raw, output)
    frame = load_pairs(output)
    assert frame.loc[0, "split"] == "final"
    assert frame.loc[0, "gold_preference"] == "B"
    assert frame.loc[0, "source_group"] == "judgebench::question-1"
    assert frame.loc[0, "raw_judge_status"] == "unscored"


def test_normalize_judgebench_uses_pair_id_when_original_id_is_null(tmp_path: Path) -> None:
    raw = tmp_path / "judgebench_null.jsonl"
    raw.write_text(
        json.dumps(
            {
                "pair_id": "pair-null",
                "original_id": None,
                "source": "livecodebench",
                "question": "What is correct?",
                "response_A": "A response",
                "response_B": "B response",
                "label": "A>B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized_null.csv"
    normalize_judgebench_jsonl(raw, output)
    frame = load_pairs(output)
    assert frame.loc[0, "source_group"] == "judgebench::pair-null"
