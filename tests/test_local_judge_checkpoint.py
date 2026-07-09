from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evalforge.data import build_synthetic_fixture, load_pairs
from evalforge.local_judge import (
    LocalPairwiseJudge,
    score_local_judge_frame,
    validate_completed_local_judge_scoring,
)


class _FakeJudge:
    init_calls = 0

    def __init__(self, model_id: str, **_: object) -> None:
        type(self).init_calls += 1
        self.model_id = model_id

    def predict_p_a(self, frame):
        return np.clip(frame["raw_judge_p_a"].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)


def test_checkpointed_local_judge_scoring_resumes_without_new_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evalforge.local_judge.LocalPairwiseJudge", _FakeJudge)
    _FakeJudge.init_calls = 0
    source = tmp_path / "pairs.csv"
    build_synthetic_fixture(source, groups_per_domain=4)
    checkpoint_dir = tmp_path / "checkpoints"
    first_output = tmp_path / "scored_first.csv"

    score_local_judge_frame(
        source,
        first_output,
        "fake-judge",
        chunk_size=5,
        checkpoint_dir=checkpoint_dir,
    )
    assert _FakeJudge.init_calls == 1
    first = load_pairs(first_output)
    assert set(first["raw_judge_status"]) == {"local_scored"}
    assert len(list(checkpoint_dir.glob("chunk_*.csv"))) >= 2

    resumed_output = tmp_path / "scored_resumed.csv"
    score_local_judge_frame(
        source,
        resumed_output,
        "fake-judge",
        chunk_size=5,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )
    assert _FakeJudge.init_calls == 1
    resumed = load_pairs(resumed_output)
    assert resumed["item_id"].tolist() == first["item_id"].tolist()

    with pytest.raises(ValueError, match="Checkpoint manifest"):
        score_local_judge_frame(
            source,
            tmp_path / "bad_resume.csv",
            "fake-judge",
            chunk_size=6,
            checkpoint_dir=checkpoint_dir,
            resume=True,
        )


def test_completed_score_validation_rejects_stale_or_tampered_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evalforge.local_judge.LocalPairwiseJudge", _FakeJudge)
    _FakeJudge.init_calls = 0
    source = tmp_path / "pairs.csv"
    build_synthetic_fixture(source, groups_per_domain=4)
    checkpoint_dir = tmp_path / "checkpoints"
    output = tmp_path / "scored.csv"

    score_local_judge_frame(
        source,
        output,
        "fake-judge",
        chunk_size=5,
        checkpoint_dir=checkpoint_dir,
    )
    result = validate_completed_local_judge_scoring(
        source,
        output,
        "fake-judge",
        chunk_size=5,
        checkpoint_dir=checkpoint_dir,
    )
    assert result["status"] == "VALID"

    output.write_bytes(output.read_bytes() + b"\n# tamper")
    with pytest.raises(ValueError, match="checksum"):
        validate_completed_local_judge_scoring(
            source,
            output,
            "fake-judge",
            chunk_size=5,
            checkpoint_dir=checkpoint_dir,
        )


def test_windows_safe_local_judge_default_is_not_4bit() -> None:
    assert LocalPairwiseJudge.__dataclass_fields__["load_in_4bit"].default is False
