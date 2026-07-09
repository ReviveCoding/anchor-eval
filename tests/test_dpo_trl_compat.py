from __future__ import annotations

from pathlib import Path

import pandas as pd

from evalforge.dpo import audit_dpo_sequence_contract, make_trl_dpo_config


class _Tokenizer:
    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> dict[str, list[int]]:
        del add_special_tokens, truncation
        return {"input_ids": list(range(len(text.split())))}


class _ModernDPOConfig:
    def __init__(
        self,
        output_dir: str,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        learning_rate: float,
        logging_steps: int,
        save_strategy: str,
        bf16: bool,
        fp16: bool,
        report_to: str,
        seed: int,
        max_length: int,
        truncation_mode: str,
    ) -> None:
        self.values = locals()


class _LegacyDPOConfig:
    def __init__(
        self,
        output_dir: str,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        learning_rate: float,
        logging_steps: int,
        save_strategy: str,
        bf16: bool,
        fp16: bool,
        report_to: str,
        seed: int,
        max_length: int,
        truncation_mode: str,
        max_prompt_length: int,
    ) -> None:
        self.values = locals()


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_id": "compat",
        "label_source": "raw_judge",
        "model_id": "model",
        "pairs_path": str(tmp_path / "pairs.jsonl"),
        "output_dir": str(tmp_path / "out"),
        "seed": 1729,
        "max_length": 12,
        "max_prompt_length": 6,
    }


def test_modern_trl_config_uses_strict_pretrainer_sequence_audit(tmp_path: Path) -> None:
    args, details = make_trl_dpo_config(
        _ModernDPOConfig,
        config=_config(tmp_path),
        output_dir=str(tmp_path / "trainer"),
    )

    assert details["trl_dpo_config_sequence_length_mode"] == "strict_pretrainer_sequence_audit"
    assert "max_prompt_length" not in args.values
    assert args.values["max_length"] == 12


def test_legacy_trl_config_keeps_native_max_prompt_length(tmp_path: Path) -> None:
    args, details = make_trl_dpo_config(
        _LegacyDPOConfig,
        config=_config(tmp_path),
        output_dir=str(tmp_path / "trainer"),
    )

    assert details["trl_dpo_config_sequence_length_mode"] == "native_max_prompt_length"
    assert args.values["max_prompt_length"] == 6


def test_sequence_contract_accepts_fitting_rows_and_rejects_overflow() -> None:
    tokenizer = _Tokenizer()
    pairs = pd.DataFrame(
        [
            {
                "item_id": "a",
                "prompt": "one two",
                "chosen": "three",
                "rejected": "four five",
            }
        ]
    )

    report = audit_dpo_sequence_contract(
        pairs,
        tokenizer,
        max_length=8,
        max_prompt_length=4,
    )

    assert report["rows_checked"] == 1
    assert report["max_observed_prompt_tokens"] == 2

    overflow = pairs.copy()
    overflow.loc[0, "prompt"] = "one two three four five"
    try:
        audit_dpo_sequence_contract(
            overflow,
            tokenizer,
            max_length=8,
            max_prompt_length=4,
        )
    except ValueError as exc:
        assert "sequence-length contract" in str(exc)
    else:
        raise AssertionError("Expected sequence-length contract overflow to fail closed.")
