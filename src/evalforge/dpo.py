from __future__ import annotations

import inspect
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DPO_REQUIRED_COLUMNS = {"item_id", "source_group", "domain", "prompt", "chosen", "rejected"}


def _validate_dpo_boundary(predictions: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    if not set(predictions["item_id"]).issubset(set(full_frame["item_id"])):
        raise ValueError("Predictions contain unknown item_id values.")
    lookup = full_frame.set_index("item_id")
    splits = lookup.loc[predictions["item_id"], "split"].astype(str)
    invalid = sorted(set(splits) - {"dpo"})
    if invalid:
        raise ValueError(
            f"Only isolated dpo split items may enter preference training; found {invalid}."
        )


def build_preference_training_pairs(
    predictions: pd.DataFrame,
    full_frame: pd.DataFrame,
    output_path: str | Path,
    candidate_id: str,
    require_auto_accept: bool,
    max_pairs: int | None = None,
    seed: int = 1729,
) -> pd.DataFrame:
    """Materialize a DPO-ready pool exclusively from the protected dpo split."""
    work = predictions.loc[predictions["candidate_id"] == candidate_id].copy()
    if require_auto_accept:
        work = work.loc[work["route"] == "AUTO_ACCEPT"].copy()
    if work.empty:
        raise ValueError(f"No eligible {candidate_id} rows available for preference training.")
    _validate_dpo_boundary(work, full_frame)
    lookup = full_frame.set_index("item_id")
    if max_pairs is not None and len(work) > max_pairs:
        rng = np.random.default_rng(seed)
        sampled_ids: list[str] = []
        for _, group in work.groupby("domain", sort=True):
            target = max(1, int(round(max_pairs * len(group) / len(work))))
            sampled_ids.extend(
                rng.choice(
                    group["item_id"].to_numpy(), size=min(target, len(group)), replace=False
                ).tolist()
            )
        sampled_ids = list(dict.fromkeys(sampled_ids))[:max_pairs]
        work = work.loc[work["item_id"].isin(sampled_ids)].copy()
    records: list[dict[str, str | float]] = []
    for row in work.itertuples(index=False):
        source = lookup.loc[row.item_id]
        chosen = str(source["response_a"] if row.prediction == "A" else source["response_b"])
        rejected = str(source["response_b"] if row.prediction == "A" else source["response_a"])
        records.append(
            {
                "item_id": str(row.item_id),
                "source_group": str(source["source_group"]),
                "domain": str(source["domain"]),
                "prompt": str(source["prompt"]),
                "chosen": chosen,
                "rejected": rejected,
                "evaluator_candidate_id": candidate_id,
                "evaluator_p_correct": float(row.p_correct),
            }
        )
    result = (
        pd.DataFrame(records)
        .sort_values(["domain", "source_group", "item_id"])
        .reset_index(drop=True)
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(path, orient="records", lines=True, force_ascii=False)
    return result


def build_equal_budget_label_pools(
    raw_predictions: pd.DataFrame,
    routed_predictions: pd.DataFrame,
    full_frame: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 1729,
) -> dict[str, object]:
    """Create raw-judge and routed-evaluator DPO pools on identical isolated item IDs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    routed = routed_predictions.loc[routed_predictions["route"] == "AUTO_ACCEPT"].copy()
    _validate_dpo_boundary(routed, full_frame)
    raw_path = output_dir / "raw_judge_equal_budget.jsonl"
    routed_path = output_dir / "routed_equal_budget.jsonl"
    if routed.empty:
        # No safe routed labels is a valid reliability outcome, not a pipeline crash.
        # Materialize explicit empty pools so the evidence contract stays complete while
        # downstream DPO validation correctly refuses to train on an empty dataset.
        raw_path.write_text("", encoding="utf-8")
        routed_path.write_text("", encoding="utf-8")
        return {
            "status": "unavailable_zero_routed_labels",
            "budget": 0,
            "pool_dir": str(output_dir.name),
            "raw_path": str(raw_path.name),
            "routed_path": str(routed_path.name),
            "raw_fingerprint_sha256": sha256(raw_path.read_bytes()).hexdigest(),
            "routed_fingerprint_sha256": sha256(routed_path.read_bytes()).hexdigest(),
            "same_item_ids": True,
            "claim_boundary": (
                "No routed labels met the safety policy. DPO training is unavailable, and this run "
                "must not be used for a DPO-transfer claim."
            ),
        }
    budget = int(len(routed))
    raw = raw_predictions.loc[raw_predictions["item_id"].isin(routed["item_id"])].copy()
    _validate_dpo_boundary(raw, full_frame)
    if len(raw) != budget:
        raise AssertionError("Raw and routed pools must share the same isolated DPO item set.")
    raw_pool = build_preference_training_pairs(
        raw,
        full_frame,
        raw_path,
        candidate_id="C1_raw_judge",
        require_auto_accept=False,
        max_pairs=budget,
        seed=seed,
    )
    routed_pool = build_preference_training_pairs(
        routed,
        full_frame,
        routed_path,
        candidate_id="C5_routed_production",
        require_auto_accept=True,
        max_pairs=budget,
        seed=seed,
    )
    if set(raw_pool["item_id"]) != set(routed_pool["item_id"]):
        raise AssertionError("Equal-budget label pools must contain exactly the same dpo item IDs.")
    return {
        "budget": budget,
        "pool_dir": str(output_dir.name),
        "raw_path": str(raw_path.name),
        "routed_path": str(routed_path.name),
        "raw_fingerprint_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "routed_fingerprint_sha256": sha256(routed_path.read_bytes()).hexdigest(),
        "same_item_ids": True,
        "claim_boundary": "Pools only. Equal-budget DPO training and independent downstream evaluation remain required.",
    }


def validate_dpo_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "experiment_id",
        "label_source",
        "model_id",
        "pairs_path",
        "output_dir",
        "seed",
        "max_length",
        "max_prompt_length",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"DPO config is missing: {sorted(missing)}")
    if str(config["label_source"]) not in {"raw_judge", "routed_evaluator"}:
        raise ValueError("label_source must be raw_judge or routed_evaluator.")
    if int(config["max_prompt_length"]) >= int(config["max_length"]):
        raise ValueError("max_prompt_length must be smaller than max_length.")
    pairs_path = Path(str(config["pairs_path"]))
    if not pairs_path.exists():
        raise FileNotFoundError(f"DPO pairs file does not exist: {pairs_path}")
    pairs = pd.read_json(pairs_path, lines=True)
    missing_columns = DPO_REQUIRED_COLUMNS.difference(pairs.columns)
    if missing_columns:
        raise ValueError(f"DPO pairs file is missing: {sorted(missing_columns)}")
    if pairs["item_id"].duplicated().any() or pairs["source_group"].duplicated().any():
        raise ValueError("DPO pairs must have unique item_id and source_group values.")
    if (
        pairs[["prompt", "chosen", "rejected"]]
        .astype(str)
        .apply(lambda col: col.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError("DPO prompt, chosen, and rejected fields must be non-empty.")
    destination = Path(str(config["output_dir"]))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "DPO output_dir must be absent or empty to avoid overwriting an experiment."
        )
    return config


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    """Return a single token-id sequence without truncation."""

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Tokenizer returned more than one sequence for one text input.")
        token_ids = token_ids[0]
    return list(token_ids)


def audit_dpo_sequence_contract(
    pairs: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, int]:
    """Fail closed when a frozen DPO pair would require implicit truncation.

    TRL 0.29 removed ``max_prompt_length`` from ``DPOConfig``. The project keeps
    that field as an explicit experiment contract, then verifies every frozen
    row fits both the prompt and complete-sequence budgets before trainer
    construction. This preserves the original no-silent-truncation guarantee.
    """

    if max_prompt_length >= max_length:
        raise ValueError("max_prompt_length must be smaller than max_length.")
    eos_token = str(getattr(tokenizer, "eos_token", "") or "")
    violations: list[dict[str, int | str]] = []
    max_prompt_tokens = 0
    max_chosen_tokens = 0
    max_rejected_tokens = 0

    for row in pairs.itertuples(index=False):
        prompt = str(row.prompt)
        chosen = str(row.chosen)
        rejected = str(row.rejected)
        if eos_token:
            if not chosen.endswith(eos_token):
                chosen = chosen + eos_token
            if not rejected.endswith(eos_token):
                rejected = rejected + eos_token

        prompt_tokens = len(_token_ids(tokenizer, prompt))
        chosen_tokens = len(_token_ids(tokenizer, prompt + chosen))
        rejected_tokens = len(_token_ids(tokenizer, prompt + rejected))
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        max_chosen_tokens = max(max_chosen_tokens, chosen_tokens)
        max_rejected_tokens = max(max_rejected_tokens, rejected_tokens)

        if (
            prompt_tokens > max_prompt_length
            or chosen_tokens > max_length
            or rejected_tokens > max_length
        ):
            violations.append(
                {
                    "item_id": str(row.item_id),
                    "prompt_tokens": prompt_tokens,
                    "chosen_total_tokens": chosen_tokens,
                    "rejected_total_tokens": rejected_tokens,
                }
            )

    if violations:
        preview = violations[:5]
        raise ValueError(
            "Frozen DPO pool violates the sequence-length contract before trainer construction. "
            f"max_length={max_length}, max_prompt_length={max_prompt_length}, "
            f"violations={len(violations)}, examples={preview}"
        )

    return {
        "rows_checked": int(len(pairs)),
        "max_length": int(max_length),
        "max_prompt_length": int(max_prompt_length),
        "max_observed_prompt_tokens": int(max_prompt_tokens),
        "max_observed_chosen_total_tokens": int(max_chosen_tokens),
        "max_observed_rejected_total_tokens": int(max_rejected_tokens),
    }


def make_trl_dpo_config(
    dpo_config_class: Any,
    *,
    config: dict[str, Any],
    output_dir: str,
) -> tuple[Any, dict[str, str]]:
    """Create DPOConfig across legacy and TRL 0.29 sequence-length APIs."""

    parameters = set(inspect.signature(dpo_config_class).parameters)
    required = {"output_dir", "max_length"}
    missing = sorted(required.difference(parameters))
    if missing:
        raise RuntimeError(
            f"Installed TRL DPOConfig does not expose the required DPO arguments: {missing}."
        )

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config.get("batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation", 8)),
        "num_train_epochs": float(config.get("epochs", 1.0)),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_strategy": "no",
        "bf16": bool(config.get("bf16", True)),
        "fp16": False,
        "report_to": "none",
        "seed": int(config["seed"]),
        "max_length": int(config["max_length"]),
        "truncation_mode": "keep_start",
    }
    if "max_prompt_length" in parameters:
        kwargs["max_prompt_length"] = int(config["max_prompt_length"])
        sequence_length_mode = "native_max_prompt_length"
    else:
        sequence_length_mode = "strict_pretrainer_sequence_audit"

    kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    unsupported = sorted({"output_dir", "max_length"}.difference(kwargs))
    if unsupported:
        raise RuntimeError(
            "Installed TRL DPOConfig dropped critical arguments after compatibility filtering: "
            f"{unsupported}."
        )

    return dpo_config_class(**kwargs), {
        "trl_dpo_config_sequence_length_mode": sequence_length_mode,
        "trl_dpo_config_parameters": ",".join(sorted(parameters)),
    }


def make_trl_dpo_trainer(
    dpo_trainer_class: Any,
    *,
    model: Any,
    dpo_args: Any,
    dataset: Any,
    peft_config: Any,
    tokenizer: Any,
) -> tuple[Any, str]:
    """Build DPOTrainer while handling tokenizer API renames."""

    parameters = set(inspect.signature(dpo_trainer_class).parameters)
    kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
        processing_argument_name = "processing_class"
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
        processing_argument_name = "tokenizer"
    else:
        raise RuntimeError(
            "Installed TRL DPOTrainer exposes neither processing_class nor tokenizer."
        )

    unsupported = sorted(set(kwargs).difference(parameters))
    if unsupported:
        raise RuntimeError(
            f"Installed TRL DPOTrainer does not accept required execution arguments: {unsupported}."
        )
    return dpo_trainer_class(**kwargs), processing_argument_name


def run_optional_dpo(config: dict[str, Any]) -> None:
    """Run local DPO after explicit immutable-pair and API-compatibility checks."""

    validate_dpo_config(config)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover - GPU optional path
        raise RuntimeError(
            "Optional GPU dependencies are missing. Install `pip install -e .[gpu]` before DPO execution."
        ) from exc

    if not torch.cuda.is_available():  # pragma: no cover - GPU optional path
        raise RuntimeError(
            "DPO is GPU-only. CUDA was not detected, and no CPU fallback is allowed."
        )

    model_id = str(config["model_id"])
    pairs_path = str(config["pairs_path"])
    output_dir = str(config["output_dir"])
    pairs = pd.read_json(pairs_path, lines=True)

    compute_dtype = (
        torch.bfloat16
        if bool(config.get("bf16", True)) and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    sequence_length_audit = audit_dpo_sequence_contract(
        pairs,
        tokenizer,
        max_length=int(config["max_length"]),
        max_prompt_length=int(config["max_prompt_length"]),
    )

    dataset = load_dataset("json", data_files=pairs_path, split="train")
    model_kwargs: dict[str, Any] = {"device_map": {"": 0}}
    if bool(config.get("load_in_4bit", True)):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    if bool(config.get("load_in_4bit", True)):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    args, trl_api = make_trl_dpo_config(
        DPOConfig,
        config=config,
        output_dir=output_dir,
    )
    trainer, processing_argument_name = make_trl_dpo_trainer(
        DPOTrainer,
        model=model,
        dpo_args=args,
        dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    manifest = {
        **config,
        "pairs_fingerprint_sha256": sha256(Path(pairs_path).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "bf16": compute_dtype == torch.bfloat16,
        "sequence_length_audit": sequence_length_audit,
        "trl_api_compatibility": {
            **trl_api,
            "dpo_trainer_processing_argument": processing_argument_name,
        },
    }
    Path(output_dir, "dpo_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
