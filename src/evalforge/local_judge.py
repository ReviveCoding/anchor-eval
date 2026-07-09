from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dataset_fingerprint, load_pairs, validate_pairs


@dataclass
class LocalPairwiseJudge:
    model_id: str
    load_in_4bit: bool = False
    max_input_tokens: int = 2048

    def __post_init__(self) -> None:  # pragma: no cover - GPU-only path
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install `pip install -e .[gpu]` to use the local LLM judge adapter."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LocalPairwiseJudge is GPU-only; CUDA was not detected. No CPU fallback is allowed."
            )
        if self.max_input_tokens < 64:
            raise ValueError("max_input_tokens must be at least 64.")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        kwargs: dict[str, object] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if self.load_in_4bit:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            # Explicitly use PyTorch SDPA for the Windows BF16 fallback.
            kwargs["attn_implementation"] = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        embedding = self.model.get_input_embeddings()
        self._input_device = embedding.weight.device
        if self._input_device.type != "cuda":
            raise RuntimeError("LocalPairwiseJudge input embeddings were not placed on CUDA.")
        self._a_tokens = self._continuation_tokens("A")
        self._b_tokens = self._continuation_tokens("B")
        if not self._a_tokens or not self._b_tokens:
            raise RuntimeError("Tokenizer produced an empty A/B continuation token sequence.")
        model_limit_value = getattr(self.model.config, "max_position_embeddings", None)
        model_limit = int(model_limit_value or self.max_input_tokens)
        continuation_room = max(len(self._a_tokens), len(self._b_tokens))
        self._prompt_token_limit = min(self.max_input_tokens, model_limit) - continuation_room
        if self._prompt_token_limit < 32:
            raise RuntimeError(
                "Model context limit leaves insufficient room for pairwise-judge prompts."
            )

    def _continuation_tokens(self, text: str) -> list[int]:  # pragma: no cover - GPU-only path
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _render_prompt(self, task: str, response_a: str, response_b: str) -> str:
        user_content = (
            "You are a strict pairwise evaluator. Compare response A and response B for correctness, "
            "instruction following, grounding, and safety. Return exactly one letter: A or B.\n\n"
            f"Task:\n{task}\n\nResponse A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\nPreferred response:"
        )
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content

    def _shorten_text(self, text: str, token_budget: int) -> str:
        if token_budget < 8:
            return ""
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= token_budget:
            return text
        marker = self.tokenizer("\n[...truncated...]\n", add_special_tokens=False)["input_ids"]
        remaining = max(2, token_budget - len(marker))
        head = max(1, remaining // 2)
        tail = max(1, remaining - head)
        compact_ids = token_ids[:head] + marker + token_ids[-tail:]
        return self.tokenizer.decode(compact_ids, skip_special_tokens=True)

    def _prompt_ids(self, row: pd.Series):
        task = str(row.get("prompt", row["item_id"]))
        response_a = str(row["response_a"])
        response_b = str(row["response_b"])
        rendered = self._render_prompt(task, response_a, response_b)
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].shape[1] <= self._prompt_token_limit:
            return encoded["input_ids"].to(self._input_device)

        # Preserve the evaluator instruction, both alternatives, and the final decision cue.
        # A right-truncated prompt can silently remove response B or the answer position cue.
        for scale in (0.95, 0.75, 0.60, 0.45, 0.30, 0.20, 0.12):
            task_budget = max(24, int(self._prompt_token_limit * 0.18 * scale))
            response_budget = max(32, int(self._prompt_token_limit * 0.38 * scale))
            candidate = self._render_prompt(
                self._shorten_text(task, task_budget),
                self._shorten_text(response_a, response_budget),
                self._shorten_text(response_b, response_budget),
            )
            candidate_ids = self.tokenizer(
                candidate, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            if candidate_ids.shape[1] <= self._prompt_token_limit:
                return candidate_ids.to(self._input_device)
        raise RuntimeError(
            "Could not construct a bounded pairwise prompt without dropping the decision cue. "
            "Increase --max-input-tokens or shorten the input pair."
        )

    def _continuation_logprob(
        self, prompt_ids, continuation: list[int]
    ) -> float:  # pragma: no cover - GPU-only path
        torch = self._torch
        continuation_ids = torch.tensor(
            [continuation], device=self._input_device, dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
        with torch.inference_mode():
            logits = self.model(input_ids=full_ids).logits
        prompt_length = prompt_ids.shape[1]
        continuation_logits = logits[
            :, prompt_length - 1 : prompt_length - 1 + len(continuation), :
        ]
        log_probs = torch.log_softmax(continuation_logits.float(), dim=-1)
        selected = log_probs.gather(
            dim=-1,
            index=continuation_ids.unsqueeze(-1),
        ).squeeze(-1)
        return float(selected.sum().item())

    def predict_p_a(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover - GPU-only path
        scores: list[float] = []
        with self._torch.inference_mode():
            for _, row in frame.iterrows():
                prompt_ids = self._prompt_ids(row)
                log_a = self._continuation_logprob(prompt_ids, self._a_tokens)
                log_b = self._continuation_logprob(prompt_ids, self._b_tokens)
                scores.append(float(1.0 / (1.0 + np.exp(np.clip(log_b - log_a, -30.0, 30.0)))))
        return np.asarray(scores, dtype=float)


def probe_local_judge_load(
    model_id: str,
    load_in_4bit: bool = False,
    max_input_tokens: int = 2048,
) -> dict[str, object]:
    """Load the exact scoring model in an isolated process before scoring.

    The Windows runner executes this command in a child process. A native CUDA
    extension failure then produces a clean child exit code rather than leaving
    a partially initialized scoring process or corrupted checkpoint state.
    """
    judge = LocalPairwiseJudge(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        max_input_tokens=max_input_tokens,
    )
    torch = judge._torch
    return {
        "status": "PASS",
        "model_id": model_id,
        "precision_mode": "4bit"
        if load_in_4bit
        else ("bf16" if torch.cuda.is_bf16_supported() else "fp16"),
        "input_device": str(judge._input_device),
        "parameter_count": int(sum(parameter.numel() for parameter in judge.model.parameters())),
        "max_input_tokens": max_input_tokens,
    }


def _score_chunk(frame: pd.DataFrame, judge: LocalPairwiseJudge) -> pd.DataFrame:
    """Score the complete order/style presentation orbit for one input chunk."""
    scored = frame.copy()
    original = judge.predict_p_a(scored)

    swapped = scored.copy()
    swapped[["response_a", "response_b"]] = scored[["response_b", "response_a"]].to_numpy()
    swapped[["model_family_a", "model_family_b"]] = scored[
        ["model_family_b", "model_family_a"]
    ].to_numpy()
    swapped_scores = judge.predict_p_a(swapped)

    styled = scored.copy()
    styled["response_a"] = (
        "## Executive Summary\n" + scored["response_a"].astype(str) + "\n**Conclusion:** see above."
    )
    style_scores = judge.predict_p_a(styled)

    styled_swapped = swapped.copy()
    styled_swapped["response_a"] = (
        "## Executive Summary\n"
        + swapped["response_a"].astype(str)
        + "\n**Conclusion:** see above."
    )
    style_swapped_scores = judge.predict_p_a(styled_swapped)

    scored["raw_judge_p_a"] = np.clip(original, 1e-4, 1.0 - 1e-4)
    # The swapped channels score original B in the displayed A position. Downstream converts them.
    scored["raw_judge_p_a_swapped"] = np.clip(swapped_scores, 1e-4, 1.0 - 1e-4)
    scored["raw_judge_p_a_style"] = np.clip(style_scores, 1e-4, 1.0 - 1e-4)
    scored["raw_judge_p_a_style_swapped"] = np.clip(style_swapped_scores, 1e-4, 1.0 - 1e-4)
    scored["raw_judge_status"] = "local_scored"
    scored["raw_judge_model_id"] = judge.model_id
    validate_pairs(scored)
    return scored


def _chunk_plan(frame: pd.DataFrame, chunk_size: int) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for chunk_id, start in enumerate(range(0, len(frame), chunk_size)):
        stop = min(start + chunk_size, len(frame))
        item_ids = frame.iloc[start:stop]["item_id"].astype(str).tolist()
        plan.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "stop": stop,
                "row_count": stop - start,
                "item_id_sha256": sha256("\n".join(item_ids).encode("utf-8")).hexdigest(),
            }
        )
    return plan


def _checkpoint_payload(
    input_path: str | Path,
    frame: pd.DataFrame,
    model_id: str,
    load_in_4bit: bool,
    max_input_tokens: int,
    chunk_size: int,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "input_fingerprint_sha256": dataset_fingerprint(input_path),
        "row_count": int(len(frame)),
        "model_id": model_id,
        "load_in_4bit": bool(load_in_4bit),
        "max_input_tokens": int(max_input_tokens),
        "chunk_size": int(chunk_size),
        "chunks": _chunk_plan(frame, chunk_size),
    }


def _load_or_create_checkpoint_manifest(
    checkpoint_dir: Path,
    payload: dict[str, object],
    resume: bool,
) -> Path:
    manifest_path = checkpoint_dir / "scoring_checkpoint_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Checkpoint manifest does not match this scoring request. Use a new checkpoint directory "
                "or remove the stale directory after inspection."
            )
        if not resume:
            raise FileExistsError(
                "Checkpoint directory already contains a matching manifest. Pass --resume to reuse completed "
                "chunks or select a new checkpoint directory."
            )
        return manifest_path
    if resume:
        raise FileNotFoundError(
            "--resume was requested, but the scoring checkpoint manifest does not exist."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _validate_cached_chunk(path: Path, expected: pd.DataFrame, model_id: str) -> pd.DataFrame:
    cached = load_pairs(path)
    if cached["item_id"].astype(str).tolist() != expected["item_id"].astype(str).tolist():
        raise ValueError(f"Cached chunk item order does not match current input: {path}")
    if cached["raw_judge_status"].astype(str).ne("local_scored").any():
        raise ValueError(f"Cached chunk is not fully local_scored: {path}")
    if cached["raw_judge_model_id"].astype(str).ne(model_id).any():
        raise ValueError(f"Cached chunk was scored by another model: {path}")
    return cached


def validate_completed_local_judge_scoring(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str,
    load_in_4bit: bool = False,
    max_input_tokens: int = 2048,
    chunk_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, object]:
    """Validate a completed checkpointed local-judge artifact before reuse.

    A finished CSV is reusable only when the immutable request contract, output
    checksum, item order, score status, and judge model all match the current
    invocation. This prevents a stale but same-row-count CSV from silently
    entering a new evidence run.
    """
    if checkpoint_dir is None:
        raise ValueError("Completed-score validation requires checkpoint_dir.")

    input_frame = load_pairs(input_path)
    if input_frame.empty:
        raise ValueError("Local judge scoring requires at least one input row.")
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(input_frame)

    checkpoint_root = Path(checkpoint_dir)
    manifest_path = checkpoint_root / "scoring_checkpoint_manifest.json"
    completion_path = checkpoint_root / "scoring_completion_manifest.json"
    output = Path(output_path)
    if not manifest_path.exists() or not completion_path.exists() or not output.exists():
        raise FileNotFoundError(
            "Completed local-score reuse requires output CSV plus checkpoint and completion manifests."
        )

    expected = _checkpoint_payload(
        input_path,
        input_frame,
        model_id,
        load_in_4bit,
        max_input_tokens,
        chunk_size,
    )
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            "Completed scoring checkpoint does not match this input, model, quantization, token limit, "
            "or chunk plan. Use a new run ID or remove the stale evidence directory after inspection."
        )

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE":
        raise ValueError("Completed scoring manifest does not report COMPLETE status.")
    if completion.get("completed_chunk_count") != len(expected["chunks"]):
        raise ValueError("Completed scoring manifest has an unexpected chunk count.")
    output_fingerprint = dataset_fingerprint(output)
    if completion.get("output_fingerprint_sha256") != output_fingerprint:
        raise ValueError(
            "Completed scoring output checksum does not match its completion manifest."
        )

    scored = load_pairs(output)
    if len(scored) != len(input_frame):
        raise ValueError("Completed score row count does not match the current input.")
    if scored["item_id"].astype(str).tolist() != input_frame["item_id"].astype(str).tolist():
        raise ValueError("Completed score item order does not match the current input.")
    if scored["raw_judge_status"].astype(str).ne("local_scored").any():
        raise ValueError("Completed score contains rows that are not local_scored.")
    if scored["raw_judge_model_id"].astype(str).ne(model_id).any():
        raise ValueError("Completed score was produced by a different judge model.")

    return {
        "status": "VALID",
        "rows": int(len(scored)),
        "output_fingerprint_sha256": output_fingerprint,
        "checkpoint_manifest": str(manifest_path),
        "completion_manifest": str(completion_path),
    }


def score_local_judge_frame(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str,
    load_in_4bit: bool = False,
    max_input_tokens: int = 2048,
    chunk_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Populate raw local-judge channels with optional crash-safe chunk checkpoints.

    The model is loaded once per command. Completed chunks are only reused when their manifest,
    row order, local-score status, and model ID exactly match the new request.
    """
    frame = load_pairs(input_path)
    if frame.empty:
        raise ValueError("Local judge scoring requires at least one input row.")
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(frame)
    if checkpoint_dir is None and resume:
        raise ValueError("--resume requires --checkpoint-dir.")

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    plan = _chunk_plan(frame, chunk_size)
    if checkpoint_root is not None:
        payload = _checkpoint_payload(
            input_path,
            frame,
            model_id,
            load_in_4bit,
            max_input_tokens,
            chunk_size,
        )
        _load_or_create_checkpoint_manifest(checkpoint_root, payload, resume)

    chunks: list[pd.DataFrame] = []
    judge: LocalPairwiseJudge | None = None
    for entry in plan:
        chunk_id = int(entry["chunk_id"])
        start = int(entry["start"])
        stop = int(entry["stop"])
        expected = frame.iloc[start:stop].copy()
        chunk_path = (
            checkpoint_root / f"chunk_{chunk_id:05d}.csv" if checkpoint_root is not None else None
        )
        if chunk_path is not None and chunk_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Completed chunk already exists: {chunk_path}. Pass --resume or use a new checkpoint directory."
                )
            chunks.append(_validate_cached_chunk(chunk_path, expected, model_id))
            continue
        if judge is None:
            judge = LocalPairwiseJudge(
                model_id=model_id,
                load_in_4bit=load_in_4bit,
                max_input_tokens=max_input_tokens,
            )
        scored = _score_chunk(expected, judge)
        if chunk_path is not None:
            temporary_path = chunk_path.with_suffix(".csv.tmp")
            scored.to_csv(temporary_path, index=False)
            temporary_path.replace(chunk_path)
        chunks.append(scored)

    combined = pd.concat(chunks, ignore_index=True)
    if combined["item_id"].astype(str).tolist() != frame["item_id"].astype(str).tolist():
        raise AssertionError("Merged local-judge chunks do not preserve input item order.")
    validate_pairs(combined)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    combined.to_csv(temporary_output, index=False)
    temporary_output.replace(output)

    if checkpoint_root is not None:
        completion = {
            "output_path": str(output),
            "output_fingerprint_sha256": dataset_fingerprint(output),
            "completed_chunk_count": len(plan),
            "status": "COMPLETE",
        }
        (checkpoint_root / "scoring_completion_manifest.json").write_text(
            json.dumps(completion, indent=2), encoding="utf-8"
        )
    return output
