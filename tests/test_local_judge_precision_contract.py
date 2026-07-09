from __future__ import annotations

import sys
import types

import pytest

from evalforge.local_judge import LocalPairwiseJudge


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    chat_template = None

    @classmethod
    def from_pretrained(cls, _: str):
        return cls()

    def __call__(self, _: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": [1]}


class _FakeModel:
    config = types.SimpleNamespace(max_position_embeddings=4096)

    def eval(self):
        return self

    @staticmethod
    def get_input_embeddings():
        return types.SimpleNamespace(
            weight=types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        )


class _FakeAutoModel:
    received: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, _: str, **kwargs):
        cls.received = kwargs
        return _FakeModel()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_gpu_imports(monkeypatch: pytest.MonkeyPatch):
    torch = types.SimpleNamespace(cuda=_FakeCuda(), bfloat16="bf16", float16="fp16")
    transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_FakeAutoModel,
        AutoTokenizer=_FakeTokenizer,
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _FakeAutoModel.received = None
    yield
    _FakeAutoModel.received = None


def test_bf16_default_loader_contract_avoids_bitsandbytes(fake_gpu_imports) -> None:
    judge = LocalPairwiseJudge("fake-qwen")
    assert judge.load_in_4bit is False
    kwargs = _FakeAutoModel.received
    assert kwargs is not None
    assert kwargs["device_map"] == "auto"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["use_safetensors"] is True
    assert kwargs["torch_dtype"] == "bf16"
    assert kwargs["attn_implementation"] == "sdpa"
    assert "quantization_config" not in kwargs


def test_4bit_mode_remains_explicit_opt_in(fake_gpu_imports) -> None:
    LocalPairwiseJudge("fake-qwen", load_in_4bit=True)
    kwargs = _FakeAutoModel.received
    assert kwargs is not None
    config = kwargs["quantization_config"]
    assert isinstance(config, _FakeBitsAndBytesConfig)
    assert config.kwargs["load_in_4bit"] is True
    assert config.kwargs["bnb_4bit_compute_dtype"] == "bf16"
