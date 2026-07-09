from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_verifier_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "verify_distribution_install.py"
    spec = importlib.util.spec_from_file_location("evalforge_distribution_verifier", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_venv_paths_include_exe_suffix(monkeypatch, tmp_path: Path) -> None:
    module = _load_verifier_module()
    monkeypatch.setattr(module.os, "name", "nt")
    assert module._venv_executable(tmp_path, "python").as_posix().endswith("Scripts/python.exe")
    assert (
        module._venv_executable(tmp_path, "evalforge").as_posix().endswith("Scripts/evalforge.exe")
    )


def test_distribution_selector_requires_exactly_one_matching_file(tmp_path: Path) -> None:
    module = _load_verifier_module()
    wheel = tmp_path / "evalforge-0.5.0-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    assert module._select_distribution(tmp_path, "wheel") == wheel
    (tmp_path / "extra.whl").write_bytes(b"placeholder")
    try:
        module._select_distribution(tmp_path, "wheel")
    except ValueError as exc:
        assert "Expected exactly one wheel" in str(exc)
    else:
        raise AssertionError("The verifier must reject ambiguous distributions.")


def test_writable_pip_cache_survives_work_directory_cleanup(monkeypatch, tmp_path: Path) -> None:
    module = _load_verifier_module()
    configured = tmp_path / "persistent-pip-cache"
    cache = Path(module._writable_pip_cache({"PIP_CACHE_DIR": str(configured)}))
    assert cache == configured
    assert cache.exists()
    assert not (cache / ".evalforge-write-probe").exists()
