from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_import_does_not_eagerly_import_visualization_or_gpu_modules() -> None:
    code = """
import sys
import evalforge.cli
assert 'matplotlib.pyplot' not in sys.modules
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "PASS"


def test_lightweight_cli_commands_run_without_visualization_import(tmp_path: Path) -> None:
    output = tmp_path / "smoke.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "evalforge", "write-smoke-config", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert result.stdout.strip() == str(output)
