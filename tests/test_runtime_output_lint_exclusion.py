from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_runtime_outputs_are_excluded_from_lint_and_format_even_when_explicitly_named() -> None:
    root = Path(__file__).resolve().parents[1]
    generated_root = root / "outputs" / "__ruff_runtime_contract__"
    generated_file = generated_root / "runtime_helper.py"
    generated_root.mkdir(parents=True, exist_ok=True)
    generated_file.write_text(
        "import sys\n\n# Deliberately not Ruff-clean: runtime helper artifact only.\n",
        encoding="utf-8",
    )
    try:
        commands = (
            [sys.executable, "-m", "ruff", "check", "."],
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            [sys.executable, "-m", "ruff", "check", str(generated_file)],
        )
        for command in commands:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
    finally:
        shutil.rmtree(generated_root, ignore_errors=True)
