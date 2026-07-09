from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ci_contract_script_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/validate_ci_contract.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS:" in result.stdout
