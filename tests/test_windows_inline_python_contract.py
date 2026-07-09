from __future__ import annotations

import re
from pathlib import Path


def test_inline_python_payloads_are_compilable_and_file_backed() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_anchor_eval_v15.ps1"
    text = runner.read_text(encoding="utf-8")

    # Windows PowerShell native-argument reconstruction corrupted the quotes in
    # Python -c payloads in v1.2. The runner must therefore never use -c.
    assert '"-c"' not in text
    assert "function Write-InlinePythonScript" in text
    assert "function Invoke-InlinePythonScript" in text
    assert "System.Text.UTF8Encoding($false)" in text
    assert '$InlinePythonRoot = Join-Path $ExecutionRoot "inline_python"' in text

    payloads = re.findall(r"\$[A-Za-z][A-Za-z0-9_]*\s*=\s*@'\n(.*?)\n'@", text, flags=re.DOTALL)
    assert payloads, "Expected inline Python payloads in the Windows runner."
    for index, payload in enumerate(payloads):
        compile(payload, f"inline_payload_{index}.py", "exec")


def test_all_checked_in_powershell_files_have_balanced_here_strings_and_clean_continuations() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "scripts").glob("*.ps1"))
    assert scripts
    for script in scripts:
        lines = script.read_text(encoding="utf-8").splitlines()
        assert sum(line.strip() == "'@" for line in lines) == sum(
            line.rstrip().endswith("@'") for line in lines
        ), script.name
        assert sum(line.strip() == '"@' for line in lines) == sum(
            line.rstrip().endswith('@"') for line in lines
        ), script.name
        assert not [
            line_number
            for line_number, line in enumerate(lines, start=1)
            if line.rstrip().endswith("`") and line != line.rstrip()
        ], script.name
