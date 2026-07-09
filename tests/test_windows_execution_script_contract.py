from __future__ import annotations

import re
from pathlib import Path


def test_windows_execution_script_has_fail_closed_native_and_reuse_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_anchor_eval_v15.ps1"
    assert script.exists()
    assert (root / "scripts" / "verify_powershell_syntax.ps1").exists()
    assert (root / "scripts" / "verify_python_file_invocation.ps1").exists()
    assert not (root / "run_anchor_eval_v10.ps1").exists()
    assert not (root / "scripts" / "run_anchor_eval_v11.ps1").exists()
    text = script.read_text(encoding="utf-8")

    assert '$ExpectedVersion = "0.15.0"' in text
    assert "function Invoke-NativeChecked" in text
    assert "function Invoke-Python" in text
    assert "function Test-CompletedHeldoutEvaluation" in text
    assert '$JudgePrecision = "bf16"' in text
    assert '"probe-local-judge"' in text
    assert '"--precision", $JudgePrecision' in text
    assert "Probe Qwen BF16 model load in isolated child process" in text
    assert '"-m", "bitsandbytes"' not in text
    assert "$TorchProbePath = Write-InlinePythonScript" in text
    assert "return ($TorchExitCode -eq 0)" in text
    assert '"validate-heldout"' in text
    assert "Native command failed:" in text
    assert "function Write-InlinePythonScript" in text
    assert "function Invoke-InlinePythonScript" in text
    assert '"-c"' not in text
    assert "Get-CsvRowCount" in text
    assert "finqa_gpu_probe_${FinQAInputHash}_local_scored.csv" in text
    assert "finqa_derived_${FinQAInputHash}_local_scored.csv" in text
    assert "judgebench_final_${JudgeBenchInputHash}_local_scored.csv" in text
    assert "JudgeBench held-out directory is partial or invalid" in text
    assert "Start-Transcript" in text
    assert '$env:MPLBACKEND = "Agg"' in text
    assert '$env:MPLCONFIGDIR = Join-Path $ExecutionRoot ".mplconfig"' in text
    assert "Read JudgeBench git revision" in text
    assert "(& git" not in text
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'extend-exclude = ["outputs"]' in pyproject
    assert "force-exclude = true" in pyproject


def test_power_shell_expandable_strings_do_not_use_unbraced_variable_colons() -> None:
    root = Path(__file__).resolve().parents[1]
    known_scopes = {"global", "script", "local", "private", "env", "using", "function", "variable"}
    for script in (root / "scripts").glob("*.ps1"):
        for line_number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), start=1
        ):
            matches = re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*):", line)
            unsafe = [name for name in matches if name.lower() not in known_scopes]
            assert not unsafe, (
                f"{script.name}:{line_number} has unsafe $variable: interpolation: {line}"
            )


def test_standalone_syntax_verifier_uses_native_parser_api() -> None:
    root = Path(__file__).resolve().parents[1]
    verifier = (root / "scripts" / "verify_powershell_syntax.ps1").read_text(encoding="utf-8")
    assert "System.Management.Automation.Language.Parser]::ParseFile" in verifier
    assert "parseErrors.Count" in verifier
    assert "run_anchor_eval_v15.ps1" in verifier


def test_python_file_invocation_verifier_uses_a_script_file_not_dash_c() -> None:
    root = Path(__file__).resolve().parents[1]
    verifier = (root / "scripts" / "verify_python_file_invocation.ps1").read_text(encoding="utf-8")
    assert "python_file_invocation_probe.py" in verifier
    assert "-c" not in verifier
    assert "ANCHOREVAL_PYTHON_FILE_INVOKE=PASS" in verifier
