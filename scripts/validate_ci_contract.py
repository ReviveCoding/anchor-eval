"""Fail fast when the checked-in GitHub Actions workflow loses critical EvalForge validation stages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REQUIRED_ACTIONS = {
    "actions/checkout@v6",
    "actions/setup-python@v6",
    "actions/upload-artifact@v4",
    "actions/download-artifact@v4",
}


def _step_values(steps: Iterable[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for step in steps:
        for key in ("uses", "run", "name"):
            value = step.get(key)
            if isinstance(value, str):
                values.add(value)
    return values


def _contains(values: set[str], text: str) -> bool:
    return any(text in value for value in values)


def _matrix_includes(job: dict[str, Any], expected: dict[str, str]) -> bool:
    matrix = job.get("strategy", {}).get("matrix", {})
    include = matrix.get("include", []) if isinstance(matrix, dict) else []
    return isinstance(include, list) and any(
        isinstance(row, dict) and all(str(row.get(key)) == value for key, value in expected.items())
        for row in include
    )


def main() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    payload = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(payload, dict):
        raise RuntimeError("CI workflow must parse to a mapping.")
    if "push" not in payload.get("on", {}) or "pull_request" not in payload.get("on", {}):
        raise RuntimeError("CI workflow must run on both push and pull_request.")
    if payload.get("permissions", {}).get("contents") != "read":
        raise RuntimeError("CI workflow must use least-privilege contents: read permissions.")
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        raise RuntimeError("CI workflow must define jobs.")
    for name in ("quality-and-smoke", "package-build", "package-install"):
        if name not in jobs:
            raise RuntimeError(f"CI workflow is missing required job: {name}")
    quality = jobs["quality-and-smoke"]
    package_build = jobs["package-build"]
    package_install = jobs["package-install"]
    if not _matrix_includes(quality, {"os": "ubuntu-latest", "python-version": "3.11"}):
        raise RuntimeError("quality-and-smoke must include Ubuntu Python 3.11.")
    if not _matrix_includes(quality, {"os": "ubuntu-latest", "python-version": "3.12"}):
        raise RuntimeError("quality-and-smoke must include Ubuntu Python 3.12.")
    if not _matrix_includes(quality, {"os": "ubuntu-latest", "python-version": "3.13"}):
        raise RuntimeError("quality-and-smoke must include Ubuntu Python 3.13.")
    if not _matrix_includes(quality, {"os": "windows-latest", "python-version": "3.11"}):
        raise RuntimeError("quality-and-smoke must include Windows Python 3.11.")
    if not _matrix_includes(package_install, {"os": "windows-latest", "distribution": "wheel"}):
        raise RuntimeError("package-install must include a Windows wheel installation.")
    if not _matrix_includes(package_install, {"os": "ubuntu-latest", "distribution": "sdist"}):
        raise RuntimeError(
            "package-install must include an Ubuntu source-distribution installation."
        )

    all_steps = []
    for job in (quality, package_build, package_install):
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            raise RuntimeError("Every required CI job must define a step list.")
        all_steps.extend(step for step in steps if isinstance(step, dict))
    values = _step_values(all_steps)
    missing_actions = sorted(action for action in REQUIRED_ACTIONS if action not in values)
    if missing_actions:
        raise RuntimeError(f"CI workflow is missing required actions: {missing_actions}")
    required_commands = (
        "python -m pip check",
        "python -m evalforge doctor --json",
        "python -m pytest --junitxml=junit.xml",
        "python -m evalforge run --config configs/smoke.yaml",
        "ruff check .",
        "ruff format --check .",
        "python -m build --sdist --wheel --outdir dist",
        "python -m twine check dist/*",
        "python scripts/verify_distribution_install.py",
        "--packaged-smoke-config",
        "verify_powershell_syntax.ps1",
        "Parse checked-in PowerShell scripts",
        "verify_python_file_invocation.ps1",
        "Verify Python file invocation contract",
    )
    missing_commands = [command for command in required_commands if not _contains(values, command)]
    if missing_commands:
        raise RuntimeError(
            f"CI workflow is missing required validation commands: {missing_commands}"
        )
    if package_install.get("needs") != "package-build":
        raise RuntimeError("package-install must consume distributions from package-build.")
    if payload.get("env", {}).get("PIP_NO_INPUT") != "1":
        raise RuntimeError("CI must set PIP_NO_INPUT=1 to avoid interactive installation stalls.")
    package_install_steps = package_install.get("steps", [])
    evidence_step = next(
        (
            step
            for step in package_install_steps
            if isinstance(step, dict) and step.get("name") == "Upload distribution-install evidence"
        ),
        None,
    )
    if not isinstance(evidence_step, dict):
        raise RuntimeError("package-install must upload slim distribution-install evidence.")
    evidence_path = str(evidence_step.get("with", {}).get("path", ""))
    if (
        "distribution_validation.json" not in evidence_path
        or "outside_source_tree/outputs/" not in evidence_path
    ):
        raise RuntimeError(
            "distribution-install evidence must include summary JSON and published outputs."
        )
    if "distribution-validation-${{ matrix.os }}-${{ matrix.distribution }}/\n" in evidence_path:
        raise RuntimeError(
            "CI must not upload the complete clean-install work directory or its virtualenv."
        )
    print(f"PASS: {workflow.relative_to(workflow.parents[2])} satisfies the EvalForge CI contract.")


if __name__ == "__main__":
    main()
