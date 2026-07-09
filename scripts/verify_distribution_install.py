"""Install a built wheel or source distribution in a clean venv and run EvalForge outside the source tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

PATTERNS = {"wheel": "*.whl", "sdist": "*.tar.gz"}


def _venv_executable(root: Path, name: str) -> Path:
    directory = root / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    return directory / f"{name}{suffix}"


def _run(command: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _writable_pip_cache(env: dict[str, str]) -> str:
    """Return a persistent writable pip cache without bypassing a configured CI cache."""
    configured = env.get("PIP_CACHE_DIR")
    candidates = (
        [Path(configured)]
        if configured
        else [Path.home() / ".cache" / "pip", Path(tempfile.gettempdir()) / "evalforge-pip-cache"]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".evalforge-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return str(candidate)
        except OSError:
            continue
    raise RuntimeError("Could not create a writable persistent pip cache directory.")


def _select_distribution(dist_dir: Path, distribution: str) -> Path:
    pattern = PATTERNS[distribution]
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {distribution} matching {pattern!r} in {dist_dir}, found {len(matches)}."
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a clean wheel or source-distribution install outside the source tree."
    )
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--distribution", choices=sorted(PATTERNS), required=True)
    parser.add_argument(
        "--config",
        help="Optional local config to copy into the isolated run. Mutually exclusive with --packaged-smoke-config.",
    )
    parser.add_argument(
        "--packaged-smoke-config",
        action="store_true",
        help="Export the installed package's bundled smoke config instead of borrowing a source-tree config.",
    )
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    if bool(args.config) == bool(args.packaged_smoke_config):
        raise ValueError("Supply exactly one of --config or --packaged-smoke-config.")
    dist_dir = Path(args.dist_dir).resolve()
    distribution = _select_distribution(dist_dir, args.distribution)
    config = Path(args.config).resolve() if args.config else None
    if config is not None and not config.exists():
        raise FileNotFoundError(config)

    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    venv_dir = work_dir / "venv"
    install_env = os.environ.copy()
    # Keep the isolated verifier non-interactive and give pip a writable cache even
    # when it runs from a constrained CI/sandbox account.
    install_env.setdefault("PIP_NO_INPUT", "1")
    install_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    install_env["PIP_CACHE_DIR"] = _writable_pip_cache(install_env)
    # Keep the installed smoke run fully headless and avoid sharing a global
    # Matplotlib font-cache lock with parallel CI jobs or local processes.
    run_env = install_env.copy()
    run_env["MPLBACKEND"] = "Agg"
    run_env["MPLCONFIGDIR"] = str(work_dir / ".mplconfig")
    Path(run_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    python = _venv_executable(venv_dir, "python")
    cli = _venv_executable(venv_dir, "evalforge")

    _run(
        [str(python), "-m", "pip", "install", "--prefer-binary", str(distribution)],
        work_dir,
        env=install_env,
    )
    _run([str(python), "-m", "pip", "check"], work_dir, env=install_env)
    if not cli.exists():
        raise FileNotFoundError(f"Installed CLI was not created: {cli}")

    doctor_json = subprocess.check_output(
        [str(cli), "doctor", "--json"], cwd=work_dir, text=True, env=run_env
    )
    doctor = json.loads(doctor_json)
    module_file = Path(str(doctor["module_file"])).resolve()
    if "site-packages" not in module_file.as_posix().lower():
        raise RuntimeError(f"Expected installed package in site-packages, got {module_file}")
    if str(doctor["python_executable"]) != str(python):
        raise RuntimeError(
            "EvalForge CLI was not executed with the clean verification environment."
        )

    run_dir = work_dir / "outside_source_tree"
    run_dir.mkdir()
    config_copy = run_dir / "smoke.yaml"
    if config is not None:
        shutil.copy2(config, config_copy)
    else:
        _run([str(cli), "write-smoke-config", "--output", str(config_copy)], run_dir, env=run_env)
    _run(
        [
            str(cli),
            "run",
            "--config",
            str(config_copy),
            "--set",
            "run.output_root=outputs",
            "--set",
            f"run.run_name={args.distribution}_smoke",
        ],
        run_dir,
        env=run_env,
    )
    published = sorted(
        (run_dir / "outputs").glob(f"{args.distribution}_smoke_*/numeric/metrics.json")
    )
    if len(published) != 1:
        raise RuntimeError(
            "Expected exactly one published metrics artifact from the installed distribution."
        )
    partial = list((run_dir / "outputs").glob(".*.partial"))
    if partial:
        raise RuntimeError(
            f"Installed-distribution smoke left unpublished staging directories: {partial}"
        )

    (work_dir / "distribution_validation.json").write_text(
        json.dumps(
            {
                "distribution": distribution.name,
                "distribution_kind": args.distribution,
                "doctor": doctor,
                "metrics_path": str(published[0].relative_to(work_dir)),
                "config_origin": "packaged_resource"
                if args.packaged_smoke_config
                else "caller_supplied",
                "status": "PASS",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
