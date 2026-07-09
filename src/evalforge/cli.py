from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path

from . import __version__


def _build_finqa(args: argparse.Namespace) -> None:
    from .data import build_finqa_pair_rows

    source = Path(args.input)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            candidate = next(
                (name for name in archive.namelist() if name.endswith("train.json")), None
            )
            if candidate is None:
                raise ValueError("Could not find train.json within the supplied FinQA archive.")
            records = json.loads(archive.read(candidate).decode("utf-8"))
    else:
        records = json.loads(source.read_text(encoding="utf-8"))
    frame = build_finqa_pair_rows(records, max_examples=args.max_examples, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(output)


def _doctor_payload() -> dict[str, str]:
    return {
        "evalforge_version": __version__,
        "module_file": str(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="evalforge")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("build-fixture", help="Create deterministic CPU smoke fixture.")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=1729)
    fixture.add_argument("--groups-per-domain", type=int, default=48)

    finqa = sub.add_parser(
        "build-finqa-pairs", help="Create derived verifier-backed pairs from local FinQA data."
    )
    finqa.add_argument("--input", required=True)
    finqa.add_argument("--output", required=True)
    finqa.add_argument("--max-examples", type=int, default=250)
    finqa.add_argument("--seed", type=int, default=1729)

    judgebench = sub.add_parser(
        "normalize-judgebench",
        help="Normalize locally downloaded JudgeBench JSONL for immutable evaluation.",
    )
    judgebench.add_argument("--input", required=True)
    judgebench.add_argument("--output", required=True)

    score = sub.add_parser(
        "score-local-judge", help="GPU-only: populate raw original/swap/style local judge scores."
    )
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument(
        "--precision",
        choices=("bf16", "4bit"),
        default="bf16",
        help="BF16 is the Windows default; 4bit is opt-in after a successful model-load probe.",
    )
    score.add_argument("--no-4bit", action="store_true", help=argparse.SUPPRESS)
    score.add_argument("--max-input-tokens", type=int, default=2048)
    score.add_argument("--chunk-size", type=int, default=0)
    score.add_argument("--checkpoint-dir")
    score.add_argument("--resume", action="store_true")

    validate_score = sub.add_parser(
        "validate-local-score",
        help="Validate a completed checkpointed local-judge CSV before evidence reuse.",
    )
    validate_score.add_argument("--input", required=True)
    validate_score.add_argument("--output", required=True)
    validate_score.add_argument("--model-id", required=True)
    validate_score.add_argument(
        "--precision", choices=("bf16", "4bit"), default="bf16", help=argparse.SUPPRESS
    )
    validate_score.add_argument("--no-4bit", action="store_true", help=argparse.SUPPRESS)
    validate_score.add_argument("--max-input-tokens", type=int, default=2048)
    validate_score.add_argument("--chunk-size", type=int, default=0)
    validate_score.add_argument("--checkpoint-dir", required=True)

    probe_judge = sub.add_parser(
        "probe-local-judge",
        help="Load the exact local scoring model in a child process before scoring.",
    )
    probe_judge.add_argument("--model-id", required=True)
    probe_judge.add_argument("--precision", choices=("bf16", "4bit"), default="bf16")
    probe_judge.add_argument("--max-input-tokens", type=int, default=2048)

    heldout = sub.add_parser(
        "evaluate-heldout", help="Evaluate saved candidates on a locked final-only input."
    )
    heldout.add_argument("--bundle", required=True)
    heldout.add_argument("--input", required=True)
    heldout.add_argument("--output", required=True)
    heldout.add_argument("--source-type", default="external_benchmark")
    heldout.add_argument("--provenance-verified", action="store_true")
    heldout.add_argument("--provenance-manifest", required=True)
    heldout.add_argument(
        "--trust-candidate-bundle",
        action="store_true",
        help="Required acknowledgement: candidate bundles use joblib and must be locally trusted.",
    )

    validate_heldout = sub.add_parser(
        "validate-heldout",
        help="Validate a completed held-out evaluation before reusing its artifacts.",
    )
    validate_heldout.add_argument("--bundle", required=True)
    validate_heldout.add_argument("--input", required=True)
    validate_heldout.add_argument("--output", required=True)
    validate_heldout.add_argument("--source-type", default="external_benchmark")
    validate_heldout.add_argument("--provenance-verified", action="store_true")
    validate_heldout.add_argument("--provenance-manifest", required=True)

    doctor = sub.add_parser("doctor", help="Print installed package and interpreter diagnostics.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics.")

    export_smoke = sub.add_parser(
        "write-smoke-config",
        help="Write the packaged canonical CPU smoke config to a user-selected path.",
    )
    export_smoke.add_argument("--output", required=True)
    export_smoke.add_argument(
        "--force", action="store_true", help="Allow replacing an existing file."
    )

    execute = sub.add_parser(
        "run", help="Run candidate comparison, routing, artifacts, and report generation."
    )
    execute.add_argument("--config", required=True)
    execute.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")

    controlled = sub.add_parser(
        "controlled-validation", help="Run controlled synthetic multi-seed validation."
    )
    controlled.add_argument("--config", required=True)
    controlled.add_argument("--seeds", default="1729,1730,1731,1732,1733")
    controlled.add_argument("--output", required=True)
    controlled.add_argument("--groups-per-domain", type=int, default=48)

    clean_staging = sub.add_parser(
        "clean-staging",
        help="Inspect or explicitly remove stale unpublished .partial run directories.",
    )
    clean_staging.add_argument("--output-root", required=True)
    clean_staging.add_argument("--older-than-hours", type=float, default=24.0)
    clean_staging.add_argument("--apply", action="store_true")

    args = parser.parse_args()
    if args.command == "build-fixture":
        from .data import build_synthetic_fixture

        build_synthetic_fixture(
            args.output, seed=args.seed, groups_per_domain=args.groups_per_domain
        )
        print(args.output)
    elif args.command == "build-finqa-pairs":
        _build_finqa(args)
    elif args.command == "normalize-judgebench":
        from .adapters import normalize_judgebench_jsonl

        print(normalize_judgebench_jsonl(args.input, args.output))
    elif args.command == "score-local-judge":
        from .local_judge import score_local_judge_frame

        print(
            score_local_judge_frame(
                args.input,
                args.output,
                args.model_id,
                load_in_4bit=(args.precision == "4bit") and not args.no_4bit,
                max_input_tokens=args.max_input_tokens,
                chunk_size=args.chunk_size,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
            )
        )
    elif args.command == "validate-local-score":
        from .local_judge import validate_completed_local_judge_scoring

        print(
            json.dumps(
                validate_completed_local_judge_scoring(
                    args.input,
                    args.output,
                    args.model_id,
                    load_in_4bit=(args.precision == "4bit") and not args.no_4bit,
                    max_input_tokens=args.max_input_tokens,
                    chunk_size=args.chunk_size,
                    checkpoint_dir=args.checkpoint_dir,
                ),
                indent=2,
            )
        )
    elif args.command == "probe-local-judge":
        from .local_judge import probe_local_judge_load

        print(
            json.dumps(
                probe_local_judge_load(
                    args.model_id,
                    load_in_4bit=args.precision == "4bit",
                    max_input_tokens=args.max_input_tokens,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate-heldout":
        from .heldout import evaluate_heldout

        print(
            evaluate_heldout(
                args.bundle,
                args.input,
                args.output,
                source_type=args.source_type,
                provenance_verified=args.provenance_verified,
                provenance_manifest_path=args.provenance_manifest,
                trust_candidate_bundle=args.trust_candidate_bundle,
            )
        )
    elif args.command == "validate-heldout":
        from .heldout import validate_completed_heldout_evaluation

        print(
            json.dumps(
                validate_completed_heldout_evaluation(
                    args.bundle,
                    args.input,
                    args.output,
                    source_type=args.source_type,
                    provenance_verified=args.provenance_verified,
                    provenance_manifest_path=args.provenance_manifest,
                ),
                indent=2,
            )
        )
    elif args.command == "write-smoke-config":
        from .config import write_packaged_smoke_config

        print(write_packaged_smoke_config(args.output, overwrite=args.force))
    elif args.command == "doctor":
        payload = _doctor_payload()
        print(
            json.dumps(payload, indent=2)
            if args.json
            else "\n".join(f"{key}: {value}" for key, value in payload.items())
        )
    elif args.command == "run":
        from .config import apply_overrides, load_config
        from .pipeline import run

        print(run(apply_overrides(load_config(args.config), args.set)))
    elif args.command == "controlled-validation":
        from .config import load_config
        from .controlled_validation import run_controlled_validation

        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        print(
            run_controlled_validation(
                load_config(args.config), seeds, args.output, args.groups_per_domain
            )
        )
    elif args.command == "clean-staging":
        from .staging import cleanup_staging

        print(
            json.dumps(
                cleanup_staging(
                    args.output_root,
                    older_than_hours=args.older_than_hours,
                    apply=args.apply,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
