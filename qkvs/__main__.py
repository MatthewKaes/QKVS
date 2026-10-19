from __future__ import annotations

import argparse

from qkvs.harness import (
    dump_json,
    export_snapshot_data,
    load_run,
    print_status,
    print_summary,
    run_experiment,
    status,
    validate_config_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qkvs",
        description="QKVS causal heat-field research harness",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="execute or resume a configured experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument(
        "--snapshot-csv",
        help="append completed snapshot feasibility cells directly to this CSV",
    )

    validate = commands.add_parser(
        "validate",
        help="validate configuration without model execution",
    )
    validate.add_argument("--config", required=True)

    status_parser = commands.add_parser("status", help="read progress from a saved run")
    status_parser.add_argument("--run", required=True)
    status_parser.add_argument("--json", action="store_true")

    summarize = commands.add_parser("summarize", help="summarize measurements from a saved run")
    summarize.add_argument("--run", required=True)
    summarize.add_argument("--json", action="store_true")

    export = commands.add_parser(
        "export-snapshot",
        help="export a completed snapshot study as compact CSV research data",
    )
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--compress-trials",
        action="store_true",
        help="publish the candidate table as deterministic H1a_trials.csv.gz",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "validate":
        report = validate_config_file(args.config)
        dump_json(report)
        if not report["valid"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        value = status(args.run)
        dump_json(value) if args.json else print_status(value)
        return
    if args.command == "summarize":
        result = load_run(args.run)
        dump_json(result["summary"]) if args.json else print_summary(result)
        return
    if args.command == "export-snapshot":
        manifest = export_snapshot_data(
            args.run,
            args.output,
            compress_trials=args.compress_trials,
        )
        trial_record = next(
            record for name, record in manifest["csv"].items() if name.startswith("H1a_trials.csv")
        )
        print(f"Exported {trial_record['rows']} candidates to {args.output}")
        return
    if args.command == "run":
        _, result = run_experiment(
            args.config,
            device=args.device,
            snapshot_csv=args.snapshot_csv,
        )
        print_summary(result)
        return
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
