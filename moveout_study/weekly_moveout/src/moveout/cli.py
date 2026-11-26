"""Move-out study source summary.
Purpose: Provide the command-line interface for preparation, training, reconciliation, reporting, and prediction.
Inputs: Parses subcommands, validates required paths/options, and dispatches to the shared data/model pipelines.
Outputs: Returns process status codes and JSON summaries so batch runners can detect partial or complete failures.
Workflow: Model-specific imports are delayed until needed to keep lightweight commands fast.
Method note: Expected user/data errors are reported cleanly on standard error.
"""

import argparse
from pathlib import Path
import json
import sys
import pandas as pd
from .config import load_config
from .data import prepare

def main(argv=None):
    """Execute the complete analysis workflow in dependency order and leave a self-contained plot/table directory.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    # Expose each major workflow stage as an explicit subcommand for reproducible batch execution.
    parser = argparse.ArgumentParser(description="Weekly move-out and survival model comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Validate, build broad features, endpoints and deterministic folds")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--config"); p.add_argument("--endpoint-overrides")
    for name in ["run-weekly", "run-survival"]:
        p = sub.add_parser(name); p.add_argument("--prepared", required=True); p.add_argument("--out", required=True)
        p.add_argument("--config"); p.add_argument("--repeat", type=int); p.add_argument("--fold", type=int)
        p.add_argument("--drop-group", choices=["commute", "income", "payments", "relocation", "household"])
        p.add_argument("--models", default=("constant,tenure_site,logistic,ffnn" if name == "run-weekly" else "km,classical_cox,pycox_linear,pycox_neural"))
        if name == "run-survival": p.add_argument("--gate"); p.add_argument("--exploratory", action="store_true")
    p = sub.add_parser("reconcile-midterm"); p.add_argument("--midterm-path", required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("assess-logistic-hazard"); p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--evidence-json")
    p = sub.add_parser("report"); p.add_argument("--run-dir", required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True)
    p.add_argument("--input", required=True); p.add_argument("--out", required=True)
    p.add_argument("--horizons", default="1,4,8,12")
    args = parser.parse_args(argv)
    # Convert expected operational errors into a clear message and nonzero process status.
    try:
        if args.command == "prepare":
            result = prepare(args.data, args.out, load_config(args.config), args.endpoint_overrides)
        elif args.command in {"run-weekly", "run-survival"}:
            from .pipeline import run_weekly, run_survival
            cfg = load_config(args.config); kinds = args.models.split(",")
            common = dict(repeat=args.repeat, fold=args.fold, drop_group=args.drop_group)
            if args.command == "run-weekly": run_weekly(args.prepared, args.out, cfg, kinds, **common)
            else: run_survival(args.prepared, args.out, cfg, kinds, gate=args.gate, exploratory=args.exploratory, **common)
            failures = json.loads((Path(args.out)/"failures.json").read_text())
            result = dict(output=args.out, failed_models=len(failures), status="partial_failure" if failures else "completed")
        elif args.command == "reconcile-midterm":
            from .reconciliation import reconcile
            result = reconcile(args.midterm_path, args.data, args.out)
        elif args.command == "assess-logistic-hazard":
            from .pipeline import assess_gate
            result = assess_gate(args.run_dir, args.out, args.evidence_json)
        elif args.command == "report":
            from .pipeline import report
            report(args.run_dir); result = dict(output=args.run_dir)
        # Prediction loads persisted adapters and preserves available identity columns in the output.
        elif args.command == "predict":
            from .models import Adapter
            frame = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
            model = Adapter.load(args.model); out = frame[[c for c in ["resident_id", "week_index"] if c in frame]].copy()
            if model.kind in {"reference", "logistic", "ffnn", "weighted_logistic", "weighted_ffnn"}:
                out["next_week_risk"] = model.predict_weekly_probability(frame)
            else:
                horizons = [float(h) for h in args.horizons.split(",")]
                survival = model.predict_survival(frame, horizons)
                for j, horizon in enumerate(horizons): out[f"risk_{horizon:g}_weeks"] = 1-survival[:, j]
            out.to_csv(args.out, index=False); result = dict(output=args.out)
        print(json.dumps(result, indent=2, default=str))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr); return 2
    return 1 if result.get("failed_models", 0) else 0

if __name__ == "__main__": raise SystemExit(main())
