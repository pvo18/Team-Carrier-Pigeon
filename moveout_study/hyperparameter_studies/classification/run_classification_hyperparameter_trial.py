#!/usr/bin/env python3
"""Move-out study source summary.
Purpose: Run one controlled one-factor-at-a-time hyperparameter trial for the November 15 weekly models.
Inputs: Accepts one named variant, overlays its single change on the fixed study configuration, and uses the prepared cohort.
Outputs: Writes predictions, fold metrics, comparisons, manifests, and failure records to a variant-specific result directory.
Workflow: Keeping all non-target settings fixed makes each result attributable to the named hyperparameter change.
Method note: The script refuses unknown variants and existing output directories to protect experiment traceability.
"""


from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# The package is intentionally self-contained.  Resolving every path from this
# script means the experiment works after the ZIP is moved or extracted on a
# different computer; it no longer depends on the original analysis workspace.
CODE = ROOT / "source" / "weekly_moveout"
sys.path.insert(0, str(CODE / "src"))

from moveout.pipeline import run_weekly  # noqa: E402


BASE_CONFIG = ROOT / "config" / "20261115.yaml"
PREPARED = ROOT / "prepared"
OUTPUT_ROOT = ROOT / "new_trial_results"


# Each entry changes exactly one model setting from the archived baseline configuration.
VARIANTS = {
    # Logistic: one-factor-at-a-time changes from the archived selection procedure.
    "log_lr_00003": ("logistic", {"learning_rate": 0.0003}),
    "log_lr_0003": ("logistic", {"learning_rate": 0.003}),
    "log_wd_0": ("logistic", {"weight_decays": [0.0]}),
    "log_wd_000001": ("logistic", {"weight_decays": [0.00001]}),
    "log_wd_0001": ("logistic", {"weight_decays": [0.001]}),
    "log_batch_32": ("logistic", {"batch_size": 32}),
    "log_batch_128": ("logistic", {"batch_size": 128}),
    "log_patience_10": ("logistic", {"patience": 10}),
    "log_patience_50": ("logistic", {"patience": 50}),
    "log_weight_sqrt": ("weighted_logistic", {"class_weight_mode": "sqrt"}),
    "log_weight_full": ("weighted_logistic", {"class_weight_mode": "full"}),

    # FFNN: one-factor-at-a-time changes from the archived selection procedure.
    "nn_arch_16": ("ffnn", {"hidden_layers": [[16]]}),
    "nn_arch_8_4": ("ffnn", {"hidden_layers": [[8, 4]]}),
    "nn_arch_16_8": ("ffnn", {"hidden_layers": [[16, 8]]}),
    "nn_dropout_0": ("ffnn", {"dropout": 0.0}),
    "nn_dropout_01": ("ffnn", {"dropout": 0.1}),
    "nn_dropout_04": ("ffnn", {"dropout": 0.4}),
    "nn_lr_00003": ("ffnn", {"learning_rate": 0.0003}),
    "nn_lr_0003": ("ffnn", {"learning_rate": 0.003}),
    "nn_wd_0": ("ffnn", {"weight_decays": [0.0]}),
    "nn_wd_000001": ("ffnn", {"weight_decays": [0.00001]}),
    "nn_wd_0001": ("ffnn", {"weight_decays": [0.001]}),
    "nn_batch_32": ("ffnn", {"batch_size": 32}),
    "nn_batch_128": ("ffnn", {"batch_size": 128}),
    "nn_patience_10": ("ffnn", {"patience": 10}),
    "nn_patience_50": ("ffnn", {"patience": 50}),
    "nn_ensemble_1": ("ffnn", {"neural_seeds": [42]}),
    "nn_ensemble_3": ("ffnn", {"neural_seeds": [42, 43, 44]}),
    "nn_weight_sqrt": ("weighted_ffnn", {"class_weight_mode": "sqrt"}),
    "nn_weight_full": ("weighted_ffnn", {"class_weight_mode": "full"}),
}


def main() -> int:
    """Run one named controlled trial and write its complete result directory.

    The required command-line argument must be one of the keys in ``VARIANTS``.
    Returning ``2`` indicates invalid usage; returning ``0`` indicates that the
    model pipeline completed and wrote its normal artifacts.
    """
    # Reject misspelled or undocumented trials instead of silently running an
    # unintended configuration.  Printing the accepted names also makes the
    # script function as a simple experiment catalog.
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print("usage: hyperparameter_study.py VARIANT", file=sys.stderr)
        print("\n".join(sorted(VARIANTS)), file=sys.stderr)
        return 2
    name = sys.argv[1]
    kind, changes = VARIANTS[name]

    # Reload the fixed configuration for every invocation so trials cannot contaminate one another.
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg = copy.deepcopy(cfg)

    # Apply only the single documented variation.  All data, folds,
    # preprocessing, random seeds, metrics, and remaining model settings stay
    # fixed, which is the basis for attributing any difference to this factor.
    cfg.update(changes)

    # Each trial receives a separate directory.  Existing folders are kept so
    # rerunning one setting cannot overwrite evidence from an earlier run.
    out = OUTPUT_ROOT / name
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing trial results: {out}. "
            "Move or rename that directory before intentionally rerunning it."
        )
    out.mkdir(parents=True, exist_ok=False)

    # Write the changed value before fitting.  This small manifest provides a
    # human-readable audit trail even if model training later fails.
    (out / "variation.json").write_text(json.dumps({
        "variant": name,
        "model_kind": kind,
        "changes": changes,
        "base_config": str(BASE_CONFIG),
    }, indent=2))
    # Use the same prepared cohort, household-held-out folds, training-only
    # preprocessing, metric calculations, and baseline comparisons as the
    # qualified study.  ``run_weekly`` writes predictions and fold metrics.
    run_weekly(PREPARED, out, cfg, [kind])
    print(json.dumps({"variant": name, "output": str(out), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
