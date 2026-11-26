"""Run controlled hyperparameter trials for the tenant move-out survival models.

SUMMARY
-------
This script evaluates Classical Cox and Neural CoxPH models on the November 15
tenant snapshot.  Every hyperparameter setting uses the same five repeats of
five household-held-out folds, the same feature construction, and the same
12-week endpoint.  The Classical Cox trials vary only the L2 penalizer.  The
Neural CoxPH trials change one factor at a time from a documented baseline:
learning rate, hidden-layer architecture, dropout, weight decay, or early-
stopping patience.  This makes each reported change directly interpretable.

Each fold contains a training set, an inner validation set used for early
stopping, and an untouched outer test set used for the reported Brier score,
log loss, and AUC.  Tenant IDs—not weekly rows—define the splits, preventing
the same tenancy from appearing in both training and test data.

The analysis writes fold-level metrics, aggregate tables, baseline-relative
changes, a manifest, and a concise Markdown results summary.  Neural models
use one deterministic seed per fold so the requested grid is computationally
tractable; this limitation is stated in the results and the best setting can
later be confirmed with a multi-seed ensemble.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths once so worker processes import the recovered study package.
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR / "source" / "weekly_moveout"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "results"
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from moveout.config import load_config  # noqa: E402
from moveout.data import (  # noqa: E402
    build_master,
    landmark,
    make_endpoints,
    make_folds,
    read_sources,
    split_ids,
    structural_validation,
)
from moveout.metrics import horizon_metrics  # noqa: E402
from moveout.models import fit_survival, transformer  # noqa: E402


# The baseline settings match the original November survival configuration.
NEURAL_BASELINE = {
    "learning_rate": 0.001,
    "hidden": [8],
    "dropout": 0.2,
    "weight_decay": 0.0001,
    "patience": 25,
}


def classical_trials() -> list[dict]:
    """Return a focused L2-penalty grid for the Classical Cox model."""
    return [
        {
            "trial_id": f"cox_penalty_{penalty:g}",
            "factor": "penalizer",
            "value": penalty,
            "penalizer": penalty,
            "is_baseline": penalty == 0.1,
        }
        for penalty in [0.001, 0.01, 0.1, 1.0, 10.0]
    ]


def neural_trials() -> list[dict]:
    """Create one-factor-at-a-time Neural CoxPH trials around the baseline."""
    trials: list[dict] = []

    def add(trial_id: str, factor: str, value, **updates) -> None:
        """Copy the neural control, apply one change, and register the trial."""
        settings = copy.deepcopy(NEURAL_BASELINE)
        settings.update(updates)
        trials.append(
            {
                "trial_id": trial_id,
                "factor": factor,
                "value": value,
                "is_baseline": factor == "baseline",
                **settings,
            }
        )

    # Record the unchanged reference configuration once.
    add("neural_baseline", "baseline", "original")

    # Change one training or architecture choice at a time.
    add("neural_lr_0.0003", "learning_rate", 0.0003, learning_rate=0.0003)
    add("neural_lr_0.003", "learning_rate", 0.003, learning_rate=0.003)
    add("neural_hidden_4", "hidden", "[4]", hidden=[4])
    add("neural_hidden_16", "hidden", "[16]", hidden=[16])
    add("neural_hidden_16_8", "hidden", "[16,8]", hidden=[16, 8])
    add("neural_dropout_0", "dropout", 0.0, dropout=0.0)
    add("neural_dropout_0.4", "dropout", 0.4, dropout=0.4)
    add("neural_weight_decay_0", "weight_decay", 0.0, weight_decay=0.0)
    add("neural_weight_decay_0.001", "weight_decay", 0.001, weight_decay=0.001)
    add("neural_weight_decay_0.01", "weight_decay", 0.01, weight_decay=0.01)
    add("neural_patience_10", "patience", 10, patience=10)
    add("neural_patience_50", "patience", 50, patience=50)
    return trials


def prepare_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, dict]:
    """Rebuild the exact leakage-controlled feature table and household folds."""
    # This configuration is bundled beside the script so the recovered trial
    # is reproducible without any paths from the original working directory.
    config_path = SCRIPT_DIR / "config" / "20261115.yaml"
    config = load_config(config_path)
    residents, weekly, invoices, source_hashes = read_sources(DATA_DIR)

    validation = structural_validation(residents, weekly, config)
    if not validation["passed"]:
        raise ValueError(f"Source validation failed: {validation['errors']}")

    endpoints = make_endpoints(residents, weekly, config)
    master, features, _ = build_master(residents, weekly, invoices, config)
    master = master.merge(
        endpoints[["resident_id", "qualified", "duration", "event", "cluster_id"]],
        on="resident_id",
        validate="many_to_one",
    )
    folds = make_folds(master, config)
    metadata = {
        "config": config,
        "source_hashes": source_hashes,
        "validation": validation,
        "qualified_households": int(endpoints.qualified.sum()),
        "qualified_events": int(endpoints.loc[endpoints.qualified, "event"].sum()),
        "feature_count": len(features),
    }
    return master, endpoints, features, folds, metadata


def evaluate_one_fold(payload: dict) -> list[dict]:
    """Evaluate every trial on one fixed origin/repeat/fold split."""
    # Limit native math and Torch thread pools because folds run in parallel.
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    master = payload["master"]
    endpoints = payload["endpoints"]
    features = payload["features"]
    folds = payload["folds"]
    base_config = payload["config"]
    origin = payload["origin"]
    repeat = payload["repeat"]
    fold = payload["fold"]

    frame = landmark(master, endpoints, origin)
    usable_features = [feature for feature in features if feature != "week_index"]
    train_ids, validation_ids, test_ids = split_ids(frame, folds, repeat, fold, base_config)
    train = frame[frame.resident_id.isin(train_ids)].copy()
    validation = frame[frame.resident_id.isin(validation_ids)].copy()
    test = frame[frame.resident_id.isin(test_ids)].copy()

    # Fit preprocessing only on the training portion and reuse it for all trials.
    processor = transformer(train, usable_features, expanded=True).fit(train[usable_features])
    fold_seed = int(base_config["seed"] + 100 * repeat + fold)
    rows: list[dict] = []

    for trial in classical_trials():
        started = time.time()
        try:
            model = fit_survival(
                "classical_cox",
                train,
                validation,
                usable_features,
                base_config,
                processor=processor,
                penalizer=trial["penalizer"],
                seed=fold_seed,
            )
            risk = 1.0 - model.predict_survival(test, [12])[:, 0]
            metrics = horizon_metrics(train, test, risk, 12, base_config["minimum_censor_survival"], base_config["minimum_cases"])
            rows.append(
                {
                    "origin": origin,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Classical Cox",
                    **trial,
                    **metrics,
                    "best_epoch": np.nan,
                    "runtime_seconds": time.time() - started,
                    "error": "",
                }
            )
        except Exception as error:  # Preserve a complete failure ledger instead of stopping the grid.
            rows.append(
                {
                    "origin": origin,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Classical Cox",
                    **trial,
                    "status": "failed",
                    "runtime_seconds": time.time() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    for trial in neural_trials():
        started = time.time()
        try:
            trial_config = copy.deepcopy(base_config)
            trial_config.update(
                learning_rate=trial["learning_rate"],
                dropout=trial["dropout"],
                patience=trial["patience"],
            )
            model = fit_survival(
                "pycox_neural",
                train,
                validation,
                usable_features,
                trial_config,
                processor=processor,
                hidden=trial["hidden"],
                penalizer=trial["weight_decay"],
                seed=fold_seed,
            )
            risk = 1.0 - model.predict_survival(test, [12])[:, 0]
            metrics = horizon_metrics(train, test, risk, 12, trial_config["minimum_censor_survival"], trial_config["minimum_cases"])
            rows.append(
                {
                    "origin": origin,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Neural CoxPH",
                    **trial,
                    **metrics,
                    "best_epoch": model.best_epoch,
                    "runtime_seconds": time.time() - started,
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "origin": origin,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Neural CoxPH",
                    **trial,
                    "status": "failed",
                    "runtime_seconds": time.time() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aggregate and baseline-relative tables from successful fold results."""
    ok = metrics[metrics.status.eq("ok")].copy()
    group_columns = ["origin", "model", "trial_id", "factor", "value", "is_baseline"]
    summary = (
        ok.groupby(group_columns, dropna=False)
        .agg(
            evaluations=("brier", "size"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            runtime_seconds=("runtime_seconds", "sum"),
        )
        .reset_index()
    )

    # Use a model-specific baseline because Cox and Neural CoxPH have different controls.
    baseline = summary[summary.is_baseline].set_index(["origin", "model"])
    comparisons = []
    for row in summary.to_dict("records"):
        control = baseline.loc[(row["origin"], row["model"])]
        comparisons.append(
            {
                **row,
                "brier_improvement_pct": 100.0 * (control.brier_mean - row["brier_mean"]) / control.brier_mean,
                "log_loss_improvement_pct": 100.0 * (control.log_loss_mean - row["log_loss_mean"]) / control.log_loss_mean,
                "auc_change": row["auc_mean"] - control.auc_mean,
            }
        )
    comparisons = pd.DataFrame(comparisons)

    # Select the lowest mean 12-week Brier score separately by origin and model.
    best = (
        comparisons.sort_values(["origin", "model", "brier_mean", "log_loss_mean"])
        .groupby(["origin", "model"], as_index=False)
        .first()
    )
    return summary, comparisons, best


def paired_fold_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every trial with its model baseline on exactly matched test folds.

    A positive metric difference means the trial improved over the baseline:
    baseline minus trial for Brier/log loss, and trial minus baseline for AUC.
    The win counts show whether an average gain is broadly repeated or driven by
    only a few folds.
    """
    ok = metrics[metrics.status.eq("ok")].copy()
    key = ["origin", "repeat", "fold", "model"]
    baseline = ok[ok.is_baseline][key + ["brier", "log_loss", "roc_auc"]].rename(
        columns={
            "brier": "baseline_brier",
            "log_loss": "baseline_log_loss",
            "roc_auc": "baseline_auc",
        }
    )
    paired = ok.merge(baseline, on=key, validate="many_to_one")
    paired["brier_gain"] = paired.baseline_brier - paired.brier
    paired["log_loss_gain"] = paired.baseline_log_loss - paired.log_loss
    paired["auc_gain"] = paired.roc_auc - paired.baseline_auc

    group_columns = ["origin", "model", "trial_id", "factor", "value", "is_baseline"]
    return (
        paired.groupby(group_columns, dropna=False)
        .agg(
            matched_folds=("brier_gain", "size"),
            brier_gain_mean=("brier_gain", "mean"),
            brier_gain_std=("brier_gain", "std"),
            brier_fold_wins=("brier_gain", lambda values: int((values > 0).sum())),
            log_loss_gain_mean=("log_loss_gain", "mean"),
            log_loss_fold_wins=("log_loss_gain", lambda values: int((values > 0).sum())),
            auc_gain_mean=("auc_gain", "mean"),
            auc_fold_wins=("auc_gain", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )


def write_markdown(best: pd.DataFrame, comparisons: pd.DataFrame, metrics: pd.DataFrame, metadata: dict) -> None:
    """Create a concise, student-readable account of the executed results."""
    # Build the small Markdown table locally so producing the results does not
    # depend on pandas' optional ``tabulate`` package. Keeping report generation
    # dependency-free makes the supplied analysis easier to reproduce.
    display_columns = [
        "origin", "model", "trial_id", "brier_mean", "log_loss_mean",
        "auc_mean", "brier_improvement_pct", "log_loss_improvement_pct",
        "auc_change",
    ]
    headings = [
        "Origin", "Model", "Trial", "Brier", "Log loss", "AUC",
        "Brier improvement (%)", "Log-loss improvement (%)", "AUC change",
    ]
    table_lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for row in best[display_columns].itertuples(index=False, name=None):
        formatted = [str(row[0]), str(row[1]), str(row[2])] + [f"{float(value):.6f}" for value in row[3:]]
        table_lines.append("| " + " | ".join(formatted) + " |")

    lines = [
        "# Survival hyperparameter trial results",
        "",
        "## Scope",
        "",
        f"- November 15 snapshot: {metadata['qualified_households']} qualified tenancies and {metadata['qualified_events']} observed move-outs.",
        "- Outcome: probability of move-out within 12 weeks.",
        "- Evaluation: five repeats of five household-held-out folds (25 test evaluations per trial) at entry and week 4.",
        "- Selection rule: lowest mean 12-week Brier score, with log loss and AUC retained as guardrails.",
        "",
        "## Best setting by model and prediction origin",
        "",
        "\n".join(table_lines),
        "",
        "## Interpretation",
        "",
        "Positive Brier and log-loss improvement percentages mean the trial improved probability quality relative to that model's documented baseline. A positive AUC change means better tenant-risk ranking; a negative value means worse ranking.",
        "",
        "These trials estimate predictive association, not why a tenant decides to leave. The results are retrospective and come from two sites. Neural trials use one deterministic seed per fold; the selected configuration should be confirmed with a multi-seed ensemble before operational use.",
        "",
        "## Execution checks",
        "",
        f"- Successful fold-level evaluations: {int(metrics.status.eq('ok').sum())}",
        f"- Failed fold-level evaluations: {int(metrics.status.ne('ok').sum())}",
        f"- Unique trial definitions: {comparisons.trial_id.nunique()}",
    ]
    (OUTPUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build fixed folds, execute the full grid, and write audit-ready results.

    ``--workers`` controls only process-level parallelism; it does not change
    folds, seeds, model definitions, or scores.  ``--smoke`` executes one fold
    as an installation check and deliberately skips aggregate result tables.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5, help="Parallel origin/fold workers.")
    parser.add_argument("--smoke", action="store_true", help="Run only entry, repeat 0, fold 0.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, endpoints, features, folds, metadata = prepare_analysis_data()
    config = metadata["config"]

    tasks = []
    for origin in config["landmarks"]:
        for repeat in range(config["repeats"]):
            for fold in range(config["folds"]):
                if args.smoke and (origin != 0 or repeat != 0 or fold != 0):
                    continue
                tasks.append(
                    {
                        "master": master,
                        "endpoints": endpoints,
                        "features": features,
                        "folds": folds,
                        "config": config,
                        "origin": origin,
                        "repeat": repeat,
                        "fold": fold,
                    }
                )

    all_rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_one_fold, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "completed_origin": task["origin"],
                        "completed_repeat": task["repeat"],
                        "completed_fold": task["fold"],
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )

    metrics = pd.DataFrame(all_rows).sort_values(["origin", "model", "trial_id", "repeat", "fold"])
    metrics.to_csv(OUTPUT_DIR / "trial_metrics_by_fold.csv", index=False)

    if args.smoke:
        print(metrics[["model", "trial_id", "status", "brier", "log_loss", "roc_auc", "error"]].to_string(index=False))
        return

    summary, comparisons, best = summarize(metrics)
    paired = paired_fold_comparisons(metrics)
    summary.to_csv(OUTPUT_DIR / "trial_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "comparison_to_model_baseline.csv", index=False)
    best.to_csv(OUTPUT_DIR / "best_settings.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_fold_comparisons.csv", index=False)
    manifest = {
        **{key: value for key, value in metadata.items() if key != "config"},
        "config": config,
        "classical_trials": classical_trials(),
        "neural_trials": neural_trials(),
        "runtime_seconds": time.time() - started,
        "neural_seed_policy": "one deterministic seed per fold",
        "selection_metric": "mean outer-test 12-week Brier score",
    }
    (OUTPUT_DIR / "trial_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_markdown(best, comparisons, metrics, metadata)


if __name__ == "__main__":
    main()

