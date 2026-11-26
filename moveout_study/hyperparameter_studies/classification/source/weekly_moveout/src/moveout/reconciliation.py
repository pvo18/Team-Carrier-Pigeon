"""Move-out study source summary.
Purpose: Verify supplied midterm predictions and configurations without loading untrusted checkpoints or retraining models.
Inputs: Reconstructs deterministic folds, checks source hashes/features/seeds, and recomputes saved test metrics.
Outputs: Writes reconciliation tables, configurations, checks, and explicit limitations of the recoverable evidence.
Workflow: AST literal extraction reads constants without executing the supplied source files.
Method note: The procedure distinguishes verified predictions from assumptions about unavailable training code.
"""

from pathlib import Path
import ast
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from .config import load_config
from .data import read_sources, make_folds, REPORTED
from .metrics import weekly_metrics
from .pipeline import write_json

def reconcile(midterm_path, data_path, output):
    """Verify midterm sources, fold membership, labels, saved predictions, and scores without retraining.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    root = Path(midterm_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    residents, weekly, _, source_hashes = read_sources(data_path)
    cfg = load_config(); cfg.update(seed=42, folds=5, repeats=5)
    folds = make_folds(weekly, cfg)
    folds.to_csv(output / "recovered_fold_assignments.csv", index=False)
    tests = []; results = []; configurations = []
    def constant(name, path):
        """Implement the constant step used by this module's documented workflow.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        # Parse constants as syntax and literals; never execute the supplied midterm source during verification.
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        return None
    prep_source = root / "src/preprocess.py"
    ffnn_source = root / "src/ffnn.py"
    source_features = constant("NUMERIC_FEATURES", ffnn_source) + constant("CATEGORICAL_FEATURES", ffnn_source)
    tests.append(dict(check="reported_feature_order", passed=source_features == REPORTED))
    tests.append(dict(check="preparation_seed", passed=constant("SEED", prep_source) == 42))
    # Compare embedded raw files byte-for-byte against the independently supplied provider extracts.
    with zipfile.ZipFile(root / "Gated_Rentals_Dataset.zip") as archive:
        for name, expected in source_hashes.items():
            candidates = [n for n in archive.namelist() if n.endswith("/"+name)]
            tests.append(dict(check="source_hash:"+name,
                              passed=len(candidates) == 1 and hashlib.sha256(archive.read(candidates[0])).hexdigest() == expected))
    # Recompute every saved FFNN fold metric on its original held-out prediction rows.
    for directory in sorted((root / "results/ffnn").glob("repeat_*/fold_*")):
        rep = int(directory.parent.name.split("_")[-1]); fold = int(directory.name.split("_")[-1])
        frame = pd.read_parquet(directory / "test_weekly_predictions.parquet")
        ids = set(folds.loc[folds.repeat.eq(rep) & folds.fold.eq(fold), "resident_id"])
        expected = weekly[weekly.resident_id.isin(ids)][["resident_id", "week_index", "churned_this_week"]]
        joined = expected.merge(frame, on=["resident_id", "week_index"], how="outer", indicator=True,
                                suffixes=("_source", "_saved"), validate="one_to_one")
        matches = set(frame.resident_id) == ids and joined._merge.eq("both").all() and joined.churned_this_week_source.eq(joined.churned_this_week_saved).all()
        tests.append(dict(check=f"saved_prediction_keys_and_labels:{rep}:{fold}", passed=bool(matches)))
        metrics = weekly_metrics(frame.churned_this_week, frame.predicted_hazard)
        saved = json.loads((directory / "metrics.json").read_text())
        differences = {k: abs(metrics[k]-saved["test"][k]) for k in ["brier", "log_loss", "roc_auc", "pr_auc"]}
        tests.append(dict(check=f"saved_test_metrics:{rep}:{fold}", passed=max(differences.values()) < 1e-6,
                          maximum_absolute_difference=max(differences.values())))
        train_rate = float(weekly.loc[~weekly.resident_id.isin(ids), "churned_this_week"].mean())
        null = weekly_metrics(frame.churned_this_week, np.full(len(frame), train_rate))
        inner_null = weekly_metrics(frame.churned_this_week, np.full(len(frame), saved["train"]["positive_rate"]))
        results.append(dict(repeat=rep, fold=fold, **metrics, null_brier=null["brier"], null_log_loss=null["log_loss"],
                            inner_null_brier=inner_null["brier"], inner_null_log_loss=inner_null["log_loss"]))
        configurations.append(dict(repeat=rep, fold=fold, **json.loads((directory / "config.json").read_text())))
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output / "verified_saved_metrics.csv", index=False)
    write_json(output / "saved_run_configurations.json", configurations)
    summary = dict(checks=tests, all_checks_passed=all(t["passed"] for t in tests),
                   verified_saved_folds=len(results), mean_test_metrics=result_frame.select_dtypes("number").mean().to_dict(),
                   later_training_runner_present="loss_mode" in ffnn_source.read_text() and "EarlyStopping" in ffnn_source.read_text(),
                   limitations=["Included FFNN source has weighted fixed-epoch training; later unweighted/early-stopping runner is absent.",
                                "Original logistic initialization seed is not set in its source.",
                                "Saved results verified by scoring supplied predictions; no models retrained."])
    write_json(output / "midterm_reconciliation.json", summary)
    summary_table = result_frame[["brier", "null_brier", "log_loss", "null_log_loss", "roc_auc", "pr_auc"]].mean().to_frame("mean").to_markdown()
    (output / "midterm_reconciliation.md").write_text("# Midterm reconciliation\n\n"
        + f"Verified saved folds: {len(results)}. All source/fold/metric checks pass: {summary['all_checks_passed']}.\n\n"
        + summary_table + "\n\n" + "\n".join("- "+x for x in summary["limitations"]))
    return summary
