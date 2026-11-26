"""Move-out study source summary.
Purpose: Orchestrate leakage-controlled weekly and survival training, scoring, diagnostics, and reporting.
Inputs: Reads prepared artifacts, verifies configuration compatibility, loops over held-out folds, and writes full provenance manifests.
Outputs: Inner validation selects candidates; outer test predictions are used only for final scoring and paired comparisons.
Workflow: Failures are isolated and recorded so a partial run cannot masquerade as a complete experiment.
Method note: Feature importance is exploratory inner-validation permutation importance and is never causal or test-selected.
"""

from pathlib import Path
import hashlib
import importlib.metadata
import itertools
import json
import platform
import numpy as np
import pandas as pd
from . import __version__
from .data import landmark, split_ids, NEVER
from .models import transformer, fit_reference, fit_weekly, fit_survival, ph_diagnostics
from .metrics import weekly_metrics, horizon_metrics, ipcw_components, calibration_table, integrated_brier, paired_bootstrap

def write_json(path, value):
    """Serialize structured run evidence with stable indentation and NumPy-safe value conversion.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))

def read_prepared(path, cfg):
    """Load prepared artifacts and reject configuration settings inconsistent with preparation.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    path = Path(path); preparation = json.loads((path / "preparation.json").read_text())
    for key in ["feature_set", "endpoint_policy", "include_sensitive_research_features"]:
        if preparation["config"].get(key) != cfg.get(key):
            raise ValueError(f"Preparation/run configuration differs for {key}; prepare a separate dataset")
    master = pd.read_parquet(path / "weekly_master.parquet")
    endpoints = pd.read_csv(path / "cohort_ledger.csv")
    endpoints["qualified"] = endpoints.qualified.astype(str).str.lower().isin(["true", "1"])
    features = json.loads((path / "features.json").read_text())
    folds = pd.read_csv(path / "fold_assignments.csv")
    return master, endpoints, features, folds, preparation

def manifest(output, cfg, preparation, command):
    """Create an immutable run manifest containing configuration, environment versions, and source hashes.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise ValueError("Run directory already contains results; choose a fresh output directory")
    dependencies = {}
    # Record exact dependency versions so a reported run can be reconstructed later.
    for package in ["numpy", "pandas", "scikit-learn", "torch", "pycox", "torchtuples", "lifelines", "pyarrow"]:
        try: dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: dependencies[package] = "unavailable"
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    write_json(output / "run_manifest.json", dict(package_version=__version__, python=platform.python_version(),
               command=command, config=cfg, preparation=preparation, dependencies=dependencies, code_hashes=code_hashes))

def filter_features(features, drop_group=None):
    """Apply a named ablation by removing all feature names associated with that conceptual group.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    if not drop_group: return features
    groups = dict(commute=["work_", "commute_"], income=["income_", "affordability", "rent_income"],
                  payments=["paid", "billed", "outstanding", "collection", "delinquent", "invoice", "cycle", "past_weekly_rent"],
                  relocation=["prior_home_", "relocated_"], household=["n_fobs", "n_pets", "has_pet", "pc_"])
    if drop_group not in groups: raise ValueError("Unknown ablation group")
    return [f for f in features if not any(token in f for token in groups[drop_group])]

def selected_folds(folds, repeat=None, fold=None):
    """Yield the requested repeat/fold pairs after validating that assignments exist.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    pairs = folds[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    if repeat is not None: pairs = pairs[pairs.repeat.eq(repeat)]
    if fold is not None: pairs = pairs[pairs.fold.eq(fold)]
    if pairs.empty: raise ValueError("No requested fold assignments")
    return pairs.itertuples(index=False, name=None)

def save_calibration(table, directory):
    """Persist a calibration table and, when supported, a corresponding observed-versus-predicted plot.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / "calibration.csv", index=False)
    if table.empty: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4)); ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.plot(table.predicted, table.observed, "o-"); ax.set(xlabel="Predicted risk", ylabel="Observed risk")
    fig.tight_layout(); fig.savefig(directory / "calibration.png", dpi=140); plt.close(fig)

def feature_importance(models, validation, features, directory, seed=42, survival_train=None):
    """Exploratory inner-validation permutation; never used to select on the outer test."""
    rng = np.random.default_rng(seed)
    def score(frame):
        """Implement the score step used by this module's documented workflow.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        if survival_train is None:
            p = np.mean([m.predict_weekly_probability(frame) for m in models], axis=0)
            return float(np.mean((p-validation.churned_this_week.to_numpy())**2))
        risk = 1 - np.mean([m.predict_survival(frame, [1]) for m in models], axis=0)[:, 0]
        r = horizon_metrics(survival_train, validation, risk, 1)
        return r.get("brier", np.nan)
    baseline = score(validation); rows = []
    for feature in features:
        deltas = []
        for _ in range(3):
            altered = validation.copy()
            altered[feature] = rng.permutation(altered[feature].to_numpy())
            deltas.append(score(altered)-baseline)
        rows.append(dict(feature=feature, validation_brier_increase=float(np.nanmean(deltas)),
                         interpretation="exploratory_prediction_importance_not_causal"))
    pd.DataFrame(rows).sort_values("validation_brier_increase", ascending=False).to_csv(Path(directory) / "validation_feature_importance.csv", index=False)

def run_weekly(prepared, output, cfg, kinds, repeat=None, fold=None, drop_group=None):
    """Train, select, persist, and score every requested weekly model on the specified held-out folds.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    master, endpoints, features, folds, prep = read_prepared(prepared, cfg)
    features = filter_features(features, drop_group)
    output = Path(output); manifest(output, cfg, prep, "run-weekly")
    data = master[master.eligible_weekly].copy()
    all_metrics = []; all_predictions = []; selections = []; failures = []
    for rep, fo in selected_folds(folds, repeat, fold):
        train_ids, val_ids, test_ids = split_ids(data, folds, rep, fo, cfg)
        if cfg["mode"] == "historical_reproduction": train_ids |= val_ids
        train = data[data.resident_id.isin(train_ids)]; val = data[data.resident_id.isin(val_ids)]
        test = data[data.resident_id.isin(test_ids)]
        base = output / f"repeat_{rep}" / f"fold_{fo}"; base.mkdir(parents=True, exist_ok=True)
        assignments = pd.DataFrame([dict(resident_id=x, split=s) for s, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)] for x in sorted(ids)])
        if cfg["mode"] == "historical_reproduction":
            # No model selection or early stopping is performed on these overlap-only logging rows.
            assignments = assignments[~((assignments.split == "validation") & assignments.resident_id.isin(train_ids))]
        assignments.to_csv(base / "inner_assignments.csv", index=False)
        processor = transformer(train, features, cfg["feature_set"] == "expanded").fit(train[features])
        # Fit each requested model family against the same split and feature contract.
        for kind in kinds:
            print(f"weekly repeat={rep} fold={fo} model={kind}", flush=True)
            try:
                if kind in {"constant", "tenure_site"}:
                    models = [fit_reference(train, kind == "tenure_site")]; chosen = {"reference": kind}
                else:
                    historical = cfg["mode"] == "historical_reproduction"
                    if kind not in {"logistic", "ffnn", "weighted_logistic", "weighted_ffnn"}: raise ValueError(f"Unknown weekly model {kind}")
                    hidden_grid = [[]] if "logistic" in kind else ([[16, 8]] if historical else cfg["hidden_layers"])
                    wd_grid = [0.0] if historical else cfg["weight_decays"]
                    seeds = [cfg["seed"]+100*rep+fo] if historical or "logistic" in kind else cfg["neural_seeds"]
                    best = np.inf; models = None; chosen = None
                    for hidden, wd in itertools.product(hidden_grid, wd_grid):
                        fitted = [fit_weekly(kind, train, val, features, cfg, processor, hidden, wd,
                                             seed+100*rep+fo if not historical else seed, historical) for seed in seeds]
                        pval = np.mean([m.predict_weekly_probability(val) for m in fitted], axis=0)
                        score = float(np.mean((pval-val.churned_this_week.to_numpy())**2))
                        if historical or score < best:
                            best = score; models = fitted; chosen = dict(hidden=hidden, weight_decay=wd,
                                  seeds=[m.config["seed"] for m in fitted], validation_brier=score)
                for ix, model in enumerate(models): model.save(base / kind / f"member_{ix}")
                write_json(base / kind / "ensemble.json", dict(members=[f"member_{i}" for i in range(len(models))]))
                if kind not in {"constant", "tenure_site"} and cfg["mode"] != "historical_reproduction":
                    feature_importance(models, val, features, base / kind, cfg["seed"])
                probabilities = np.mean([m.predict_weekly_probability(test) for m in models], axis=0)
                result = weekly_metrics(test.churned_this_week, probabilities, test)
                result.update(repeat=rep, fold=fo, model=kind, task="weekly", split="test")
                all_metrics.append(result)
                predictions = test[["resident_id", "week_index", "site", "cluster_id", "churned_this_week"]].copy()
                predictions["predicted_probability"] = probabilities
                predictions["repeat"] = rep; predictions["fold"] = fo; predictions["model"] = kind
                all_predictions.append(predictions)
                save_calibration(calibration_table(test.churned_this_week, probabilities), base / kind)
                selections.append(dict(repeat=rep, fold=fo, model=kind, parameters=chosen))
                for site, g in predictions.groupby("site"):
                    sm = weekly_metrics(g.churned_this_week, g.predicted_probability, g)
                    sm.update(repeat=rep, fold=fo, model=kind, task="weekly_site", site=site, split="test")
                    all_metrics.append(sm)
            except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
                # Record model-level exceptions explicitly instead of dropping failed candidates silently.
                failures.append(dict(repeat=rep, fold=fo, model=kind, error=str(error)))
                write_json(output / "failures.json", failures)
    pd.DataFrame(all_metrics).to_csv(output / "metrics_by_fold.csv", index=False)
    if all_predictions: pd.concat(all_predictions).to_parquet(output / "weekly_predictions.parquet", index=False)
    write_json(output / "selected_candidates.json", selections); write_json(output / "failures.json", failures)
    if not all_predictions: raise RuntimeError("No models succeeded; inspect failures.json")
    report(output)

def run_survival(prepared, output, cfg, kinds, repeat=None, fold=None, gate=None, exploratory=False, drop_group=None):
    """Train, select, persist, and score every requested survival model at each landmark and horizon.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    master, endpoints, features, folds, prep = read_prepared(prepared, cfg)
    features = filter_features(features, drop_group)
    if "logistic_hazard" in kinds and not exploratory:
        if gate is None: raise ValueError("LogisticHazard requires a passed development gate or --exploratory")
        gate_value = json.loads(Path(gate).read_text())
        if gate_value.get("status") != "passed" or not gate_value.get("development_only"):
            raise ValueError("LogisticHazard gate is not passed")
    output = Path(output); manifest(output, cfg, prep, "run-survival")
    write_json(output / "logistic_hazard_status.json", dict(exploratory=exploratory, gate=str(gate) if gate else None))
    all_metrics = []; all_predictions = []; selections = []; failures = []
    for origin in cfg["landmarks"]:
        frame = landmark(master, endpoints, origin)
        usable_features = [f for f in features if f != "week_index"]
        for rep, fo in selected_folds(folds, repeat, fold):
            try: train_ids, val_ids, test_ids = split_ids(frame, folds, rep, fo, cfg)
            except ValueError as error:
                failures.append(dict(origin=origin, repeat=rep, fold=fo, model="all", error=str(error))); continue
            train = frame[frame.resident_id.isin(train_ids)]; val = frame[frame.resident_id.isin(val_ids)]
            test = frame[frame.resident_id.isin(test_ids)]
            base = output / f"origin_{origin}" / f"repeat_{rep}" / f"fold_{fo}"; base.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(resident_id=x, split=s) for s, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)] for x in sorted(ids)]).to_csv(base / "inner_assignments.csv", index=False)
            processor = transformer(train, usable_features, cfg["feature_set"] == "expanded").fit(train[usable_features])
            for kind in kinds:
                print(f"survival origin={origin} repeat={rep} fold={fo} model={kind}", flush=True)
                try:
                    hidden_grid = [None] if kind in {"km", "classical_cox", "pycox_linear"} else cfg["hidden_layers"]
                    penalties = [0.0] if kind == "km" else (cfg["cox_penalizers"] if kind == "classical_cox" else cfg["weight_decays"])
                    seeds = cfg["neural_seeds"] if kind in {"pycox_neural", "logistic_hazard"} else [cfg["seed"]]
                    best = np.inf; models = None; chosen = None
                    for hidden, penalty in itertools.product(hidden_grid, penalties):
                        fitted = [fit_survival(kind, train, val, usable_features, cfg, processor,
                                               hidden, penalty, seed+100*rep+fo) for seed in seeds]
                        risk_val = 1 - np.mean([m.predict_survival(val, [1]) for m in fitted], axis=0)[:, 0]
                        result = horizon_metrics(train, val, risk_val, 1, cfg["minimum_censor_survival"])
                        if result["status"] != "ok": continue
                        if result["brier"] < best:
                            best = result["brier"]; models = fitted
                            chosen = dict(hidden=hidden, penalty=penalty, seeds=seeds, validation_brier=best)
                    if models is None: raise ValueError("No candidate has supported validation follow-up")
                    for ix, model in enumerate(models): model.save(base / kind / f"member_{ix}")
                    write_json(base / kind / "ensemble.json", dict(members=[f"member_{i}" for i in range(len(models))]))
                    if kind != "km": feature_importance(models, val, usable_features, base / kind, cfg["seed"], train)
                    if kind == "classical_cox":
                        try: ph_diagnostics(models[0], base / kind / "diagnostics")
                        except (ValueError, RuntimeError, KeyError) as error:
                            write_json(base / kind / "diagnostic_issue.json", dict(error=str(error)))
                    survival = np.mean([m.predict_survival(test, cfg["horizons"]) for m in models], axis=0)
                    results = []
                    for j, horizon in enumerate(cfg["horizons"]):
                        risk = 1-survival[:, j]
                        result = horizon_metrics(train, test, risk, horizon, cfg["minimum_censor_survival"], cfg["minimum_cases"])
                        result.update(repeat=rep, fold=fo, model=kind, origin=origin, task="survival", split="test")
                        results.append(result); all_metrics.append(result)
                        predictions = test[["resident_id", "site", "cluster_id", "duration", "event"]].copy()
                        predictions["risk"] = risk; predictions["horizon"] = horizon; predictions["origin"] = origin
                        predictions["model"] = kind; predictions["repeat"] = rep; predictions["fold"] = fo
                        comp = ipcw_components(train, test, risk, horizon, cfg["minimum_censor_survival"])
                        if comp is not None:
                            y, w, _, _ = comp
                            predictions["ipcw_target"] = y; predictions["ipcw_weight"] = w
                            save_calibration(calibration_table(y, risk, w), base / kind / f"horizon_{horizon}")
                        else: predictions["ipcw_target"] = np.nan; predictions["ipcw_weight"] = np.nan
                        all_predictions.append(predictions)
                    selections.append(dict(origin=origin, repeat=rep, fold=fo, model=kind, parameters=chosen,
                                           integrated_brier=integrated_brier(results)))
                except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
                    failures.append(dict(origin=origin, repeat=rep, fold=fo, model=kind, error=str(error)))
                    write_json(output / "failures.json", failures)
    pd.DataFrame(all_metrics).to_csv(output / "metrics_by_fold.csv", index=False)
    if all_predictions: pd.concat(all_predictions).to_parquet(output / "survival_predictions.parquet", index=False)
    write_json(output / "selected_candidates.json", selections); write_json(output / "failures.json", failures)
    if not all_predictions: raise RuntimeError("No survival models succeeded; inspect failures.json")
    report(output)

def assess_gate(run_directory, output, evidence=None):
    """Record scientific review explicitly; an isolated PH p-value never auto-enables a model."""
    directory = Path(run_directory)
    diagnostics = list(directory.glob("origin_*/repeat_*/fold_*/classical_cox/diagnostics/ph_tests.csv"))
    flags = []
    for path in diagnostics:
        d = pd.read_csv(path, index_col=0)
        for col, row in d.iterrows():
            if row.get("p", 1) < 0.05: flags.append(dict(file=str(path), feature=str(col), p=float(row["p"])))
    review = json.loads(Path(evidence).read_text()) if evidence else {}
    required = (review.get("development_only") is True and review.get("functional_form_checked") is True
                and review.get("site_checked") is True and bool(review.get("reason"))
                and (review.get("time_varying_effects") is True or review.get("persistent_miscalibration") is True))
    files = review.get("evidence_files", [])
    passed = required and bool(files) and all(Path(p).is_file() for p in files)
    result = dict(status="passed" if passed else "review_required", development_only=True,
                  n_training_diagnostics=len(diagnostics), exploratory_ph_flags=flags, review=review,
                  explanation="Training diagnostic flags alone are insufficient; document functional-form/site review.")
    write_json(output, result); return result

def report(directory):
    """Aggregate fold metrics/predictions into comparisons, uncertainty intervals, and human-readable summaries.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    directory = Path(directory); manifest_value = json.loads((directory / "run_manifest.json").read_text())
    metrics = pd.read_csv(directory / "metrics_by_fold.csv")
    if metrics.empty: return
    numeric = [c for c in ["brier", "log_loss", "roc_auc", "pr_auc", "tenancy_weighted_brier", "tenancy_weighted_log_loss"] if c in metrics]
    groups = [c for c in ["task", "model", "origin", "horizon", "site", "repeat"] if c in metrics]
    means = metrics.groupby(groups, dropna=False)[numeric].mean().reset_index()
    means.to_csv(directory / "metrics_by_repeat.csv", index=False)
    # Paired comparisons reuse identical held-out observations to reduce avoidable variance.
    comparisons = []
    if (directory / "weekly_predictions.parquet").exists():
        p = pd.read_parquet(directory / "weekly_predictions.parquet")
        for model in sorted(set(p.model)-{"constant"}):
            keys = ["resident_id", "week_index", "repeat", "fold"]
            a = p[p.model.eq(model)]; b = p[p.model.eq("constant")]
            pair = a.merge(b[keys+["predicted_probability"]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
            if pair.empty: continue
            y = pair.churned_this_week.to_numpy()
            for suffix in ["a", "b"]:
                pred = pair["predicted_probability_"+suffix].to_numpy().clip(1e-7, 1-1e-7)
                pair["squared_error_"+suffix] = (pred-y)**2
                pair["log_loss_"+suffix] = -y*np.log(pred)-(1-y)*np.log1p(-pred)
            for group in ["resident_id", "cluster_id"]:
                result = paired_bootstrap(pair, group, manifest_value["config"]["bootstrap_replicates"], manifest_value["config"]["seed"])
                comparisons.append(dict(model=model, comparator="constant", **result))
    write_json(directory / "paired_comparisons.json", comparisons)
    if (directory / "survival_predictions.parquet").exists():
        p = pd.read_parquet(directory / "survival_predictions.parquet")
        comparator = "classical_cox" if "classical_cox" in set(p.model) else "km"
        keys = ["resident_id", "repeat", "fold", "origin", "horizon"]
        for model in sorted(set(p.model)-{comparator}):
            a = p[p.model.eq(model)]
            b = p[p.model.eq(comparator)]
            pair = a.merge(b[keys+["risk", "ipcw_weight"]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
            pair = pair[pair.ipcw_weight_a.notna() & pair.ipcw_weight_b.notna()].copy()
            if pair.empty: continue
            if not np.allclose(pair.ipcw_weight_a, pair.ipcw_weight_b): raise ValueError("Censor weights differ between compared models")
            y = pair.ipcw_target.to_numpy(); weight = pair.ipcw_weight_a.to_numpy()
            for suffix in ["a", "b"]:
                pred = pair["risk_"+suffix].to_numpy().clip(1e-7, 1-1e-7)
                pair["squared_error_"+suffix] = weight*(pred-y)**2
                pair["log_loss_"+suffix] = weight*(-y*np.log(pred)-(1-y)*np.log1p(-pred))
            for (origin, horizon), group_frame in pair.groupby(["origin", "horizon"]):
                for group in ["resident_id", "cluster_id"]:
                    result = paired_bootstrap(group_frame, group, manifest_value["config"]["bootstrap_replicates"], manifest_value["config"]["seed"])
                    comparisons.append(dict(model=model, comparator=comparator, origin=origin, horizon=horizon, **result))
        write_json(directory / "paired_comparisons.json", comparisons)
    # Never turn repeated-fold dispersion into an independent-sample standard error.
    table = means.to_markdown(index=False, floatfmt=".6f")
    failures = json.loads((directory / "failures.json").read_text())
    text = ("# Weekly move-out comparison\n\n" + f"Package {__version__}. Feature set: {manifest_value['config']['feature_set']}.\n\n"
            + "Results are retrospective; billing-start timing and static-field as-of provenance remain assumptions.\n\n"
            + "## Metrics by repeat\n\n" + table + "\n\n"
            + "Fold/repeat scores share tenancies and are not independent experiments. Unknown horizon outcomes are handled using training-fitted censoring weights.\n\n"
            + f"Model/fold failures: {len(failures)}. See failures.json and paired_comparisons.json for details.\n")
    (directory / "final_comparison.md").write_text(text)
    write_json(directory / "final_comparison.json", dict(metrics=means.to_dict("records"), paired_comparisons=comparisons, failures=failures))
