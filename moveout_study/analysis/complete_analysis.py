#!/usr/bin/env python3
"""Move-out study source summary.
Purpose: Create the final slide-ready plots and supporting tables for the move-out study.
Inputs: Reads the qualified November 15 weekly/survival results, completed hyperparameter trials, and held-out predictions.
Outputs: Produces eight PNG figures plus CSV/JSON summary tables in move_out_analysis_plots.
Workflow: The workflow loads inputs, calculates descriptive/model summaries, renders diagnostics, and writes reproducible outputs.
Method note: This script does not train models; it visualizes already qualified held-out results.
"""


from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


# Resolve all inputs relative to this script so the analysis is portable as a folder.
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "inspect/revised/results"
HYPER = ROOT / "hyperparameter_results"
OUT = ROOT / "move_out_analysis_plots"

BLUE = "#2F5597"
ORANGE = "#ED7D31"
GRAY = "#A5A5A5"
GREEN = "#548235"
RED = "#C00000"
LIGHT_BLUE = "#9DC3E6"


def setup_plotting() -> None:
    """Set global Matplotlib defaults so every exported figure uses the same slide-ready typography, colors, and background.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    """Write one figure to the analysis output directory at presentation resolution, then close it to release memory.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def input_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load snapshot validation counts plus the November weekly master panel and tenancy endpoint ledger used by input plots.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    # Use human-readable labels while retaining sortable snapshot identifiers in saved tables.
    labels = {"20260915": "Sep 15", "20261031": "Oct 31", "20261115": "Nov 15"}
    rows = []
    for snapshot, label in labels.items():
        report = json.loads((RESULTS / snapshot / "prepared/validation_report.json").read_text())
        rows.append({"snapshot": snapshot, "label": label, **report["counts"]})
    growth = pd.DataFrame(rows)

    panel = pd.read_parquet(RESULTS / "20261115/prepared/weekly_master.parquet")
    ledger = pd.read_csv(RESULTS / "20261115/prepared/cohort_ledger.csv")
    return growth, panel, ledger


def input_overview_plot(growth: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Plot cohort growth across snapshots beside the severe November weekly class imbalance.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Input data: coverage grew while weekly move-outs remained rare", fontsize=24, weight="bold")

    ax = axes[0]
    x = np.arange(len(growth))
    width = 0.34
    bars1 = ax.bar(x - width / 2, growth["households"], width, label="Household tenancies", color=BLUE)
    bars2 = ax.bar(x + width / 2, growth["events"], width, label="Observed move-outs", color=ORANGE)
    ax.set_xticks(x, growth["label"])
    ax.set_ylabel("Count")
    ax.set_title("Three analysis snapshots")
    ax.legend(frameon=False, loc="upper left")
    ax.bar_label(bars1, padding=3, fmt="%.0f")
    ax.bar_label(bars2, padding=3, fmt="%.0f")
    for xi, weekly_rows in zip(x, growth["rows"]):
        ax.text(xi, 15, f"{weekly_rows:,.0f}\nweekly rows", ha="center", va="bottom", color="white", fontsize=11, weight="bold")

    ax = axes[1]
    # Restrict descriptive class counts to complete weeks that were eligible for weekly scoring.
    eligible = panel.loc[panel["eligible_weekly"].astype(bool)].copy()
    counts = eligible["churned_this_week"].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["No move-out", "Move-out"], counts.values, color=[GRAY, ORANGE], width=0.58)
    ax.set_ylabel("Household-week observations")
    ax.set_title("Weekly label balance, Nov 15 snapshot")
    ax.bar_label(bars, padding=4, labels=[f"{v:,.0f}" for v in counts.values])
    event_rate = counts.loc[1] / counts.sum()
    ax.text(0.98, 0.92, f"Move-out rate: {event_rate:.2%}", transform=ax.transAxes, ha="right", va="top", fontsize=15, weight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "01_input_data_overview.png")


def tenure_plot(panel: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Plot observed tenancy-duration/censoring distributions and empirical move-out frequency by tenure band.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Input data: tenancy duration and observed move-out timing", fontsize=24, weight="bold")

    ax = axes[0]
    # Use four-week duration bins to show the distribution without implying day-level precision.
    bins = np.arange(0, math.ceil(ledger["duration"].max() / 4) * 4 + 4, 4)
    censored = ledger.loc[ledger["event"] == 0, "duration"]
    events = ledger.loc[ledger["event"] == 1, "duration"]
    ax.hist([censored, events], bins=bins, stacked=True, color=[LIGHT_BLUE, ORANGE], label=["Still active/censored", "Moved out"])
    ax.set_xlabel("Observed tenancy duration (weeks)")
    ax.set_ylabel("Household tenancies")
    ax.set_title("Duration distribution, Nov 15 snapshot")
    ax.legend(frameon=False)

    ax = axes[1]
    eligible = panel.loc[panel["eligible_weekly"].astype(bool)].copy()
    eligible["tenure_bin"] = pd.cut(
        eligible["week_index"],
        bins=[-1, 3, 7, 11, 15, 19, 25, 31, 39, np.inf],
        labels=["0–3", "4–7", "8–11", "12–15", "16–19", "20–25", "26–31", "32–39", "40+"],
    )
    rate = eligible.groupby("tenure_bin", observed=True)["churned_this_week"].agg(["mean", "sum", "count"]).reset_index()
    bars = ax.bar(rate["tenure_bin"].astype(str), rate["mean"] * 100, color=BLUE)
    ax.set_xlabel("Tenure week")
    ax.set_ylabel("Observed weekly move-out rate (%)")
    ax.set_title("Move-out frequency by tenure")
    ax.tick_params(axis="x", rotation=35)
    for bar, n in zip(bars, rate["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"n={n:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "02_tenure_and_moveout_timing.png")


def weekly_results() -> pd.DataFrame:
    """Read held-out weekly fold metrics and average the constant, logistic, and FFNN scores across folds.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    path = RESULTS / "20261115/combined_weekly/metrics_by_fold.csv"
    df = pd.read_csv(path)
    # Exclude subgroup rows so each plotted value represents the overall held-out cohort.
    df = df[(df["task"] == "weekly") & (df["site"].isna()) & df["model"].isin(["constant", "logistic", "ffnn"])]
    return df.groupby("model", as_index=False)[["brier", "log_loss", "roc_auc"]].mean()


def weekly_results_plot(summary: pd.DataFrame) -> None:
    """Render side-by-side Brier, log-loss, and AUC comparisons for the weekly models.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    order = ["constant", "logistic", "ffnn"]
    names = {"constant": "Constant", "logistic": "Logistic", "ffnn": "FFNN"}
    colors = [GRAY, LIGHT_BLUE, BLUE]
    data = summary.set_index("model").loc[order]
    metrics = [("brier", "Brier score ↓", 5), ("log_loss", "Log loss ↓", 5), ("roc_auc", "AUC ↑", 3)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8))
    fig.suptitle("Weekly prediction: FFNN delivered the best probability accuracy", fontsize=24, weight="bold")
    for ax, (metric, title, decimals) in zip(axes, metrics):
        bars = ax.bar([names[m] for m in order], data[metric].values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylim(0, max(data[metric]) * 1.22)
        ax.bar_label(bars, padding=4, fmt=f"%.{decimals}f")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "03_weekly_model_results.png")


def survival_results() -> pd.DataFrame:
    """Load the qualified 12-week classical and neural CoxPH summaries at entry and week 4.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    df = pd.read_csv(RESULTS / "20261115/combined_survival/summary_mean_folds.csv")
    return df[(df["horizon"] == 12) & df["model"].isin(["classical_cox", "pycox_neural"])].copy()


def survival_results_plot(summary: pd.DataFrame) -> None:
    """Render 12-week survival probability accuracy and discrimination by prediction origin.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    metrics = [("brier", "Brier score ↓"), ("log_loss", "Log loss ↓"), ("roc_auc", "AUC ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8))
    fig.suptitle("12-week survival prediction: neural CoxPH improved probability accuracy", fontsize=24, weight="bold")
    origins = [0, 4]
    x = np.arange(len(origins))
    width = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        classical = summary[summary["model"] == "classical_cox"].set_index("origin").loc[origins, metric]
        neural = summary[summary["model"] == "pycox_neural"].set_index("origin").loc[origins, metric]
        b1 = ax.bar(x - width / 2, classical, width, label="Classical Cox", color=GRAY)
        b2 = ax.bar(x + width / 2, neural, width, label="Neural CoxPH", color=BLUE)
        ax.set_xticks(x, ["Entry", "Week 4"])
        ax.set_title(title)
        ax.set_ylim(0, max(classical.max(), neural.max()) * 1.25)
        ax.bar_label(b1, padding=3, fmt="%.3f", fontsize=10)
        ax.bar_label(b2, padding=3, fmt="%.3f", fontsize=10)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "04_survival_12week_results.png")


def hyperparameter_results(baseline: pd.DataFrame) -> pd.DataFrame:
    """Collect every completed controlled trial and calculate score changes relative to its fixed model baseline.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    baseline_by_model = baseline.set_index("model")
    rows: list[dict] = []
    for variation_path in sorted(HYPER.glob("*/variation.json")):
        variant_dir = variation_path.parent
        metrics_path = variant_dir / "metrics_by_fold.csv"
        if not metrics_path.exists():
            continue
        variation = json.loads(variation_path.read_text())
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[(metrics["task"] == "weekly") & metrics["site"].isna()]
        if metrics.empty:
            continue
        means = metrics[["brier", "log_loss", "roc_auc"]].mean()
        family = "logistic" if variation["model_kind"] in {"logistic", "weighted_logistic"} else "ffnn"
        base = baseline_by_model.loc[family]
        rows.append(
            {
                "variant": variation["variant"],
                "family": family,
                "change": json.dumps(variation["changes"], sort_keys=True),
                **means.to_dict(),
                "delta_brier": means["brier"] - base["brier"],
                "delta_log_loss": means["log_loss"] - base["log_loss"],
                "delta_auc": means["roc_auc"] - base["roc_auc"],
            }
        )
    result = pd.DataFrame(rows).sort_values(["family", "brier"])
    result.to_csv(OUT / "hyperparameter_trial_results.csv", index=False)
    return result


def friendly_variant(value: str) -> str:
    """Translate compact experiment identifiers into labels suitable for a presentation legend.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    mapping = {
        "log_wd_0001": "Weight decay .001",
        "nn_wd_0001": "Weight decay .001",
        "log_weight_sqrt": "√ class weight",
        "log_weight_full": "Full class weight",
        "nn_weight_sqrt": "√ class weight",
        "nn_weight_full": "Full class weight",
        "nn_arch_16_8": "Layers 16→8",
        "nn_arch_8_4": "Layers 8→4",
        "nn_batch_32": "Batch 32",
        "nn_ensemble_3": "3-seed ensemble",
        "nn_patience_50": "Patience 50",
    }
    return mapping.get(value, value.replace("log_", "").replace("nn_", "").replace("_", " "))


def hyperparameter_plot(results: pd.DataFrame) -> None:
    """Show the joint Brier/AUC tradeoff for logistic and FFNN variants while flagging off-scale harmful weights.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.suptitle("Controlled hyperparameter trials: calibration mattered more than AUC alone", fontsize=24, weight="bold")
    for ax, family, title in zip(axes, ["logistic", "ffnn"], ["Logistic regression", "FFNN"]):
        d = results[results["family"] == family].copy()
        weighted = d["variant"].str.contains("weight")
        standard = d.loc[~weighted].copy()
        improved_both = (standard["delta_brier"] < 0) & (standard["delta_auc"] > 0)
        ax.scatter(
            standard.loc[~improved_both, "delta_brier"],
            standard.loc[~improved_both, "delta_auc"],
            s=70,
            color=BLUE,
            label="Mixed or worse",
        )
        ax.scatter(
            standard.loc[improved_both, "delta_brier"],
            standard.loc[improved_both, "delta_auc"],
            s=85,
            color=GREEN,
            label="Brier and AUC improved",
        )
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_xlabel("Change in Brier score (left is better)")
        ax.set_ylabel("Change in AUC (up is better)")
        ax.set_title(title)
        best = standard.nsmallest(2, "brier")
        for _, row in best.iterrows():
            ax.annotate(friendly_variant(row["variant"]), (row["delta_brier"], row["delta_auc"]), xytext=(6, 6), textcoords="offset points", fontsize=10)
        weighted_rows = d.loc[weighted].sort_values("delta_brier")
        if not weighted_rows.empty:
            lines = ["Class weighting (outside scale):"]
            for _, row in weighted_rows.iterrows():
                lines.append(f"{friendly_variant(row['variant'])}: Δ Brier {row['delta_brier']:+.3f}")
            ax.text(
                0.98,
                0.04,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color=ORANGE,
            )
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "05_hyperparameter_effects.png")


def feature_importance_plot() -> pd.DataFrame:
    """Aggregate held-out-model permutation evidence and plot the strongest predictive associations for both weekly models.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    df = pd.read_csv(RESULTS / "feature_importance_summary.csv")
    df = df[(df["snapshot"] == 20261115) & (df["task"] == "weekly") & df["model"].isin(["logistic", "ffnn"])].copy()
    label_map = {
        "billed_this_week_lag1": "Prior-week amount billed",
        "billed_this_week_past4_mean": "4-week average billed",
        "billed_this_week_past8_mean": "8-week average billed",
        "paid_this_week_lag1": "Prior-week amount paid",
        "paid_this_week_past4_mean": "4-week average paid",
        "paid_this_week_past8_mean": "8-week average paid",
        "outstanding_lag1": "Prior-week balance",
        "outstanding_past4_mean": "4-week average balance",
        "outstanding_past8_mean": "8-week average balance",
        "past_collection_ratio": "Historical collection ratio",
        "first_responder": "First-responder indicator",
        "commute_band": "Commute-distance band",
        "prior_rent_income_ratio": "Rent-to-income ratio",
        "historical_affordability": "Historical affordability",
        "income_monthly_est": "Estimated monthly income",
        "prior_rent_monthly": "Prior monthly rent",
        "initial_discount": "Initial discount",
        "week_index": "Tenure week",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    fig.suptitle("Which inputs most changed weekly predictions?", fontsize=24, weight="bold")
    exported = []
    for ax, model, title, color in zip(axes, ["logistic", "ffnn"], ["Logistic regression", "FFNN"], [LIGHT_BLUE, BLUE]):
        d = df[df["model"] == model].nlargest(10, "mean_increase").sort_values("mean_increase")
        labels = [label_map.get(v, v.replace("_", " ")) for v in d["feature"]]
        ax.barh(labels, d["mean_increase"], color=color)
        ax.set_xlabel("Increase in Brier score when permuted")
        ax.set_title(title)
        exported.append(d.assign(feature_label=labels))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "06_feature_importance.png")
    result = pd.concat(exported, ignore_index=True)
    result.to_csv(OUT / "top_feature_importance.csv", index=False)
    return result


def weekly_diagnostic_curves() -> None:
    """Build ROC, precision-recall, and calibration curves from averaged held-out weekly predictions.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    predictions = pd.read_parquet(RESULTS / "20261115/combined_weekly/weekly_predictions.parquet")
    predictions = predictions[predictions["model"].isin(["logistic", "ffnn"])]
    averaged = (
        predictions.groupby(["resident_id", "week_index", "model"], as_index=False)
        .agg(churned_this_week=("churned_this_week", "first"), predicted_probability=("predicted_probability", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.suptitle("Weekly prediction diagnostics: discrimination, rare-event retrieval, and calibration", fontsize=23, weight="bold")
    colors = {"logistic": LIGHT_BLUE, "ffnn": BLUE}
    labels = {"logistic": "Logistic", "ffnn": "FFNN"}

    for model in ["logistic", "ffnn"]:
        d = averaged[averaged["model"] == model]
        y = d["churned_this_week"].to_numpy()
        p = d["predicted_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[model], linewidth=2.5, label=f"{labels[model]} (AUC {roc_auc_score(y, p):.3f})")
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, color=colors[model], linewidth=2.5, label=f"{labels[model]} (AP {average_precision_score(y, p):.3f})")
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[2].plot(predicted, observed, marker="o", color=colors[model], linewidth=2.2, label=labels[model])

    axes[0].plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1].axhline(averaged["churned_this_week"].mean(), color=GRAY, linewidth=1.2, linestyle="--", label="Move-out prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    maximum = max(axes[2].get_xlim()[1], axes[2].get_ylim()[1])
    axes[2].plot([0, maximum], [0, maximum], color=GRAY, linewidth=1.2, linestyle="--", label="Perfect calibration")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed move-out rate", title="Calibration by risk decile")
    for ax in axes:
        ax.legend(frameon=False, fontsize=10)
        ax.grid(axis="both", color="#E6E6E6", linewidth=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "07_weekly_diagnostic_curves.png")


def paired_uncertainty_plot() -> None:
    """Visualize clustered-bootstrap confidence intervals for each model's paired improvement over its comparator.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    weekly = json.loads((RESULTS / "20261115/combined_weekly/paired_comparisons.json").read_text())
    survival = json.loads((RESULTS / "20261115/combined_survival/paired_comparisons.json").read_text())
    rows = []
    w = next(r for r in weekly if r["model"] == "ffnn" and r["comparator"] == "constant" and r["unit"] == "resident_id")
    rows.append({"label": "Weekly FFNN vs constant", **w})
    for origin, label in [(0, "Neural CoxPH vs Cox, entry"), (4, "Neural CoxPH vs Cox, week 4")]:
        r = next(
            r
            for r in survival
            if r["model"] == "pycox_neural"
            and r["comparator"] == "classical_cox"
            and r["origin"] == origin
            and r["horizon"] == 12
            and r["unit"] == "resident_id"
        )
        rows.append({"label": label, **r})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Paired bootstrap uncertainty: negative differences favor the neural model", fontsize=23, weight="bold")
    metrics = [("brier_delta", "brier_ci", "Difference in Brier score"), ("log_loss_delta", "log_loss_ci", "Difference in log loss")]
    y = np.arange(len(rows))
    for ax, (metric, ci_name, title) in zip(axes, metrics):
        values = np.array([r[metric] for r in rows])
        lows = np.array([r[ci_name][0] for r in rows])
        highs = np.array([r[ci_name][1] for r in rows])
        errors = np.vstack([values - lows, highs - values])
        ax.errorbar(values, y, xerr=errors, fmt="o", color=BLUE, ecolor=BLUE, capsize=5, markersize=8)
        ax.axvline(0, color=GRAY, linewidth=1.2, linestyle="--")
        ax.set_yticks(y, [r["label"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_title("95% bootstrap confidence intervals")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "08_paired_improvement_uncertainty.png")


def write_summary_tables(
    growth: pd.DataFrame,
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    weekly: pd.DataFrame,
    survival: pd.DataFrame,
    hyper: pd.DataFrame,
) -> None:
    """Write the exact data/model/trial/importance values behind the figures to reusable CSV and JSON files.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    growth.to_csv(OUT / "input_snapshot_summary.csv", index=False)
    weekly.to_csv(OUT / "weekly_model_summary.csv", index=False)
    survival.to_csv(OUT / "survival_12week_summary.csv", index=False)
    eligible = panel.loc[panel["eligible_weekly"].astype(bool)]
    summary = {
        "november_15_resident_records": 572,
        "november_15_household_tenancies": int(ledger.shape[0]),
        "november_15_household_week_rows": int(panel.shape[0]),
        "november_15_observed_moveouts": int(ledger["event"].sum()),
        "november_15_censored_active_tenancies": int((ledger["event"] == 0).sum()),
        "eligible_weekly_rows": int(eligible.shape[0]),
        "weekly_positive_labels": int(eligible["churned_this_week"].sum()),
        "weekly_moveout_rate": float(eligible["churned_this_week"].mean()),
        "completed_hyperparameter_trials": int(hyper.shape[0]),
    }
    (OUT / "analysis_counts.json").write_text(json.dumps(summary, indent=2))

    captions = """01 Input data overview — analysis coverage across three snapshots and the rare weekly move-out label.\n
02 Tenure and move-out timing — observed duration/censoring and weekly move-out frequency by tenure band.\n
03 Weekly model results — constant baseline, logistic regression, and FFNN on Brier score, log loss, and AUC.\n
04 Survival results — classical Cox versus neural CoxPH for 12-week prediction at entry and week 4.\n
05 Hyperparameter effects — each controlled variation versus its original model; left/up is the preferred quadrant.\n
06 Feature importance — top inputs by permutation increase in Brier score for logistic regression and FFNN.\n"""
    captions += "07 Weekly diagnostic curves — ROC, precision–recall, and calibration from averaged held-out predictions.\n"
    captions += "08 Paired improvement uncertainty — household-clustered bootstrap intervals for neural-versus-baseline score differences.\n"
    (OUT / "plot_captions.txt").write_text(captions)


def main() -> None:
    """Execute the complete analysis workflow in dependency order and leave a self-contained plot/table directory.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    setup_plotting()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    growth, panel, ledger = input_data()
    weekly = weekly_results()
    survival = survival_results()
    hyper = hyperparameter_results(weekly)

    input_overview_plot(growth, panel)
    tenure_plot(panel, ledger)
    weekly_results_plot(weekly)
    survival_results_plot(survival)
    hyperparameter_plot(hyper)
    feature_importance_plot()
    weekly_diagnostic_curves()
    paired_uncertainty_plot()
    write_summary_tables(growth, panel, ledger, weekly, survival, hyper)
    shutil.make_archive(str(ROOT / "Move_Out_Analysis_Plots_and_Tables"), "zip", OUT)
    print(json.dumps({"output": str(OUT), "trials": int(hyper.shape[0])}))


if __name__ == "__main__":
    main()
