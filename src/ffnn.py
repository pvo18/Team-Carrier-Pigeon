"""Feed-forward neural-network model for Team Carrier Pigeon's tenant project.

This model is intentionally aligned with baseline_logistic_regression_torch.py
so the two models can be compared fairly.

Both scripts:
  * read the parquet files produced by preprocess.py;
  * use the exact same input columns;
  * fit imputation/scaling/one-hot encoding on training rows only;
  * hold out one complete resident-level k-fold split for evaluation;
  * use class-weighted BCEWithLogitsLoss for the rare weekly churn event; and
  * report n_rows, n_events, Brier score, log loss, ROC-AUC, and calibration.

The key modeling difference is that the baseline has one linear layer, while
this file adds hidden ReLU layers (default: 16 -> 8) and optional dropout.

Repository layout expected by this file:
    Team-Carrier-Pigeon/
    ├── preprocessing/
    │   ├── preprocess.py
    │   └── ffnn.py
    ├── data/
    │   └── processed/
    └── results/

Because ffnn.py and preprocess.py are in the same folder, this file imports
PROCESSED, REPO_ROOT, and SEED directly from preprocess.py.

Default run from the repository root (same repeat/fold as the baseline):
    python preprocessing/ffnn.py

Architecture experiment examples:
    python preprocessing/ffnn.py --hidden-dims 8
    python preprocessing/ffnn.py --hidden-dims 16,8
    python preprocessing/ffnn.py --hidden-dims 32,16 --dropout 0.2

IMPORTANT: Run preprocessing/preprocess.py at least once first so these files exist:
    data/processed/model_table.parquet
    data/processed/splits/kfold.parquet

You can also tell ffnn.py to run preprocessing first with:
    python preprocessing/ffnn.py --run-preprocess
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Repository / preprocess.py integration
# -----------------------------------------------------------------------------
# ffnn.py and preprocess.py live in the SAME directory. When this file is run
# directly (for example: python preprocessing/ffnn.py), Python automatically
# places that directory on sys.path, so a normal sibling-module import works.
#
# Using REPO_ROOT and PROCESSED from preprocess.py keeps both scripts pointed at
# the exact same data/ and results/ repository locations.
from preprocess import PROCESSED, REPO_ROOT, SEED, main as preprocess_main

REPO_ROOT = Path(REPO_ROOT)
PROCESSED = Path(PROCESSED)
PREPROCESS_FILE = Path(__file__).resolve().with_name("preprocess.py")


# -----------------------------------------------------------------------------
# Feature definitions
# -----------------------------------------------------------------------------
# These deliberately match baseline_logistic_regression_torch.py exactly.
LABEL_COLUMN = "churned_this_week"
ID_COLUMN = "resident_id"
WEEK_COLUMN = "week_index"

NUMERIC_FEATURES = [
    "week_index",
    "outstanding_lag1",
    "delinquent_lag1",
    "rent_to_income_ratio",
    "work_drive_min",
    "prior_home_distance_mi",
    "n_fobs",
    "n_pets",
    "income_monthly_est",
]

CATEGORICAL_FEATURES = ["site"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Household-level hard classification threshold. The model itself remains
# probabilistic; this cutoff is used only for predicted-move-out counts and the
# confusion matrix.
HOUSEHOLD_MOVEOUT_THRESHOLD = 0.50


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model-side feature preprocessing
# -----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Create the same feature preprocessing used by the logistic baseline.

    Numeric values:
      1. Missing values -> training-column median.
      2. StandardScaler.

    Categorical values:
      1. Missing values -> "missing".
      2. One-hot encoding.

    This object MUST be fit on the training rows only.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _dataframe_to_arrays(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a DataFrame to model-ready X/y arrays.

    fit=True is used ONLY for the training set. Evaluation data is transformed
    using the already-fitted training preprocessor, preventing leakage.
    """
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    X_cols = df[FEATURE_COLUMNS]
    X = preprocessor.fit_transform(X_cols) if fit else preprocessor.transform(X_cols)

    # ColumnTransformer may return either a dense array or sparse matrix.
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32)
    return X, y


# -----------------------------------------------------------------------------
# Feed-forward neural network
# -----------------------------------------------------------------------------
class TenantHazardFFNN(nn.Module):
    """Feed-forward network that outputs one raw weekly churn logit."""

    def __init__(
        self,
        n_input_features: int,
        hidden_dims: tuple[int, ...] = (16, 8),
        dropout: float = 0.20,
    ):
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("Every hidden-layer width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        previous_width = n_input_features

        for width in hidden_dims:
            layers.append(nn.Linear(previous_width, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_width = width

        # One logit = weekly move-out hazard before sigmoid.
        layers.append(nn.Linear(previous_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_ffnn(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    *,
    hidden_dims: tuple[int, ...] = (16, 8),
    dropout: float = 0.20,
    epochs: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[nn.Module, ColumnTransformer]:
    """Train the FFNN on an already-split training DataFrame.

    For comparability with the logistic baseline:
      * the same preprocessing is used;
      * positive events get n_negative / n_positive loss weight;
      * Adam is used;
      * eval_df is never used to update weights or fit preprocessing;
      * training runs for the requested fixed number of epochs.

    eval_df is only used for printing loss so you can inspect overfitting.
    """
    set_seed(seed)

    preprocessor = build_preprocessor()
    X_train, y_train = _dataframe_to_arrays(train_df, preprocessor, fit=True)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    if n_pos == 0:
        raise ValueError("Training data contains no churn events")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TenantHazardFFNN(
        n_input_features=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # Same class-imbalance strategy as the baseline.
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Prepare full train/eval tensors once for progress reporting.
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    X_eval_t = None
    y_eval_t = None
    if eval_df is not None:
        X_eval, y_eval = _dataframe_to_arrays(eval_df, preprocessor, fit=False)
        X_eval_t = torch.from_numpy(X_eval).to(device)
        y_eval_t = torch.from_numpy(y_eval).to(device)

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Match the baseline's progress cadence and print format closely.
        if verbose and eval_df is not None and (epoch % 20 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                train_loss = loss_fn(model(X_train_t), y_train_t).item()
                eval_loss = loss_fn(model(X_eval_t), y_eval_t).item()
            print(
                f"epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"eval_loss={eval_loss:.4f}"
            )

    return model, preprocessor


# -----------------------------------------------------------------------------
# Evaluation -- intentionally matches baseline output structure
# -----------------------------------------------------------------------------
def evaluate_ffnn(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    eval_df: pd.DataFrame,
) -> dict:
    """Evaluate one split using the same metrics as the logistic baseline."""
    X_eval, y_eval = _dataframe_to_arrays(eval_df, preprocessor, fit=False)

    device = next(model.parameters()).device
    X_eval_t = torch.from_numpy(X_eval).to(device)

    model.eval()
    with torch.no_grad():
        predicted_prob = torch.sigmoid(model(X_eval_t)).cpu().numpy()

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_eval,
        predicted_prob,
        n_bins=10,
        strategy="quantile",
    )

    return {
        "n_rows": len(eval_df),
        "n_events": int(y_eval.sum()),
        "brier_score": brier_score_loss(y_eval, predicted_prob),
        "log_loss": log_loss(y_eval, predicted_prob),
        "roc_auc": roc_auc_score(y_eval, predicted_prob),
        "calibration_curve": {
            "mean_predicted_value": mean_predicted_value.tolist(),
            "fraction_of_positives": fraction_of_positives.tolist(),
        },
    }


def compare_train_vs_eval(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict:
    """Return the same train/eval result dictionary as the baseline."""
    return {
        "train": evaluate_ffnn(model, preprocessor, train_df),
        "eval": evaluate_ffnn(model, preprocessor, eval_df),
    }


# -----------------------------------------------------------------------------
# Household-level move-out counts / classification
# -----------------------------------------------------------------------------
def _predict_weekly_probabilities(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one weekly move-out probability for every resident-week row."""
    X, y = _dataframe_to_arrays(df, preprocessor, fit=False)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.from_numpy(X).to(device))
        ).cpu().numpy()

    out = df[[ID_COLUMN, WEEK_COLUMN]].copy()
    out[LABEL_COLUMN] = y.astype(int)
    out["predicted_hazard"] = probs.astype(float)
    return out


def summarize_household_moveout_classification(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    *,
    threshold: float = HOUSEHOLD_MOVEOUT_THRESHOLD,
) -> tuple[dict, pd.DataFrame]:
    """Convert weekly hazards into one move-out prediction per resident.

    For a resident with weekly hazards h_1, ..., h_T, cumulative move-out risk
    over the resident's supplied/observed weeks is

        1 - product_t(1 - h_t).

    Two useful counts are reported:
      * expected_moveouts: sum of cumulative household risks (no threshold);
      * predicted_moveouts: number of households with cumulative risk >= threshold.

    The confusion matrix, precision, recall, and F1 use the thresholded household
    prediction. Each resident is counted at most once.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    weekly = _predict_weekly_probabilities(model, preprocessor, df)
    rows = []

    for resident_id, group in weekly.groupby(ID_COLUMN, sort=False):
        group = group.sort_values(WEEK_COLUMN)
        hazards = np.clip(
            group["predicted_hazard"].to_numpy(dtype=float),
            0.0,
            1.0 - 1e-12,
        )

        # log1p is numerically stable for product(1 - hazard).
        survival = float(np.exp(np.log1p(-hazards).sum()))
        cumulative_risk = 1.0 - survival
        actual_moveout = int(group[LABEL_COLUMN].max())
        predicted_moveout = int(cumulative_risk >= threshold)

        rows.append(
            {
                ID_COLUMN: resident_id,
                "actual_moveout": actual_moveout,
                "cumulative_moveout_risk": cumulative_risk,
                "predicted_moveout": predicted_moveout,
            }
        )

    households = pd.DataFrame(rows)
    y_true = households["actual_moveout"].to_numpy(dtype=int)
    y_pred = households["predicted_moveout"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n_households": int(len(households)),
        "actual_moveouts": int(y_true.sum()),
        "expected_moveouts": float(households["cumulative_moveout_risk"].sum()),
        "predicted_moveouts": int(y_pred.sum()),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, households


def print_household_moveout_summary(
    split_name: str,
    metrics: dict,
) -> None:
    """Print household-level counts in a compact, baseline-comparable format."""
    print(
        f"  {split_name}: n_households={metrics['n_households']}  "
        f"actual_moveouts={metrics['actual_moveouts']}  "
        f"expected_moveouts={metrics['expected_moveouts']:.2f}  "
        f"predicted_moveouts@{metrics['threshold']:.2f}={metrics['predicted_moveouts']}"
    )
    print(
        f"           TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
        f"TN={metrics['true_negatives']}  FN={metrics['false_negatives']}  "
        f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
        f"f1={metrics['f1']:.4f}"
    )
    print(
        "           confusion_matrix=[[TN, FP], [FN, TP]] = "
        f"{metrics['confusion_matrix']}"
    )


# -----------------------------------------------------------------------------
# Optional hazard -> move-out-week helpers
# -----------------------------------------------------------------------------
def hazards_to_week_distribution(
    weeks: np.ndarray,
    hazards: np.ndarray,
) -> pd.DataFrame:
    """Convert weekly hazards into probabilities of moving out in each week.

    P(T=t) = P(survive all earlier supplied weeks) * hazard_t
    """
    weeks = np.asarray(weeks, dtype=int)
    hazards = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)

    order = np.argsort(weeks)
    weeks = weeks[order]
    hazards = hazards[order]

    survival_before = 1.0
    event_probabilities = []
    survival_after = []

    for hazard in hazards:
        p_event = survival_before * hazard
        event_probabilities.append(p_event)
        survival_before *= 1.0 - hazard
        survival_after.append(survival_before)

    return pd.DataFrame(
        {
            WEEK_COLUMN: weeks,
            "predicted_hazard": hazards,
            "moveout_probability": event_probabilities,
            "survival_after_week": survival_after,
        }
    )


def predict_weekly_hazards(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one predicted weekly hazard for each supplied resident-week row."""
    return _predict_weekly_probabilities(model, preprocessor, df)


def summarize_moveout_weeks(weekly_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize predicted move-out-week distributions resident by resident."""
    rows = []

    for resident_id, group in weekly_predictions.groupby(ID_COLUMN, sort=False):
        group = group.sort_values(WEEK_COLUMN)
        distribution = hazards_to_week_distribution(
            group[WEEK_COLUMN].to_numpy(),
            group["predicted_hazard"].to_numpy(),
        )

        total_moveout_risk = float(distribution["moveout_probability"].sum())
        survival_past_horizon = float(distribution["survival_after_week"].iloc[-1])

        if total_moveout_risk > 0:
            most_likely_idx = distribution["moveout_probability"].to_numpy().argmax()
            most_likely_week = int(distribution.iloc[most_likely_idx][WEEK_COLUMN])
            expected_week = float(
                np.sum(
                    distribution[WEEK_COLUMN].to_numpy(dtype=float)
                    * distribution["moveout_probability"].to_numpy(dtype=float)
                )
                / total_moveout_risk
            )
        else:
            most_likely_week = None
            expected_week = np.nan

        actual_events = group[group[LABEL_COLUMN] == 1]
        actual_week = (
            int(actual_events.iloc[0][WEEK_COLUMN])
            if len(actual_events) == 1
            else None
        )

        rows.append(
            {
                ID_COLUMN: resident_id,
                "actual_moveout_week": actual_week,
                "predicted_week_conditional": most_likely_week,
                "expected_week_conditional": expected_week,
                "risk_within_observed_horizon": total_moveout_risk,
                "survival_past_observed": survival_past_horizon,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Result persistence
# -----------------------------------------------------------------------------
def save_run_outputs(
    output_dir: Path,
    model_name: str,
    repeat: int,
    fold: int,
    weekly_results: dict,
    household_results: dict,
    household_predictions: dict[str, pd.DataFrame],
) -> tuple[Path, Path]:
    """Always save comparable metrics and household predictions for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for split_name in ("train", "eval"):
        weekly = weekly_results[split_name]
        household = household_results[split_name]
        metric_rows.append({
            "model": model_name,
            "repeat": repeat,
            "fold": fold,
            "split": split_name,
            "n_rows": weekly["n_rows"],
            "n_events": weekly["n_events"],
            "brier_score": weekly["brier_score"],
            "log_loss": weekly["log_loss"],
            "roc_auc": weekly["roc_auc"],
            "n_households": household["n_households"],
            "actual_moveouts": household["actual_moveouts"],
            "expected_moveouts": household["expected_moveouts"],
            "predicted_moveouts": household["predicted_moveouts"],
            "threshold": household["threshold"],
            "true_positives": household["true_positives"],
            "false_positives": household["false_positives"],
            "true_negatives": household["true_negatives"],
            "false_negatives": household["false_negatives"],
            "precision": household["precision"],
            "recall": household["recall"],
            "f1": household["f1"],
        })

    metrics_path = output_dir / f"repeat_{repeat}_fold_{fold}_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    pred_frames = []
    for split_name in ("train", "eval"):
        frame = household_predictions[split_name].copy()
        frame.insert(0, "split", split_name)
        pred_frames.append(frame)

    household_path = output_dir / f"repeat_{repeat}_fold_{fold}_household_classification.csv"
    pd.concat(pred_frames, ignore_index=True).to_csv(household_path, index=False)

    return metrics_path, household_path


# -----------------------------------------------------------------------------
# Command-line execution
# -----------------------------------------------------------------------------
def parse_hidden_dims(text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(piece.strip()) for piece in text.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--hidden-dims must look like 8 or 16,8 or 64,32,16"
        ) from exc

    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive integers")
    return dims


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tenant weekly-hazard FFNN on preprocess.py outputs."
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(16, 8))
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save held-out weekly hazards and resident move-out-week summaries.",
    )
    parser.add_argument(
        "--run-preprocess",
        action="store_true",
        help=(
            "Run the sibling preprocess.py before training. Useful after the raw "
            "CSV files have changed."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("\n=== Team Carrier Pigeon: FFNN ===", flush=True)
    print(f"Repository root: {REPO_ROOT}", flush=True)
    print(f"Processed data directory: {PROCESSED}", flush=True)

    if args.run_preprocess:
        print(f"Running preprocessing: {PREPROCESS_FILE}", flush=True)
        preprocess_main()

    model_table_path = PROCESSED / "model_table.parquet"
    kfold_path = PROCESSED / "splits" / "kfold.parquet"

    if not model_table_path.exists() or not kfold_path.exists():
        raise FileNotFoundError(
            f"Missing {model_table_path} or {kfold_path}. "
            "Run the sibling `preprocess.py` first, or rerun this command "
            "with `--run-preprocess`."
        )

    model_table = pd.read_parquet(model_table_path)
    kfold = pd.read_parquet(kfold_path)

    # EXACTLY the same outer train/eval split logic as the logistic baseline.
    this_repeat = kfold[kfold["repeat"] == args.repeat]
    if this_repeat.empty:
        raise ValueError(f"Repeat {args.repeat} does not exist in kfold.parquet")

    eval_ids = this_repeat.loc[
        this_repeat["fold"] == args.fold,
        ID_COLUMN,
    ]
    train_ids = this_repeat.loc[
        this_repeat["fold"] != args.fold,
        ID_COLUMN,
    ]

    if eval_ids.empty:
        raise ValueError(
            f"Fold {args.fold} does not exist for repeat {args.repeat} in kfold.parquet"
        )

    train_df = model_table[model_table[ID_COLUMN].isin(train_ids)].copy()
    eval_df = model_table[model_table[ID_COLUMN].isin(eval_ids)].copy()

    print(
        f"Loaded split: train_rows={len(train_df)}, eval_rows={len(eval_df)}, "
        f"train_households={train_df[ID_COLUMN].nunique()}, "
        f"eval_households={eval_df[ID_COLUMN].nunique()}",
        flush=True,
    )

    # Safety checks: whole residents stay on one side of the split.
    train_residents = set(train_df[ID_COLUMN].unique())
    eval_residents = set(eval_df[ID_COLUMN].unique())
    assert train_residents.isdisjoint(eval_residents), "Resident leakage across train/eval"

    print(
        f"FFNN architecture: {X_architecture(args.hidden_dims)}  "
        f"dropout={args.dropout:.2f}"
        , flush=True
    )
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    trained_model, fitted_preprocessor = train_ffnn(
        train_df,
        eval_df=eval_df,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=True,
    )

    results = compare_train_vs_eval(
        trained_model,
        fitted_preprocessor,
        train_df,
        eval_df,
    )

    # Match the baseline's final console output so results can be placed side-by-side.
    print(
        f"\nResults -- repeat {args.repeat}, fold {args.fold} held out as eval:",
        flush=True,
    )
    for split_name in ("train", "eval"):
        r = results[split_name]
        print(
            f"  {split_name}: n_rows={r['n_rows']}  n_events={r['n_events']}  "
            f"brier={r['brier_score']:.4f}  "
            f"log_loss={r['log_loss']:.4f}  "
            f"roc_auc={r['roc_auc']:.4f}",
            flush=True,
        )

    print(
        f"\nHousehold move-out summary -- cumulative risk over each resident's "
        f"observed weeks, threshold={HOUSEHOLD_MOVEOUT_THRESHOLD:.2f}:",
        flush=True,
    )
    household_results = {}
    household_predictions = {}
    for split_name, split_df in (("train", train_df), ("eval", eval_df)):
        hm, hp = summarize_household_moveout_classification(
            trained_model,
            fitted_preprocessor,
            split_df,
            threshold=HOUSEHOLD_MOVEOUT_THRESHOLD,
        )
        household_results[split_name] = hm
        household_predictions[split_name] = hp
        print_household_moveout_summary(split_name, hm)

    # Always save the comparable metrics and one-row-per-household classifications.
    output_dir = REPO_ROOT / "results" / "ffnn"
    metrics_path, classification_path = save_run_outputs(
        output_dir=output_dir,
        model_name=f"FFNN_{'x'.join(map(str, args.hidden_dims))}",
        repeat=args.repeat,
        fold=args.fold,
        weekly_results=results,
        household_results=household_results,
        household_predictions=household_predictions,
    )
    print(f"\nSaved metrics to: {metrics_path}", flush=True)
    print(f"Saved household classifications to: {classification_path}", flush=True)

    if args.save_predictions:
        output_dir = REPO_ROOT / "results" / "ffnn"
        output_dir.mkdir(parents=True, exist_ok=True)

        weekly = predict_weekly_hazards(
            trained_model,
            fitted_preprocessor,
            eval_df,
        )
        household = summarize_moveout_weeks(weekly)

        weekly_path = output_dir / (
            f"repeat_{args.repeat}_fold_{args.fold}_weekly_predictions.parquet"
        )
        household_path = output_dir / (
            f"repeat_{args.repeat}_fold_{args.fold}_household_predictions.parquet"
        )
        weekly.to_parquet(weekly_path, index=False)
        household.to_parquet(household_path, index=False)

        print(f"Saved weekly predictions to: {weekly_path}")
        print(f"Saved household summaries to: {household_path}")


def X_architecture(hidden_dims: tuple[int, ...]) -> str:
    return " -> ".join(["input", *[str(x) for x in hidden_dims], "1"])


if __name__ == "__main__":
    main()
