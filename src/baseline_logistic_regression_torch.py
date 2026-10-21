"""
Baseline model for the Gated Rentals tenant retention project (Team Carrier Pigeon):
logistic regression implemented in PyTorch.

This file does not load the raw CSVs and does not build the modeling table
itself. train_baseline() / evaluate_baseline() take already-split
DataFrames. The __main__ block below reads the parquet files that
preprocess.py writes (data/processed/model_table.parquet and data/processed/splits/kfold.parquet),
so preprocess.py's main() has to be run at least once first to produce them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from torch.utils.data import DataLoader, TensorDataset

#preprocess.py doesn't export these, so they're matched here to its
#WEEKLY_COLS / RESIDENT_COLS rather than imported.
#Name of target column: 1 if resident churned that week, 0 otherwise.
LABEL_COLUMN = "churned_this_week"

#Numeric input columns that get imputed and standardized.
NUMERIC_FEATURES = [
    "week_index", #Which week of tenancy is this row
    # Balance owed as of the previous week.
    "outstanding_lag1", #Balanced owed as of the previous week.
    "delinquent_lag1", #Was the resident delinquent the previous week
    "rent_to_income_ratio", #Rent divcided by income
    "work_drive_min", #Commute time to work (minutes)
    "prior_home_distance_mi", #Distance from resident's previous home (miles)
    "n_fobs", #Number of key fobs issued.
    "n_pets", #Number of pets
    "income_monthly_est", #Estimated monthly income
]

#"site" is the only categorical column in preprocess.py's output. The
#provider README flags site as worth controlling for (payment rails and
#application-link coverage differ between Willis Lakes and Crosby
#Heights), so it's included as a feature.

#Categorical input columns that get imputed and one-hot encoded.
CATEGORICAL_FEATURES = ["site"]


#Builds the object that turns raw DataFrame cols into a numeric matrix.
def build_preprocessor() -> ColumnTransformer:
    #Create the pipeline used for numeric cols.
    numeric_pipeline = Pipeline(
        #Steps run in order
        steps=[
            #Step 1: replace missing numbers with column's median (learned from training data).
            ("impute", SimpleImputer(strategy="median")),
            #Step 2: subtract the mean and divide by the std dev so features on comparable scales.
            ("scale", StandardScaler()),
        ]
    )
    #Create pipeline used for categorical columns.
    categorical_pipeline = Pipeline(
        #Steps run in order.
        steps=[
            #Step 1: replace missing categories with string "missing".
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            #Step 2: one 0/1 column per category; categories not seen in training become all zeros.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    #Combine both pipelines, each applied to its own set of columns.
    return ColumnTransformer(
        transformers=[
            #Run the numeric pipeline on the numeric feature columns.
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            #Run the categorical pipeline on the categorical feature columns.
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


#Defines the model as a PyTorch module
class LogisticRegressionTorch(nn.Module):
    """One linear layer mapping input features to a raw logit. Pair with
    BCEWithLogitsLoss (not a manual sigmoid + BCELoss) for numerical
    stability."""

    #Takes the number of input features after preprocessing.
    def __init__(self, n_input_features: int):
        #Initialize the nn.Module base class.
        super().__init__()
        #One linear layer: weighted sum of all features plus a bias, producing 1 output.
        self.linear = nn.Linear(n_input_features, 1)

    #Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Compute logits of shape (batch, 1), then drop the last dim to get shape (batch,).
        return self.linear(x).squeeze(-1)


#Helper that turns a DataFrame into a feature matrix X and label vector y.
def _dataframe_to_arrays(
    df: pd.DataFrame, preprocessor: ColumnTransformer, fit: bool
#Returns a (features, labels) pair of NumPy arrays.
) -> tuple[np.ndarray, np.ndarray]:
    #Select only the feature columns
    X_cols = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    #Fit and transform on training data; only transform on other data so nothing leaks from it.
    X = preprocessor.fit_transform(X_cols) if fit else preprocessor.transform(X_cols)
    #Convert to a float array.
    X = np.asarray(X, dtype=np.float32)
    #Pull out the label column as a float array.
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32)

    return X, y


#Trains the logistic regression and returns the model plus the fitted preprocessor.
def train_baseline(
    #Training rows.
    train_df: pd.DataFrame,
    #Optional evaluation rows, used only for printing progress.
    eval_df: pd.DataFrame | None = None,
    #Number of full passes over the training data.
    epochs: int = 200,
    #Step size for the optimizer.
    learning_rate: float = 0.05,
    #L2 penalty strength; 0 means no regularization.
    weight_decay: float = 0.0,
    #Rows per mini-batch.
    batch_size: int = 64,
    #Whether to print train/eval loss during training.
    verbose: bool = False,
#Returns the trained model and the preprocessor fitted on training data.
) -> tuple[nn.Module, ColumnTransformer]:
    """Fit on an already-split, already-prepared training set. eval_df, if
    given, is only used to print loss each epoch (watch for overfitting) --
    never to update weights. weight_decay is the L2 regularization
    strength and is worth sweeping."""
    #Make anunfitted preprocessor.
    preprocessor = build_preprocessor()
    #Fit the preprocessor on training data and get the training arrays.
    X_train, y_train = _dataframe_to_arrays(train_df, preprocessor, fit=True)

    #Count positive rows (churn events); labels are 0/1 so the sum is the count.
    n_pos = y_train.sum()
    #Count negative rows (no churn).
    n_neg = len(y_train) - n_pos
    #Weight positives by the negative/positive ratio to offset class imbalance (max avoids divide-by-zero).
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    #Build the model with one input per preprocessed feature column.
    model = LogisticRegressionTorch(n_input_features=X_train.shape[1])
    #Loss: sigmoid + binary cross-entropy in one step.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #Adam optimizer over the model's weights and bias, with optional L2 via weight_decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Wrap the training data so it can be iterated in mini-batches.
    train_loader = DataLoader(
        #Pair each feature row with its label, converting NumPy arrays to tensors.
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        #Rows per batch.
        batch_size=batch_size,
        #Reshuffle row order every epoch.
        shuffle=True,
    )

    #Only prepare eval data if it was provided.
    if eval_df is not None:
        #Transform eval data with the already-fitted preprocessor (no refitting).
        X_eval, y_eval = _dataframe_to_arrays(eval_df, preprocessor, fit=False)
        #Convert eval arrays to tensors once, up front.
        X_eval_t, y_eval_t = torch.from_numpy(X_eval), torch.from_numpy(y_eval)

    #Loop over epochs.
    for epoch in range(epochs):
        #Put the model in training mode.
        model.train()
        #Loop over mini-batches.
        for X_batch, y_batch in train_loader:
            #Clear gradients left over from the previous batch.
            optimizer.zero_grad()
            #Run the model on the batch and compute the loss against the true labels.
            loss = loss_fn(model(X_batch), y_batch)
            #Backpropagate: compute gradients of the loss with respect to each parameter.
            loss.backward()
            #Update the parameters using those gradients.
            optimizer.step()

        #Every 20 epochs and on the last epoch, report losses if verbose and eval data exists.
        if verbose and eval_df is not None and (epoch % 20 == 0 or epoch == epochs - 1):
            #Switch to evaluation mode.
            model.eval()
            #Turn off gradient tracking: faster, and these computations shouldn't affect training.
            with torch.no_grad():
                #Loss over the full training set
                train_loss = loss_fn(model(torch.from_numpy(X_train)), torch.from_numpy(y_train)).item()
                #Loss over the full eval set.
                eval_loss = loss_fn(model(X_eval_t), y_eval_t).item()
            #Print both losses; eval loss rising while train loss falls signals overfitting.
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}")

    #Return the trained model and the preprocessor needed to transform new data the same way.
    return model, preprocessor


#Scores a trained model on one DataFrame and returns a dict of metrics.
def evaluate_baseline(model: nn.Module, preprocessor: ColumnTransformer, eval_df: pd.DataFrame) -> dict:
    """Evaluate a trained model on one data split. Does not retrain and does
    not refit the preprocessor."""
    #Transform the data with the fitted preprocessor (fit=False, so nothing is learned from it).
    X_eval, y_eval = _dataframe_to_arrays(eval_df, preprocessor, fit=False)
    #Switch to evaluation mode.
    model.eval()
    #Disable gradient tracking for inference.
    with torch.no_grad():
        #Get logits, apply sigmoid to turn them into probabilities (0-1), convert to NumPy.
        predicted_prob = torch.sigmoid(model(torch.from_numpy(X_eval))).numpy()

    #Build calibration data: per bin, the actual event rate and the average predicted probability.
    fraction_of_positives, mean_predicted_value = calibration_curve(
        #10 bins with equal row counts (quantile), rather than equal-width probability ranges.
        y_eval, predicted_prob, n_bins=10, strategy="quantile"
    )
    #Collect results in a dictionary.
    return {
        #Number of rows evaluated.
        "n_rows": len(eval_df),
        #Number of churn events in those rows.
        "n_events": int(y_eval.sum()),
        #Mean squared error between predicted probability and 0/1 outcome (lower is better).
        "brier_score": brier_score_loss(y_eval, predicted_prob),
        #Cross-entropy of the predictions; heavily penalizes confident wrong predictions (lower is better).
        "log_loss": log_loss(y_eval, predicted_prob),
        #How well predictions rank churners above non-churners (0.5 = random, 1.0 = perfect).
        "roc_auc": roc_auc_score(y_eval, predicted_prob),
        #Calibration points, stored as plain lists so they're easy to serialize or plot.
        "calibration_curve": {
            #x-axis: average predicted probability per bin.
            "mean_predicted_value": mean_predicted_value.tolist(),
            #y-axis: observed fraction of churn events per bin.
            "fraction_of_positives": fraction_of_positives.tolist(),
        },
    }


#Run evaluation on both the training and eval sets.
def compare_train_vs_eval(
    #Trained model, fitted preprocessor, and two data splits.
    model: nn.Module, preprocessor: ColumnTransformer, train_df: pd.DataFrame, eval_df: pd.DataFrame
#Returns a dict with "train" and "eval" metric dicts.
) -> dict:
    #Build and return the comparison.
    return {
        #Metrics on the data the model was trained on.
        "train": evaluate_baseline(model, preprocessor, train_df),
        #Metrics on held-out data; a large gap from train suggests overfitting.
        "eval": evaluate_baseline(model, preprocessor, eval_df),
    }


#Only runs when the file is executed directly, not when it's imported.
if __name__ == "__main__":
    #Reads the parquet files preprocess.py writes; run
    #`python preprocess.py` first if these don't exist yet.
    #Import the processed-data directory path from preprocess.py.
    from preprocess import PROCESSED

    #Path to the modeling table (one row per resident per week).
    model_table_path = PROCESSED / "model_table.parquet"
    #Path to the cross-validation fold assignments.
    kfold_path = PROCESSED / "splits" / "kfold.parquet"

    #Stop early with a clear message if either input file is missing.
    if not model_table_path.exists() or not kfold_path.exists():
        #Raise an error explaining which files are missing.
        raise FileNotFoundError(
            #First part of the message: the two expected paths.
            f"Missing {model_table_path} or {kfold_path}. "
            #Second part: how to fix it.
            "Run preprocess.py's main() first -- it builds both from the raw CSVs."
        )

    #Load the modeling table into a DataFrame.
    model_table = pd.read_parquet(model_table_path)
    #Load the fold assignments (resident_id, repeat, fold).
    kfold = pd.read_parquet(kfold_path)

    #Hold out repeat 0 / fold 0 as eval, train on the other folds of that
    #same repeat. The split is by resident_id (household), so every week
    #belonging to one household lands entirely on one side -- no leakage
    #across train/eval. Choose which CV repeat to use and which fold within it to hold out.
    REPEAT, HELD_OUT_FOLD = 0, 0
    #Keep only fold assignments from the chosen repeat.
    this_repeat = kfold[kfold["repeat"] == REPEAT]
    #Resident IDs assigned to the held-out fold.
    eval_ids = this_repeat.loc[this_repeat["fold"] == HELD_OUT_FOLD, "resident_id"]
    #Resident IDs in every other fold.
    train_ids = this_repeat.loc[this_repeat["fold"] != HELD_OUT_FOLD, "resident_id"]

    #All weekly rows for training residents.
    train_df = model_table[model_table["resident_id"].isin(train_ids)]
    #All weekly rows for eval residents.
    eval_df = model_table[model_table["resident_id"].isin(eval_ids)]

    #Train the model, printing train/eval loss along the way.
    trained_model, fitted_preprocessor = train_baseline(train_df, eval_df=eval_df, verbose=True)
    #Compute metrics on both splits.
    results = compare_train_vs_eval(trained_model, fitted_preprocessor, train_df, eval_df)

    #Print a header saying which fold was held out.
    print(f"\nResults -- repeat {REPEAT}, fold {HELD_OUT_FOLD} held out as eval:")
    #Print one summary line for each split.
    for split_name in ("train", "eval"):
        #Grab that split's metrics dict.
        r = results[split_name]
        #Print the metrics on one line.
        print(
            #Split name, row count, and event count.
            f"  {split_name}: n_rows={r['n_rows']}  n_events={r['n_events']}  "
            #Brier score, log loss, and ROC AUC, each to 4 decimal places.
            f"brier={r['brier_score']:.4f}  log_loss={r['log_loss']:.4f}  roc_auc={r['roc_auc']:.4f}"
        )