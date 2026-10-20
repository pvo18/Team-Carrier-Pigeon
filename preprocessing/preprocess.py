import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
PATHS_TO_DATA = [REPO_ROOT / "data/raw/residents.csv", REPO_ROOT / "data/raw/weekly_panel.csv"]
PROCESSED = REPO_ROOT / "data/processed" 

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
N_BOOTSTRAP = 100

# expected counts from the provider's files, NOT computed by this script
EXPECTED_RESIDENTS = 455    # provider DATA_DICTIONARY.md: "residents.csv — 455 rows"
EXPECTED_WEEKLY_ROWS = 6731 # provider DATA_DICTIONARY.md: "weekly_panel.csv — 6731 rows"
EXPECTED_HOUSEHOLDS = 444   # data/reference/km_survival.csv: ALL, week 0, n_at_risk
EXPECTED_MOVE_OUTS = 125    # km_survival.csv: sum of n_events for ALL; provider README limitation 7


WEEKLY_COLS = [
      "resident_id",
      "site",
      "week_index",
      "churned_this_week",
      "outstanding",
      "delinquent",
  ]

RESIDENT_COLS = [
      "resident_id",
      "rent_to_income_ratio",
      "work_drive_min",
      "prior_home_distance_mi",
      "n_fobs",
      "n_pets",
      "income_monthly_est"
  ]


def load_csv_as_pd(path, cols):
    '''
    path: path to the csv to read in 
    type: either the weekly.csv cols or the resident.csv cols
    '''
    return pd.read_csv(Path(path), engine="pyarrow", usecols=cols)


def check_resident_ids(df):
    assert df["resident_id"].is_unique, f"Resident Id's are not unique"
    print(f"Residents are all unique :)") 

def check_for_dup_household_weeks(df):
    assert (not df.duplicated(["resident_id", "week_index"]).any()), f"There are duplicated household weeks in weekly_panel"
    print(f"Weekly panel has no duplicated household-weeks :)")

def check_at_most_one_move_out(df):
    # group by household
    grouped_weekly = df.groupby("resident_id")["churned_this_week"].sum()
    assert (grouped_weekly <= 1).all(), f"There is more than one move out for a house-hold, {grouped_weekly[grouped_weekly > 1]}"
    print(f"There is at most 1 move out per household :)")

def check_is_move_out_on_last_week(df):
    last_week = df.groupby("resident_id")["week_index"].transform("max")
    churned_rows = df["churned_this_week"] == 1

    # get the week index for each of the churned rows
    assert (df.loc[churned_rows, "week_index"] == last_week[churned_rows]).all(), f"There is a row in weekly in which the churn week index is not the last week of data they have"
    print(f"Every churned person moves out on the last week of data we have for them :)")

def check_households_table(df):
    # check to make sure we have the right number of households
    assert len(df) == EXPECTED_HOUSEHOLDS and df['moved_out'].sum() == EXPECTED_MOVE_OUTS, f"Expected {EXPECTED_HOUSEHOLDS} households and {EXPECTED_MOVE_OUTS} move-outs, got {len(df)} households and {df['moved_out'].sum()} move-outs"
    print(f"{EXPECTED_HOUSEHOLDS} households, {EXPECTED_MOVE_OUTS} move-outs :)")

def check_shape(df, expected_shape, name):
    # expected_shape is built from the EXPECTED_* constants and the column lists, never from df itself
    assert df.shape == expected_shape, f"{name} has shape {df.shape}, expected {expected_shape}"
    print(f"{name} has the expected shape {expected_shape} :)")

def check_splits(kfold, bootstrap, households):

    per_fold = kfold.merge(households, on="resident_id").groupby(["repeat", "fold"])["moved_out"].agg(["size", "sum"])
    print(per_fold)
    assert len(kfold) == EXPECTED_HOUSEHOLDS * N_REPEATS, \
        f"k-fold has {len(kfold)} rows, expected {EXPECTED_HOUSEHOLDS * N_REPEATS}"
    assert per_fold["size"].between(EXPECTED_HOUSEHOLDS // N_SPLITS, EXPECTED_HOUSEHOLDS // N_SPLITS + 1).all(), \
        f"A fold does not hold {EXPECTED_HOUSEHOLDS // N_SPLITS}-{EXPECTED_HOUSEHOLDS // N_SPLITS + 1} households"
    # exact, because the folds are stratified on moved_out
    assert (per_fold["sum"] == EXPECTED_MOVE_OUTS // N_SPLITS).all(), \
        f"A fold does not hold exactly {EXPECTED_MOVE_OUTS // N_SPLITS} move-outs"
    assert kfold.groupby("repeat")["resident_id"].agg(lambda s: s.is_unique and len(s) == len(households)).all(), \
        "A household is missing or repeated within a k-fold repeat"

    assert len(bootstrap) == EXPECTED_HOUSEHOLDS * N_BOOTSTRAP, \
        f"bootstrap has {len(bootstrap)} rows, expected {EXPECTED_HOUSEHOLDS * N_BOOTSTRAP}"
    assert (bootstrap.groupby("replicate")["n_draws"].sum() == len(households)).all(), \
        f"A bootstrap replicate does not have exactly {EXPECTED_HOUSEHOLDS} households"
    # random, so printed, not asserted: (1 - 1/n)^n -> 1/e ~ 0.368
    print(f"Mean out-of-bag fraction: {(bootstrap['n_draws'] == 0).mean():.3f}")


def clean_rent_income_ratio_renters(df):
    # use a threshold (noninclusive) since you cannot have a rent/income that negative
    # use a upper threshold (noninclusive) since this means the rent is above income
    lower_threshold = 0.0
    upper_threshold = 1.0

    # filter the ratios of this column to ensure that none of them are below 0 or above 1 
    renters_to_clean = ((df["rent_to_income_ratio"] < lower_threshold) | (df["rent_to_income_ratio"] > upper_threshold)).sum()
    df.loc[(df["rent_to_income_ratio"] < lower_threshold) | (df["rent_to_income_ratio"] > upper_threshold) , "rent_to_income_ratio"] = np.nan

    print(f'There are {renters_to_clean} renters to clean. Their rent to income ratio has been set to nan')
    print(f'In the model creation is where we need to use the median of the TRAINING PORTION of the data to prevent leakage and since the column is right skewed')

def sort_weekly(df):
    # sort weekly by ["resident_id", "week_index"]
    df = df.sort_values(by=["resident_id", "week_index"])
    return df

def define_lags(df):
    # want to use last weeks value instead of the current week since the model should not have access to this weeks
    # information to predict if they will move out that week
    # prevents data leakage
    df = sort_weekly(df)
    df["outstanding_lag1"] = df.groupby("resident_id")["outstanding"].shift(1)
    df["delinquent_lag1"] = df.groupby("resident_id")["delinquent"].shift(1)

    # add 0 to the first week for these outstanding columns
    df.loc[(df["week_index"] == 0), "outstanding_lag1"] = 0
    df.loc[(df["week_index"] == 0) & df['delinquent'].notna(), "delinquent_lag1"] = 0

    # drop the other outstanding and deliquent columns
    df = df.drop(columns=["outstanding", "delinquent"])
    return df

def merge_weekly_resdients(weekly, residents):
    merged = weekly.merge(
        residents,
        on="resident_id",
        how="left",
        validate="many_to_one"
    )
    return merged

def get_k_fold(households):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    pieces = []
    for i, (_, test_index) in enumerate(rskf.split(households, households["moved_out"])):
        pieces.append(pd.DataFrame({"resident_id": households["resident_id"].iloc[test_index].values, 
                                    "repeat": i // N_SPLITS, 
                                    "fold": i % N_SPLITS}))
    kfold = pd.concat(pieces, ignore_index=True)
    return kfold

def get_bootstrap(households):
    rng = np.random.default_rng(SEED)
    n = len(households)
    pieces = []
    for b in range(N_BOOTSTRAP):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        pieces.append(pd.DataFrame({"resident_id": households["resident_id"].values,
                                    "replicate": b,
                                    "n_draws": counts
                                    }))
    bootstrap = pd.concat(pieces, ignore_index=True)
    return bootstrap



def main():
    # load the csv's as a dataframe
    residents = load_csv_as_pd(PATHS_TO_DATA[0], RESIDENT_COLS)
    weekly_panel = load_csv_as_pd(PATHS_TO_DATA[1], WEEKLY_COLS)

    # check to make sure that there are no residents with the same id
    check_resident_ids(residents)

    # check the weekly panel to make sure this is at most one move out
    check_at_most_one_move_out(weekly_panel)

    # check to make sure that there are no duplicate households
    check_for_dup_household_weeks(weekly_panel)

    # check to make sure that the last week of data is the week a household moved out
    check_is_move_out_on_last_week(weekly_panel)

    # printing shape and sample data to see what the data looks like
    print(f"Shape of residents: {residents.shape}")
    print(f"Before cleaning: {residents.head(5)}")
    check_shape(residents, (EXPECTED_RESIDENTS, len(RESIDENT_COLS)), "residents")

    print(f"Shape of weekly: {weekly_panel.shape}")
    check_shape(weekly_panel, (EXPECTED_WEEKLY_ROWS, len(WEEKLY_COLS)), "weekly_panel")
    print(f"Before cleaning: {weekly_panel.head(5)}")

    # clean the residents on the rent-income ratio
    clean_rent_income_ratio_renters(residents)

    # create the lags on weekly since we are only predicting with knowledge of week prior
    weekly_panel = define_lags(weekly_panel)

    # merging the two datasets together
    model_table = merge_weekly_resdients(weekly=weekly_panel, residents=residents)

    # printing out stats on the merged data
    print(f'Shape of the merged data: {model_table.shape}')
    # lags swap 2 raw columns for 2 lag columns, and resident_id is shared, so - 1
    check_shape(model_table, (EXPECTED_WEEKLY_ROWS, len(WEEKLY_COLS) + len(RESIDENT_COLS) - 1), "model_table")
    print(f'Number of uniques in the merged data: {model_table["resident_id"].nunique()}')
    print(f'Example of some data: {model_table.head(5)}')
    print(f'Number of nans: {model_table.isna().sum()}')
 
    # saving the file in a parquet in the processed
    path = PROCESSED / "model_table.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    households = model_table.groupby("resident_id")["churned_this_week"].max().reset_index(name="moved_out")
    check_households_table(households)

    model_table.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(model_table.reset_index(drop=True), pd.read_parquet(path))
    print("model_table.parquet round-trips exactly :)")


    # saving the k-fold and boostrapping samples
    split_path_k_fold = PROCESSED / "splits/kfold.parquet"
    split_path_bootstrap = PROCESSED / "splits/bootstrap.parquet"

    split_path_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    split_path_k_fold.parent.mkdir(parents=True, exist_ok=True)

    k_fold = get_k_fold(households)
    bootstrap = get_bootstrap(households)

    check_splits(kfold=k_fold, bootstrap=bootstrap, households=households)

    k_fold.to_parquet(split_path_k_fold, index=False)
    bootstrap.to_parquet(split_path_bootstrap, index=False)



















if __name__ == "__main__":
    main()
