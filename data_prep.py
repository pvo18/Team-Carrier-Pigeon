"""
Data prep for the Gated Rentals tenant retention baselines (Team Carrier Pigeon).
Column names etc. are based on DATA_DICTIONARY.md.

Builds the modeling table from residents.csv and weekly_panel.csv. One row per
household-week, with the features joined on and lagged properly.

Note: this does NOT do the train/val/test split. That's Matt's
K-fold / bootstrap utility. This file only joins the two tables and
makes the lagged features, so everything that comes out of here still has to
go through the split function before it goes into a model.

Stuff fixed from what we thought of earlier:

1. We were planning on using `pc_n_children` as a model input, which we cannot do. The
   data dictionary says anything starting with `pc_` is a protected class
   (or a proxy for one) under the Fair Housing Act, so those are only for
   controls / bias checks. I switched to `n_fobs` as the household size
   proxy. It is kind of a weak proxy, but the dictionary suggests it and it
   is not protected.
2. `weekly_panel.csv` does not actually have the income/commute/household
   columns. It only has the ids, the week info, the label, and the billing
   fields. The other features are in `residents.csv` so we have to join on
   `resident_id`.
3. The summary columns in `residents.csv` (total_billed, n_late_payments,
   mean_days_late, collection_rate, etc.) are calculated over the household's
   whole tenancy. If we used them for an early week we'd be leaking info
   from weeks that haven't happened yet. I took them out of the features
   completely. They're still fine for descriptive stats, just not for
   predicting week by week.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMN = "resident_id"      #One household. Do not split a household's weeks across folds
WEEK_COLUMN = "week_index"        #Weeks since move-in, starts at 0
LABEL_COLUMN = "churned_this_week"

#Static features from residents.csv. Ok to use since we know them at signup / move-in
STATIC_NUMERIC_FEATURES = [
    "income_monthly_est",
    "work_drive_min",
    "n_fobs",   #Rough proxy for household size, instead of pc_n_children
    "n_pets",   #Regular pets only. Not the same as pc_n_esa_pets (protected)
]
STATIC_CATEGORICAL_FEATURES = [
    "income_band_hourly_equiv",  #Same bands as the hypothesis in our proposal
    "commute_band",              #Main explanatory variable from the proposal, but banded
]

#Pulling these from residents.csv but only for filtering/checking
#Data quality, not model inputs.
RESIDENT_DIAGNOSTIC_COLUMNS = [
    "income_quality_flag",     #Filter to 'ok' for the high-confidence subset sensitivity check
    "work_location_source",    #job_address is better, employer_name is noisier
    "work_geocode_confidence", #Drop low confidence rows before trusting commute
]

RESIDENT_COLUMNS_TO_JOIN = (
    [GROUP_COLUMN]
    + STATIC_NUMERIC_FEATURES
    + STATIC_CATEGORICAL_FEATURES
    + RESIDENT_DIAGNOSTIC_COLUMNS
)

#Time-varying features, made from the billing columns in weekly_panel.csv.
#These are lagged by one week (the previous week's values). We do not use
#the same-week columns (billed_this_week, paid_this_week, cum_billed,
#cum_paid, outstanding, delinquent) because in the week someone moves out
#those can include deposit refunds/reversals (README limitation #5), and
#then the model could tell that they're leaving. Lagging by a week means 
#every feature only uses info from before the current week, same as week_index.
TIME_VARYING_NUMERIC_FEATURES = [
    "outstanding_lag1",
]
TIME_VARYING_BINARY_FEATURES = [
    "delinquent_lag1",
]

NUMERIC_FEATURES = (
    STATIC_NUMERIC_FEATURES
    + [WEEK_COLUMN]
    + TIME_VARYING_NUMERIC_FEATURES
    + TIME_VARYING_BINARY_FEATURES  #Just 0/1 so treating it as numeric is fine
)
CATEGORICAL_FEATURES = STATIC_CATEGORICAL_FEATURES

#Columns we cannot use as model inputs, either because they leak the outcome
#or because they're protected / screening attributes (HUD disparate-impact
#guidance). Keeping the list here so if someone adds a feature later they
#can check it against this first.
LEAKAGE_OR_RESTRICTED_COLUMNS = [
    #Leakage: either gives away the outcome or is calculated from it
    "move_in_week",
    "move_out_week",
    "tenure_days",
    "tenure_weeks",
    "is_active",
    "is_censored",
    "lease_status",
    "churn_type",
    "move_out_reason",
    "notice_days_before_moveout",
    "active_end_of_week",
    "censored_this_week",
    #Whole-tenancy summaries (would leak future weeks into early rows)
    "n_invoice_cycles",
    "total_billed",
    "total_paid",
    "total_outstanding",
    "collection_rate",
    "n_cycles_unpaid",
    "n_cycles_partial",
    "pct_cycles_paid",
    "n_late_payments",
    "mean_days_late",
    "max_days_late",
    #Same-week billing (use the _lag1 versions, explained above)
    "billed_this_week",
    "paid_this_week",
    "cum_billed",
    "cum_paid",
    "outstanding",
    "delinquent",
    #Protected class / proxies under the Fair Housing Act (pc_ prefix).
    #Only for diagnostics and fairness checks, never as inputs
    "pc_n_adults",
    "pc_n_children",
    "pc_has_children",
    "pc_age_band_at_movein",
    "pc_n_esa_pets",
    "pc_has_esa_pet",
    #Screening attribute, HUD disparate-impact guidance (sc_ prefix)
    "sc_eviction_history",
]


def build_modeling_table(residents_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the static features from residents.csv onto the weekly rows from
    weekly_panel.csv and adds the lagged billing features.

    Does not split, shuffle, or filter anything, and does not touch
    train/val/test. It returns the full table with every week for every
    household, sorted by household and then week. Matt's split utility takes
    the output of this.
    """
    weekly_df = weekly_df.sort_values([GROUP_COLUMN, WEEK_COLUMN]).copy()

    #Shift the billing columns down by one week within each household, so
    #week w gets the billing from week w-1 and not its own (week w's own
    #values might have a move-out refund/reversal in them).
    grouped = weekly_df.groupby(GROUP_COLUMN, sort=False)
    weekly_df["outstanding_lag1"] = grouped["outstanding"].shift(1)
    weekly_df["delinquent_lag1"] = grouped["delinquent"].shift(1)
    #Week 0 is the move-in week so there's no week before it and the shift
    #gives NaN. Filling with 0 because they really did owe nothing before
    #they lived here, so it is the actual value and not something we need to impute.
    weekly_df["outstanding_lag1"] = weekly_df["outstanding_lag1"].fillna(0.0)
    weekly_df["delinquent_lag1"] = weekly_df["delinquent_lag1"].fillna(0)

    resident_static = residents_df[RESIDENT_COLUMNS_TO_JOIN]
    modeling_table = weekly_df.merge(resident_static, on=GROUP_COLUMN, how="left", validate="many_to_one")

    return modeling_table
