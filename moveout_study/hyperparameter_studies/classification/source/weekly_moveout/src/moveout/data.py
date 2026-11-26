"""Move-out study source summary.
Purpose: Load provider extracts and build the single leakage-controlled dataset shared by all models.
Inputs: Validates panel structure/cutoffs, constructs endpoint ledgers, engineers lagged history, and assigns grouped repeated folds.
Outputs: Writes prepared Parquet/CSV/JSON artifacts plus feature and data-quality audits.
Workflow: Features use information available at or before each prediction origin; target and identity columns are forbidden predictors.
Method note: Weekly and survival analyses therefore differ in outcomes while sharing source facts and split assignments.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

# The reported feature set reproduces the narrower feature contract used in the first study run.
REPORTED = ["week_index", "outstanding_lag1", "delinquent_lag1",
            "rent_to_income_ratio", "work_drive_min", "prior_home_distance_mi",
            "n_fobs", "n_pets", "income_monthly_est", "site"]
# Static candidate fields are copied from the resident record and do not change by panel week.
STATIC = ["unit_known", "occupancy_seq", "n_pets", "has_pet", "n_fobs",
          "income_monthly_est", "income_band_monthly", "income_band_hourly_equiv",
          "income_basis", "income_quality_flag", "prior_rent_monthly",
          "payment_frequency", "security_deposit", "pet_deposit", "total_deposits",
          "initial_discount", "discount_group", "first_responder", "prior_home_zip3",
          "prior_home_distance_mi", "prior_home_drive_min", "prior_home_straight_mi",
          "relocated_over_25mi", "work_distance_mi", "work_drive_min",
          "work_straight_mi", "commute_band", "prior_home_distance_band",
          "work_location_source", "work_geocode_confidence", "application_linked",
          "application_link_method"]
SENSITIVE = ["pc_n_adults", "pc_n_children", "pc_has_children",
             "pc_age_band_at_movein", "pc_n_esa_pets", "pc_has_esa_pet", "sc_eviction_history"]
CATEGORICAL = {"site", "income_band_monthly", "income_band_hourly_equiv", "income_basis",
               "income_quality_flag", "payment_frequency", "initial_discount", "discount_group",
               "first_responder", "prior_home_zip3", "commute_band", "prior_home_distance_band",
               "work_location_source", "application_link_method", "pc_age_band_at_movein",
               "sc_eviction_history"}
# These identity, outcome, and post-outcome columns are prohibited to prevent target leakage.
NEVER = {"resident_id", "unit_key", "unit_total_occupancies", "move_in_week", "move_out_week",
         "tenure_days", "tenure_weeks", "is_active", "is_censored", "lease_status", "churn_type",
         "move_out_reason", "notice_days_before_moveout", "data_quality_flags", "event", "duration",
         "churned_this_week", "censored_this_week", "active_end_of_week", "at_risk", "week_start",
         "cohort_week", "n_invoice_cycles", "total_billed", "total_paid", "total_outstanding",
         "collection_rate", "n_cycles_unpaid", "n_cycles_partial", "pct_cycles_paid", "n_late_payments",
         "mean_days_late", "max_days_late", "primary_payment_method", "weekly_rent", "weekly_rent_modal"}
RAW_WEEKLY = ["billed_this_week", "paid_this_week", "cum_billed", "cum_paid", "outstanding"]

def read_sources(directory):
    """Read provider resident, weekly, and optional invoice files and compute source-file SHA-256 hashes.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    directory = Path(directory)
    if (directory / "data/residents.csv").exists(): directory /= "data"
    r = pd.read_csv(directory / "residents.csv")
    w = pd.read_csv(directory / "weekly_panel.csv")
    ipath = directory / "invoice_cycles.csv"
    invoices = pd.read_csv(ipath) if ipath.exists() else None
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [directory / "residents.csv", directory / "weekly_panel.csv", ipath] if p.exists()}
    return r, w, invoices, hashes

def structural_validation(residents, weekly, cfg):
    """Check identities, panel continuity, binary/terminal outcomes, cutoff compliance, and optional expected counts.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    errors = []
    cutoff = pd.Timestamp(cfg["censor_date"])
    if pd.to_datetime(weekly.week_start, errors="coerce").gt(cutoff).any():
        errors.append("weekly_row_after_cutoff")
    if "move_in_week" in residents and pd.to_datetime(residents.move_in_week, errors="coerce").gt(cutoff).any():
        errors.append("move_in_after_cutoff")
    if not residents.resident_id.is_unique: errors.append("duplicate_resident_id")
    if weekly.duplicated(["resident_id", "week_index"]).any(): errors.append("duplicate_week_key")
    if not set(weekly.resident_id).issubset(set(residents.resident_id)): errors.append("orphan_weekly_id")
    if not weekly.churned_this_week.isin([0, 1]).all(): errors.append("nonbinary_target")
    if not weekly.censored_this_week.isin([0, 1]).all(): errors.append("nonbinary_censor")
    # Validate the full ordered history of each tenancy, including terminal event/censor rules.
    for rid, g in weekly.groupby("resident_id", sort=True):
        g = g.sort_values("week_index")
        if not np.array_equal(g.week_index, np.arange(len(g))): errors.append(f"week_gap:{rid}")
        if g.churned_this_week.sum() > 1: errors.append(f"multiple_events:{rid}")
        if g.churned_this_week.sum() and g.churned_this_week.iloc[-1] != 1:
            errors.append(f"nonterminal_event:{rid}")
        if g.censored_this_week.sum() != 1 - g.churned_this_week.sum(): errors.append(f"terminal_conflict:{rid}")
        if g.censored_this_week.sum() and g.censored_this_week.iloc[-1] != 1:
            errors.append(f"nonterminal_censor:{rid}")
    counts = dict(residents=len(residents), households=weekly.resident_id.nunique(),
                  rows=len(weekly), events=int(weekly.churned_this_week.sum()),
                  censors=int(weekly.censored_this_week.sum()))
    if cfg.get("strict_counts"):
        for key in ["households", "rows", "events"]:
            expected = cfg.get("expected_" + key)
            if expected is not None and counts[key] != expected: errors.append(f"count_{key}:{counts[key]}!={expected}")
    return dict(counts=counts, errors=errors, passed=not errors)

def make_endpoints(residents, weekly, cfg, overrides=None):
    """Use one panel-week clock for predictors and targets; keep source days diagnostic."""
    rows = []
    source = residents.set_index("resident_id")
    cutoff = pd.Timestamp(cfg["censor_date"])
    for rid, group in weekly.groupby("resident_id", sort=True):
        g = group.sort_values("week_index"); last = g.iloc[-1]; r = source.loc[rid]
        event = int(g.churned_this_week.max()); terminal = float(last.week_index)
        duration = terminal + 1.0
        issues = []
        days = pd.to_numeric(r.get("tenure_days"), errors="coerce")
        if pd.notna(days) and days < 0: issues.append("negative_source_duration")
        event_date = pd.to_datetime(r.get("move_out_week"), errors="coerce")
        if event and pd.notna(event_date) and event_date > cutoff: issues.append("event_after_cutoff")
        if cfg["endpoint_policy"] != "panel" and not event:
            start = pd.to_datetime(last.get("week_start"), errors="coerce")
            if pd.notna(start): duration = terminal + float(np.clip((cutoff-start).days / 7, 0, 1))
        known = int(r.get("unit_known", 0) or 0) if pd.notna(r.get("unit_known")) else 0
        rows.append(dict(resident_id=rid, site=r.get("site"), unit_key=r.get("unit_key", rid),
                         cluster_id=r.get("unit_key", rid) if known else rid,
                         event=event, duration=duration, source_duration_days=days,
                         source_duration_weeks=days/7 if pd.notna(days) else np.nan,
                         last_week_index=terminal, endpoint_issues=";".join(issues),
                         qualified=not issues or cfg["endpoint_policy"] == "panel"))
    endpoints = pd.DataFrame(rows)
    if overrides is not None:
        o = pd.read_csv(overrides)
        if not o.resident_id.is_unique: raise ValueError("Duplicate endpoint overrides")
        if not set(o.resident_id).issubset(set(endpoints.resident_id)): raise ValueError("Unknown override ID")
        for row in o.to_dict("records"):
            match = endpoints.resident_id.eq(row["resident_id"])
            for col in ["duration", "event", "qualified"]:
                if col in row and pd.notna(row[col]): endpoints.loc[match, col] = row[col]
            if not row.get("reason"): raise ValueError("Each endpoint override requires a reason")
            endpoints.loc[match, "endpoint_issues"] = "override:" + str(row["reason"])
    endpoints["qualified"] = endpoints.qualified.astype(bool) & endpoints.duration.gt(0)
    if not endpoints.event.isin([0, 1]).all(): raise ValueError("Invalid overridden event")
    if cfg["endpoint_policy"] == "error" and not endpoints.qualified.all():
        raise ValueError("Endpoint issues require overrides or explicit endpoint_policy=exclude/panel")
    return endpoints

def invoice_history(weekly, invoices):
    """Only billing cycles strictly before origin; never eventual payments/status/lateness."""
    names = ["prior_invoice_count", "prior_completed_cycle_count", "prior_cycle_days_median",
             "past_weekly_rent_estimate", "prior_credit_cycle_fraction", "prior_weekly_cycle_fraction",
             "prior_monthly_cycle_fraction"]
    out = pd.DataFrame(np.nan, index=weekly.index, columns=names)
    if invoices is None: return out
    invoices = invoices.copy()
    invoices["start"] = pd.to_datetime(invoices.billing_start_week, errors="coerce")
    invoices["end"] = pd.to_datetime(invoices.billing_end_week, errors="coerce")
    sources = dict(tuple(invoices.groupby("resident_id")))
    for rid, group in weekly.groupby("resident_id", sort=False):
        inv = sources.get(rid)
        if inv is None: continue
        beginning = pd.to_datetime(group.week_start, errors="coerce").min()
        inv = inv[inv.start.ge(beginning)]
        for ix, row in group.iterrows():
            origin = pd.to_datetime(row.week_start, errors="coerce")
            prior = inv[inv.start.lt(origin)]
            completed = prior[prior.end.lt(origin) & prior.billing_days.between(1, 62)]
            regular = completed[completed.amount_billed.gt(0) & completed.cycle_type.isin(["weekly", "bi_weekly", "monthly"])]
            rent = (regular.amount_billed * 7 / regular.billing_days).median()
            out.loc[ix] = [len(prior), len(completed), completed.billing_days.median(), rent,
                           prior.amount_billed.lt(0).mean(), prior.cycle_type.eq("weekly").mean(),
                           prior.cycle_type.eq("monthly").mean()]
    return out

def build_master(residents, weekly, invoices, cfg):
    """Merge static and time-varying data, engineer lagged/rolling features, and return the approved feature contract.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    weekly = weekly.sort_values(["resident_id", "week_index"]).copy().reset_index(drop=True)
    numeric_resident = [c for c in residents if c not in CATEGORICAL and c not in NEVER and c not in {"site"}]
    residents = residents.copy()
    for c in CATEGORICAL & set(residents):
        residents[c] = residents[c].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    for c in ["outstanding", "delinquent"]:
        weekly[c + "_lag1"] = weekly.groupby("resident_id")[c].shift(1)
    weekly.loc[weekly.week_index.eq(0), "outstanding_lag1"] = 0.0
    weekly.loc[weekly.week_index.eq(0) & weekly.delinquent.notna(), "delinquent_lag1"] = 0.0
    if cfg["feature_set"] == "reported":
        residents.loc[~residents.rent_to_income_ratio.between(0, 1), "rent_to_income_ratio"] = np.nan
        columns = ["resident_id"] + [c for c in REPORTED if c in residents and c != "site"]
        master = weekly.merge(residents[columns], on="resident_id", how="left", validate="many_to_one")
        features = REPORTED.copy()
    else:
        static = [c for c in STATIC if c in residents]
        if cfg.get("include_sensitive_research_features"): static += [c for c in SENSITIVE if c in residents]
        master = weekly.merge(residents[["resident_id"] + static], on="resident_id", how="left", validate="many_to_one")
        features = ["week_index", "site"] + static
        if "work_geocode_confidence" in master:
            master["geocode_confidence_invalid"] = ~master.work_geocode_confidence.between(0, 1) & master.work_geocode_confidence.notna()
            master.loc[master.geocode_confidence_invalid, "work_geocode_confidence"] = np.nan
            features.append("geocode_confidence_invalid")
        group = master.groupby("resident_id", sort=False)
        # Lag and roll raw financial activity so a prediction never uses information from its target week.
        for c in RAW_WEEKLY:
            if c not in master: continue
            lag = c + "_lag1"
            master[lag] = group[c].shift(1)
            master.loc[master.week_index.eq(0), lag] = 0.0
            features.append(lag)
            if c in {"billed_this_week", "paid_this_week", "outstanding"}:
                for window in [4, 8]:
                    for stat in ["mean", "std"]:
                        name = f"{c}_past{window}_{stat}"
                        master[name] = master.groupby("resident_id")[lag].transform(
                            lambda x: getattr(x.rolling(window, min_periods=1), stat)())
                        features.append(name)
        master["outstanding_change_lag1"] = master.groupby("resident_id").outstanding_lag1.diff()
        master["past_collection_ratio"] = master.cum_paid_lag1 / master.cum_billed_lag1.replace(0, np.nan)
        master["prior_rent_income_ratio"] = master.prior_rent_monthly / master.income_monthly_est.replace(0, np.nan)
        features += ["outstanding_change_lag1", "past_collection_ratio", "prior_rent_income_ratio"]
        inv = invoice_history(master, invoices)
        master = pd.concat([master, inv], axis=1); features += list(inv)
        master["historical_affordability"] = master.past_weekly_rent_estimate * (52/12) / master.income_monthly_est.replace(0, np.nan)
        master["historical_delinquent"] = np.where(master.past_weekly_rent_estimate.notna(),
                                                   (master.outstanding_lag1 > master.past_weekly_rent_estimate).astype(float), np.nan)
        features += ["historical_affordability", "historical_delinquent"]
        # Add trainable missingness indicators without converting absent history to invented values.
        for c in list(dict.fromkeys(features)):
            if c not in CATEGORICAL and master[c].isna().any():
                master[c + "_missing"] = master[c].isna().astype(float); features.append(c + "_missing")
    features = list(dict.fromkeys(features))
    if NEVER & set(features): raise ValueError("Target/identifier leakage in feature list")
    master[features] = master[features].replace([np.inf, -np.inf], np.nan)
    audit = []
    for c in sorted(set(residents) | set(weekly) | set(features)):
        included = c in features
        reason = "available_static_or_prior_week"
        if c in NEVER or c in RAW_WEEKLY or c == "delinquent": reason = "identifier_target_or_future_summary"
        if c in SENSITIVE and not cfg.get("include_sensitive_research_features"): reason = "source_restricted_diagnostics_only"
        if c in {"rent_to_income_ratio", "rent_to_income_band", "delinquent_lag1"}: reason = "full_ledger_threshold_historical_only"
        if c.startswith("prior_") and "cycle" in c or c in set(inv.columns if cfg["feature_set"] != "reported" else []):
            reason = "prior_billing_start_proxy_not_transaction_audit"
        if c in {"n_fobs", "n_pets", "has_pet", "initial_discount", "security_deposit", "pet_deposit", "total_deposits"}:
            reason = "static_asof_assumed_needs_provider_confirmation"
        audit.append(dict(feature=c, included=included, reason=reason,
                          missing_fraction=float(master[c].isna().mean()) if c in master else None))
    return master, features, pd.DataFrame(audit)

def make_folds(master, cfg):
    """Assign resident-level repeated stratified folds so no tenancy appears in both training and test data.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    h = master.groupby("resident_id")["churned_this_week"].max().reset_index(name="event")
    if h.event.value_counts().min() < cfg["folds"]: raise ValueError("Too few events/non-events for outer folds")
    splitter = RepeatedStratifiedKFold(n_splits=cfg["folds"], n_repeats=cfg["repeats"], random_state=cfg["seed"])
    frames = []
    for n, (_, ix) in enumerate(splitter.split(h, h.event)):
        frames.append(pd.DataFrame(dict(resident_id=h.resident_id.iloc[ix], repeat=n//cfg["folds"], fold=n%cfg["folds"])))
    return pd.concat(frames, ignore_index=True)

def split_ids(frame, folds, repeat, fold, cfg):
    """Split one outer training set into deterministic train/validation residents while retaining the held-out test fold.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    mapping = folds[folds.repeat.eq(repeat)]
    if not mapping.resident_id.is_unique: raise ValueError("Duplicate fold IDs")
    eligible = set(frame.resident_id)
    test = set(mapping.loc[mapping.fold.eq(fold), "resident_id"]) & eligible
    train_outer = set(mapping.loc[mapping.fold.ne(fold), "resident_id"]) & eligible
    if test & train_outer: raise ValueError("Household leakage")
    households = frame[frame.resident_id.isin(train_outer)].groupby("resident_id").agg(
        event=("event" if "event" in frame else "churned_this_week", "max"))
    if len(households) < 10 or households.event.nunique() < 2 or households.event.value_counts().min() < 2:
        raise ValueError("Insufficient households/events for inner validation")
    train, val = train_test_split(households.index, test_size=cfg["validation_fraction"],
                                 random_state=cfg["seed"] + 100*repeat + fold, stratify=households.event)
    return set(train), set(val), test

def landmark(master, endpoints, week):
    """Create a survival risk set at a requested week and express remaining duration relative to that origin.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    ep = endpoints[endpoints.qualified & endpoints.duration.gt(week)].copy()
    # Prepared masters already carry endpoint metadata; use the ledger once, never suffix it.
    snapshot = master[master.week_index.eq(week)].drop(columns=[c for c in ep if c not in {"resident_id", "site"} and c in master])
    result = snapshot.merge(ep.drop(columns=["site"]), on="resident_id", validate="one_to_one")
    result["duration"] -= week
    result["prediction_origin"] = week
    return result

def prepare(directory, output, cfg, overrides=None):
    """Run source validation and feature preparation, then persist all reusable cohort, split, and audit artifacts.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    r, w, invoices, hashes = read_sources(directory)
    validation = structural_validation(r, w, cfg)
    (output / "validation_report.json").write_text(json.dumps(validation, indent=2))
    if not validation["passed"]: raise ValueError(f"Data validation failed: {validation['errors']}")
    if invoices is not None and cfg.get("require_snapshot_cutoff", False):
        cutoff = pd.Timestamp(cfg["censor_date"])
        if pd.to_datetime(invoices.billing_start_week, errors="coerce").gt(cutoff).any():
            raise ValueError("Billing cycles after snapshot cutoff")
        paid_after = pd.to_datetime(invoices.paid_week, errors="coerce").gt(cutoff)
        if (paid_after & invoices.amount_paid.fillna(0).ne(0)).any():
            raise ValueError("Post-cutoff payments must be rewound by provider")
    r.loc[~r.resident_id.isin(w.resident_id)].to_csv(output / "residents_without_panel.csv", index=False)
    ep = make_endpoints(r, w, cfg, overrides)
    master, features, audit = build_master(r, w, invoices, cfg)
    master = master.merge(ep[["resident_id", "qualified", "duration", "event", "cluster_id"]], on="resident_id", validate="many_to_one")
    master["eligible_weekly"] = master.qualified
    if cfg["endpoint_policy"] != "panel":
        # Censored incomplete terminal intervals cannot be labeled full negative weeks.
        master["eligible_weekly"] &= master.event.eq(1) | (master.week_index + 1).le(master.duration + 1e-9)
    folds = make_folds(master, cfg)
    master.to_parquet(output / "weekly_master.parquet", index=False)
    ep.to_csv(output / "cohort_ledger.csv", index=False)
    audit.to_csv(output / "feature_audit.csv", index=False)
    folds.to_csv(output / "fold_assignments.csv", index=False)
    (output / "features.json").write_text(json.dumps(features, indent=2))
    # Diagnostic attributes remain available for subgroup evaluation, not automatic model inputs.
    r[["resident_id"] + [c for c in SENSITIVE if c in r]].to_csv(output / "subgroup_attributes.csv", index=False)
    summary = dict(starting_households=int(len(ep)), qualified_households=int(ep.qualified.sum()),
                   qualified_events=int(ep.loc[ep.qualified, "event"].sum()), features=len(features),
                   endpoint_clock="panel week; events at interval end; partial censor exposure retained",
                   prospective_feature_provenance="not_fully_verified", source_hashes=hashes,
                   config=cfg)
    (output / "preparation.json").write_text(json.dumps(summary, indent=2))
    return summary
