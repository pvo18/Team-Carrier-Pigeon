# Weekly move-out survival comparison

Runnable Python package for the tenancy data supplied in this conversation. The primary profile searches broadly across the available features, rather than restricting the analysis to the midterm's feature list.

The package includes source code, configurations, tests, the three input CSVs, an updated code specification, and verification artifacts. Source input files are never edited by the program.

## Install

Python 3.12 was used for verification. Create a virtual environment and install CPU Torch first to avoid an unnecessary GPU dependency download:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
moveout --help
```

`requirements-tested.txt` records the direct versions used for verification. It is a tested environment reference, not a full cross-platform dependency lock. Install CPU Torch before using that reference as well.

## Prepare the expanded data

```bash
moveout prepare --data data/provider --config configs/expanded.yaml --out outputs/prepared
```

This creates the shared weekly master, 98 raw candidate predictors, fold assignments, feature audit, cohort ledger, source hashes and subgroup diagnostic attributes. One-hot dimensions vary by training fold.

Expanded inputs cover tenancy/site, income, commute, prior-home characteristics, household/lease information, deposits/discounts, missingness, lagged/rolling payments and earlier billing-cycle history. Identifiers, outcome-derived columns and future-tenancy summaries are excluded. Full-ledger modal rent and its delinquency threshold are replaced with prior-history estimates. Source-restricted fields remain diagnostic-only by default; an explicit configuration setting supports a research sensitivity.

The starting ledger retains all 444 panel IDs. The default qualified comparison excludes two unresolved endpoint conflicts and has 442 tenancies/124 events. `configs/reported.yaml` preserves the historical 444/125 panel convention. Every model in a given comparison uses the same documented exclusions. Corrected endpoints can be supplied with `--endpoint-overrides FILE.csv`; each row requires `resident_id` and a `reason`, with optional `duration`, `event` and `qualified` values.

Feature timing still depends on the supplied extract. Static-field as-of dates and billing-start-as-availability are assumptions recorded in `feature_audit.csv`. These files support retrospective experimentation; they do not independently establish transaction-time or prospective validation.

## Run the scientific comparisons

```bash
moveout run-weekly --prepared outputs/prepared --config configs/expanded.yaml --out outputs/weekly
moveout run-survival --prepared outputs/prepared --config configs/expanded.yaml --out outputs/survival
```

The defaults run five repeated household-stratified folds, with inner household validation for selection and early stopping. Weekly defaults are constant and tenure/site references, unweighted logistic and small FFNNs. Survival defaults are KM, classical Cox, linear PyCox CoxPH and neural CoxPH, at origins zero and four weeks. Neural candidates use recorded seed ensembles. All candidate selection uses validation records, not outer-test outcomes.

For an initial scientific subset, add `--repeat 0 --fold 0`. Choose a fresh output directory for every new training run. A run with some model/fold failures writes those failures and returns a nonzero exit status; other completed results remain reviewable.

Historical weighted reference, using its separate prepared profile:

```bash
moveout prepare --data data/provider --config configs/reported.yaml --out outputs/reported_prepared
moveout run-weekly --prepared outputs/reported_prepared --config configs/reported.yaml --models weighted_logistic,weighted_ffnn --repeat 0 --fold 0 --out outputs/historical
```

Historical logistic initialization was not seeded in the supplied source. The later 25-fold FFNN runner was absent. New reconstructed fits are labeled accordingly; the program does not promise bit-identical retraining of those earlier fits.

## Feature relationships and ablations

Each selected feature model saves exploratory inner-validation permutation importance. Importance is predictive, not causal, and correlated variables can substitute for one another. Evaluate group contributions with separate runs on identical fold assignments:

```bash
moveout run-weekly --prepared outputs/prepared --config configs/expanded.yaml --drop-group commute --out outputs/weekly_without_commute
```

Other groups are `income`, `payments`, `relocation` and `household`. The main expanded run includes all these groups. Do not select a winning feature set using the outer test and then describe that same test as independent confirmation.

## LogisticHazard

LogisticHazard is implemented but conditional for comparative runs:

```bash
moveout assess-logistic-hazard --run-dir outputs/survival --out outputs/gate.json
```

Training PH flags alone result in `review_required`. To document a scientific decision, pass `--evidence-json FILE.json` containing `development_only: true`, `functional_form_checked: true`, `site_checked: true`, a nonempty `reason`, a list of existing `evidence_files`, and either `time_varying_effects: true` or `persistent_miscalibration: true`. The file records scientific review; it is not an outer-test tuning shortcut.

```bash
moveout run-survival --prepared outputs/prepared --config configs/expanded.yaml --models logistic_hazard --gate outputs/gate.json --out outputs/logistic_hazard
```

`--exploratory` permits an explicitly labeled methodological experiment. A changing baseline hazard alone does not justify abandoning PH. LogisticHazard's dummy time-zero head is fixed to zero hazard; censoring contributes completed weekly intervals only.

## Predictions and saved ensembles

Pass a selected model directory containing `ensemble.json` to average all its saved seed members. A `member_0` directory gives an individual member instead. Snapshot inputs must contain that model's prepared feature columns.

```bash
moveout predict --model outputs/weekly/repeat_0/fold_0/ffnn --input outputs/prepared/weekly_master.parquet --out outputs/predictions.csv
```

The prediction command does not make its own train/test selection: supply the snapshots you actually want scored. It never fits preprocessing on prediction inputs. Torch weights use state-dictionary loading; load joblib checkpoints only from trusted runs of this package.

## Midterm reconciliation

```bash
moveout reconcile-midterm --midterm-path PATH_TO_EXTRACTED_TEAM_REPOSITORY --data data/provider --out outputs/reconciliation
```

This checks source hashes, exact reported feature order, seed-42 outer folds, all later saved test row keys/labels, and recomputed metrics. It does not execute the third-party source or load its checkpoints. The supplied code archive is not duplicated inside this package; the verified reconciliation report is included.

The later saved FFNN results average AUC 0.599669 and Brier 0.018318. A constant outer-training-rate reference averages Brier 0.018311. The later run largely resolves the original weighted inflation, but its probability-score advantage remains unproven. Repeated folds are dependent observations.

## Verification versus study results

```bash
python -m pytest -q
```

The included `verification/weekly_smoke` and `verification/survival_smoke` use three epochs on one outer fold to verify execution, reports and checkpoint paths. They are NOT the completed expanded scientific experiment. Full comparative training must be launched with the normal profile. `verification_status.md` records exactly what was exercised.

Metrics include censor-aware survival Brier/log loss, discrimination, calibration, grouped uncertainty and per-repeat summaries. Unsupported horizons receive explicit status instead of invented scores. This extract does not support a 52-week headline evaluation. Weekly classifiers do not fabricate future multiweek financial histories to produce unconditional curves.

## Layout

- `src/moveout`: implementation and command-line entry point.
- `configs`: expanded, reported and verification profiles.
- `tests`: behavioral and adapter tests.
- `data/provider`: supplied resident, weekly and invoice CSVs and provider documentation.
- `docs`: updated code specification.
- `verification`: preparation/audit, saved-result reconciliation and short execution checks.

Run directories contain manifests, source/code hashes, exact inner memberships, selected settings, ensemble members, held-out predictions, calibration/PH diagnostics, feature importance, failure records and Markdown/JSON comparison reports.
