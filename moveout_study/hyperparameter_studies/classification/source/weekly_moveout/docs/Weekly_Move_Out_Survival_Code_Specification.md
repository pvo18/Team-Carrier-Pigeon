# Weekly Move-Out Survival Analysis: Code Specification

Version 1.2 — implemented package version 0.2.0  
Date: November 16, 2026  
Status: runnable implementation; expanded feature search is the primary analysis

## 1. Objective and scope

Compare weekly probability references, logistic regression, feed-forward neural networks (FFNN), classical Cox proportional hazards, linear PyCox CoxPH, neural PyCox CoxPH, and conditionally LogisticHazard. Estimate next-week move-out probability among tenancies present at the prediction origin. Evaluate four-, eight-, and twelve-week survival probabilities only when follow-up and censoring-weight support permit.

The user's revised direction takes precedence over the earlier fixed-feature proposal: examine all provided data that can reasonably be used at the prediction origin. The primary analysis is not limited to the midterm's nine numeric predictors. Retain that original feature set as a secondary reference.

Implementation is in weekly_moveout/. Code is organized into config.py, data.py, models.py, metrics.py, pipeline.py, reconciliation.py and cli.py. The modules consolidate the earlier proposed responsibilities without changing their intended separation.

## 2. Inputs and provenance

Required inputs are residents.csv and weekly_panel.csv. invoice_cycles.csv is used when supplied to derive earlier billing-cycle features. Read the provider README and dictionary for interpretation; existing cohort, KM and CLV outputs are descriptive references, not training inputs.

The original supplied panel has 444 tenancies, 6,731 weekly rows, 125 recorded moves and 319 terminal censoring rows. The resident source contains 455 rows. The nine missing-start/duration records are already outside the 444 and must not be subtracted again.

The received Team-Carrier-Pigeon-main.zip identifies archive commit d34a090078a8fe8ef3c1866466e06078f128b276. Its resident, weekly and invoice CSVs match the earlier extract byte for byte. The code reconciles saved predictions without executing imported project scripts or loading their checkpoint/pickle files.

## 3. Execution profiles

| Profile | Purpose |
|---|---|
| configs/expanded.yaml | Primary analysis with broad candidate features, documented endpoint exclusions, grouped tuning and evaluation |
| configs/reported.yaml | Historical reference using the original feature order, cleaning rules and panel endpoints |
| configs/smoke.yaml | Three-epoch execution verification on one requested fold; not scientific model-selection evidence |

The package is not a production tenant-selection system. Its results describe exploratory retention prediction. No 52-week headline evaluation is permitted with this extract.

## 4. Feature contract

The supplied data produce 98 raw candidate predictors in the expanded profile, before one-hot encoding and training-fold removal of constant columns. This count is an observed outcome of this extract, not a hard limit.

| Group | Candidate information |
|---|---|
| Tenancy/site | Tenancy week, site, unit-link availability, occupancy sequence |
| Income/affordability | Continuous income, income bands, basis, quality, prior rent, prior-rent/income ratio, affordability from earlier billing cycles |
| Employment/commute | Routed and straight-line work distance, commute time/band, location source and validated geocode confidence |
| Relocation | Prior ZIP prefix, routed/straight-line distance, driving time, distance band and relocation indicator |
| Household/lease | Fob/pet counts, pet indicator, deposits, discount information, first-responder coding and payment frequency |
| Weekly financial history | Lagged billed/paid/cumulative amounts and balances; four/eight-week means and standard deviations; balance change and historical collection ratio |
| Invoice history | Earlier cycle counts, cadence, completed-cycle lengths, prior credits and a prior-cycle rent estimate |
| Missingness | Numeric missingness indicators plus train-fitted imputation |

Outcome dates, total tenure, move-out reasons, end-of-tenancy statuses, identifiers and full-tenancy financial summaries cannot enter the predictor matrix. They remain available for validation, targets and grouping. Unit identifiers are clustering keys rather than memorization inputs; total future occupancies are excluded.

The expanded profile replaces full-ledger modal rent and its delinquency threshold with estimates from earlier completed regular billing cycles. It uses prior-week financial values and strictly earlier invoice periods. It never uses eventual payment status, final invoice payment amounts or future lateness as early-week predictors.

Negative payments/credits are preserved. Invalid confidence values outside 0–1 become missing with an invalid-value indicator. Invoice periods with impossible or extreme lengths are omitted from cadence/rent estimation, not silently deleted from source data.

Static field timestamps and billing-start-as-availability are assumptions, not verified transaction-time provenance. The feature audit records them. Source-restricted attributes are exported for subgroup diagnostics; an explicit include_sensitive_research_features configuration enables a separate research sensitivity. They are not automatically inserted into the default predictor set.

The reported reference uses this exact numeric order: week_index, outstanding_lag1, delinquent_lag1, rent_to_income_ratio, work_drive_min, prior_home_distance_mi, n_fobs, n_pets, income_monthly_est; categorical site follows. Its rent/income values outside [0,1] become missing for faithful historical comparison.

## 5. Common data objects

1. Weekly master: one row per resident_id/week_index; source label/censor flags, candidate features, cluster identity, eligibility and endpoint metadata.
2. Cohort ledger: one row for every starting panel ID, including source days, panel-clock endpoint, event flag, issue codes, qualification and any override reason.
3. Fold assignments: resident_id, repeat and outer fold, established before fitting.
4. Landmark snapshots: one row per at-risk tenancy at a specified origin, with the same available features and residual time/event targets.
5. Feature audit: every candidate/source column, inclusion decision, timing rationale and missing fraction.

The master table and ledger never silently lose the original 444 identities. All compared models within an analysis consume common eligibility decisions and fold assignments.

## 6. Endpoint and clock policy

The default comparison profile explicitly excludes unresolved negative source duration and an event beyond the September 15 cutoff. On this extract that yields 442 qualified tenancies and 124 events. The historical panel profile retains 444/125. These are documented analysis populations, not competing counts of the supplied panel.

Supported endpoint policies:

- panel: preserve historical terminal-week labels and full intervals.
- exclude: exclude unresolved endpoint conflicts for every model, and retain the administrative fraction of the final censored week.
- error: require conflicts to be resolved before proceeding.

An optional endpoint-override CSV supplies resident_id, corrected duration/event/qualified values, and a required reason. Overrides are logged.

The implemented primary clock is the panel-week clock for both features and targets. Observed moves are assigned to the end of their reported event interval. Administrative censoring retains the observed fraction of its final calendar week. Source exact-day tenure is diagnostic and is not mixed silently with Monday-rounded predictor origins. An exact calendar move-in date is unnecessary when a consistent elapsed observation clock exists.

For weekly BCE, incomplete censored final intervals are excluded from full-week negative labels. For survival likelihoods, they contribute observed follow-up. The historical profile preserves the earlier convention for reproduction.

## 7. Splitting and preprocessing

The verified outer generator is RepeatedStratifiedKFold applied to a sorted one-row-per-household event table: seed 42, five folds, five repeats. Its assignments exactly match all 25 later saved FFNN test files. Each household appears in one test fold per repeat; folds/repeats are not independent experiments.

Within each outer training portion, stratify a 20% validation group at household level. Save exact inner membership. Use this validation portion for penalties, architecture, early stopping and ensemble selection. Outer test records never fit transformations or select models.

All model adapters share the relevant train-fitted transformer: numeric median imputation and standardization, missing categorical values labeled missing, dense one-hot encoding, rare-category pooling in the expanded profile and removal of constant training columns. The historical profile preserves its original category handling. Missing/unknown test categories do not trigger refitting.

Cox variants within a landmark use identical records and transformed fields. Landmark-specific constant week index is omitted. Weekly models train on pooled eligible weeks and are evaluated at common prediction weeks; their training structure differs from a single-landmark survival fit and is disclosed.

## 8. Model adapters

| Model | Implementation behavior |
|---|---|
| Constant reference | Training-only weekly event prevalence |
| Tenure/site reference | Four-week tenure bands by site, smoothed toward training prevalence |
| Weighted logistic | One linear Torch logit, positive-class weighting, historical fixed epochs when requested |
| Unweighted logistic | Same prepared features, ordinary BCE, validation-selected regularization |
| Weighted FFNN | Historical [16,8], ReLU/dropout, weighted BCE |
| Small unweighted FFNN | [4] or [8], ReLU/dropout, Adam, L2 and validation early stopping |
| KM | Training-only Kaplan–Meier reference |
| Classical Cox | lifelines ridge Cox, Efron event ties and Breslow baseline |
| Linear PyCox CoxPH | Linear log-relative-risk network, full batches and training-only baseline |
| Neural PyCox CoxPH | Small nonlinear network with the same landmark information |
| LogisticHazard | PyCox likelihood with weekly interval heads, no artificial event hazard at time zero |

Candidate penalty grids and seed lists are configuration values. Defaults use five neural seeds. Evaluate candidate seed ensembles on validation probabilities; save the selected members and raw per-member artifacts. Maximum epochs and early stopping govern optimization; no arbitrary wall-clock acquisition-style limiter is imposed.

Neural CoxPH retains proportional hazards. A changing baseline hazard is allowed by Cox and is not evidence that LogisticHazard is needed. PyCox's cumulative-sum risk-set limitation is recorded, especially with tied weekly durations. Training on full batches does not eliminate that limitation.

The dummy LogisticHazard origin head is fixed to zero hazard. Events map to the containing interval; censoring contributes only completed intervals. Censor follow-up beyond the configured grid at its end. Primary evaluation uses shared supported horizons.

## 9. Model interfaces and persistence

Every adapter fits using explicit training/validation tables, predicts weekly probability or survival where appropriate, and saves/reloads its artifacts. Weekly classifiers do not fabricate multiweek forecasts requiring unknown future payment trajectories.

For a survival curve starting at the prediction origin, next-week move-out risk is 1 − S(1). For a fixed-profile absolute-tenure curve conditional at a, it is 1 − S(a+1)/S(a). A landmark fit models residual time directly.

Checkpoint directories contain model metadata, fitted preprocessing, weights or lifelines estimator, histories and Cox baseline hazards as appropriate. Torch state dictionaries are loaded with weights_only=True. Joblib artifacts are trusted local package outputs, not a format for loading arbitrary third-party checkpoints.

## 10. Model selection and explanation

Choose weekly candidates using inner-validation Brier score. Choose survival candidates using training-censor-weighted validation Brier score at one week, subject to follow-up support. Preserve weighted outputs as historical scores and prioritize unweighted likelihoods for probability estimation.

Save validation-set permutation importance for candidate predictors, labeled exploratory and noncausal. Correlated inputs may share/substitute importance; a low individual permutation score does not prove irrelevance. Use --drop-group for separate commute, income, payment, relocation or household ablation runs on identical outer folds. Do not select features using outer-test importance or comparisons.

Large candidate matrices require strong regularization and conservative interpretation at approximately 125 observed moves. The expanded search does not imply that every feature should be retained in a final deployed model.

## 11. Evaluation and uncertainty

Weekly outputs include Brier score, ordinary log loss, ROC-AUC, average precision, predicted/observed event rates, calibration intercept/slope when enough cases exist, five-group calibration tables/plots, and tenancy-weighted versions of the probability scores.

Survival outputs include horizon-specific IPCW Brier/log loss, time-dependent AUC, horizon-specific IPCW concordance, cases/controls/unknown counts, probability calibration and supported integrated Brier scores. Estimate censoring from training data. Unknown outcomes before a horizon are not labeled non-movers. Return an unsupported status when follow-up, censor survival or model output support is inadequate.

Use identical scoring records, origins and horizons for paired comparisons. Never compare an ordinary weekly AUC directly with lifetime concordance as though they measured the same task.

Summarize out-of-fold results by repeat. Use paired household-cluster bootstrap uncertainty, with known-unit clustering as a sensitivity. Never derive narrow standard errors by treating all 25 folds as independent. Preserve person-week and tenancy weighting labels.

A proposed promotion screen is at least 5% relative improvement in the agreed primary probability score, acceptable calibration, stable development results and uncertainty compatible with meaningful benefit. This is a configurable practical criterion, not a universal statistical significance rule.

## 12. LogisticHazard gate

Training PH tests and scaled Schoenfeld residual plots are diagnostics, not automatic approval. assess-logistic-hazard writes a review_required artifact by default.

A passed gate requires documented development-only evidence of time-varying effects or persistent horizon-specific miscalibration, with functional-form and site checks and referenced evidence files. Comparative LogisticHazard execution requires this passed artifact. --exploratory explicitly permits a labeled methodological experiment, including adapter verification.

The original weighted inflation, a weak AUC or neural overfitting are not by themselves PH violations.

## 13. Configuration and commands

Key primary defaults: feature_set=expanded, seed=42, folds=5, repeats=5, endpoint_policy=exclude, landmarks=[0,4], horizons=[1,4,8,12], validation_fraction=0.20, epochs=300, patience=25, min_delta=1e-5, batch_size=64, learning_rate=0.001, weight_decays=[0.0001,0.01], cox_penalizers=[0.01,0.1,1], hidden_layers=[[4],[8]], neural_seeds=[42,43,44,45,46]. See packaged YAML for exact settings.

Commands from the package root:

    moveout prepare --data data/provider --config configs/expanded.yaml --out outputs/prepared
    moveout reconcile-midterm --midterm-path PATH --data data/provider --out outputs/reconciliation
    moveout run-weekly --prepared outputs/prepared --config configs/expanded.yaml --out outputs/weekly
    moveout run-survival --prepared outputs/prepared --config configs/expanded.yaml --out outputs/survival
    moveout assess-logistic-hazard --run-dir outputs/survival --out outputs/gate.json
    moveout run-survival --prepared outputs/prepared --config configs/expanded.yaml --models logistic_hazard --gate outputs/gate.json --out outputs/logistic_hazard
    moveout report --run-dir outputs/weekly
    moveout predict --model CHECKPOINT --input SNAPSHOTS.parquet --out predictions.csv

Use --repeat and --fold to run a subset. Use a fresh output directory for a new training run; previous results are not overwritten.

## 14. Outputs

Preparation produces weekly_master.parquet, cohort_ledger.csv, feature_audit.csv, fold_assignments.csv, features.json, subgroup_attributes.csv, preparation.json and structural validation results.

Runs produce a code/dependency/config manifest, exact inner memberships, candidate-selection records, saved model members, histories, held-out predictions, calibration data/figures, training PH diagnostics, exploratory feature importance, metrics by fold/repeat, paired comparisons, failures.json and final Markdown/JSON reports.

A model/fold failure is recorded and does not silently become a successful zero score. If no model succeeds the command fails. The final report states partial failures. Source data stay read-only.

## 15. Midterm reconciliation findings

The archive contains the original weighted fold-zero outputs and later unweighted/early-stopped FFNN artifacts for every fold. Saved prediction keys, household splits and labels match the source data. Recomputed test metrics agree with their saved values within 1e-6.

| Mean of 25 test-fold scores | Later saved FFNN | Constant outer-training-rate reference |
|---|---:|---:|
| Brier | 0.01831820 | 0.01831095 |
| Log loss | 0.09329158 | 0.09279502 |
| ROC-AUC | 0.59966880 | 0.50000000 |
| Average precision | 0.05839318 | Depends on test prevalence |

The later configuration is [16,8], dropout 0.20, learning rate 0.001, weight decay 0.0001, batch size 64, maximum 300 epochs, patience 25, minimum improvement 1e-5, validation fraction 0.20, ordinary BCE and seed 42 + 100×repeat + fold.

The later training runner and its inner split memberships are not in the archive; the included source still trains weighted fixed-epoch FFNNs. The original logistic source also leaves Torch initialization unseeded. Therefore saved-result verification is complete, but exact retraining of those historical fits is not claimed.

The later results largely resolve severe weighted probability inflation without establishing a probability-score advantage over the constant reference. They are historical context for the new expanded-feature experiment.

## 16. Verification and acceptance

Required behavioral checks cover: grouped splits; cross-household lag prevention; no effect of future payments on earlier features; training-only transformations; duplicate/terminal validation; endpoint qualification; landmark residual targets; incomplete censoring; known IPCW arithmetic; grouped bootstrap; model gating; unsupported 52-week horizons; all adapter fit/predict operations; and save/reload prediction equivalence.

End-to-end verification prepares the real expanded data, reconciles all saved midterm folds, and exercises one outer fold using three-epoch smoke settings. Smoke scores are execution evidence only. The full comparative 25-fold expanded experiment is not automatically launched by writing the code.

Implementation acceptance requires: all 444 starting IDs in the ledger; documented common exclusions; a shared feature contract and fold file; passing leakage/censor tests; model adapters sharing appropriate prediction origins; traceable held-out artifacts; correct censoring support; explicit conditional LogisticHazard execution; and disclosure of feature-timing assumptions and any run failures.

## 17. Provider clarifications and research limits

Useful questions remain: whether fob/pet/deposit attributes represent move-in or extract-date values; whether invoice periods establish when amounts were known; how partial payments were allocated historically; whether September 15 is authoritative despite a later event; and the meaning of out-of-range geocode confidence.

These questions do not block writing or exercising the retrospective implementation. They do limit claims that a model has been prospectively validated. Full-ledger rent and outcome-derived totals are not accepted as early-week inputs merely because a field is populated.

The received code resolves the midterm preparation and outer fold questions. No further code delivery is required before running the implemented comparison; the absent later runner is a historical reproducibility limitation rather than a software blocker.
