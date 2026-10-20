# Open Issues

**Project:** _Team Carrier Pigeon_ (tenant departure hazard model, `willis-crosby-cohort-clv-20260915`)

---

_**What this file is.** Every question about the project that you cannot answer yet, in one place, with the name of the person who can answer it_

_**Why it exists.** A draft specification with confident guesses in the gaps is more dangerous than one with holes in it, because nobody can tell the guesses from the facts. Writing "we do not know" is not an admission of failure in week 3, it is the correct state. What fails is knowing and not writing it down._

**Where entries come from.**

- Entries are made when there is something that needs to get done in the project and it is an open task.

**How they leave.**

- Entries leave once they have an opened PR on GitHub that has been merged into the main branch. After the changes are made, the entry is closed in this file with an explanation of what the task / open issue contributed.

_**Identifiers are a prefix plus a number**, e.g. `PRE-1` upward within each prefix. Numbers are fine for a list that only ever grows at the bottom and gets cited lightly._

## Current Identifiers

`PRE-X` - data preprocessing
`MOD-X` - Model creation
`EXP-X` - Anything relating to an experiment (running a pipeline for training a model)
`EVAL-X` - Model Evaluation

**Note on New Identifiers**:

- new identifiers can be placed here if this document is updated. A new section should be created for this

# Current Open Issues

## Example Template

### Workflow: open issue → merged PR

Every piece of work follows these steps, in order. Nothing is committed directly to `main`.

1. **Log it here.** Add a row to the `Open` table of the matching section, using the next number for that prefix (e.g. `PRE-5`). Phrase the question so it has a definite answer.
2. **Open a GitHub issue.** Title it `<ID>: <question>` (e.g. `PRE-1: Are raw CSVs committed or gitignored?`) and use the issue body template below. Note the GitHub number (`#N`) — it is different from our ID.
3. **Branch off an up-to-date `main`.** Branch name is `<id-lowercase>-<short-topic>`:
   ```bash
   git switch main && git pull
   git switch -c pre-1-data-layout
   ```
4. **Do the work.** Start each commit message with the ID (e.g. `PRE-1: add data/raw and data/processed`). If the work raises a new question, log it as a new row instead of growing this branch.
5. **Open a PR and link the issue.** Title it the same as the issue. Put `Closes #N` in the PR body so GitHub closes the issue automatically when the PR merges.
6. **Resolve it here, on the same branch.** Move the row from `Open` to `Resolved` and fill in Answer, Answered by, Date, and Filed in. Push to the PR so the resolution merges together with the work.
7. **Review and merge.** At least one teammate reviews the PR before it merges into `main`. Delete the branch after merging.

### GitHub issue body template

```markdown
## Question
<Same question as the Open row.>

## Why it matters
<What breaks or gets decided wrong if we guess. Name what it blocks.>

## Context
- <Relevant files, docs, related IDs>

## Done when
- [ ] <Concrete, checkable deliverable>
- [ ] Row moved to Resolved in `docs/OPEN-ISSUES.md`
```

### Table format

- Each open issue MUST FOLLOW THIS FORMAT

### Open

| ID   | Question                                                                                                                                              | Why it matters                                                              | Who can answer                                              | Raised         |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------- |
| OI-1 | _[The question, phrased so it has a definite answer. Not "tell us about reporting" but "who is allowed to see another student's evaluation scores?"]_ | _[What we get wrong if we guess. Name the artifact or decision it blocks.]_ | _[Client, instructor, a specific user, or the team itself]_ | _[YYYY-MM-DD]_ |

### Resolved

| ID   | Question                   | Answer                | Answered by | Date           | Filed in                                                    |
| ---- | -------------------------- | --------------------- | ----------- | -------------- | ----------------------------------------------------------- |
| OI-0 | _[Example row, delete it]_ | _[What we were told]_ | _[Name]_    | _[YYYY-MM-DD]_ | _[`BO-grading-time`, or the glossary term, or the section]_ |

## Data Preprocessing - PRE-X

### Open

| ID    | Question | Why it matters | Who can answer | Raised |
| ----- | -------- | -------------- | -------------- | ------ |
| PRE-3 | Which columns are used as model inputs and which are dropped, each with a reason? | `docs/README.md` lists features from `residents.csv` (child / no child, pet / no pet, income, work distance, work minutes); the updated proposal (2026-10-16) names income, family size, commute, "and more". **Update 2026-10-20:** column decisions made and filed in `docs/architecture/data-dictionary.md` (tagged `[T]`): 10 features (`week_index`, `income_monthly_est`, `rent_to_income_ratio`, `work_drive_min`, `prior_home_distance_mi`, `n_fobs`, `n_pets`, `outstanding_lag1`, `delinquent_lag1`, `site`); `income_band_hourly_equiv`, `commute_band`, `work_distance_mi` dropped as redundant. Closes when Abbie and Peter sign off. Still to do: remove "Child / No Child" (a protected column) from `docs/README.md`. Quality flags (`income_quality_flag`, `work_location_source`, `work_geocode_confidence`) go in the audit table for the sensitivity reruns. | Team | 2026-10-15 |
| PRE-4 | Is every scaling and encoding step fit inside each training fold only, never on the full dataset? | Fitting a scaler or encoder on all 455 tenancies leaks the held-out folds' statistics into training and inflates every metric. This means scaling/encoding lives in the model pipeline (refit per fold / per bootstrap sample), not in `preprocess.py`. The same goes for filling blanks: median fill plus a `_missing` indicator (e.g. `SimpleImputer(strategy="median", add_indicator=True)`), fit on each split's training households. `preprocess.py` leaves NaNs in place for this reason. Applies to the k-fold and bootstrap splits (EVAL-3, EVAL-4). | _[TBD]_ | 2026-10-15 |
| PRE-5 | Do we keep `rent_to_income_ratio`, and if so, is `weekly_rent` a valid stand-in for the rent known at signup? | **Kept** (decided 2026-10-20, see the data dictionary); the remaining question is whether `weekly_rent` is valid. There is no rent field on the lease; `weekly_rent` is the modal weekly billed amount over the household's whole invoice ledger. If rent changed during the tenancy, or short tenancies' modal bill is mostly fees, the ratio uses information from after the prediction week. Check: compare `weekly_rent` against each household's first regular billing cycle in `invoice_cycles.csv` (in the zip; `scripts/set-up-repo.py` doesn't extract it). | Team | 2026-10-15 |
| PRE-6 | Do we build `commute_cost_to_income` at all, and if so with what $/mile and workdays-per-month? | The updated proposal's ablation is with vs. without "commute-related features"; it no longer names commute cost relative to income, so `work_drive_min` alone may be enough. If built, it needs `work_distance_mi` × 2 × workdays × $/mile ÷ income, and the two constants (e.g. IRS mileage rate, ~21 workdays) must be fixed and cited first. See `docs/architecture/data-dictionary.md`, Derived Features. `work_distance_mi` was dropped from the model table 2026-10-20; add it back to `RESIDENT_COLS` if this gets built. | Team | 2026-10-15 |
| PRE-7 | How do we represent the proposal's "family size" feature without using protected columns? | The updated proposal names family size as a model input, but the columns that measure it (`pc_n_adults`, `pc_n_children`, `pc_has_children`) are flagged by the provider as protected class (familial status, Fair Housing Act): controls and bias checks only. Current plan follows `data_prep.py`: use `n_fobs` (access fobs issued) as a weak proxy. Decide whether that's acceptable, and have the fairness check test whether `n_fobs` ends up acting as a stand-in for familial status. | Team / instructor | 2026-10-16 |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |
| PRE-1 | What is the `data/raw` → `data/processed` layout, and are the raw CSVs extracted from `Gated_Rentals_Dataset.zip` committed or gitignored? | Gitignored. The zip is the only committed copy; `scripts/set-up-repo.py` verifies it against the provider's `SHA256SUMS` and extracts the files we use into `data/raw/` (residents, weekly_panel), `data/reference/` (km_survival, sensitivity_checks), and `data/provider-docs/`, and creates an empty `data/processed/`. Setup steps are in the root `README.md`. | Matt | 2026-10-15 | `scripts/set-up-repo.py`, `README.md` |
| PRE-2 | What does `preprocessing/preprocess.py` produce from `data/raw` into `data/processed` (which files, columns, filters, format)? | Three parquet files. `model_table.parquet`: one row per household-week (6,731 × 12), the target plus the PRE-3 features, unscaled, NaNs kept. Filters: `rent_to_income_ratio` < 0 or > 1 set to NaN (10 households); `outstanding`/`delinquent` replaced by one-week lags (week 0 = 0, except `delinquent_lag1` stays blank for the 64 households with no rent); the 11 residents with no weekly rows drop out in the join (444 households, 125 move-outs). `splits/kfold.parquet` and `splits/bootstrap.parquet` hold the split assignments (EVAL-3, EVAL-4). The script asserts the raw-data checks, every shape, the parquet round trip, and the split properties; a fresh run passes all of them. No scaling, encoding, or imputation (PRE-4). | Matt | 2026-10-20 | `preprocessing/preprocess.py`; data dictionary, Processed Files |

## Model Creation - MOD-X

### Open

| ID  | Question | Why it matters | Who can answer | Raised |
| --- | -------- | -------------- | -------------- | ------ |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |

## Experiments - EXP-X

### Open

| ID  | Question | Why it matters | Who can answer | Raised |
| --- | -------- | -------------- | -------------- | ------ |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |

## Model Evaluation - EVAL-X

With only 125 observed move-outs across 455 tenancies, how the data is split matters as much as which model is used. EVAL-1 (unit of resampling), EVAL-3 and EVAL-4 (the two resampling methods) are resolved and implemented in `preprocessing/preprocess.py`; EVAL-5 and EVAL-6 pin down the baseline and the horizons.

### Open

| ID     | Question | Why it matters | Who can answer | Raised |
| ------ | -------- | -------------- | -------------- | ------ |
| EVAL-5 | Is the Kaplan-Meier baseline the provider's `km_survival.csv`, or our own KM fit on each training fold? | The proposal says both baselines are "provided by the client". Only the KM curve is provided, and it is fit on all 446 households, so it has seen every test fold. Fitting our own KM per fold (e.g. `lifelines`) is the fair comparison; the provider's curve stays as a sanity check. Logistic regression is not provided by the client; we build it, and the midterm report should say so. | Team | 2026-10-16 |
| EVAL-6 | How do we evaluate the proposal's 52-week horizon when the longest observed tenancy is 46 weeks? | The proposal evaluates at 26 and 52 weeks and defines success as the model closely matching historical tenure. No household is observed past week 46, so there are no real outcomes at 52 weeks to score against. Options: score at 26 weeks and the latest supported horizon (e.g. 40 or 46), and report 52-week predictions only as an extrapolation compared with KM. Also define "closely matches" (e.g. calibration against observed KM survival). | Team | 2026-10-16 |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |
| EVAL-2 | Which split strategy is primary: the proposal's hold-out (one property or recent cohorts) or the README's bootstrap + k-fold? | The updated proposal (2026-10-16) drops the hold-out. Bootstrap + grouped, stratified k-fold (`docs/README.md`) is the split strategy; leave-one-site-out stays available as an optional extra check. | Team (proposal update) | 2026-10-16 | `docs/project-proposal-initial.md` |
| EVAL-1 | What is the unit of resampling for bootstrap and k-fold? | The household (`resident_id`). Both splits are built on a 444-row household table, so all of a household's weeks always land on the same side of a split. | Matt | 2026-10-20 | `preprocessing/preprocess.py` (`get_k_fold`, `get_bootstrap`) |
| EVAL-3 | How is k-fold cross-validation implemented? | `RepeatedStratifiedKFold`, 5 folds × 5 repeats (`random_state = 42`), on households, stratified on household-level `moved_out`: 25 splits, each test fold 88–89 households with exactly 25 move-outs. Saved as `data/processed/splits/kfold.parquet` (test households per `repeat`, `fold`). Per-fold scaling/imputation is PRE-4. | Matt | 2026-10-20 | `preprocessing/preprocess.py`; data dictionary, Processed Files |
| EVAL-4 | How is bootstrap sampling implemented? | 100 replicates; each draws 444 households with replacement (`np.random.default_rng(42)`). Train on drawn households (repeated `n_draws` times), test on the out-of-bag ones (`n_draws == 0`, 36.8% on average). Saved as `data/processed/splits/bootstrap.parquet`. Per-replicate scaling/imputation is PRE-4. | Matt | 2026-10-20 | `preprocessing/preprocess.py`; data dictionary, Processed Files |
