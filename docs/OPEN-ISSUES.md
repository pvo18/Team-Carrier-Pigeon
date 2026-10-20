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
| PRE-2 | What does `preprocessing/preprocess.py` produce from `data/raw` into `data/processed` (which files, columns, filters, format)? | The file exists but is empty. Every model (Kaplan-Meier, logistic regression, feed-forward NN) must read the same processed data so comparisons are fair. Must not write scaled or encoded values (see PRE-4). Depends on PRE-3. | _[TBD]_ | 2026-10-15 |
| PRE-3 | Which columns are used as model inputs and which are dropped, each with a reason? | `docs/README.md` lists features from `residents.csv` (child / no child, pet / no pet, income, work distance, work minutes); the updated proposal (2026-10-16) names income, family size, commute, "and more". A full draft answer is in `docs/architecture/data-dictionary.md` and needs team review; "Child / No Child" in `docs/README.md` is a protected column and should come out. Coverage gaps and quality flags (`income_quality_flag`, `work_location_source`, `work_geocode_confidence`) also decide which rows and columns are usable. Blocks PRE-2. | Team | 2026-10-15 |
| PRE-4 | Is every scaling and encoding step fit inside each training fold only, never on the full dataset? | Fitting a scaler or encoder on all 455 tenancies leaks the held-out folds' statistics into training and inflates every metric. This means scaling/encoding lives in the model pipeline (refit per fold / per bootstrap sample), not in `preprocess.py`. Applies to EVAL-3 and EVAL-4. | _[TBD]_ | 2026-10-15 |
| PRE-5 | Do we keep `rent_to_income_ratio`, and if so, is `weekly_rent` a valid stand-in for the rent known at signup? | The updated proposal no longer names affordability (it falls under "and more"), so first decide whether the feature stays. If it does: there is no rent field on the lease; `weekly_rent` is the modal weekly billed amount over the household's whole invoice ledger. If rent changed during the tenancy, or short tenancies' modal bill is mostly fees, the ratio uses information from after the prediction week. Check: compare `weekly_rent` against each household's first regular billing cycle in `invoice_cycles.csv` (in the zip; `scripts/set-up-repo.py` doesn't extract it). | Team | 2026-10-15 |
| PRE-6 | Do we build `commute_cost_to_income` at all, and if so with what $/mile and workdays-per-month? | The updated proposal's ablation is with vs. without "commute-related features"; it no longer names commute cost relative to income, so `work_drive_min` alone may be enough. If built, it needs `work_distance_mi` × 2 × workdays × $/mile ÷ income, and the two constants (e.g. IRS mileage rate, ~21 workdays) must be fixed and cited first. See `docs/architecture/data-dictionary.md`, Derived Features. | Team | 2026-10-15 |
| PRE-7 | How do we represent the proposal's "family size" feature without using protected columns? | The updated proposal names family size as a model input, but the columns that measure it (`pc_n_adults`, `pc_n_children`, `pc_has_children`) are flagged by the provider as protected class (familial status, Fair Housing Act): controls and bias checks only. Current plan follows `data_prep.py`: use `n_fobs` (access fobs issued) as a weak proxy. Decide whether that's acceptable, and have the fairness check test whether `n_fobs` ends up acting as a stand-in for familial status. | Team / instructor | 2026-10-16 |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |
| PRE-1 | What is the `data/raw` → `data/processed` layout, and are the raw CSVs extracted from `Gated_Rentals_Dataset.zip` committed or gitignored? | Gitignored. The zip is the only committed copy; `scripts/set-up-repo.py` verifies it against the provider's `SHA256SUMS` and extracts the files we use into `data/raw/` (residents, weekly_panel), `data/reference/` (km_survival, sensitivity_checks), and `data/provider-docs/`, and creates an empty `data/processed/`. Setup steps are in the root `README.md`. | Matt | 2026-10-15 | `scripts/set-up-repo.py`, `README.md` |

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

With only 125 observed move-outs across 455 tenancies, how the data is split matters as much as which model is used. EVAL-1 (unit of resampling) applies to every split; EVAL-3 and EVAL-4 implement the two resampling methods (EVAL-2, resolved); EVAL-5 and EVAL-6 pin down the baseline and the horizons.

### Open

| ID     | Question | Why it matters | Who can answer | Raised |
| ------ | -------- | -------------- | -------------- | ------ |
| EVAL-1 | What is the unit of resampling for bootstrap and k-fold? Proposed: the household tenancy (group-level), not the person-week row. | `weekly_panel.csv` has one row per household per week (6,731 rows for the 444 tenancies that have panel rows; 11 of the 455 in `residents.csv` have none). Splitting by row puts different weeks of the same tenancy in both train and test, which leaks the tenant's identity and outcome across folds and inflates calibration and Brier score. Every split must keep all weeks of a tenancy together. Blocks EVAL-3 and EVAL-4. | _[TBD]_ | 2026-10-15 |
| EVAL-3 | How is k-fold cross-validation implemented? Proposed: grouped by tenancy (EVAL-1) and stratified by whether the tenant moved out. | Without stratification, some folds may hold very few of the 125 move-outs, making per-fold metrics noisy or undefined. Grouping and stratifying together means stratifying on a tenancy-level label, not the weekly event flag. Scaling/encoding refit per fold (PRE-4). Number of folds still to decide. | _[TBD]_ | 2026-10-15 |
| EVAL-4 | How is bootstrap sampling implemented? Proposed: resample tenancies with replacement and evaluate on the out-of-bag tenancies, to get confidence intervals for calibration and Brier score. | Gives the uncertainty bands needed to say whether the NN actually beats logistic regression and Kaplan-Meier, rather than reporting a single point estimate. Resampling must be at the tenancy level (EVAL-1), with all of a tenancy's weeks moving together. Scaling/encoding refit per bootstrap sample (PRE-4). Number of bootstrap replicates still to decide. | _[TBD]_ | 2026-10-15 |
| EVAL-5 | Is the Kaplan-Meier baseline the provider's `km_survival.csv`, or our own KM fit on each training fold? | The proposal says both baselines are "provided by the client". Only the KM curve is provided, and it is fit on all 446 households, so it has seen every test fold. Fitting our own KM per fold (e.g. `lifelines`) is the fair comparison; the provider's curve stays as a sanity check. Logistic regression is not provided by the client; we build it, and the midterm report should say so. | Team | 2026-10-16 |
| EVAL-6 | How do we evaluate the proposal's 52-week horizon when the longest observed tenancy is 46 weeks? | The proposal evaluates at 26 and 52 weeks and defines success as the model closely matching historical tenure. No household is observed past week 46, so there are no real outcomes at 52 weeks to score against. Options: score at 26 weeks and the latest supported horizon (e.g. 40 or 46), and report 52-week predictions only as an extrapolation compared with KM. Also define "closely matches" (e.g. calibration against observed KM survival). | Team | 2026-10-16 |

### Resolved

| ID  | Question | Answer | Answered by | Date | Filed in |
| --- | -------- | ------ | ----------- | ---- | -------- |
| EVAL-2 | Which split strategy is primary: the proposal's hold-out (one property or recent cohorts) or the README's bootstrap + k-fold? | The updated proposal (2026-10-16) drops the hold-out. Bootstrap + grouped, stratified k-fold (`docs/README.md`) is the split strategy; leave-one-site-out stays available as an optional extra check. | Team (proposal update) | 2026-10-16 | `docs/project-proposal-initial.md` |
