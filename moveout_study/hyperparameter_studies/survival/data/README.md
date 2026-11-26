# Willis / Crosby resident cohort + CLV dataset

De-identified extract for the retention / customer-lifetime-value study
described in *Resident Cohort Analysis: Commute and Affordability Retention
Calibration*. Built directly from the two live Dynamics 365 environments
(`willisprod`, `crosbyprod`) — read-only — on **2026-09-15**, which is also the
censoring date for every open tenancy.

## Files

| file | grain | rows |
|---|---|---|
| `residents.csv` | one row per household tenancy (move-ins only) | 455 |
| `invoice_cycles.csv` | one row per billing cycle | 7566 |
| `weekly_panel.csv` | one row per household per week since move-in | 6731 |
| `cohort_retention_weekly.csv` | move-in week x weeks-since-move-in | 1605 |
| `cohort_retention_monthly.csv` | move-in month x weeks-since-move-in | 404 |
| `km_survival.csv` | Kaplan-Meier survival, overall and per site | 128 |
| `clv_baseline_by_segment.csv` | reference CLV per segment | 46 |
| `sensitivity_checks.csv` | each headline re-run on its high-confidence subset | 15 |

`weekly_panel.csv` is the analysis-ready object. Kaplan-Meier, Cox regression
and cohort CLV all read off it without further reshaping.

## Population

Move-ins only — applicants who never took occupancy are excluded by design.

| | Willis Lakes | Crosby Heights |
|---|---|---|
| tenancies | 252 | 203 |
| moved out (observed churn events) | 71 | 54 |
| still in place (right-censored) | 181 | 149 |
| application record linked | 246 | 152 |

Billing cycles present: weekly 4394, monthly 1374, one_time 784, bi_weekly 672, other 340.
Weekly is the dominant cycle, as the proposal assumes.

## Read this before modelling

**1. The history is short and heavily censored.** First move-in is
2025-10-24 (Willis) / 2026-01-31 (Crosby). No tenancy in this dataset is older
than about 11 months, and 73%
of tenancies are still open. Mean observed tenure is therefore biased sharply
downward and must not be used as expected lifetime. Use survival methods
(Kaplan-Meier / Cox) and state the extrapolation horizon explicitly.
`clv_baseline_by_segment.csv` reports 52-week restricted mean survival time
(`rmst_52w`) rather than a mean, for this reason.

**2. Field coverage is uneven, and differs by site.** The two sites hold
applications in different entities (Willis `crc7a_portal`, Crosby
`gr_application`) but, with few exceptions, in the *same* `gr_*`-prefixed field
set, which was mapped to common columns. Willis's `crc7a_portal` carries fields
from two publishers and the `crc7a_*` income/address columns are near-empty
decoys — the live values are the `gr_*` ones. Remaining coverage gaps:

| column | Willis filled | Crosby filled | overall |
|---|---|---|---|
| `income_monthly_est` | 230 / 252 | 139 / 203 | 81% |
| `pc_age_band_at_movein` | 242 / 252 | 148 / 203 | 86% |
| `pc_n_children` | 252 / 252 | 203 / 203 | 100% |
| `n_pets` | 219 / 252 | 146 / 203 | 80% |
| `work_drive_min` | 233 / 252 | 145 / 203 | 83% |
| `prior_home_drive_min` | 243 / 252 | 144 / 203 | 85% |
| `prior_home_zip3` | 246 / 252 | 151 / 203 | 87% |
| `weekly_rent` | 212 / 252 | 172 / 203 | 84% |
| `rent_to_income_ratio` | 195 / 252 | 116 / 203 | 68% |
| `sc_eviction_history` | 245 / 252 | 150 / 203 | 87% |
| `move_out_reason` | 19 / 252 | 3 / 203 | 5% |

Income is usable at both sites (Willis 230/252, Crosby 139/203) but is graded —
see limitation 6. Treat site as a control, not a nuisance: payment rails
(Willis is Zelle-heavy, Crosby debit-card-heavy), application-link rate, and
`first_responder` coding all differ materially between the two.

**3. Commute is derived, not collected.** The dedicated employer-address field
is empty in both environments, so work location comes from `gr_jobaddress` —
a stated work address, present for 252 of 455 households and recorded as
`work_location_source='job_address'`. For a further 126 households with no
stated work address, the free-text employer name was geocoded to the nearest
matching point of interest within 120 km of the site
(`work_location_source='employer_name'`); those rows are materially noisier.
Filter on `work_location_source` and `work_geocode_confidence` before fitting
anything commute-related, and check whether the employer-name rows move the
curve — if they do, trust the `job_address` subset.

**4. Move-out reasons are mostly missing.** Only a minority of the observed
move-outs carry a reason code, so `churn_type` is `unknown` for most events.
Voluntary-vs-involuntary churn cannot be modelled separately at this sample
size; use payment-behaviour columns as the proxy for distress instead.

**5. Refunds and credits appear as negative amounts.** 73 cycles carry a
negative payment and 83 a negative billed amount (move-out deposit returns,
reversals). `cum_paid` in the panel therefore steps down in those weeks. Do not
clamp these to zero — they are real cash movements — but exclude them if you
want gross-billing-only revenue.

**6. Income is graded, and you should use the grade.** The amount field and the
pay-frequency picklist are not reliably consistent: the picklist describes the
applicant's pay *cadence*, which is often not the unit of the number they
typed (Willis's field is even named `gr_hourlyincome` but holds hourly wages,
weekly paychecks, monthly figures and annual salaries in roughly equal measure).
Every value was normalised to USD/month and graded in `income_quality_flag`:

| flag | n | meaning |
|---|---|---|
| `ok` | 143 | stated unit is plausible and agrees with the magnitude of the number |
| `ok_unit_ambiguous` | 122 | stated unit is plausible, but the magnitude suggests a different unit |
| `unit_reconciled` | 104 | stated unit gave an impossible monthly figure; unit re-inferred from magnitude (sub-$100 = hourly, five figures = annual) |
| `implausible_after_normalisation` | 8 | no reading produced a credible figure; income left blank |
| `income_missing` / `no_application` | 21 / 57 | not captured |

Run the affordability analysis on `ok` alone (n=143) and on all 369 with a
value, and report both. If the curves diverge, the income data is the binding
constraint on the result, not the method. Monthly income was capped at $25,000
($300k/yr) as implausible for this population before reconciliation.

**7. Sample size caps model complexity.** 125
churn events across 455 tenancies supports roughly 6-12 covariates in a
Cox model before overfitting. Weekly cohorts average about
9 households, which is
too thin to read individually — aggregate to monthly cohorts
(`cohort_retention_monthly.csv`) or to income/commute bands for any curve you
intend to trust.

## Fair-housing constraint on how the output may be used

Columns prefixed `pc_` are, or directly proxy, characteristics protected under
the Fair Housing Act: `pc_n_children` / `pc_has_children` (familial status),
`pc_age_band_at_movein` (age), and `pc_has_esa_pet` / `pc_n_esa_pets`
(emotional-support animals — a disability proxy). `sc_eviction_history` is a
screening attribute subject to HUD disparate-impact guidance.

They are included because retrospective measurement of *our own* outcomes is a
legitimate research use, and because leaving them out would hide confounding.
They are **not** inputs for a scoring function that decides who gets housed or
which parcels get bought. Use them as controls and as bias diagnostics — fit
the commute/affordability curve the proposal asks for, then check whether that
curve differs across protected groups. Any model output intended to feed the
Buy Box Score or tenant selection should be reviewed by counsel first, and
should be specified on commute, income and payment behaviour only.

Criminal-history and domestic-violence fields exist in the Crosby application
record. They were deliberately **excluded** from this extract: not requested,
highest legal sensitivity, and available for only part of one site. Ask if the
analysis genuinely needs them.

## What was removed

No names, emails, phone numbers, SSNs, dates of birth, street addresses,
employer names, or real unit numbers appear in any shipped file. Specifically:

- Households are keyed `{SITE}-U{nnn}-{k}` — a pseudonymous unit label plus
  the occupancy sequence for that unit over time (`occupancy_seq`), so unit
  fixed effects and turnover-per-unit remain analysable.
- Dates are truncated to the **Monday of the ISO week**.
- Age and income are banded; exact DOB and exact stated income are dropped
  (`income_monthly_est` is a normalised monthly figure, retained because the
  proposal needs a continuous affordability ratio).
- Addresses are reduced to `prior_home_zip3` plus derived distance and routed
  drive time. Coordinates are not shipped.
- The unit-label and employer mappings stay in the internal working directory
  and are not part of the deliverable.

## What the baseline already shows — and what doesn't survive checking

`clv_baseline_by_segment.csv` produces two findings that look like headlines and
are **not** robust. `sensitivity_checks.csv` re-runs each one on the subset where
the underlying variable is trustworthy, and both collapse:

- **An income gradient.** Across all 364 households with an income figure,
  52-week RMST falls from 36 weeks in the lowest bands to 24.4 weeks for $30+/hr
  — i.e. higher earners appearing to leave faster, which would invert a Buy Box
  assumption. Restricted to the 141 households whose income carries
  `income_quality_flag='ok'`, the spread is 33.2-37.3 weeks with no ordering.
  The effect lives in the reconciled and ambiguous income rows, not in the data.
- **A 20-minute commute breakpoint.** Across all 372 households with a commute,
  RMST steps down about 8 weeks at exactly 20 minutes and then stays flat —
  suspiciously close to the hypothesis the proposal set out to test. Restricted
  to the 246 households with a *stated* work address, the pattern is
  non-monotonic (35.4 / 41.3 / 34.7 / 42.4 / 29.8) and only the 45-minute-plus
  band remains clearly worse. The breakpoint is an artifact of the employer-name
  geocodes.

One finding **is** stable, because it rests on the best-populated variable in
the dataset (prior address, ~99% filled, routed distance) and needs no grading:

- **Distance moved predicts tenure, monotonically.** RMST by prior-home
  distance: 36.5w (<10mi), 34.2w (10-25mi), 33.4w (25-50mi), 24.3w (50-100mi),
  28.0w (100mi+). Households recruited from far away leave sooner.

Treat all three as hypotheses to test properly with confidence intervals, not as
results. The first job is establishing which survive a Cox model with
site controls — at 125 events and 72% censoring, none of them is settled.

## Mapping to the intended analyses

- **Cohort analysis** — `cohort_retention_weekly.csv` / `_monthly.csv` are the
  retention triangles; `weekly_panel.csv` if you want to build your own.
- **Customer lifetime value** — `weekly_panel.csv` gives billed/paid/cumulative
  revenue per week; `km_survival.csv` gives the survival curve. CLV =
  expected weeks (RMST or fitted survival) x weekly rent x collection rate.
  `clv_baseline_by_segment.csv` is the reference implementation.
- **Segmentation / clustering** — `residents.csv` carries the signup-time
  features (income band, affordability ratio, commute minutes, household
  composition, deposits, discount group).
- **Churn prediction** — `weekly_panel.csv` in discrete-time form
  (`churned_this_week` as the label), or `residents.csv` with
  `tenure_weeks` + `is_censored` for a Cox model.

## Open questions from the proposal, answered against the data

1. *Nearest major employment cluster per site* — not needed as an assumption
   after all. 252 households state an actual work address and another 126 have a
   geocodable employer, so commute is measured per resident rather than per site.
   The gap worth closing going forward is the dedicated employer-address field,
   which is collected nowhere: it would remove the employer-name fallback.
2. *Income band boundaries* — `income_band_hourly_equiv` uses <15 / 15-20 /
   20-25 / 25-30 / 30+ $/hr to match the proposal's stated hypothesis;
   `income_band_monthly` is also provided. Both are derived from
   `income_monthly_est`, so you can re-band freely.
3. *Which retention and payment fields pull cleanly across sites and periods* —
   move-in/move-out dates, lease status, household composition, deposits and the
   full invoice ledger pull cleanly for both sites and the whole window. Income
   and work location pull for most households but need the quality flags
   (limitations 3 and 6). Move-out reason does not pull usefully at all
   (limitation 4).
4. *Minimum cohort size* — at this volume, do not read a weekly cohort cell
   with fewer than ~20 at-risk households. Monthly cohorts, or income/commute
   bands pooled across both sites, are the smallest units worth a curve point.

## Provenance

| | |
|---|---|
| Source systems | Dynamics 365 `willisprod`, `crosbyprod` |
| Access mode | read-only (the platform never writes to Dataverse) |
| Extracted | 2026-09-15 23:00 UTC |
| Censoring date | 2026-09-15 (every open tenancy is right-censored here) |
| Population | move-in tenancies only; non-converting applicants excluded |
| Tenancies | 455 (Willis 252, Crosby 203) |
| Billing cycles | 7,566 |

Pipeline stages, in order: entity/attribute discovery, lease-to-application
linkage, resident assembly, invoice cycle derivation, weekly panel expansion,
survival and CLV aggregation, then document generation. The Dataverse client is
`dv/dv_client.py`.

Lease-to-application linkage: Willis 98%, Crosby 75%. Willis joins on
`_gr_application_value` with a reverse `_gr_unitleaseid_value` check and a
contact match; Crosby's lookup field is populated on only 7 of 203 leases, so it
joins on contact (`_gr_applicant_value` to `_gr_applicantcontact_value`) plus
co-applicant cross-paths. `application_link_method` records which key matched on
every row, and `application_linked` marks rows with no link at all. Filter on
these before running anything that depends on application-sourced variables
(income, commute, household composition) — 246 of 252 Willis and 152 of 203
Crosby tenancies carry one.

Two source-side quirks are already corrected in this extract and are recorded
here because they change values rather than only coverage:

- Willis applications carry fields under two publisher prefixes on one entity.
  The analytic values are on the `gr_*` fields; the similarly-named `crc7a_*`
  income, address and ZIP columns are near-empty and are not used.
- Pay-frequency labels differ by site and by vintage (`BiWeekly`, `Bi-Weekly`,
  `Bi Weekly`). All are mapped explicitly; the build fails rather than falling
  back to a default, because an unmapped label silently inflates normalised
  monthly income by about 2.17x into a range that still looks plausible.

## Verifying this archive

Checksums for every shipped file are in `SHA256SUMS`, in the standard format:

    sha256sum -c SHA256SUMS

`MANIFEST.md` carries the same digests alongside row counts and byte sizes.

### On Windows

Windows has no `sha256sum`. Use the included PowerShell script instead — open
PowerShell in the unzipped folder and run:

    powershell -ExecutionPolicy Bypass -File verify.ps1

It prints a line per file and exits non-zero if any file is missing or altered.
(`-ExecutionPolicy Bypass` is needed only because the script is unsigned; it
applies to that one command, not to your machine.)

To check a single file by hand, either of these is built in:

    certutil -hashfile data\residents.csv SHA256
    Get-FileHash data\residents.csv -Algorithm SHA256

Compare the result against `SHA256SUMS` or `MANIFEST.md`. The comparison is
case-insensitive — PowerShell prints uppercase, the file lists lowercase.

On macOS or Linux, `sha256sum -c SHA256SUMS` from the archive root does the
same thing. (macOS ships `shasum -a 256 -c SHA256SUMS`.)

