# Data dictionary

Generated from the shipped files on 2026-09-15.

Column-name prefixes: **`pc_`** = protected class or direct proxy under the
Fair Housing Act; **`sc_`** = screening attribute subject to HUD
disparate-impact guidance. See the fair-housing section of `README.md` — these
are controls and bias diagnostics, not scoring inputs.

## residents.csv — 455 rows, one per household tenancy

| column | meaning |
|---|---|
| `resident_id` | Pseudonymous household key: {site}-U{unit}-{occupancy sequence}. |
| `site` | WIL = Willis Lakes, CRO = Crosby Heights. |
| `unit_key` | Pseudonymous unit label. Stable within this extract; not the real unit number. `*-UNKnn` means the source lease carried no unit link. |
| `unit_known` | 1 if the tenancy is tied to a known unit. 0 for the 30 leases with no unit lookup in the source — each holds its own `UNKnn` label, so do not read them as repeat occupancies of one unit. |
| `occupancy_seq` | 1 = first household in that unit in the data, 2 = next, ordered by move-in. |
| `unit_total_occupancies` | How many households have occupied that unit in the window. |
| `move_in_week` | Monday of the ISO week of move-in. |
| `move_out_week` | Monday of the ISO week of move-out; blank if still in place. |
| `tenure_days` | Days from move-in to move-out, or to 2026-09-15 if still in place. |
| `tenure_weeks` | tenure_days / 7. |
| `is_active` | 1 = still in place at the extract date. |
| `is_censored` | 1 = right-censored (tenure is a lower bound). Same as is_active here. |
| `lease_status` | Source lease status (Active / Moved-Out / Refund Phase / Closed Lease / Future Move In). |
| `churn_type` | involuntary (reason coded on the involuntary field), voluntary, or unknown. Blank if active. |
| `move_out_reason` | Coded reason where present; mostly missing. |
| `notice_days_before_moveout` | Days between notice date and move-out. |
| `pc_n_adults` | PROTECTED-CLASS PROXY. Adults in the household at signup. |
| `pc_n_children` | PROTECTED-CLASS (familial status). Children in the household at signup; 0 where "N/A". |
| `pc_has_children` | PROTECTED-CLASS (familial status). 1 if any children. |
| `pc_age_band_at_movein` | PROTECTED-CLASS (age). Primary applicant age band at move-in. |
| `pc_n_esa_pets` | PROTECTED-CLASS PROXY (disability). Emotional-support animals on the lease. |
| `pc_has_esa_pet` | PROTECTED-CLASS PROXY (disability). 1 if any ESA. |
| `n_pets` | Non-ESA pets on the lease. |
| `has_pet` | 1 if any non-ESA pet. |
| `n_fobs` | Access fobs issued — a weak proxy for household size. |
| `income_monthly_est` | Stated income normalised to USD/month using the stated pay frequency. Blank where absent or implausible. |
| `income_band_monthly` | Banded income_monthly_est. |
| `income_band_hourly_equiv` | income_monthly_est expressed as $/hr at 40 h/wk, banded to match the proposal hypothesis. |
| `income_basis` | The pay frequency the applicant stated (Hourly / Weekly / Bi-Weekly / Monthly / Yearly Salary). |
| `income_quality_flag` | ok | unit_inferred (frequency missing, inferred from magnitude) | income_missing | implausible_after_normalisation | no_application. |
| `prior_rent_monthly` | Rent the applicant reported paying before moving in. |
| `weekly_rent` | Modal weekly billed amount from the invoice ledger — the effective rent, since no rent field exists on the lease. |
| `rent_to_income_ratio` | (weekly_rent x 52/12) / income_monthly_est. The affordability variable. |
| `rent_to_income_band` | Banded rent_to_income_ratio (<25% / 25-35% / 35-45% / 45-60% / 60%+). |
| `payment_frequency` | Billing cadence on the lease (Weekly / Bi-Weekly / Monthly). |
| `security_deposit` | Security deposit amount. |
| `pet_deposit` | Pet deposit amount. |
| `total_deposits` | Total deposits held. |
| `initial_discount` | Whether a move-in discount was applied. |
| `discount_group` | Qualified discount group (Teachers, First Responders, Veterans/Military, etc.). Sparse. |
| `first_responder` | First-responder flag on the lease. Coded very differently between the two sites — do not pool naively. |
| `prior_home_zip3` | First 3 digits of the prior home ZIP. |
| `prior_home_distance_mi` | Routed driving miles from prior home to the site; falls back to straight-line where routing failed. |
| `prior_home_drive_min` | Routed driving minutes from prior home to the site. |
| `prior_home_straight_mi` | Great-circle miles from prior home to the site. |
| `relocated_over_25mi` | 1 if the prior home was more than 25 straight-line miles from the site. |
| `work_distance_mi` | Routed driving miles from the reconstructed work location to the site. |
| `work_drive_min` | Routed driving minutes from work to the site. The commute variable. |
| `work_straight_mi` | Great-circle miles from work to the site. |
| `commute_band` | Banded work_drive_min (<10 / 10-20 / 20-30 / 30-45 / 45+ min). The proposal's primary explanatory variable. |
| `prior_home_distance_band` | Banded driving distance from the prior home to the site. |
| `work_location_source` | job_address (stated work address, stronger) or employer_name (employer geocoded to a POI, noisy). |
| `work_geocode_confidence` | Geocoder confidence 0-1. Filter before modelling. |
| `sc_eviction_history` | SCREENING ATTRIBUTE, HUD disparate-impact sensitive. Self-reported prior eviction (Crosby only). |
| `n_invoice_cycles` | Billing cycles issued to this household. |
| `total_billed` | Sum of invoiced amounts. |
| `total_paid` | Sum of payments applied. |
| `total_outstanding` | total_billed - total_paid. |
| `collection_rate` | total_paid / total_billed. |
| `weekly_rent_modal` | Same as weekly_rent (kept for traceability). |
| `n_cycles_unpaid` | Cycles with status Not Paid. |
| `n_cycles_partial` | Cycles with status Partially Paid. |
| `pct_cycles_paid` | Share of cycles fully paid. |
| `n_late_payments` | Cycles paid after the due date. |
| `mean_days_late` | Mean lateness across late cycles. |
| `max_days_late` | Worst lateness observed. |
| `primary_payment_method` | Most-used payment method. |
| `application_linked` | 1 if an application record was matched to this tenancy. |
| `application_link_method` | direct_link | reverse_link | contact_match | none. |
| `data_quality_flags` | Semicolon-separated issues, e.g. moveout_before_movein. |

## invoice_cycles.csv — 7566 rows, one per billing cycle

| column | meaning |
|---|---|
| `resident_id` | Pseudonymous household key: {site}-U{unit}-{occupancy sequence}. |
| `site` | WIL = Willis Lakes, CRO = Crosby Heights. |
| `cycle_seq` | Sequence of this cycle within the tenancy. |
| `weeks_since_move_in` | Weeks between move-in and the start of this billing period. |
| `billing_start_week` | Monday of the billing-period start week. |
| `billing_end_week` | Monday of the billing-period end week. |
| `billing_days` | Length of the billing period in days. |
| `cycle_type` | weekly | bi_weekly | monthly | one_time | other, derived from billing_days. |
| `amount_billed` | Invoiced amount. Negative on credit memos / reversals (83 cycles). |
| `amount_paid` | Amount applied. Negative on refunds (73 cycles). |
| `balance` | billed - paid. |
| `payment_status` | Paid / Partially Paid / Not Paid / Pending. |
| `payment_method` | Cash / Debit Card / Credit Card / ACH / Zelle / Cashiers Check / Money Order. |
| `payment_gateway` | Cash Kiosk / Authorize.net / Chase / Zelle. |
| `paid_week` | Monday of the week payment was recorded. |
| `days_late_vs_due` | Days between due date and payment. Negative = early. |
| `is_late` | 1 if paid after the due date. |
| `discount_amount` | Discount applied to the cycle. |
| `credit_amount` | Credit applied to the cycle. |

## weekly_panel.csv — 6731 rows, one per household-week

| column | meaning |
|---|---|
| `resident_id` | Pseudonymous household key: {site}-U{unit}-{occupancy sequence}. |
| `site` | WIL = Willis Lakes, CRO = Crosby Heights. |
| `cohort_week` | Move-in week — the cohort label. |
| `week_index` | Weeks since move-in (0-based). |
| `week_start` | Monday of that week. |
| `at_risk` | Always 1; the household is under observation this week. |
| `active_end_of_week` | 1 unless this is the week they moved out. |
| `churned_this_week` | 1 in the move-out week. The discrete-time survival label. |
| `censored_this_week` | 1 in the final week for households still in place. |
| `billed_this_week` | Amount invoiced with a billing period starting this week. |
| `paid_this_week` | Amount paid this week. |
| `cum_billed` | Cumulative billed to date. |
| `cum_paid` | Cumulative paid to date — observed revenue. Can step down where a refund was issued. |
| `outstanding` | cum_billed - cum_paid. |
| `delinquent` | 1 if outstanding exceeds one week of rent. |

## cohort_retention_weekly.csv / cohort_retention_monthly.csv

| column | meaning |
|---|---|
| `cohort` | Move-in week (Monday) or month. |
| `site` | WIL or CRO. |
| `cohort_size` | Households in the cohort. |
| `week_index` | Weeks since move-in. |
| `n_active` | Still in place at that week. |
| `retention_rate` | n_active / cohort_size. |
| `mature_through_week` | The last week every member of the cohort could have reached by the censoring date. Rows stop there, so `retention_rate` is never depressed by households that simply haven't aged that far. |

## km_survival.csv

| column | meaning |
|---|---|
| `population` | ALL, WIL or CRO. |
| `week_index` | Weeks since move-in. |
| `n_at_risk` | Households at risk at the start of the week. |
| `n_events` | Move-outs observed that week. |
| `hazard` | n_events / n_at_risk. |
| `survival` | Kaplan-Meier survival probability. |

## sensitivity_checks.csv

Each candidate finding computed twice: once on every row where the variable is
present, once on the subset where it is trustworthy. `rmst_shift` is the
difference. A large shift means the finding is an artifact of the weak rows.

| column | meaning |
|---|---|
| `variable` / `band` | The segmentation being tested. |
| `n_all` / `events_all` / `rmst52_all` | Full-coverage figures. |
| `subset_all` | What "all" means for this variable. |
| `n_hiconf` / `events_hiconf` / `rmst52_hiconf` | High-confidence-subset figures. |
| `subset_hiconf` | The restriction applied. |
| `rmst_shift` | rmst52_hiconf - rmst52_all. |

## clv_baseline_by_segment.csv

| column | meaning |
|---|---|
| `segment` | ALL, site=X, or field=value. |
| `n` / `n_churned` / `pct_censored` | Segment size and how much of it is still open. |
| `mean_tenure_weeks_observed` | Naive mean — biased low by censoring. Shown for contrast. |
| `rmst_52w` | Restricted mean survival time over 52 weeks from the KM curve. The defensible expected-tenure figure. |
| `mean_weekly_rent` / `mean_collection_rate` | Revenue inputs. |
| `mean_collected_to_date` | Observed revenue per household so far. |
| `clv_52w_expected` | rmst_52w x mean_weekly_rent x mean_collection_rate. A 52-week-horizon CLV, not a perpetuity. |
