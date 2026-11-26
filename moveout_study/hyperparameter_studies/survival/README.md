# Survival Hyperparameter Study — Commented Code and Results

## Purpose

This package reproduces the survival hyperparameter study for the question:
**among tenants who have not yet moved out at the prediction origin, what is
their probability of moving out within the next 12 weeks?** Models are scored
at tenancy entry and after the tenant has remained through week 4.

Classical Cox varies only its L2 penalizer. Neural CoxPH changes learning rate,
hidden-layer architecture, dropout, weight decay, or early-stopping patience
one factor at a time. Every setting uses identical tenant-held-out folds and
the same 25 repeat-fold evaluations at each prediction origin.

## Run the study

Install `requirements.txt`, then run:

```bash
python run_survival_hyperparameter_trials.py --workers 5
```

Use `--smoke` to validate one entry/fold before a full run. Worker count changes
execution speed only; it does not change folds, seeds, or model settings.

## Files

- `run_survival_hyperparameter_trials.py`: heavily commented complete runner.
- `source/weekly_moveout/`: commented source for cohort creation, censoring,
  preprocessing, Cox/Neural CoxPH fitting, and horizon metrics.
- `config/20261115.yaml`: fixed analysis settings.
- `data/`: November 15 source tables and data documentation.
- `results/`: all 900 fold-level evaluations, summaries, comparisons, best
  settings, trial manifest, and concise interpretation.
- `graphics/`: labeled Classical Cox and Neural CoxPH trial figures.
- `docs/Survival_Hyperparameter_Study_Explanation.docx`: Word narrative.

## Important interpretation

Lower Brier score and log loss indicate better 12-week probability estimates;
higher AUC indicates better risk ranking. The selected setting may differ at
entry and week 4 because week-4 predictions condition on the tenant having
remained event-free for four weeks. These results show predictive association,
not a causal explanation of why a tenant chose to leave.
