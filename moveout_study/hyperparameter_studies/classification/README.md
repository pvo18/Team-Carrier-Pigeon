# Classification Hyperparameter Study — Commented Code and Results

## Purpose

This package reproduces the controlled hyperparameter study for the weekly
move-out classification question: **given information available at the start
of a completed tenant-week, what is the probability of move-out during the
following week?** It compares logistic regression and a feed-forward neural
network (FFNN).

Every trial changes one factor at a time while retaining the November 15
cohort, the same 25 repeat-fold evaluations, training-only preprocessing,
random-seed policy, scoring code, and all other settings. A result row is the
mean of those 25 outer-test evaluations—not one tenant and not one training
run.

## Run a trial

Install the dependencies in `source/weekly_moveout/requirements-tested.txt`,
then provide one trial name:

```bash
python run_classification_hyperparameter_trial.py nn_wd_0001
```

Run the script without a valid name to print the complete list. New outputs
are written to `new_trial_results/<trial-name>/`. The script refuses to
overwrite an existing trial folder so the experiment trail is preserved.

## Files

- `run_classification_hyperparameter_trial.py`: heavily commented experiment
  catalog and one-factor-at-a-time runner.
- `source/weekly_moveout/`: commented data, model, metric, and pipeline code.
- `config/20261115.yaml`: fixed November 15 settings.
- `prepared/`: fixed prepared cohort and household-held-out fold assignments.
- `results/hyperparameter_trial_results.csv`: completed 30-trial comparison.
- `results/RESULTS.md`: concise numerical interpretation.
- `graphics/`: labeled figures used to explain the controlled trials.
- `docs/Classification_Hyperparameter_Study_Explanation.docx`: Word narrative.

## Important interpretation

Lower Brier score and log loss indicate better probability estimates; higher
AUC indicates better ranking. A setting should not be selected from AUC alone:
class weighting showed that ranking can remain similar while probability
quality deteriorates substantially. These results are retrospective model
comparisons and do not establish why an individual tenant moved out.
