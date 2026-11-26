# Survival hyperparameter trial results

## Scope

- November 15 snapshot: 563 qualified tenancies and 176 observed move-outs.
- Outcome: probability of move-out within 12 weeks.
- Evaluation: five repeats of five household-held-out folds (25 test evaluations per trial) at entry and week 4.
- Selection rule: lowest mean 12-week Brier score, with log loss and AUC retained as guardrails.

## Best setting by model and prediction origin

| Origin | Model | Trial | Brier | Log loss | AUC | Brier improvement (%) | Log-loss improvement (%) | AUC change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Classical Cox | cox_penalty_0.1 | 0.140200 | 0.450341 | 0.711718 | 0.000000 | 0.000000 | 0.000000 |
| 0 | Neural CoxPH | neural_dropout_0.4 | 0.141983 | 0.451915 | 0.685938 | 1.851447 | 1.375047 | 0.017063 |
| 4 | Classical Cox | cox_penalty_0.1 | 0.165668 | 0.522032 | 0.735156 | 0.000000 | 0.000000 | 0.000000 |
| 4 | Neural CoxPH | neural_hidden_16_8 | 0.160638 | 0.497820 | 0.737703 | 2.782597 | 2.526311 | 0.022917 |

## Interpretation

Positive Brier and log-loss improvement percentages mean the trial improved probability quality relative to that model's documented baseline. A positive AUC change means better tenant-risk ranking; a negative value means worse ranking.

These trials estimate predictive association, not why a tenant decides to leave. The results are retrospective and come from two sites. Neural trials use one deterministic seed per fold; the selected configuration should be confirmed with a multi-seed ensemble before operational use.

## Execution checks

- Successful fold-level evaluations: 900
- Failed fold-level evaluations: 0
- Unique trial definitions: 18
