# Classification hyperparameter results

The fixed controls were expanded logistic regression (Brier **0.018452**, log
loss **0.082975**, AUC **0.812897**) and expanded FFNN (Brier **0.017623**, log
loss **0.079899**, AUC **0.814328**).

Weight decay of **0.001** gave the best balanced setting for both model
families. For logistic regression it produced Brier **0.018219**, log loss
**0.081922**, and AUC **0.813562**. For the FFNN it produced Brier
**0.017596**, log loss **0.079690**, and AUC **0.815807**.

The FFNN architecture `[16, 8]` achieved the single lowest trial Brier score
(**0.017554**) but worsened log loss and AUC relative to the FFNN control.
Patience 50 was close to the balanced FFNN result. Aggressive class weighting
substantially worsened Brier score and log loss even when AUC changed little,
demonstrating why all three metrics were retained.

`hyperparameter_trial_results.csv` reports signed deltas as trial minus
control. Therefore, negative Brier/log-loss deltas are improvements, while a
positive AUC delta is an improvement.
