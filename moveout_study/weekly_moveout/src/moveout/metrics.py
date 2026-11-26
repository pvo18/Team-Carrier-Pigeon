"""Move-out study source summary.
Purpose: Compute weekly and censor-aware survival metrics, calibration summaries, and paired uncertainty intervals.
Inputs: Weekly metrics use direct binary outcomes; horizon metrics use inverse-probability-of-censoring weights.
Outputs: Support checks suppress discrimination measures when follow-up, cases, controls, or censor survival are inadequate.
Workflow: Clustered bootstrap resampling preserves repeated observations belonging to the same tenancy.
Method note: All probabilities are clipped only for numerical stability in logarithms.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

def calibration_coefficients(y, p, weights=None):
    """Fit weighted calibration intercept and slope when both outcome classes have adequate support.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    from scipy.optimize import minimize
    y = np.asarray(y, float); p = np.asarray(p, float).clip(1e-7, 1-1e-7)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    # Zero-weight observations have unknown horizon status and must not influence calibration fitting.
    used = w > 0
    if np.sum(y[used] == 1) < 5 or np.sum(y[used] == 0) < 5:
        return dict(calibration_intercept=None, calibration_slope=None, calibration_status="insufficient_events")
    x = np.log(p/(1-p))
    def objective(ab):
        """Implement the objective step used by this module's documented workflow.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        z = ab[0]+ab[1]*x
        return np.sum(w*(np.logaddexp(0, z)-y*z))/np.sum(w)
    fit = minimize(objective, [0., 1.], method="BFGS")
    return dict(calibration_intercept=float(fit.x[0]), calibration_slope=float(fit.x[1]),
                calibration_status="ok" if fit.success else "unstable_fit")

def weekly_metrics(y, p, frame=None):
    """Calculate binary probability accuracy, discrimination, prevalence, calibration, and optional tenancy-balanced scores.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all(): raise ValueError("Invalid predictions")
    p = p.clip(1e-7, 1-1e-7)
    result = dict(n_rows=len(y), n_events=int(y.sum()), brier=float(brier_score_loss(y, p)),
                  log_loss=float(log_loss(y, p, labels=[0, 1])),
                  roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                  pr_auc=float(average_precision_score(y, p)) if y.sum() else None,
                  predicted_rate=float(p.mean()), event_rate=float(y.mean()))
    if frame is not None:
        local = pd.DataFrame(dict(resident_id=frame.resident_id.to_numpy(), squared=(p-y)**2,
                                  loss=-(y*np.log(p)+(1-y)*np.log1p(-p))))
        result["tenancy_weighted_brier"] = float(local.groupby("resident_id").squared.mean().mean())
        result["tenancy_weighted_log_loss"] = float(local.groupby("resident_id").loss.mean().mean())
    result.update(calibration_coefficients(y, p))
    return result

class CensorKM:
    def __init__(self, duration, event):
        """Initialize the object state required by later prediction or metric operations.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        d = np.asarray(duration, float); e = np.asarray(event, int)
        self.times = np.unique(d); probabilities = []; g = 1.0
        for t in self.times:
            at_risk = np.sum(d >= t); censored = np.sum((d == t) & (e == 0))
            g *= 1 - censored/at_risk
            probabilities.append(g)
        self.values = np.array(probabilities)
    def at(self, t, left=False):
        """Implement the at step used by this module's documented workflow.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        t = np.asarray(t, float)
        ix = np.searchsorted(self.times, t, side="left" if left else "right")-1
        return np.where(ix < 0, 1.0, self.values[np.maximum(ix, 0)])

def ipcw_components(train, test, risk, horizon, minimum_g=0.05):
    """Construct censor-aware targets and inverse censoring weights for one survival horizon.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    d = test.duration.to_numpy(float); e = test.event.to_numpy(int); risk = np.asarray(risk, float)
    # Define cases as observed events by the horizon and controls as known event-free survivors beyond it.
    cases = (d <= horizon) & (e == 1); controls = d > horizon
    censor = CensorKM(train.duration, train.event)
    gh = float(censor.at(horizon)); gt = censor.at(d, left=True)
    # Reject horizons that require extrapolation or unstable inverse-censoring weights.
    if horizon >= train.duration.max() or gh < minimum_g or np.any(gt[cases] < minimum_g):
        return None
    if not np.isfinite(risk).all(): return None
    weights = np.zeros(len(test)); weights[cases] = 1 / gt[cases]; weights[controls] = 1 / gh
    target = cases.astype(int)
    return target, weights, cases, controls

def horizon_metrics(train, test, risk, horizon, minimum_g=0.05, minimum_cases=5):
    """Calculate censor-weighted Brier/log loss and supported discrimination measures at one horizon.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    components = ipcw_components(train, test, risk, horizon, minimum_g)
    if components is None: return dict(horizon=horizon, status="unsupported_followup_or_censor_weights")
    y, w, cases, controls = components
    p = np.asarray(risk, float).clip(1e-7, 1-1e-7)
    brier = float(np.mean(w*(p-y)**2))
    loss = float(np.mean(w*(-y*np.log(p)-(1-y)*np.log1p(-p))))
    support = cases.sum() >= minimum_cases and controls.sum() >= minimum_cases
    auc = float(roc_auc_score(y[w > 0], p[w > 0], sample_weight=w[w > 0])) if support else None
    # IPCW concordance at this horizon: event cases by horizon versus longer-observed comparators.
    censor = CensorKM(train.duration, train.event); numerator = denominator = 0.0
    d = test.duration.to_numpy(float); e = test.event.to_numpy(int)
    # Compare each observed event with residents known to remain at risk longer for IPCW concordance.
    for i in np.flatnonzero((e == 1) & (d <= horizon)):
        compare = d > d[i]; wi = float(1 / censor.at(d[i], left=True)**2)
        denominator += wi*compare.sum()
        numerator += wi*(np.sum(p[i] > p[compare]) + 0.5*np.sum(p[i] == p[compare]))
    return dict(horizon=horizon, status="ok", brier=brier, log_loss=loss, roc_auc=auc,
                discrimination_status="ok" if support else "insufficient_cases_or_controls",
                concordance_ipcw=numerator/denominator if denominator and support else None,
                n_cases=int(cases.sum()), n_controls=int(controls.sum()),
                n_unknown=int((w == 0).sum()), observed_risk_ipcw=float(np.mean(w*y)),
                predicted_risk=float(p.mean()))

def calibration_table(y, p, weights=None, groups=5):
    """Bin supported predictions and return weighted observed-versus-predicted risk for calibration plots.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    p = np.asarray(p, float); y = np.asarray(y, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    d = pd.DataFrame(dict(observed=y, predicted=p, weight=w))
    d = d[d.weight > 0].copy()
    if d.empty: return pd.DataFrame()
    d["bin"] = pd.qcut(d.predicted, q=min(groups, len(d)), duplicates="drop")
    if d.bin.isna().all(): d["bin"] = "all"
    rows = []
    for label, group in d.groupby("bin", observed=True):
        rows.append(dict(bin=str(label), n=len(group), predicted=float(np.average(group.predicted, weights=group.weight)),
                         observed=float(np.average(group.observed, weights=group.weight))))
    return pd.DataFrame(rows)

def integrated_brier(metrics):
    """Average supported horizon-specific Brier scores into a compact overall survival score.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    good = sorted([r for r in metrics if r.get("status") == "ok"], key=lambda r: r["horizon"])
    if len(good) < 2: return None
    times = np.array([r["horizon"] for r in good]); scores = np.array([r["brier"] for r in good])
    return float(np.trapezoid(scores, times)/(times[-1]-times[0]))

def paired_bootstrap(frame, group_column="resident_id", replicates=200, seed=42):
    """One contribution per cluster; repeats and folds never become independent samples."""
    columns = ["squared_error_a", "squared_error_b", "log_loss_a", "log_loss_b"]
    grouped = frame.groupby(group_column)[columns].mean()
    rng = np.random.default_rng(seed); n = len(grouped)
    if not n: return {}
    delta_b = (grouped.squared_error_a-grouped.squared_error_b).to_numpy()
    delta_l = (grouped.log_loss_a-grouped.log_loss_b).to_numpy()
    samples = rng.integers(0, n, size=(replicates, n))
    b = delta_b[samples].mean(axis=1); l = delta_l[samples].mean(axis=1)
    return dict(unit=group_column, n_clusters=n, brier_delta=float(delta_b.mean()),
                brier_ci=np.quantile(b, [0.025, 0.975]).tolist(), log_loss_delta=float(delta_l.mean()),
                log_loss_ci=np.quantile(l, [0.025, 0.975]).tolist())
