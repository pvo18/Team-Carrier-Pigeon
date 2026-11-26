"""Move-out study source summary.
Purpose: Define the reproducible default experiment configuration and validate user overrides.
Inputs: Defaults cover folds, repeats, endpoints, horizons, optimization, regularization, seeds, and support thresholds.
Outputs: A YAML file may replace individual settings while nested logistic-hazard settings are merged safely.
Workflow: Validation rejects unsupported feature/endpoint policies and nonsensical horizons or training controls.
Method note: A deep copy prevents one experiment from mutating defaults used by another.
"""

from pathlib import Path
import copy
import yaml

# Centralize every experiment-control value so manifests capture the full study configuration.
DEFAULTS = dict(mode="survival_comparison", feature_set="expanded", seed=42,
                folds=5, repeats=5, bootstrap_replicates=200, censor_date="2026-09-15",
                endpoint_policy="exclude", strict_counts=False,
                include_sensitive_research_features=False, landmarks=[0, 4],
                horizons=[1, 4, 8, 12], validation_fraction=0.2, epochs=300,
                patience=25, min_delta=1e-5, batch_size=64, dropout=0.2,
                learning_rate=0.001, weight_decays=[1e-4, 0.01],
                cox_penalizers=[0.01, 0.1, 1.0], hidden_layers=[[4], [8]],
                neural_seeds=[42, 43, 44, 45, 46], minimum_censor_survival=0.05,
                minimum_cases=5,
                logistic_hazard=dict(enabled="conditional", max_weeks=12,
                                     practical_relative_gain=0.05))

def load_config(path=None):
    """Return a validated deep copy of defaults with optional YAML overrides applied.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    # Never allow nested overrides in one run to mutate module-level defaults for later runs.
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        supplied = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in supplied.items():
            if k == "logistic_hazard": cfg[k].update(v)
            else: cfg[k] = v
    if cfg["feature_set"] not in {"expanded", "reported"}:
        raise ValueError("feature_set must be expanded or reported")
    if cfg["endpoint_policy"] not in {"panel", "exclude", "error"}:
        raise ValueError("endpoint_policy must be panel, exclude, or error")
    if any(h <= 0 or h >= 52 for h in cfg["horizons"]):
        raise ValueError("This extract supports positive horizons below 52 weeks only")
    if not 0 < cfg["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if cfg["epochs"] < 1 or cfg["patience"] < 1:
        raise ValueError("epochs and patience must be positive")
    return cfg
