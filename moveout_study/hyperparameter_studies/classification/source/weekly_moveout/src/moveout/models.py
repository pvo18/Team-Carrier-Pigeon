"""Move-out study source summary.
Purpose: Implement shared preprocessing, weekly classifiers, survival models, persistence adapters, and deterministic ensembles.
Inputs: Supports constant/tenure references, logistic and FFNN weekly models, Kaplan-Meier, classical Cox, and PyCox variants.
Outputs: The Adapter interface standardizes prediction and serialization across libraries and model families.
Workflow: Preprocessing is fit only on training data, and deterministic seeds/algorithms support reproducible comparisons.
Method note: Survival probabilities use right-continuous step evaluation and leave unsupported tails missing.
"""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from .data import CATEGORICAL, NEVER

def transformer(frame, features, expanded=True):
    """Build the train-only numeric/categorical preprocessing pipeline and reject forbidden predictors.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    if NEVER & set(features): raise ValueError("Forbidden predictor in feature contract")
    # Determine column treatment from the approved contract and observed pandas data types.
    cat = [c for c in features if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(frame[c])]
    num = [c for c in features if c not in cat]
    encoding = OneHotEncoder(handle_unknown="infrequent_if_exist" if expanded else "ignore",
                             min_frequency=5 if expanded else None, sparse_output=False)
    columns = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                               ("scale", StandardScaler())]), num),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                   ("encode", encoding)]), cat)])
    return Pipeline([("columns", columns), ("nonconstant", VarianceThreshold())])

def seed_all(seed):
    """Seed Python, NumPy, and PyTorch and request deterministic single-threaded torch operations.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def network(n_features, hidden, dropout, outputs=1, dummy_zero=False, final_bias=True):
    """Construct a configurable feed-forward network, optionally adding a fixed zero-origin survival output.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    import torch
    from torch import nn
    layers = []; n = n_features
    # Build hidden layers in declared order so architecture variants remain directly traceable.
    for width in hidden:
        layers += [nn.Linear(n, width), nn.ReLU()]
        if dropout: layers.append(nn.Dropout(dropout))
        n = width
    layers.append(nn.Linear(n, outputs, bias=final_bias))
    base = nn.Sequential(*layers)
    if not dummy_zero: return base
    class ZeroOrigin(nn.Module):
        """Wrap a network with a fixed near-zero logit at the survival time origin."""

        def __init__(self, base):
            """Store the learned base network used for all non-origin logits."""
            super().__init__()
            self.base = base

        def forward(self, x):
            """Prepend the fixed origin logit and return the complete logit sequence."""
            logits = self.base(x)
            return torch.cat([logits.new_full((len(x), 1), -1e9), logits], dim=1)
    return ZeroOrigin(base)

def sample_survival(survival, horizons):
    """Right-continuous step evaluation; unsupported tails remain NaN."""
    index = survival.index.to_numpy(dtype=float)
    values = survival.to_numpy(dtype=float)
    result = np.ones((values.shape[1], len(horizons)))
    for j, h in enumerate(horizons):
        if h > index.max() + 1e-8: result[:, j] = np.nan; continue
        i = np.searchsorted(index, h, side="right") - 1
        if i >= 0: result[:, j] = values[i]
    return np.clip(result, 0, 1)

class Adapter:
    def __init__(self, kind, processor=None, features=None, config=None):
        """Initialize the object state required by later prediction or metric operations.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        self.kind = kind; self.processor = processor; self.features = features or []
        self.config = config or {}; self.history = []; self.best_epoch = None
        self.estimator = None; self.reference = None

    def matrix(self, frame):
        """Transform the adapter's approved feature columns into the numeric matrix expected by its estimator.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        return np.asarray(self.processor.transform(frame[self.features]), dtype=np.float32)

    def predict_weekly_probability(self, frame):
        """Return next-week event probabilities through a common interface for references, classifiers, or survival models.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        # References bypass learned preprocessing and return rates estimated from training data only.
        if self.kind == "reference":
            global_rate = self.reference["global"]
            return np.array([self.reference["cells"].get(f"{r.site}|{int(r.week_index)//4}", global_rate)
                             if self.config.get("reference") == "tenure_site" else global_rate
                             for r in frame.itertuples()], dtype=float)
        if self.kind in {"logistic", "ffnn", "weighted_logistic", "weighted_ffnn"}:
            import torch
            self.estimator.eval()
            # Disable gradient tracking during scoring for deterministic, memory-efficient inference.
            with torch.no_grad():
                return torch.sigmoid(self.estimator(torch.from_numpy(self.matrix(frame))).reshape(-1)).numpy()
        return 1 - self.predict_survival(frame, [1])[:, 0]

    def predict_survival(self, frame, horizons):
        """Return survival probabilities at requested horizons through the model-family-specific implementation.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        if self.kind == "km":
            s = self.estimator.survival_function_.copy()
            return np.repeat(sample_survival(s, horizons), len(frame), axis=0)
        x = self.matrix(frame)
        if self.kind == "classical_cox":
            return sample_survival(self.estimator.predict_survival_function(pd.DataFrame(x)), horizons)
        if self.kind in {"pycox_linear", "pycox_neural", "logistic_hazard"}:
            return sample_survival(self.estimator.predict_surv_df(x), horizons)
        raise ValueError("Weekly classifiers do not forecast multiweek curves without future covariate paths")

    def save(self, directory):
        """Write one figure to the analysis output directory at presentation resolution, then close it to release memory.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(kind=self.kind, features=self.features, config=self.config,
                        best_epoch=self.best_epoch, history=self.history, reference=self.reference)
        if self.processor is not None: joblib.dump(self.processor, directory / "preprocessor.joblib")
        if self.kind in {"reference"}: pass
        elif self.kind in {"classical_cox", "km"}: joblib.dump(self.estimator, directory / "estimator.joblib")
        else:
            import torch
            net = self.estimator.net if self.kind.startswith("pycox") or self.kind == "logistic_hazard" else self.estimator
            torch.save(net.state_dict(), directory / "weights.pt")
            if self.kind.startswith("pycox"):
                self.estimator.baseline_hazards_.to_csv(directory / "baseline_hazards.csv")
        (directory / "model.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, directory):
        """Load only artifacts produced by this package; joblib is not an untrusted-file format."""
        directory = Path(directory)
        if (directory / "ensemble.json").exists():
            metadata = json.loads((directory / "ensemble.json").read_text())
            return Ensemble([cls.load(directory / member) for member in metadata["members"]])
        meta = json.loads((directory / "model.json").read_text())
        obj = cls(meta["kind"], features=meta["features"], config=meta["config"])
        obj.best_epoch = meta["best_epoch"]; obj.history = meta["history"]; obj.reference = meta["reference"]
        if (directory / "preprocessor.joblib").exists(): obj.processor = joblib.load(directory / "preprocessor.joblib")
        if obj.kind == "reference": return obj
        if obj.kind in {"km", "classical_cox"}:
            obj.estimator = joblib.load(directory / "estimator.joblib"); return obj
        import torch
        cfg = obj.config
        is_survival = obj.kind.startswith("pycox") or obj.kind == "logistic_hazard"
        net = network(cfg["n_features"], cfg["hidden"], cfg["dropout"],
                      outputs=cfg.get("outputs", 1), dummy_zero=obj.kind == "logistic_hazard",
                      final_bias=not obj.kind.startswith("pycox"))
        net.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
        if not is_survival: obj.estimator = net
        else:
            import torchtuples as tt
            from pycox.models import CoxPH, LogisticHazard
            obj.estimator = (LogisticHazard(net, tt.optim.Adam, duration_index=np.arange(cfg["outputs"]+1), device="cpu")
                             if obj.kind == "logistic_hazard" else CoxPH(net, tt.optim.Adam, device="cpu"))
            if obj.kind.startswith("pycox"):
                baseline = pd.read_csv(directory / "baseline_hazards.csv", index_col=0).iloc[:, 0]
                baseline.index = baseline.index.astype(float)
                obj.estimator.baseline_hazards_ = baseline
                obj.estimator.baseline_cumulative_hazards_ = baseline.cumsum()
        return obj

class Ensemble:
    """Prediction-equivalent reload of the selected seed ensemble."""
    def __init__(self, members):
        """Initialize the object state required by later prediction or metric operations.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        if not members: raise ValueError("Empty ensemble")
        self.members = members; self.kind = members[0].kind; self.features = members[0].features
    def predict_weekly_probability(self, frame):
        """Return next-week event probabilities through a common interface for references, classifiers, or survival models.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        return np.mean([m.predict_weekly_probability(frame) for m in self.members], axis=0)
    def predict_survival(self, frame, horizons):
        """Return survival probabilities at requested horizons through the model-family-specific implementation.

        Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
        Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
        """
        return np.mean([m.predict_survival(frame, horizons) for m in self.members], axis=0)

def fit_reference(train, tenure_site=False):
    """Fit either a global constant event-rate reference or a tenure-by-site smoothed reference.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    obj = Adapter("reference", config=dict(reference="tenure_site" if tenure_site else "constant"))
    rate = float(train.churned_this_week.mean()); cells = {}
    if tenure_site:
        for (site, band), g in train.groupby(["site", train.week_index//4]):
            # Twenty prior pseudo-weeks, using only the training global event rate.
            cells[f"{site}|{int(band)}"] = float((g.churned_this_week.sum()+20*rate)/(len(g)+20))
    obj.reference = dict(global_rate=rate, cells=cells)
    # JSON uses a stable human-readable field; prediction keeps backward-safe access below.
    obj.reference["global"] = obj.reference.pop("global_rate")
    return obj

def fit_weekly(kind, train, val, features, cfg, processor=None, hidden=None, weight_decay=0.0,
               seed=42, historical=False):
    """Train one weekly logistic/FFNN candidate with early stopping and optional class weighting.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    import torch
    from torch import nn
    seed_all(seed)
    processor = processor or transformer(train, features, cfg["feature_set"] == "expanded").fit(train[features])
    obj = Adapter(kind, processor, features)
    x = torch.from_numpy(obj.matrix(train)); y = torch.tensor(train.churned_this_week.to_numpy(), dtype=torch.float32)
    xv = torch.from_numpy(obj.matrix(val)); yv = torch.tensor(val.churned_this_week.to_numpy(), dtype=torch.float32)
    hidden = hidden if hidden is not None else ([] if "logistic" in kind else [8])
    dropout = cfg["dropout"] if hidden else 0.0
    obj.estimator = network(x.shape[1], hidden, dropout)
    weighted = kind.startswith("weighted")
    pos = float(y.sum())
    if pos <= 0 or pos == len(y): raise ValueError("Weekly training requires both classes")
    if weighted:
        imbalance_weight = (len(y)-pos)/pos
        weight_mode = cfg.get("class_weight_mode", "full")
        if weight_mode == "sqrt":
            weight = float(np.sqrt(imbalance_weight))
        elif weight_mode == "full":
            weight = imbalance_weight
        else:
            raise ValueError(f"Unknown class_weight_mode {weight_mode}")
    else:
        weight = 1.0
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    lr = 0.05 if kind == "weighted_logistic" and historical else cfg["learning_rate"]
    optimizer = torch.optim.Adam(obj.estimator.parameters(), lr=lr, weight_decay=weight_decay)
    batch = cfg["batch_size"]
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch,
                                         shuffle=True, generator=generator)
    best = np.inf; stale = 0; best_state = None
    # Train up to the configured maximum while validation-based early stopping controls overfit.
    for epoch in range(cfg["epochs"]):
        obj.estimator.train()
        for xb, yb in loader:
            optimizer.zero_grad(); loss = objective(obj.estimator(xb).reshape(-1), yb)
            if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
            loss.backward(); optimizer.step()
        obj.estimator.eval()
        with torch.no_grad():
            train_loss = float(objective(obj.estimator(x).reshape(-1), y))
            # Select probabilities with ordinary likelihood, not weighted validation BCE.
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(obj.estimator(xv).reshape(-1), yv))
        obj.history.append(dict(epoch=epoch, train_objective=train_loss, validation_log_loss=val_loss))
        if val_loss < best-cfg["min_delta"]:
            best = val_loss; stale = 0; obj.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in obj.estimator.state_dict().items()}
        else: stale += 1
        if not historical and stale >= cfg["patience"]: break
    if not historical and best_state is not None: obj.estimator.load_state_dict(best_state)
    obj.config = dict(n_features=x.shape[1], hidden=hidden, dropout=dropout, outputs=1,
                      weight_decay=weight_decay, seed=seed, learning_rate=lr, pos_weight=weight,
                      class_weight_mode=cfg.get("class_weight_mode", "none") if weighted else "none",
                      historical_fixed_epochs=historical)
    return obj

def fit_survival(kind, train, val, features, cfg, processor=None, hidden=None, penalizer=0.1, seed=42):
    """Train one Kaplan-Meier, classical Cox, or PyCox survival candidate using the common feature contract.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    if train.event.sum() < 2: raise ValueError("At least two training events required")
    if kind == "km":
        from lifelines import KaplanMeierFitter
        obj = Adapter(kind); obj.estimator = KaplanMeierFitter().fit(train.duration, train.event); return obj
    processor = processor or transformer(train, features, cfg["feature_set"] == "expanded").fit(train[features])
    obj = Adapter(kind, processor, features)
    x = obj.matrix(train); xv = obj.matrix(val)
    durations = train.duration.to_numpy(dtype=np.float32); events = train.event.to_numpy(dtype=np.float32)
    vd = val.duration.to_numpy(dtype=np.float32); ve = val.event.to_numpy(dtype=np.float32)
    if kind == "classical_cox":
        from lifelines import CoxPHFitter
        obj.estimator = CoxPHFitter(penalizer=penalizer)
        frame = pd.DataFrame(x); frame["duration"] = durations; frame["event"] = events
        obj.estimator.fit(frame, duration_col="duration", event_col="event")
        obj.training_matrix = frame
        obj.config = dict(penalizer=penalizer, tie_method="Efron", baseline_method="Breslow")
        return obj
    seed_all(seed)
    import torchtuples as tt
    from pycox.models import CoxPH, LogisticHazard
    hidden = [] if kind == "pycox_linear" else (hidden or [8])
    dropout = cfg["dropout"] if hidden else 0.0
    outputs = cfg["logistic_hazard"]["max_weeks"] if kind == "logistic_hazard" else 1
    net = network(x.shape[1], hidden, dropout, outputs, dummy_zero=kind == "logistic_hazard",
                  final_bias=not kind.startswith("pycox"))
    optim = tt.optim.Adam(lr=cfg["learning_rate"], weight_decay=penalizer)
    if kind == "logistic_hazard":
        obj.estimator = LogisticHazard(net, optim, duration_index=np.arange(outputs+1), device="cpu")
        def discretize(d, e):
            # Events enter containing intervals; censoring contributes complete intervals only.
            """Implement the discretize step used by this module's documented workflow.

            Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
            Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
            """
            ev = ((e == 1) & (d <= outputs)).astype(np.float32)
            ix = np.where(ev > 0, np.ceil(d), np.floor(d)).clip(0, outputs).astype(np.int64)
            return ix, ev
        target = discretize(durations, events); validation = (xv, discretize(vd, ve))
    else:
        if ve.sum() == 0: raise ValueError("Cox validation requires an event")
        obj.estimator = CoxPH(net, optim, device="cpu")
        target = (durations, events); validation = (xv, (vd, ve))
    log = obj.estimator.fit(x, target, batch_size=len(x), epochs=cfg["epochs"], verbose=False,
                            callbacks=[tt.callbacks.EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])],
                            val_data=validation, val_batch_size=len(xv))
    if kind.startswith("pycox"): obj.estimator.compute_baseline_hazards()
    obj.history = json.loads(log.to_pandas().reset_index().to_json(orient="records"))
    obj.best_epoch = int(np.argmin(log.to_pandas()["val_loss"])) if len(log.to_pandas()) else None
    obj.config = dict(n_features=x.shape[1], hidden=hidden, dropout=dropout, outputs=outputs,
                      penalizer=penalizer, seed=seed, learning_rate=cfg["learning_rate"])
    return obj

def ph_diagnostics(adapter, directory):
    """Write proportional-hazards diagnostics for a fitted classical Cox adapter when supported.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    from lifelines.statistics import proportional_hazard_test
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    tested = proportional_hazard_test(adapter.estimator, adapter.training_matrix, time_transform="rank")
    tested.summary.to_csv(directory / "ph_tests.csv")
    residual = adapter.estimator.compute_residuals(adapter.training_matrix, kind="scaled_schoenfeld")
    residual.to_csv(directory / "scaled_schoenfeld.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for col in residual.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        times = adapter.training_matrix.loc[residual.index, "duration"].to_numpy()
        ax.scatter(times, residual[col], alpha=0.5, s=10); ax.axhline(0, color="grey")
        ax.set(xlabel="Residual tenancy weeks", ylabel="Scaled Schoenfeld residual", title=str(col))
        fig.tight_layout(); fig.savefig(directory / f"ph_{str(col)}.png", dpi=120); plt.close(fig)
    return tested.summary
