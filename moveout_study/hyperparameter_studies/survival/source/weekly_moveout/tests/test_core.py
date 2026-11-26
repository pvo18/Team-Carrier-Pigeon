"""Move-out study source summary.
Purpose: Test the core data, split, metric, model, persistence, and gating behavior of the move-out package.
Inputs: Synthetic fixtures create deterministic tenancies with both events and censoring across two sites.
Outputs: Tests verify leakage controls, endpoint handling, IPCW behavior, reference arithmetic, clustered bootstrap, and adapters.
Workflow: Round-trip persistence checks ensure saved models reproduce valid predictions.
Method note: These tests exercise behavior; they do not assert the study's final empirical scores.
"""

import numpy as np
import pandas as pd
import pytest
from moveout.config import load_config
from moveout.data import make_folds, split_ids, make_endpoints, landmark, build_master, structural_validation, NEVER
from moveout.metrics import CensorKM, horizon_metrics, weekly_metrics, paired_bootstrap
from moveout.models import transformer, fit_weekly, fit_survival, Adapter
from moveout.pipeline import assess_gate

@pytest.fixture
def weekly():
    """Implement the weekly step used by this module's documented workflow.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    rows = []
    # Generate enough synthetic tenancies to exercise grouped splits with both events and censoring.
    for i in range(60):
        event = i % 3 == 0; terminal = 2 + i % 12
        for week in range(terminal+1):
            rows.append(dict(resident_id=f"T{i:03}", site="WIL" if i%2 else "CRO", week_index=week,
                             week_start=(pd.Timestamp("2026-01-05")+pd.Timedelta(weeks=week)).strftime("%Y-%m-%d"),
                             churned_this_week=int(event and week==terminal), censored_this_week=int(not event and week==terminal),
                             outstanding=float(i*10+week*2), delinquent=int(i%4==0), billed_this_week=200., paid_this_week=190.,
                             cum_billed=200.*(week+1), cum_paid=190.*(week+1)))
    return pd.DataFrame(rows)

@pytest.fixture
def residents(weekly):
    """Implement the residents step used by this module's documented workflow.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    ids = sorted(weekly.resident_id.unique())
    return pd.DataFrame([dict(resident_id=rid, site="WIL" if i%2 else "CRO", unit_key=f"U{i//2}", unit_known=1,
                              tenure_days=(weekly[weekly.resident_id.eq(rid)].week_index.max()+1)*7,
                              move_out_week=np.nan, income_monthly_est=3000.+i*20, rent_to_income_ratio=.3,
                              work_drive_min=10.+i, prior_home_distance_mi=20.+i, n_fobs=i%3, n_pets=i%2,
                              prior_rent_monthly=900., work_geocode_confidence=.8)
                         for i,rid in enumerate(ids)])

def test_grouped_folds(weekly):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg=load_config(); folds=make_folds(weekly,cfg)
    assert len(folds)==60*5
    for rep in range(5): assert folds[folds.repeat.eq(rep)].resident_id.nunique()==60
    train,val,test=split_ids(weekly,folds,0,0,cfg)
    assert not(train&val or train&test or val&test)

def test_lags_and_expansion(residents,weekly):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg=load_config(); master,features,audit=build_master(residents,weekly,None,cfg)
    for rid,g in master.groupby("resident_id"):
        assert g.outstanding_lag1.iloc[0]==0
        assert np.allclose(g.outstanding_lag1.iloc[1:],g.outstanding.iloc[:-1])
    assert "prior_rent_income_ratio" in features and "paid_this_week_past4_mean" in features
    assert not NEVER.intersection(features)
    # Leakage check: a transaction in week 10 must not alter features calculated for earlier weeks.
    # Changing a future payment cannot change earlier features.
    changed=weekly.copy(); changed.loc[changed.week_index.eq(10),"paid_this_week"]=99999
    other,_,_=build_master(residents,changed,None,cfg)
    early=master.week_index.lt(10)
    pd.testing.assert_frame_equal(master.loc[early,features],other.loc[early,features])

def test_validation_fails_duplicate(residents,weekly):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    bad=pd.concat([weekly,weekly.iloc[:1]],ignore_index=True)
    assert not structural_validation(residents,bad,load_config())["passed"]

def test_endpoint_ledger_and_landmarks(residents,weekly):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg=load_config(); cfg["endpoint_policy"]="exclude"
    residents.loc[0,"tenure_days"]=-7
    ep=make_endpoints(residents,weekly,cfg)
    assert len(ep)==60 and ep.qualified.sum()==59
    result=landmark(weekly,ep,4)
    assert result.resident_id.is_unique and result.duration.gt(0).all()
    assert "T000" not in set(result.resident_id)
    prepared=weekly.merge(ep[["resident_id","duration","event","qualified","cluster_id"]],on="resident_id")
    repeated=landmark(prepared,ep,4)
    assert np.allclose(result.duration,repeated.duration)

def test_train_only_transform():
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    train=pd.DataFrame(dict(work_drive_min=[1.,2.,3.],site=["A"]*3))
    proc=transformer(train,list(train)).fit(train)
    before=proc.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    proc.transform(pd.DataFrame(dict(work_drive_min=[999999.],site=["A"])))
    assert np.array_equal(before,proc.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"].statistics_)

def test_censor_unknown_not_negative():
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    train=pd.DataFrame(dict(duration=[1.,2.,3.,4.,5.],event=[1,0,1,1,0]))
    test=pd.DataFrame(dict(duration=[1.,1.,4.],event=[1,0,0]))
    r=horizon_metrics(train,test,np.array([.8,.9,.2]),2.,.01)
    assert r["n_unknown"]==1 and r["n_cases"]==1 and r["n_controls"]==1
    assert r["brier"]==pytest.approx((.2**2 + .2**2/.75)/3)
    assert horizon_metrics(train,test,np.ones(3)*.1,5.)["status"]!="ok"

def test_constant_reference_arithmetic():
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    p=100/5523; y=np.r_[np.ones(25),np.zeros(1183)]
    m=weekly_metrics(y,np.full(1208,p))
    assert m["brier"]==pytest.approx(.02027377041765142)
    assert m["log_loss"]==pytest.approx(.10091346057777945)

def test_bootstrap_clusters():
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    d=pd.DataFrame(dict(resident_id=["a","a","b","b"],squared_error_a=[.1,.1,.2,.2],
                        squared_error_b=[.2,.2,.3,.3],log_loss_a=[.1]*4,log_loss_b=[.2]*4))
    b=paired_bootstrap(d,replicates=20)
    assert b["n_clusters"]==2 and b["brier_delta"]==pytest.approx(-.1)

def test_gating_and_horizon(tmp_path):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    result=assess_gate(tmp_path,tmp_path/"gate.json")
    assert result["status"]=="review_required"
    import yaml
    path=tmp_path/"bad.yaml";path.write_text(yaml.safe_dump(dict(horizons=[52])))
    with pytest.raises(ValueError):load_config(path)

@pytest.mark.parametrize("kind",["logistic","ffnn","weighted_logistic","weighted_ffnn"])
def test_weekly_adapter(kind,weekly,residents,tmp_path):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg=load_config(); cfg.update(epochs=3,patience=2,feature_set="reported")
    master,features,_=build_master(residents,weekly,None,cfg)
    train=master[master.resident_id.lt("T040")];val=master[master.resident_id.ge("T040")]
    model=fit_weekly(kind,train,val,features,cfg,hidden=[] if "logistic" in kind else [4])
    pred=model.predict_weekly_probability(val)
    assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
    model.save(tmp_path/kind);other=Adapter.load(tmp_path/kind)
    assert np.allclose(pred,other.predict_weekly_probability(val))
    import json
    (tmp_path/"ensemble.json").write_text(json.dumps(dict(members=[kind,kind])))
    ensemble=Adapter.load(tmp_path)
    assert np.allclose(pred,ensemble.predict_weekly_probability(val))

@pytest.mark.parametrize("kind",["km","classical_cox","pycox_linear","pycox_neural","logistic_hazard"])
def test_survival_adapter(kind,weekly,residents,tmp_path):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg=load_config();cfg.update(epochs=3,patience=2,feature_set="reported",endpoint_policy="panel")
    master,features,_=build_master(residents,weekly,None,cfg)
    ep=make_endpoints(residents,weekly,cfg);frame=landmark(master,ep,0)
    train=frame[frame.resident_id.lt("T040")];val=frame[frame.resident_id.ge("T040")]
    features=[f for f in features if f!="week_index"]
    model=fit_survival(kind,train,val,features,cfg,hidden=[4],penalizer=.1)
    pred=model.predict_survival(val,[1,2,4])
    assert np.isfinite(pred).all() and np.all(np.diff(pred,axis=1)<=1e-6)
    model.save(tmp_path/kind);other=Adapter.load(tmp_path/kind)
    assert np.allclose(pred,other.predict_survival(val,[1,2,4]),atol=1e-6)
