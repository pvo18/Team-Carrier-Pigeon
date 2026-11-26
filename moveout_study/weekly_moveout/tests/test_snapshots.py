"""Move-out study source summary.
Purpose: Test revised-study safeguards for snapshot cutoffs and sparse survival discrimination.
Inputs: Constructs minimal synthetic inputs that isolate each edge condition.
Outputs: Confirms post-cutoff records fail validation and insufficient case/control support suppresses AUC.
Workflow: The tests protect methodological constraints added for the refreshed snapshots.
Method note: No external data or trained artifacts are required.
"""

from test_core import residents, weekly
import pandas as pd
import numpy as np
from moveout.data import structural_validation
from moveout.metrics import horizon_metrics

# Create an intentionally post-cutoff row and confirm structural validation rejects it.
def test_snapshot_cutoff(residents,weekly):
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    cfg={'censor_date':'2026-01-01'}
    result=structural_validation(residents,weekly,cfg)
    assert 'weekly_row_after_cutoff' in result['errors']

# Require adequate cases and controls before reporting survival AUC/concordance.
def test_sparse_discrimination_is_suppressed():
    """Verify the named behavioral requirement using isolated synthetic or persisted study evidence.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    train=pd.DataFrame(dict(duration=[1.,2.,3.,4.,5.],event=[1,0,1,1,0]))
    test=pd.DataFrame(dict(duration=[1.,1.,4.],event=[1,0,0]))
    result=horizon_metrics(train,test,np.array([.8,.9,.2]),2.,.01)
    assert result['roc_auc'] is None
    assert result['discrimination_status']=='insufficient_cases_or_controls'
    assert np.isfinite(result['brier'])
