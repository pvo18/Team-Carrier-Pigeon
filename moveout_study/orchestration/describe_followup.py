"""Move-out study source summary.
Purpose: Summarize censor-aware follow-up and cohort retention across all study snapshots.
Inputs: Fits descriptive Kaplan-Meier curves by snapshot and site at prespecified horizons.
Outputs: Writes survival, at-risk, and September-cohort follow-up tables for reporting and interpretation.
Workflow: This is descriptive analysis only and is separate from cross-validated prediction scoring.
Method note: Unsupported horizons are omitted when a cohort lacks sufficient observed duration.
"""

from pathlib import Path
import pandas as pd,numpy as np,json
from lifelines import KaplanMeierFitter
root=Path(__file__).resolve().parent
rows=[];cohorts=[]
for label in ['original','20260915','20261031','20261115']:
 if label=='original':p=root.parent/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/results/prepared'
 else:p=root/'results'/label/'prepared'
 ep=pd.read_csv(p/'cohort_ledger.csv');ep=ep[ep.qualified]
 for site in ['all','WIL','CRO']:
  e=ep if site=='all' else ep[ep.site.eq(site)]
  km=KaplanMeierFitter().fit(e.duration,e.event)
  for t in [4,8,12,24,36]:
   if t<=e.duration.max():rows.append(dict(snapshot=label,site=site,horizon=t,survival=float(km.predict(t)),at_risk=int(e.duration.ge(t).sum()),households=len(e)))
 if label!='original':
  sep=pd.read_csv(root/'results/20260915/prepared/cohort_ledger.csv');common=ep[ep.resident_id.isin(sep.resident_id)]
  cohorts.append(dict(snapshot=label,september_panel_cohort=len(common),events=int(common.event.sum()),still_censored=int(common.event.eq(0).sum()),new_panel_households=int((~ep.resident_id.isin(sep.resident_id)).sum())))
pd.DataFrame(rows).to_csv(root/'results/km_comparison.csv',index=False)
pd.DataFrame(cohorts).to_csv(root/'results/september_cohort_followup.csv',index=False)
print(pd.DataFrame(cohorts).to_string(index=False))
