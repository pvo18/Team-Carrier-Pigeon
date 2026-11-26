"""Move-out study source summary.
Purpose: Quantify coverage and descriptive retention sensitivity for income and commute variables.
Inputs: Creates broad available-case and conservative high-quality subsets at prespecified thresholds.
Outputs: Writes coverage and 12-week Kaplan-Meier summaries for each snapshot, variable, subset, and group.
Workflow: These comparisons describe data sensitivity and are not adjusted causal effect estimates.
Method note: Unsupported 12-week estimates remain missing instead of being extrapolated.
"""

from pathlib import Path
import pandas as pd,json
from lifelines import KaplanMeierFitter
root=Path(__file__).resolve().parent;rows=[];coverage=[]
for d in ['original','20260915','20261031','20261115']:
 src=root.parent/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/code/weekly_moveout/data/provider' if d=='original' else root.parent/'snapshots'/f'willis-crosby-resident-data-{d}'
 epdir=root.parent/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/results/prepared' if d=='original' else root/'results'/d/'prepared'
 r=pd.read_csv(src/'residents.csv');ep=pd.read_csv(epdir/'cohort_ledger.csv');x=r.merge(ep[ep.qualified][['resident_id','duration','event']],on='resident_id')
 coverage.append(dict(snapshot=d,qualified=len(x),income_exact_ok=int(x.income_quality_flag.eq('ok').sum()),income_flag_values=json.dumps(x.income_quality_flag.value_counts(dropna=False).to_dict()),commute_job_conf08=int((x.work_location_source.eq('job_address')&x.work_geocode_confidence.ge(.8)).sum())))
 for var,col,cut,clean in [('income','income_monthly_est',3000,x.income_quality_flag.eq('ok')),('commute','work_drive_min',20,x.work_location_source.eq('job_address')&x.work_geocode_confidence.ge(.8))]:
  for subset,mask in [('available',x[col].notna()),('conservative_quality',x[col].notna()&clean)]:
   for name,gmask in [('below',x[col].lt(cut)),('at_or_above',x[col].ge(cut))]:
    g=x[mask&gmask]
    km=KaplanMeierFitter().fit(g.duration,g.event) if len(g) else None
    supported=len(g)>0 and g.duration.max()>=12
    rows.append(dict(snapshot=d,variable=var,threshold=cut,subset=subset,group=name,n=len(g),events=int(g.event.sum()),at_risk12=int(g.duration.ge(12).sum()),survival12=float(km.predict(12)) if supported else None))
pd.DataFrame(rows).to_csv(root/'results/quality_sensitivity.csv',index=False);pd.DataFrame(coverage).to_csv(root/'results/quality_coverage.csv',index=False)
print(pd.DataFrame(coverage)[['snapshot','qualified','income_exact_ok','commute_job_conf08']].to_string(index=False))
