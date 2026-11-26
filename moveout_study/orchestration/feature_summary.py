"""Move-out study source summary.
Purpose: Consolidate fold-level permutation importance files into a study-level feature summary.
Inputs: Finds every validation_feature_importance.csv produced by weekly and survival jobs across snapshots.
Outputs: Writes both the stacked fold evidence and grouped mean/positive-fold counts by model, origin, and feature.
Workflow: Importance represents predictive association on inner validation data and must not be interpreted causally.
Method note: The script produces no model selections or test-set changes.
"""

from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent
rows=[]
for date in ['20260915','20261031','20261115']:
 for kind in ['weekly','survival']:
  for p in (root/'results'/date/'folds').glob(kind+'_r*_f*'):
   for q in p.rglob('validation_feature_importance.csv'):
    df=pd.read_csv(q);df['snapshot']=date;df['task']=kind;df['model']=q.parent.name;df['fold_job']=p.name;df['origin']=int(q.relative_to(p).parts[0].split('_')[1]) if kind=='survival' else None
    rows.append(df)
if rows:
 df=pd.concat(rows,ignore_index=True);df.to_csv(root/'results/feature_importance_folds.csv',index=False)
 print(df.columns.tolist())
if rows:
 summary=df.groupby(['snapshot','task','model','origin','feature'],dropna=False).agg(mean_increase=('validation_brier_increase','mean'),positive_folds=('validation_brier_increase',lambda x:int((x>0).sum())),folds=('validation_brier_increase','size')).reset_index()
 summary.to_csv(root/'results/feature_importance_summary.csv',index=False)
