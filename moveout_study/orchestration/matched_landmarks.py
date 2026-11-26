"""Move-out study source summary.
Purpose: Compare weekly and survival predictions on exactly matched residents and prediction origins.
Inputs: Aligns one-week survival risks with weekly classifier probabilities at entry and week 4.
Outputs: Writes matched predictions, fold metrics, summaries, and sample/event counts for an apples-to-apples comparison.
Workflow: Residents without an observable one-week outcome at the landmark are excluded rather than labeled negative.
Method note: All matching keys include repeat and fold to preserve held-out evaluation boundaries.
"""

from pathlib import Path
import pandas as pd,numpy as np,sys
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'code/weekly_moveout/src'))
from moveout.metrics import weekly_metrics
rows=[];counts=[]
for date in ['20260915','20261031','20261115']:
 p=root/'results'/date
 w=pd.read_parquet(p/'combined_weekly/weekly_predictions.parquet');s=pd.read_parquet(p/'combined_survival/survival_predictions.parquet')
 for origin in [0,4]:
  ss=s[s.origin.eq(origin)&s.horizon.eq(1)].copy()
  valid=ss[(ss.event.eq(1)&ss.duration.le(1))|ss.duration.gt(1)]
  keys=['resident_id','repeat','fold']
  common=valid[keys].drop_duplicates().merge(w[w.week_index.eq(origin)][keys].drop_duplicates(),on=keys)
  a=valid.merge(common,on=keys);a['target']=(a.event.eq(1)&a.duration.le(1)).astype(int);a['probability']=a.risk
  b=w[w.week_index.eq(origin)].merge(common,on=keys).copy();b['target']=b.churned_this_week;b['probability']=b.predicted_probability
  pred=pd.concat([a,b],ignore_index=True)
  pred.to_parquet(p/f'matched_landmark_{origin}_predictions.parquet',index=False)
  unique=pred[['resident_id','target']].drop_duplicates();counts.append(dict(snapshot=date,origin=origin,households=len(unique),events=int(unique.target.sum())))
  for (model,rep,fold),g in pred.groupby(['model','repeat','fold']):
   m=weekly_metrics(g.target,g.probability);m.update(snapshot=date,origin=origin,model=model,repeat=rep,fold=fold);rows.append(m)
df=pd.DataFrame(rows);df.to_csv(root/'results/matched_landmark_metrics.csv',index=False)
df.groupby(['snapshot','origin','model'])[['brier','log_loss','roc_auc']].mean().reset_index().to_csv(root/'results/matched_landmark_summary.csv',index=False)
pd.DataFrame(counts).to_csv(root/'results/matched_landmark_counts.csv',index=False)
