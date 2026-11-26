"""Move-out study source summary.
Purpose: Combine the 25 fold jobs for each snapshot and analysis task into qualified study-level results.
Inputs: Reads per-fold weekly and survival metrics/predictions after confirming every expected job completed.
Outputs: Writes combined predictions, aggregate metrics, model summaries, support counts, and the standard comparison report.
Workflow: Aggregation never retrains a model; it only consolidates held-out artifacts created by the fold runners.
Method note: A hard completeness check prevents partially finished snapshots from being reported as final.
"""

from pathlib import Path
import sys,json,shutil,pandas as pd
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'code/weekly_moveout/src'))
from moveout.pipeline import report
for date in (sys.argv[1:] or ['20260915','20261031','20261115']):
 for kind in ['weekly','survival']:
  dirs=sorted((root/'results'/date/'folds').glob(kind+'_r*_f*'))
  # Require all five repeats by five folds before creating any combined task result.
  if len(dirs)!=25 or any(not (p/'final_comparison.json').exists() for p in dirs):raise RuntimeError(f'{date} {kind} incomplete')
  out=root/'results'/date/('combined_'+kind);out.mkdir(exist_ok=True)
  shutil.copy(dirs[0]/'run_manifest.json',out/'run_manifest.json')
  pd.concat([pd.read_csv(p/'metrics_by_fold.csv') for p in dirs],ignore_index=True).to_csv(out/'metrics_by_fold.csv',index=False)
  # Stack held-out predictions without averaging so downstream paired analyses retain fold identity.
  pd.concat([pd.read_parquet(p/f'{kind}_predictions.parquet') for p in dirs],ignore_index=True).to_parquet(out/f'{kind}_predictions.parquet',index=False)
  for n in ['failures.json','selected_candidates.json']:(out/n).write_text(json.dumps([r for p in dirs for r in json.loads((p/n).read_text())],indent=2))
  report(out)
  df=pd.read_csv(out/'metrics_by_fold.csv');df=df[df.task.eq(kind)]
  groups=['model']+(['origin','horizon'] if kind=='survival' else [])
  cols=[x for x in ['brier','log_loss','roc_auc','pr_auc','tenancy_weighted_brier','predicted_rate','event_rate','n_cases','n_controls'] if x in df]
  df.groupby(groups)[cols].mean().reset_index().to_csv(out/'summary_mean_folds.csv',index=False)
  if kind=='survival':df.groupby(groups).agg(total_folds=('fold','size'),supported_auc_folds=('roc_auc','count')).reset_index().to_csv(out/'support_counts.csv',index=False)
print('All three snapshots aggregated')
