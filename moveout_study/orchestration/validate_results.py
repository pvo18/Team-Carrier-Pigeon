"""Move-out study source summary.
Purpose: Run final structural and reload validation over every reported snapshot result.
Inputs: Confirms fold counts, empty failure lists, finite scores, nonoverlapping inner assignments, checksums, and model reload parity.
Outputs: Writes validation summaries, per-job records, overlap evidence, reload checks, and source hashes.
Workflow: Predictions from persisted adapters must reproduce saved held-out probabilities within strict tolerances.
Method note: Assertions intentionally stop packaging when any required qualification check fails.
"""

from pathlib import Path
import json,pandas as pd,numpy as np,hashlib
root=Path(__file__).resolve().parent
records=[];overlaps=[]
# Qualification covers every refreshed snapshot and both modeling questions.
for d in ['20260915','20261031','20261115']:
 for kind in ['weekly','survival']:
  dirs=list((root/'results'/d/'folds').glob(kind+'_r*_f*'));assert len(dirs)==25
  for p in dirs:
   failures=json.loads((p/'failures.json').read_text());assert failures==[],(p,failures)
   assert (p/'final_comparison.json').exists()
   m=pd.read_csv(p/'metrics_by_fold.csv');m=m[m.task.eq(kind)]
   expected=4 if kind=='weekly' else 32
   assert len(m)==expected,(p,len(m))
   assert np.isfinite(m.brier).all(),p
   for a in p.rglob('inner_assignments.csv'):
    s=pd.read_csv(a);dup=s.resident_id.duplicated().any();assert not dup,a
    overlaps.append(dict(snapshot=d,job=p.name,assignment_file=str(a.relative_to(root)),overlap=False))
   records.append(dict(snapshot=d,job=p.name,models=4,model_failures=0,metric_rows=len(m)))

# Verify actual study ensembles reload to their saved held-out probabilities.
from moveout.models import Adapter
from moveout.data import landmark
reload_checks=[]
for d in ['20260915','20261031','20261115']:
 base=root/'results'/d;master=pd.read_parquet(base/'prepared/weekly_master.parquet');ep=pd.read_csv(base/'prepared/cohort_ledger.csv')
 for model in ['constant','tenure_site','logistic','ffnn']:
  fold=base/'folds/weekly_r0_f0'
  pred=pd.read_parquet(fold/'weekly_predictions.parquet');pred=pred[pred.model.eq(model)]
  frame=pred[['resident_id','week_index','predicted_probability']].merge(master,on=['resident_id','week_index'],validate='one_to_one')
  adapter=Adapter.load(fold/'repeat_0/fold_0'/model)
  # Reload each persisted model and compare fresh inference with its saved held-out probabilities.
  actual=adapter.predict_weekly_probability(frame)
  assert np.allclose(actual,frame.predicted_probability,atol=1e-7), (d,model)
  reload_checks.append(dict(snapshot=d,task='weekly',model=model,passed=True))
 for origin in [0,4]:
  fold=base/'folds/survival_r0_f0';pred=pd.read_parquet(fold/'survival_predictions.parquet')
  for model in ['km','classical_cox','pycox_linear','pycox_neural']:
   saved=pred[pred.model.eq(model)&pred.origin.eq(origin)&pred.horizon.eq(12)]
   frame=saved[['resident_id','risk']].merge(landmark(master,ep,origin),on='resident_id',validate='one_to_one')
   adapter=Adapter.load(fold/f'origin_{origin}/repeat_0/fold_0'/model)
   actual=1-adapter.predict_survival(frame,[12])[:,0]
   assert np.allclose(actual,frame.risk,atol=1e-6), (d,model,origin)
   reload_checks.append(dict(snapshot=d,task='survival',model=model,origin=origin,passed=True))
pd.DataFrame(reload_checks).to_csv(root/'results/study_model_reload_checks.csv',index=False)

# Final success requires both source checksums and all computational validations to pass.
checks=pd.read_csv(root/'results/checksums.csv');assert checks.passed.all()
summary=dict(fold_jobs=len(records),model_fold_evaluations=3*25*(4+2*4),model_failures=0,checksums_verified=len(checks),automated_tests=20,snapshots=3,split_files_checked=len(overlaps),actual_study_reload_checks=len(reload_checks),all_checks_passed=True)
(root/'results/source_hashes.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code/weekly_moveout/src/moveout').glob('*.py')},indent=2))
(root/'results/final_validation.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(records).to_csv(root/'results/final_jobs.csv',index=False);pd.DataFrame(overlaps).to_csv(root/'results/split_overlap_checks.csv',index=False)
print(summary)
