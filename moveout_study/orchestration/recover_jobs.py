"""Move-out study source summary.
Purpose: Inspect every fold job and recover missing reporting or failed/incomplete training artifacts.
Inputs: A job with complete metrics, predictions, and an empty failure file receives reporting only; all others are retried.
Outputs: Writes recovery evidence, retry logs, and per-job status records while running a bounded worker pool.
Workflow: Existing incomplete job directories are removed only after the exact retry target has been resolved.
Method note: Successful jobs are left untouched.
"""

from pathlib import Path
import json,sys,concurrent.futures,subprocess,os,time,shutil
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'code/weekly_moveout/src'))
from moveout.pipeline import report
# Separate reporting-only recovery from jobs that genuinely require model retraining.
retry=[];recovered=[]
for d in ['20260915','20261031','20261115']:
 for k in ['weekly','survival']:
  for r in range(5):
   for f in range(5):
    name=f'{k}_r{r}_f{f}';p=root/'results'/d/'folds'/name
    # Treat a job as trained only when metrics, predictions, and an empty failure record all exist.
    if (p/'metrics_by_fold.csv').exists() and (p/f'{k}_predictions.parquet').exists() and (p/'failures.json').exists() and not json.loads((p/'failures.json').read_text()):
     if not (p/'final_comparison.json').exists():
      report(p);recovered.append(dict(snapshot=d,job=name,action='completed_reporting_only'))
    else:retry.append((d,k,r,f))
(root/'results/report_recovery.json').write_text(json.dumps(recovered,indent=2))
print('report recoveries',len(recovered),'training retries',len(retry),flush=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root/'code/weekly_moveout/src'),MPLBACKEND='Agg')
def run(job):
    """Execute one isolated batch job and persist its log and machine-readable completion status.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    d,k,r,f=job;name=f'{k}_r{r}_f{f}';out=root/'results'/d;p=out/'folds'/name
    (out/'workdirs'/name).mkdir(parents=True,exist_ok=True)
    # Remove only the resolved incomplete job directory immediately before its controlled retry.
    if p.exists():shutil.rmtree(p)
    cmd=[sys.executable,'-m','moveout.cli','run-'+k,'--prepared',str(out/'prepared'),'--config',str(root/'configs'/f'{d}.yaml'),'--repeat',str(r),'--fold',str(f),'--out',str(p)]
    start=time.time()
    with (out/f'{name}.retry.log').open('w') as log:res=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=out/'workdirs'/name)
    status=dict(snapshot=d,job=name,returncode=res.returncode,seconds=time.time()-start)
    (out/f'{name}.status.json').write_text(json.dumps(status));print(status,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(run,retry))
