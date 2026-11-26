"""Move-out study source summary.
Purpose: Recover the small set of early September jobs interrupted during the original parallel run.
Inputs: Completes reporting when training artifacts are intact and reruns only the identified survival folds.
Outputs: Writes retry logs and status JSON records for every recovered job.
Workflow: The narrowly scoped recovery avoids repeating successful work.
Method note: Controlled thread settings reduce contention during concurrent retries.
"""

from pathlib import Path
import sys,os,json,subprocess,concurrent.futures,time,shutil
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'code/weekly_moveout/src'))
from moveout.pipeline import report
for p in (root/'results/20260915/folds').glob('weekly_r0_f*'):
 if not (p/'final_comparison.json').exists():report(p)
p=root/'results/20260915/folds/weekly_r1_f0'
if not (p/'final_comparison.json').exists():report(p)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root/'code/weekly_moveout/src'),MPLBACKEND='Agg')
def run(f):
    """Execute one isolated batch job and persist its log and machine-readable completion status.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    name=f'survival_r0_f{f}';out=root/'results/20260915';p=out/'folds'/name
    shutil.rmtree(p)
    cmd=[sys.executable,'-m','moveout.cli','run-survival','--prepared',str(out/'prepared'),'--config',str(root/'configs/20260915.yaml'),'--repeat','0','--fold',str(f),'--out',str(p)]
    start=time.time()
    with (out/f'{name}.retry.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=out/'workdirs'/name)
    s=dict(snapshot='20260915',job=name,returncode=r.returncode,seconds=time.time()-start,retried=True)
    (out/f'{name}.status.json').write_text(json.dumps(s));print(s,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(run,range(5)))
