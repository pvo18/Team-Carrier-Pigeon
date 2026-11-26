"""Move-out study source summary.
Purpose: Dispatch the full repeated cross-validation experiment for all snapshots and both analysis tasks.
Inputs: Creates 150 isolated jobs: three snapshots by two tasks by five repeats by five folds.
Outputs: Each subprocess writes its own log, output directory, and status JSON before the overall run marker is saved.
Workflow: A bounded thread pool launches independent processes while numerical libraries remain single-threaded.
Method note: The script records failures rather than silently treating incomplete jobs as results.
"""

import concurrent.futures,subprocess,os,pathlib,json,time,sys
root=pathlib.Path(__file__).resolve().parent
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root/'code/weekly_moveout/src'),MPLBACKEND='Agg')
def run(job):
    """Execute one isolated batch job and persist its log and machine-readable completion status.

    Inputs are supplied by the named parameters and are not mutated unless the implementation explicitly writes a copy.
    Returns: The value described by the function signature, or no value when the function writes artifacts/asserts behavior.
    """
    date,kind,rep,fold=job;name=f'{kind}_r{rep}_f{fold}';out=root/'results'/date
    work=out/'workdirs'/name;work.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','moveout.cli','run-'+kind,'--prepared',str(out/'prepared'),'--config',str(root/'configs'/f'{date}.yaml'),'--repeat',str(rep),'--fold',str(fold),'--out',str(out/'folds'/name)]
    start=time.time()
    with (out/f'{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=work)
    result=dict(snapshot=date,job=name,returncode=r.returncode,seconds=time.time()-start)
    (out/f'{name}.status.json').write_text(json.dumps(result));print(json.dumps(result),flush=True);return result
# Materialize the complete factorial job list before launching workers so expected coverage is explicit.
jobs=[(d,k,r,f) for d in ['20260915','20261031','20261115'] for r in range(5) for f in range(5) for k in ['weekly','survival']]
# Bound concurrency while each numerical subprocess remains single-threaded.
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:results=list(ex.map(run,jobs))
(root/'results/run_status.json').write_text(json.dumps(results,indent=2))
