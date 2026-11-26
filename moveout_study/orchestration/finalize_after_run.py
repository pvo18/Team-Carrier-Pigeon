"""Move-out study source summary.
Purpose: Run the post-training completion pipeline after all snapshot jobs finish.
Inputs: Waits for the run marker unless resume mode is requested, then executes recovery, aggregation, validation, and reporting.
Outputs: Finally renders the DOCX to images/PDF so document layout can be visually checked.
Workflow: Each subprocess must succeed before the next stage starts, preventing incomplete deliverables.
Method note: Thread-count environment settings keep numerical runs controlled and resource usage predictable.
"""

from pathlib import Path
import subprocess,time,os,sys
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root/'code/weekly_moveout/src'),MPLBACKEND='Agg')
while '--resume' not in sys.argv and not (root/'results/run_status.json').exists():time.sleep(5)
for name in ['recover_jobs.py','aggregate_snapshots.py','matched_landmarks.py','feature_summary.py','validate_results.py','build_report.py']:
 print('Starting '+name,flush=True)
 res=subprocess.run([sys.executable,str(root/name)],env=env,cwd=root.parent)
 if res.returncode:raise SystemExit(res.returncode)
 print('Completed '+name,flush=True)
cmd=[os.environ['CODEX_PRIMARY_RUNTIME_PYTHON'],'/home/agent/.codex/skills/builtins/documents/render_docx.py',str(root/'deliverables/Willis_Crosby_Revised_Study_2026-11-18.docx'),'--output_dir',str(root/'rendered'),'--emit_pdf']
res=subprocess.run(cmd,env=env,cwd=root.parent);raise SystemExit(res.returncode)
