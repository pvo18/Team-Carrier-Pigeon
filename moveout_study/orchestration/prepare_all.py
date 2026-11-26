"""Move-out study source summary.
Purpose: Prepare the three refreshed snapshots with identical feature and endpoint rules.
Inputs: Builds a dated configuration from the expanded baseline, adds strict expected counts, and enforces each snapshot cutoff.
Outputs: Calls the shared preparation pipeline to write master panels, endpoint ledgers, folds, audits, and manifests.
Workflow: All snapshots use the same code path so differences reflect data/follow-up rather than preparation drift.
Method note: Provider files remain read-only.
"""

from pathlib import Path
import sys,yaml,pandas as pd
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'code/weekly_moveout/src'))
from moveout.data import prepare
from moveout.config import load_config
(root/'configs').mkdir(exist_ok=True)
for date in ['20260915','20261031','20261115']:
 p=root.parent/'snapshots'/f'willis-crosby-resident-data-{date}'
 w=pd.read_csv(p/'weekly_panel.csv')
 cfg=load_config(str(root/'code/weekly_moveout/configs/expanded.yaml'))
 cfg.update(censor_date=f'{date[:4]}-{date[4:6]}-{date[6:]}',expected_households=int(w.resident_id.nunique()),expected_rows=len(w),expected_events=int(w.churned_this_week.sum()),require_snapshot_cutoff=True)
 (root/'configs'/f'{date}.yaml').write_text(yaml.safe_dump(cfg))
 print(date,prepare(p,root/'results'/date/'prepared',cfg),flush=True)
