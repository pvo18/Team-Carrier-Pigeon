"""Move-out study source summary.
Purpose: Audit source snapshots for integrity, growth, restatements, and cutoff compliance before modeling.
Inputs: Reads resident, weekly-panel, and invoice files from the original and three refreshed snapshots.
Outputs: Writes checksum results, population/count summaries, unmatched residents, and field-level restatement counts.
Workflow: The audit distinguishes new follow-up from revisions to earlier records so comparisons are interpreted correctly.
Method note: Only descriptive evidence is produced; this script does not change provider data.
"""

from pathlib import Path
import pandas as pd,json,hashlib
root=Path(__file__).resolve().parent
base=root.parent
sources={'original':base/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/code/weekly_moveout/data/provider'}
for date in ['20260915','20261031','20261115']:sources[date]=base/'snapshots'/f'willis-crosby-resident-data-{date}'
rows=[]; data={};checks=[]
for label,p in sources.items():
 r,w,i=[pd.read_csv(p/f'{f}.csv') for f in ['residents','weekly_panel','invoice_cycles']];data[label]=(r,w,i)
 cutoff=pd.Timestamp('2026-09-15' if label=='original' else f'{label[:4]}-{label[4:6]}-{label[6:]}')
 if (p/'SHA256SUMS').exists():
  for line in (p/'SHA256SUMS').read_text().splitlines():
   h,f=line.split(maxsplit=1); f=f.lstrip('*');q=p/f
   checks.append(dict(snapshot=label,file=f,passed=hashlib.sha256(q.read_bytes()).hexdigest()==h))
 rows.append(dict(snapshot=label,residents=len(r),panel_households=w.resident_id.nunique(),panel_rows=len(w),panel_events=int(w.churned_this_week.sum()),invoices=len(i),billed=i.amount_billed.sum(),paid=i.amount_paid.sum(),future_cycles=int((pd.to_datetime(i.billing_start_week)>cutoff).sum()),future_billed=i.loc[pd.to_datetime(i.billing_start_week)>cutoff,'amount_billed'].sum(),missing_panel=len(set(r.resident_id)-set(w.resident_id))))
 r[~r.resident_id.isin(w.resident_id)].to_csv(root/'results'/f'{label}_residents_without_panel.csv',index=False)
pd.DataFrame(rows).to_csv(root/'results/snapshot_summary.csv',index=False)
pd.DataFrame(checks).to_csv(root/'results/checksums.csv',index=False)
a,b=data['original'][0].set_index('resident_id'),data['20260915'][0].set_index('resident_id');shared=a.index.intersection(b.index)
changes=[]
for c in a.columns.intersection(b.columns):
 x=a.loc[shared,c].fillna('<NA>').astype(str);y=b.loc[shared,c].fillna('<NA>').astype(str)
 changes.append(dict(field=c,changed=int((x!=y).sum())))
pd.DataFrame(changes).to_csv(root/'results/restatement_fields.csv',index=False)
identity=dict(shared=len(shared),original_only=len(a.index.difference(b.index)),restated_only=len(b.index.difference(a.index)))
(root/'results/identity_summary.json').write_text(json.dumps(identity,indent=2))
for prev,nxt in [('20260915','20261031'),('20261031','20261115')]:
 a,b=data[prev][0],data[nxt][0]
 assert set(a.resident_id)<=set(b.resident_id)
print(pd.DataFrame(rows).to_string(index=False));print(identity)
