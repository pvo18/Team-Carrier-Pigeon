"""Move-out study source summary.
Purpose: Compare the original September extract with the restated and later provider snapshots.
Inputs: Measures identity overlap, status reversals, invoice reductions, and selected field changes among stable resident IDs.
Outputs: Writes JSON and CSV evidence used to explain why snapshot results are retrospective experiments, not a time trend.
Workflow: The script deliberately avoids inferring undocumented identity reassignments without a supplied crosswalk.
Method note: No source records are modified.
"""

from pathlib import Path
import pandas as pd,json
root=Path(__file__).resolve().parent;b=root.parent
p=b/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/code/weekly_moveout/data/provider'
r0=pd.read_csv(p/'residents.csv').set_index('resident_id');i0=pd.read_csv(p/'invoice_cycles.csv')
r1=pd.read_csv(b/'snapshots/willis-crosby-resident-data-20260915/residents.csv').set_index('resident_id');i1=pd.read_csv(b/'snapshots/willis-crosby-resident-data-20260915/invoice_cycles.csv')
shared=r0.index.intersection(r1.index)
ch=r0.loc[shared].astype(str).ne(r1.loc[shared].astype(str))
result=dict(shared=len(shared),any_field_changed=int(ch.any(axis=1).sum()),moved_out_to_active=int(((r0.loc[shared,'is_active']==0)&(r1.loc[shared,'is_active']==1)).sum()),active_to_moved_out=int(((r0.loc[shared,'is_active']==1)&(r1.loc[shared,'is_active']==0)).sum()),future_cycles=int((pd.to_datetime(i0.billing_start_week)>pd.Timestamp('2026-09-15')).sum()),future_billed=float(i0.loc[pd.to_datetime(i0.billing_start_week)>pd.Timestamp('2026-09-15'),'amount_billed'].sum()))
result['net_billed_reduction']=float(i0.amount_billed.sum()-i1.amount_billed.sum());result['remaining_net_reduction']=result['net_billed_reduction']-result['future_billed']
# Do not infer the provider's 21 identity reassignments without a supplied crosswalk.
(root/'results/restatement_summary.json').write_text(json.dumps(result,indent=2))
print(result)
# Changes within later stable resident IDs, including missing move-in histories.
changes=[]
for prev,nxt in [('20260915','20261031'),('20261031','20261115')]:
 a=pd.read_csv(b/f'snapshots/willis-crosby-resident-data-{prev}/residents.csv').set_index('resident_id');z=pd.read_csv(b/f'snapshots/willis-crosby-resident-data-{nxt}/residents.csv').set_index('resident_id');ids=a.index.intersection(z.index)
 for c in ['move_in_week','unit_key','income_monthly_est','weekly_rent','initial_discount']:
  diff=a.loc[ids,c].fillna('<NA>').astype(str).ne(z.loc[ids,c].fillna('<NA>').astype(str));changes.append(dict(prior=prev,later=nxt,field=c,changed=int(diff.sum())))
pd.DataFrame(changes).to_csv(root/'results/later_snapshot_field_changes.csv',index=False)
