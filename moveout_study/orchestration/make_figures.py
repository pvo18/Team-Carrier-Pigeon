"""Move-out study source summary.
Purpose: Create the report's Kaplan-Meier retention figure across the original and refreshed snapshots.
Inputs: Reads qualified cohort ledgers, fits one descriptive survival curve per snapshot, and applies consistent styling.
Outputs: Writes a high-resolution PNG used by the revised study report.
Workflow: Confidence bands are intentionally hidden to keep the comparison legible; inferential uncertainty is reported elsewhere.
Method note: No model training or result selection occurs here.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
root=Path(__file__).resolve().parent;(root/'figures').mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,4.3))
colors=['#555555','#2675ab','#e59426','#29966e']
for label,color in zip(['original','20260915','20261031','20261115'],colors):
 p=root.parent/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17/results/prepared' if label=='original' else root/'results'/label/'prepared'
 e=pd.read_csv(p/'cohort_ledger.csv');e=e[e.qualified]
 km=KaplanMeierFitter().fit(e.duration,e.event,label={'original':'Original September','20260915':'Restated September','20261031':'October 31','20261115':'November 15'}[label])
 km.plot_survival_function(ax=ax,ci_show=False,color=color)
ax.set(xlabel='Weeks since move-in on panel clock',ylabel='Estimated probability of remaining',xlim=(0,40),ylim=(0,1));ax.grid(alpha=.18);ax.legend(frameon=False);fig.tight_layout();fig.savefig(root/'figures/retention.png',dpi=180);plt.close(fig)
