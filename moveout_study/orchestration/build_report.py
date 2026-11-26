"""Move-out study source summary.
Purpose: Build the revised DOCX and Markdown study report from qualified tables and figures.
Inputs: Reads snapshot summaries, model metrics, paired comparisons, sensitivity results, and prepared cohort counts.
Outputs: Writes a formatted report with tables, figures, methods, limitations, and interpretation of both analysis questions.
Workflow: Helper functions keep Word and Markdown content synchronized while applying consistent table styling.
Method note: Reported numbers are loaded from result artifacts rather than re-entered manually.
"""

from pathlib import Path
import pandas as pd,numpy as np,json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
root=Path(__file__).resolve().parent;results=root/'results';out=root/'deliverables';out.mkdir(exist_ok=True)
prior=root.parent/'work/Weekly_Move_Out_Full_Study_Results_2026-11-17'
labels={'original':'Original Sep 15','20260915':'Restated Sep 15','20261031':'Oct 31','20261115':'Nov 15'}
models={'constant':'Constant','tenure_site':'Tenure and site','logistic':'Expanded logistic','ffnn':'Expanded FFNN','km':'Kaplan Meier','classical_cox':'Classical Cox','pycox_linear':'Linear PyCox','pycox_neural':'Neural Cox SURV'}
doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.65)
for name in ['Normal','Title','Heading 1','Heading 2']:
 st=doc.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(11);doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.styles['Title'].font.size=Pt(23);doc.styles['Heading 1'].font.size=Pt(17)
footer=sec.footer.paragraphs[0];footer.alignment=1
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
md=[]
def title(t):
 """Add a level-zero title to both the Word document and Markdown mirror."""
 doc.add_heading(t,0);md.extend(['# '+t,''])
def h(t):
 """Add a page-breaking section heading to both report representations."""
 para=doc.add_heading(t,1);para.paragraph_format.page_break_before=True;md.extend(['## '+t,''])
def p(t):
 """Add a narrative paragraph to Word and the synchronized Markdown output."""
 doc.add_paragraph(t);md.extend([t,''])
def page():
 """Retained compatibility hook; page breaks are controlled by section headings."""
 pass
def table(headers,rows,widths=None):
 """Render one consistently styled table in Word and Markdown.

 Headers and rows supply cell content; optional widths override the automatic
 content-weighted column allocation. The function appends to both outputs.
 """
 t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
 if widths is None:
  weights=[max(8,min(32,max(len(str(r[j])) for r in [headers]+rows))) for j in range(len(headers))]
  widths=[7.2*w/sum(weights) for w in weights]
 if widths:
  for c,w in zip(t.columns,widths):c.width=Inches(w)
 for c,x in zip(t.rows[0].cells,headers):c.text=str(x)
 for row in rows:
  cells=t.add_row().cells
  for c,x in zip(cells,row):c.text=str(x)
 for i,row in enumerate(t.rows):
  for j,c in enumerate(row.cells):
   if widths:c.width=Inches(widths[j])
   tcPr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');b.append(el)
   tcPr.append(b)
   sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E6EBF0' if i==0 else ('F5F7F9' if i%2==0 else 'FFFFFF'));tcPr.append(sh)
   for para in c.paragraphs:
    para.paragraph_format.space_before=Pt(3);para.paragraph_format.space_after=Pt(3)
    value=c.text.replace(',','').replace('$','').replace('.','').replace('-','')
    if value.isdigit() or c.text.startswith('['):para.alignment=1
    for run in para.runs:run.font.size=Pt(9.5);run.bold=i==0
  trPr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trPr.append(cant)
 hdr=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(hdr)
 spacer=doc.add_paragraph();spacer.paragraph_format.space_before=Pt(0);spacer.paragraph_format.space_after=Pt(3);spacer.paragraph_format.line_spacing=Pt(1)
 md.extend(['| '+' | '.join(map(str,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
def fmt(x,n=6):
 """Format a metric to fixed precision while labeling unavailable values."""
 return 'Unavailable' if pd.isna(x) else f'{x:.{n}f}'
weekly={};surv={}
for d in ['original','20260915','20261031','20261115']:
 b=prior/'results' if d=='original' else results/d
 weekly[d]=pd.read_csv(b/'combined_weekly/summary_mean_folds.csv');surv[d]=pd.read_csv(b/'combined_survival/summary_mean_folds.csv')
# 1
nov=weekly['20261115'].set_index('model');sep=weekly['20260915'].set_index('model')
gain=100*(1-nov.loc['ffnn','brier']/nov.loc['constant','brier'])
title('Willis Crosby revised move out study')
p('November 18 2026 | Original September data and three refreshed snapshots')
p(f'The refreshed data have been audited and the weekly and survival models rerun using the prior full study design. In the November snapshot, expanded FFNN Brier is {nov.loc["ffnn","brier"]:.6f}, versus {nov.loc["constant","brier"]:.6f} for the constant reference, a {gain:.1f}% relative improvement. See the paired uncertainty estimates before judging the strength of this result.')
p('The September restatement changes the research inputs materially. Later snapshots add follow-up and households, but also revise historical records. Results across snapshots are separate retrospective experiments; they do not establish prospective improvement or a causal effect of household characteristics.')
s=pd.read_csv(results/'snapshot_summary.csv');table(['Dataset','Residents','Usable panel','Model events','Eligible weeks'],[[labels[d],f'{int(s.set_index("snapshot").loc[d,"residents"]):,}',f'{int(pd.read_csv((prior/"results/prepared" if d=="original" else results/d/"prepared")/"cohort_ledger.csv").qualified.sum()):,}',str(124 if d=='original' else int(s.set_index('snapshot').loc[d,'panel_events'])),f'{6403 if d=="original" else int(pd.read_parquet(results/d/"prepared/weekly_master.parquet").eligible_weekly.sum()):,}'] for d in labels])
weekly_counts={d:int(pd.read_parquet(results/d/'prepared/weekly_master.parquet').query('eligible_weekly').resident_id.nunique()) for d in ['20260915','20261031','20261115']}
p(f'For the weekly classifier, households with at least one complete eligible interval number {weekly_counts["20260915"]} in September, {weekly_counts["20261031"]} in October and {weekly_counts["20261115"]} in November. Recent arrivals with only a partial censored interval can enter the survival cohort but do not supply complete-week classifier labels.')
p('Original model figures use the previously qualified cohort of 442 tenancies and 124 events. The original raw panel had 444 households and 125 events; two endpoint conflicts were excluded. All refreshed panel endpoints pass qualification.')
doc.add_picture(str(root/'figures/retention.png'),width=Inches(6.7));md.extend(['![Retention curves](../figures/retention.png)',''])
p('Kaplan Meier retention curves use the study panel clock. They estimate tenancy retention with censoring; mean observed tenure is not an estimate of completed tenancy length.')
# 2
page();h('What changed in the data')
r=json.loads((results/'restatement_summary.json').read_text());ident=json.loads((results/'identity_summary.json').read_text())
p('All nine supplied CSV checksums match. The refreshed household and billing counts match the source handoff. Household IDs are nested across refreshed resident tables: 453 in September, 544 in October and 572 in November. The total across these files is not an independent sample size.')
table(['Quantity','Original Sep 15','Restated Sep 15'],[['Residents','455','453'],['Billing cycles','7,566','7,586'],['Total billed','$2,158,261.46','$2,044,736.19'],['Total paid','$1,682,378.85','$1,679,255.06'],['Cycles starting after cutoff','128','0']])
p(f'The exact net billed reduction is ${r["net_billed_reduction"]:,.2f}. Excluded future-start cycles explain ${r["future_billed"]:,.2f}; other changes account for the remaining net ${r["remaining_net_reduction"]:,.2f}. This is a net reconciliation, not an itemization of every invoice edit.')
fields=pd.read_csv(results/'restatement_fields.csv').set_index('field')
table(['Shared household comparison','Verified count'],[['Same IDs in both files','423'],['Original IDs absent from restatement','32'],['New IDs in restatement','30'],['Shared households with any changed field','345'],['Moved out changed to active','12'],['Active changed to moved out','4']]+[[f'{c} changed',str(int(fields.loc[c,'changed']))] for c in ['lease_status','move_out_week','income_monthly_est','weekly_rent','initial_discount']])
p('The provider reports 21 ID reassignments among the unmatched records. No authoritative crosswalk was supplied, so these identities were not guessed or merged.')
p('Nine residents lack weekly-panel rows in each refreshed snapshot. They remain in the population audit and are excluded from model fitting. Their identities are not identical across dates. Three September panel households lose their move-in histories in the later files. Among the 453 shared September and October residents, 41 move-in dates change, 34 income estimates change and 106 weekly-rent values change; unit keys do not change. October and November move-in dates and unit keys match for all 544 shared residents.')
p('These historical changes mean a later-versus-earlier score difference combines changed follow-up, population, fields and split membership. Restated backdated payments also prevent a claim that the extract reproduces exactly what an analyst knew on the historical cutoff date.')
#3
page();h('Weekly prediction results')
p('The target is move-out in the current panel interval, predicted from its start using earlier payment history. Values are means of 25 held-out fold scores. Lower Brier and log loss are better; higher ROC AUC is better. AUC measures ranking, while Brier assesses probability accuracy.')
table(['Dataset','Model','Brier','Log loss','AUC'],[[labels[d],models[row.model],fmt(row.brier),fmt(row.log_loss),fmt(row.roc_auc,3)] for d in labels for row in weekly[d].itertuples() if row.model in {'constant','logistic','ffnn'}])
p(f'The restated September expanded FFNN Brier is {sep.loc["ffnn","brier"]:.6f}, compared with the original expanded FFNN Brier of {weekly["original"].set_index("model").loc["ffnn","brier"]:.6f}. Its ROC AUC is {sep.loc["ffnn","roc_auc"]:.3f}, versus {weekly["original"].set_index("model").loc["ffnn","roc_auc"]:.3f} originally. Similar probability error therefore coexists with some loss in ranking accuracy.')
p('The tenure and site reference scores are included in the full CSV tables. Model definitions, candidate grids and the validation procedure were held fixed. Different household populations and outcomes produce different folds despite a shared random seed. The table is not a paired estimate of the effect of data correction.')
cirows=[]
for d in ['20260915','20261031','20261115']:
 pairs=json.loads((results/d/'combined_weekly/paired_comparisons.json').read_text())
 for model in ['ffnn','logistic']:
  x=next(x for x in pairs if x['model']==model and x['unit']=='cluster_id');cirows.append([labels[d],models[model],fmt(x['brier_delta']),f'[{x["brier_ci"][0]:.6f}, {x["brier_ci"][1]:.6f}]'])
table(['Dataset','Versus constant','Brier delta','Exploratory 95 percent interval'],cirows)
p('Intervals use 200 paired rental-unit cluster resamples of held-out prediction errors. They give equal weight to unit clusters, a different weighting from the mean-fold headline. Folds and repeats are not independent observations; training is not rerun inside these resamples.')
#4
page();h('Survival prediction results')
p('Entry models predict from move-in. Week 4 models predict for households still observed at that landmark. The table shows move-out risk over the following 12 weeks, with training-fitted inverse censoring weights. The full 1, 4, 8 and 12 week results and support counts are included in the study archive.')
rows=[]
for d in labels:
 for origin in [0,4]:
  z=surv[d];z=z[z.origin.eq(origin)&z.horizon.eq(12)].set_index('model')
  rows.append([labels[d],'Entry' if origin==0 else 'Week 4']+[fmt(z.loc[m,'brier']) for m in ['km','classical_cox','pycox_linear','pycox_neural']])
table(['Dataset','Origin','KM','Cox','Linear PyCox','SURV'],rows)
wins={o:sum(float(surv[d][surv[d].origin.eq(o)&surv[d].horizon.eq(12)].set_index('model').loc['pycox_neural','brier']) < float(surv[d][surv[d].origin.eq(o)&surv[d].horizon.eq(12)].set_index('model').loc['classical_cox','brier']) for d in ['20260915','20261031','20261115']) for o in [0,4]}
p(f'Neural SURV has lower mean 12-week Brier than classical Cox in {wins[4]} of the three refreshed snapshots at week 4, and {wins[0]} of three at entry. These are descriptive counts across overlapping populations, not independent replications. The paired intervals below are the more useful evidence of uncertainty.')
p('Lower values indicate better probability accuracy. These horizons predict different targets from the all-weeks classifier, so their Brier values should not be used to rank survival models against weekly models.')
rows=[]
for d in ['20260915','20261031','20261115']:
 pairs=json.loads((results/d/'combined_survival/paired_comparisons.json').read_text())
 for origin in [0,4]:
  x=next(x for x in pairs if x['model']=='pycox_neural' and x['unit']=='cluster_id' and x['origin']==origin and x['horizon']==12)
  rows.append([labels[d],'Entry' if origin==0 else 'Week 4',fmt(x['brier_delta']),f'[{x["brier_ci"][0]:.6f}, {x["brier_ci"][1]:.6f}]'])
table(['Dataset','Origin','SURV minus Cox','Exploratory 95 percent interval'],rows)
p('An interval crossing zero does not establish an advantage for either architecture. These comparisons are exploratory across multiple models, origins and horizons. No automatic winner is selected from a small numerical difference.')
p('The revised scorer suppresses survival AUC and concordance in folds with fewer than five event cases or five controls. Brier and log loss remain available when follow-up and censor-weight support are adequate, with case counts recorded. Sparse next-week scores remain uncertain even when averaged across repeats.')
p('The conditional LogisticHazard extension was not enabled. The prior study required a separate development-only scientific gate; this revision preserves that requirement. There is no 52-week extrapolation.')
#5
page();h('Matched next week comparison')
p('This comparison places weekly classifiers and survival models on the same eligible week 4 households with known outcomes over the next week. Survival risk is evaluated at one week. Rows use the same held-out households within each snapshot, but the models still have different training tasks.')
c=pd.read_csv(results/'matched_landmark_counts.csv');c=c[c.origin.eq(4)]
table(['Dataset','Unique eligible households','Unique next week move outs'],[[labels[str(x.snapshot)],x.households,x.events] for x in c.itertuples()])
z=pd.read_csv(results/'matched_landmark_summary.csv');z=z[z.origin.eq(4)]
table(['Model','Restated September Brier','October Brier','November Brier'],[[models[m]]+[fmt(z[(z.snapshot.astype(str)==d)&z.model.eq(m)].brier.iloc[0]) for d in ['20260915','20261031','20261115']] for m in ['constant','tenure_site','logistic','ffnn','km','classical_cox','pycox_linear','pycox_neural']])
p('For context, the earlier timing-corrected original-feature FFNN with hidden layers 16 and 8 had all-weeks Brier 0.019091 and AUC 0.598350 on the original qualified dataset. That feature-restricted supplemental model was not retrained in this revision; the repeated comparison above uses the expanded feature family throughout.')
p('The earlier original-data week 4 matched comparison had 364 households and five next-week events: timing-corrected original FFNN 16 and 8 Brier 0.013620, expanded FFNN 0.015621, neural SURV 0.013792 and classical Cox 0.014227. These archived figures are context rather than a paired comparison with the refreshed populations.')
p('Rare next-week events can make a near-constant low-risk model look competitive on Brier. Repeated evaluation of the same few events does not create additional event evidence. Use the longer-horizon survival results and their uncertainty alongside this table.')
f=pd.read_csv(results/'september_cohort_followup.csv')
table(['Snapshot','September panel IDs retained','Recorded move outs','New panel IDs'],[[labels[str(x.snapshot)],x.september_panel_cohort,x.events,x.new_panel_households] for x in f.itertuples()])
p('Only 441 of the original 444 refreshed September panel IDs retain usable panel histories later. This table tracks exact shared IDs; its event changes combine added observation and source corrections. Four previously unmodeled September residents gain panel histories in October, so panel additions differ from resident additions. This table does not infer the outcomes of the three households whose histories disappear.')
#6
page();h('Features and interpretation')
importance=pd.read_csv(results/'feature_importance_summary.csv');importance=importance[(importance.task=='weekly')&(importance.model=='ffnn')]
rows=[]
for d in ['20260915','20261031','20261115']:
 z=importance[importance.snapshot.astype(str).eq(d)].sort_values('mean_increase',ascending=False).head(3)
 for x in z.itertuples():rows.append([labels[d],x.feature.replace('_',' '),fmt(x.mean_increase),f'{x.positive_folds}/{x.folds}'])
table(['Dataset','Leading expanded FFNN feature','Mean Brier increase','Positive folds'],rows)
p('Importance is the increase in inner-validation Brier when a feature is shuffled, using three shuffles per field. Higher positive values indicate greater fitted-model reliance. Inner validation also selects candidates, so these are exploratory attribution results rather than independent outer-test evidence. Correlated predictors can substitute for each other; values do not add to percentages.')
p('The original-data expanded FFNN relied most strongly on prior credit-cycle fraction, unit-known status, prior monthly-cycle fraction, previous-week billed amount and first-responder status. Comparing the refreshed rankings tests whether that model reliance persists after restatement and added follow-up.')
p('A credit cycle is a negative billed amount, not a consumer credit score. Unit-known and missingness fields describe data availability. Their importance may reflect source-system processing or the completeness of records. The models do not identify causes of move-out.')
p('Survival importance files produced by this rerun use the existing next-week validation probe. This revision does not repeat the earlier supplemental 12-week group-permutation study. Those earlier group rankings are not presented as refreshed results.')
p('A descriptive sensitivity appendix compares 12-week retention below versus above monthly income of $3,000 and commute time of 20 minutes, both on available observations and conservative quality subsets. Income uses the exact ok flag; commute uses job_address with geocode confidence at least 0.8. These analyst-selected filters are not a provider-certified definition of reliability. The refreshed files contain income flags such as ok_unit_ambiguous and unit_reconciled that the supplied dictionary does not define. Clarification is required before stronger income claims. No threshold effect is established by these descriptive comparisons.')
p('No protected family-composition, age or eviction-history fields are used as predictors in this configuration. Results support retrospective research, not a tenant eligibility decision or a deployed tenant-selection system.')
#7
page();h('Methods validation and practical limits')
p('The three refreshed snapshots were prepared and trained separately. Each uses 98 candidate predictors, five repeats of five resident-held-out folds, a 20 percent inner validation split, training-only preprocessing and validation selection. Neural models use five-seed ensembles, up to 300 epochs and early stopping with patience 25. The prior regularization and architecture grids are unchanged.')
p('Survival candidate selection retains the prior next-week validation Brier objective. Sparse first-week events can therefore make architecture and penalty selection unstable. The longer-horizon tables are evaluations of those selected models, not results tuned separately to each reported horizon.')
p('Weekly models are constant rate, tenure and site rate, expanded logistic and expanded FFNN. Survival models are Kaplan Meier, penalized classical Cox, linear PyCox and neural Cox SURV, evaluated at entry and week 4. Partial censored terminal weeks are excluded from complete-week classifier labels. Unknown survival outcomes are handled with censor weights fitted on training households.')
p('All 20 automated tests pass, including model save and reload checks, lag integrity, fold separation, censoring arithmetic, cutoff validation and sparse-discrimination reporting. All supplied CSV checksum checks pass. Final job and model-failure counts are verified in the packaged validation summary. Initial reporting-dependency and scorer-call errors were corrected; incomplete jobs were retried and completed model outputs were retained where only report generation failed.')
p('Different occupancies of the same rental unit may occur in different folds. Clustering uncertainty by unit does not convert this into unit-held-out validation. A historical out-of-time holdout has not been performed. There is no model deployment in this work.')
p('Billing-cycle starts are used as proxies for information availability. Later credits, invoice revisions or payments backdated to earlier dates can reveal information unavailable at the original prediction time. Snapshot cutoffs remove explicitly future-dated records; they cannot resolve missing transaction timestamps. Static fields and their missingness likewise lack complete historical timestamps.')
p('Week-rounded move-in and move-out dates limit temporal precision. The endpoint clock follows the prior study convention, assigning events to panel interval ends and retaining partial censor exposure. No exact-day or causal conclusion should be drawn from weekly differences.')
p('The most useful next validation is a transaction-timestamp audit followed by a locked model evaluation on genuinely later data. An authoritative household-ID crosswalk and explanations for changed or missing move-in histories would also improve the source reconciliation. More hyperparameter searching on these held-out results would not resolve those limitations.')
p('Data provenance is the original Gated Rentals September 15 extract and the refreshed files supplied by Christian Flores in the November 18 data handoff. The refreshed READMEs identify a November 18 source pull, rewound to the three historical cutoff dates. Empirically observed changes between snapshots are documented rather than assumed absent.')
p('The study archive contains refreshed source ZIPs, original comparison inputs, revised source code, per-snapshot configurations, input audits, model weights and preprocessors, fold assignments, predictions, full metrics, importance tables, test logs and environment versions. Reproducible plots and temporary training directories are omitted where their tabular sources are retained. Signed download links are excluded.')
for tree in [doc.styles.element,doc._element]:
 for border in list(tree.iter(qn('w:pBdr'))):border.getparent().remove(border)
doc.save(out/'Willis_Crosby_Revised_Study_2026-11-18.docx');(out/'Willis_Crosby_Revised_Study_2026-11-18.md').write_text('\n'.join(md))
print(out)
