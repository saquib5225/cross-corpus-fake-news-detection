"""Stage 5: documentation/derivative tables and figures from frozen artefacts only."""
from pathlib import Path
import csv, json, hashlib
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; T=R/'tables'; F=R/'figures'/'final'
F.mkdir(parents=True,exist_ok=True)
def rows(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p, fields, data):
    with open(p,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
def write(p,s): p.write_text(s.strip()+"\n",encoding='utf-8')

master=rows(T/'final_model_comparison.csv')
out=[]
for x in master:
    for label,prefix in [('ISOT','ISOT'),('FakeNewsAMT','FakeNewsAMT')]:
        out.append({'Model':x['Model'],'Dataset':label,'Accuracy':x[prefix+' Accuracy'],'Precision':x[prefix+' Precision'],'Recall':x[prefix+' Recall'],'Macro-F1':x[prefix+' Macro_F1'],'ROC-AUC':x[prefix+' ROC_AUC']})
write_csv(T/'final_results_master.csv',['Model','Dataset','Accuracy','Precision','Recall','Macro-F1','ROC-AUC'],out)
gen=[{'Model':x['Model'],'ISOT Macro-F1':x['ISOT Macro_F1'],'External Macro-F1':x['FakeNewsAMT Macro_F1'],'Generalisation Gap (pp)':str(float(x['Generalisation Gap'])*100)} for x in master]
write_csv(T/'final_generalisation_master.csv',['Model','ISOT Macro-F1','External Macro-F1','Generalisation Gap (pp)'],gen)
datasets=[
 {'Dataset':'ISOT','Role':'training/validation/test corpus','Raw records':'44898','Usable unique non-empty content':'39101','Class distribution':'23481 fake; 21417 real raw','Independence finding':'Internal splits: zero exact-content intersections'},
 {'Dataset':'WELFake','Role':'rejected external-evaluation candidate','Raw records':'72134','Usable unique non-empty content':'63673','Class distribution':'37106 fake; 35028 real','Independence finding':'All 39101 ISOT unique cleaned articles detected (100%); not independent'},
 {'Dataset':'FakeNewsAMT','Role':'external evaluation only','Raw records':'480','Usable unique non-empty content':'430','Class distribution':'190 fake; 240 legitimate/real','Independence finding':'0 exact/normalised body and title overlaps detected with ISOT'}]
write_csv(T/'dataset_master_summary.csv',list(datasets[0]),datasets)
protocol=[
 {'Area':'Data and splits','Final protocol':'ISOT title + text; cleaning and exact duplicate handling; stratified 80/10/10 train/validation/test; seed 42.'},
 {'Area':'Classical models','Final protocol':'Saved Naive Bayes, Logistic Regression and Random Forest; TF-IDF (1,2)-grams, min_df=2, max_df=0.95, max_features=50000, fitted on ISOT train only.'},
 {'Area':'RoBERTa','Final protocol':'roberta-base; max length 64; batch 2; accumulation 8; effective batch 16; fp16; no gradient checkpointing; ISOT-only fitting.'},
 {'Area':'Selection and evaluation','Final protocol':'ISOT validation selected the checkpoint/early stopping; selected frozen model evaluated on ISOT test, then FakeNewsAMT (430 usable records).'},
 {'Area':'Statistics and Stage 4','Final protocol':'Six exact two-sided paired McNemar tests with Holm correction; post-hoc analyses read frozen predictions/models only; token Integrated Gradients not completed.'}]
write_csv(T/'methodology_master_summary.csv',['Area','Final protocol'],protocol)

# Figures are visualisations of the saved master/prediction tables, not new experiments.
names=[x['Model'] for x in master]; isot=[float(x['ISOT Macro_F1']) for x in master]; ext=[float(x['FakeNewsAMT Macro_F1']) for x in master]; gaps=[(a-b)*100 for a,b in zip(isot,ext)]
plt.rcParams.update({'font.size':10,'figure.dpi':160})
def bars(vals,title,ylabel,file,ylim=None):
    fig,ax=plt.subplots(figsize=(7,4)); c=['#4C78A8','#F58518','#54A24B','#8064A2']; ax.bar(names,vals,color=c); ax.set(title=title,ylabel=ylabel); ax.grid(axis='y',alpha=.25)
    if ylim: ax.set_ylim(*ylim)
    for i,v in enumerate(vals): ax.text(i,v+(0.01 if max(vals)<=1 else 0.5),f'{v:.3f}' if max(vals)<=1 else f'{v:.1f}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(F/file,bbox_inches='tight'); plt.close(fig)
bars(isot,'In-domain performance on ISOT test','Macro-F1','01_isot_macro_f1.png',(0.9,1.01))
bars(ext,'Cross-corpus performance on FakeNewsAMT','Macro-F1','02_fakenewsamt_macro_f1.png',(0,1))
bars(gaps,'Macro-F1 generalisation gap: ISOT to FakeNewsAMT','Percentage points','03_generalisation_gap_pp.png',(0,50))
fig,ax=plt.subplots(figsize=(7,4)); ax.bar(['ISOT\nunique non-empty','FakeNewsAMT\nevaluated','WELFake\nraw'],[39101,430,72134],color=['#4C78A8','#E45756','#999999']); ax.set(title='Dataset sizes and evaluation roles',ylabel='Records'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(F/'04_dataset_distribution_comparison.png',bbox_inches='tight'); plt.close(fig)
preds={'Naive Bayes':'stage2a_naive_bayes_fakenewsamt.csv','Logistic Regression':'stage2a_logistic_regression_fakenewsamt.csv','Random Forest':'stage2a_random_forest_fakenewsamt.csv','RoBERTa':None}
fig,axs=plt.subplots(1,4,figsize=(11,3));
for ax,n in zip(axs,names):
    p=R/'roberta/predictions_fakenewsamt.csv' if n=='RoBERTa' else R/'predictions'/preds[n]
    d=rows(p); y=[int(z['true_label']) for z in d]; q=[int(z['prediction']) if 'prediction' in d[0] else int(z['predicted_label']) for z in d]
    m=np.zeros((2,2),int)
    for a,b in zip(y,q):m[a,b]+=1
    im=ax.imshow(m,cmap='Blues'); ax.set_title(n,fontsize=9); ax.set_xticks([0,1],['fake','real']); ax.set_yticks([0,1],['fake','real'])
    for i in range(2):
        for j in range(2):ax.text(j,i,str(m[i,j]),ha='center',va='center')
axs[0].set_ylabel('Actual'); fig.suptitle('FakeNewsAMT confusion matrices (rows actual; columns predicted)'); fig.tight_layout(); fig.savefig(F/'05_fakenewsamt_confusion_comparison.png',bbox_inches='tight'); plt.close(fig)
err=rows(T/'roberta_error_analysis.csv'); cats=[z['prediction_category'] for z in err]; counts=[int(float(z['char_length_count'])) for z in err]
fig,ax=plt.subplots(figsize=(7,4)); ax.bar(cats,counts,color=['#E45756','#54A24B','#4C78A8','#F58518']); ax.set(title='RoBERTa FakeNewsAMT prediction categories',ylabel='Records'); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(F/'06_roberta_external_error_patterns.png',bbox_inches='tight'); plt.close(fig)

write(R/'FINAL_RESEARCH_AUDIT.md', '''# Final Research Audit

## Status

**Complete as a synthesis of frozen Stages 1–4.** No training, tuning, prediction regeneration, or alteration of frozen experiment artefacts was performed in Stage 5.

## Research question and objectives

The study asks how classical TF-IDF classifiers and RoBERTa trained on ISOT perform in-domain and on an independently audited cross-corpus target. Objectives were to prevent leakage, compare four models, quantify generalisation gaps, test paired accuracy differences, and inspect post-hoc error patterns.

## Data, cleaning, independence and leakage

ISOT (`Fake.csv`, `True.csv`) is the sole training corpus. The final processed corpus has 39,101 unique non-empty cleaned contents after duplicate handling; its stratified 80/10/10 splits have zero exact-content intersections. Text is represented as title + text with whitespace cleaning. WELFake was audited and rejected: all 39,101 unique cleaned ISOT contents occur in it, including every train, validation and test item. Its historical predictions are retained solely as an independence finding, never as a valid external result.

FakeNewsAMT is the external evaluation cohort. The 430 usable non-empty-body records comprise 190 fake and 240 legitimate/real items; 50 fake-labelled title-only strings were excluded by a pre-specified parsing rule. Exact and normalised body/title checks detected zero ISOT overlaps. This supports absence of direct reuse under those representations, not absence of paraphrase, shared events, or source dependence.

## Methodology and results integrity

TF-IDF was fitted on ISOT train only (1–2 grams, `min_df=2`, `max_df=0.95`, 50,000 features); saved Naive Bayes, Logistic Regression and Random Forest models were externally transformed/evaluated without fitting. RoBERTa used ISOT-only fitting, ISOT validation selection, `roberta-base`, length 64, batch 2, accumulation 8, fp16, and seed 42. Epoch 1 remained selected after epoch 2 tied validation Macro-F1, triggering configured patience-one early stopping. ISOT test and FakeNewsAMT were evaluated only after selection.

RoBERTa had the highest ISOT Macro-F1 (0.999742). Naive Bayes had the highest observed FakeNewsAMT Macro-F1 (0.570970); RoBERTa obtained 0.544063. All models declined materially externally; RoBERTa's gap was 45.568 pp. Six exact paired McNemar tests with Holm correction found RoBERTa advantages on ISOT but no significant external differences. These tests concern paired accuracy, not Macro-F1 superiority.

## Explainability, reproducibility, limitations and ethics

Stage 4 provides class/error profiles, representative cases, model-error overlap and baseline feature-weight summaries from frozen outputs. Token-level Integrated Gradients did **not** complete under the managed CPU limit; no attribution claim is made. Stage 4 checked 225 frozen files: 225 unchanged, 0 modified, 0 missing. Hashes, configurations, scripts, datasets, saved models, predictions and metadata are retained; historical documentation and configuration files are flagged where stale.

Demonstrated limitations include WELFake dependence and the 430-item external cohort. Plausible limitations include ISOT age/construction bias, short excerpts, crowdsourced fake examples, label-definition/domain mismatch, residual non-text dependence, limited power, a 64-token RoBERTa input, compute-constrained design, and limited real-world generalisability. Dataset licences/provenance must be respected; do not redistribute source text beyond permitted use. The final conclusion is cross-corpus sensitivity, not that any model has universally failed or succeeded.''')

write(R/'FINAL_RESEARCH_FINDINGS.md', '''# Final Research Findings

## 1. Executive summary

This completed study compares three ISOT-trained TF-IDF baselines with ISOT-trained RoBERTa under a leakage-aware in-domain and cross-corpus design. Very high ISOT performance did not transfer to the independent FakeNewsAMT cohort.

## 2–4. Dataset findings, WELFake discovery and FakeNewsAMT selection

WELFake was initially considered, then rejected after the audit found 100% cleaned ISOT-content inclusion. This prevented an invalid external-generalisation claim. FakeNewsAMT was selected after zero detected exact/normalised body/title overlaps. It is an independent cross-corpus evaluation dataset of 430 usable short excerpts (190 fake, 240 legitimate/real), not a large-scale benchmark; its fake examples are crowdsourced and residual dependence remains possible.

## 5–8. Model findings and generalisation

ISOT Macro-F1 values were 0.958528 (Naive Bayes), 0.987631 (Logistic Regression), 0.995879 (Random Forest), and 0.999742 (RoBERTa). FakeNewsAMT Macro-F1 values were 0.570970, 0.550823, 0.540609, and 0.544063 respectively. Naive Bayes therefore had the highest observed external Macro-F1; RoBERTa's exceptionally strong ISOT result does not establish superior cross-corpus generalisation. Its 45.568-pp gap was the largest, only slightly above Random Forest's 45.527 pp.

## 9. Statistical significance

Paired exact McNemar tests (six tests, Holm correction) found significant ISOT accuracy advantages for RoBERTa versus each baseline. No FakeNewsAMT comparison was significant after correction. With 430 external records, observed rank differences require cautious interpretation.

## 10–11. Error analysis and explainability

RoBERTa's external matrix was [[140, 50], [144, 96]], giving fake recall 0.736842 and real recall 0.400000. Stage 4's descriptive profiles, model-error overlaps and feature-weight analyses support discussion of model behaviour, not causal truth assessment. Integrated Gradients was attempted but did not complete; there are no token-attribution findings.

## 12–15. Contribution, limitations and conclusion

The main contribution is methodological: a reproducible audit exposed WELFake dependence and replaced it with a documented independently screened external cohort. The evidence demonstrates that in-domain fake-news classification metrics alone are inadequate evidence of cross-corpus robustness. It does not establish real-world misinformation performance, a universal ranking, or causal explanations.''')

write(R/'DISSERTATION_CHAPTER_GUIDE.md', '''# Dissertation Chapter Guide

## 1. Introduction
Purpose: state the cross-corpus generalisation question and objectives. Use `FINAL_RESEARCH_FINDINGS.md`; do not claim deployment-ready detection.

## 2. Literature Review
Purpose: situate datasets, leakage and transformer/classical comparison. Add only independently verified citations; do not invent citations.

## 3. Research Methodology
Purpose: document ISOT-only training, split design, TF-IDF and RoBERTa protocol. Use `configs/config.yaml`, `training_configuration.json`, and `methodology_master_summary.csv`. Do not describe the stale configuration's RoBERTa length 192/batch 4 as the executed run.

## 4. Dataset Preparation and Validation
Use leakage, duplicate, shift and external-selection reports plus `dataset_master_summary.csv`; include the WELFake overlap table and dataset figures. Do not call WELFake independent.

## 5. Experimental Design
Use Stage 2A/2B reports and model configuration. Explain validation-only selection and external evaluation after selection; do not say FakeNewsAMT tuned any model.

## 6. Results
Use `final_results_master.csv`, `final_generalisation_master.csv`, final Figures 01–05 and statistical table. Do not equate McNemar results with Macro-F1 tests.

## 7. Discussion
Interpret large gaps and non-significant external rankings using `FINAL_RESEARCH_FINDINGS.md`; do not claim RoBERTa failed.

## 8. Explainability and Error Analysis
Use Stage 4 report/tables and Figures 06 plus existing explainability figures. State Integrated Gradients did not complete.

## 9. Limitations / Threats to Validity
Use `LIMITATIONS_AND_THREATS.md`; distinguish observed limitations from plausible threats.

## 10. Conclusion and Future Work
Use the final findings and contribution. Future work may propose replication on more independent corpora; it must not be presented as completed.''')

write(R/'RESEARCH_CONTRIBUTION.md', '''# Research Contribution

The supported contribution is an MSc-scale, reproducible methodological and empirical case study of cross-corpus fake-news classification. Its strongest contribution is the dataset-validation result: the audit proved WELFake unsuitable here as an independent ISOT external set, preserving the historical outputs while preventing their misuse. The replacement FakeNewsAMT evaluation was independently screened for direct body/title reuse and applied uniformly to frozen classical and RoBERTa models.

Empirically, the study demonstrates a marked in-domain-to-external drop and an external ordering unlike the ISOT ordering. It does not prove global model rankings, real-world effectiveness, causal reasons for errors, or a novel model architecture. Practically, it supports auditing corpus dependence and using held-out cross-corpus tests before interpreting high benchmark scores as robustness.''')

write(R/'LIMITATIONS_AND_THREATS.md', '''# Limitations and Threats to Validity

## Demonstrated

- WELFake is dependent on ISOT in this workspace and cannot be an independent external test.
- FakeNewsAMT has only 430 usable records, consists of short excerpts, and contains crowdsourced deceptive-news examples.
- Large ISOT-to-FakeNewsAMT performance drops demonstrate distribution sensitivity for this design.
- Integrated Gradients did not complete; token-level attribution is unavailable.

## Possible or scope-limiting

ISOT provenance/age and dataset construction may encode source, topical or stylistic artefacts. Label definitions differ across corpora; FakeNewsAMT's crowd-authored fakes and short format create domain mismatch. Exact/normalised overlap checks cannot rule out paraphrase, shared wire material, events, or latent source dependence. The external size limits statistical power. RoBERTa uses a 64-token input and early stopping after a tie at epoch 2; these are recorded design/compute constraints, not evidence of an architecture-wide limit. Results do not generalise automatically to evolving, multilingual, long-form, or organically produced misinformation. No causal explanation follows from feature weights or error profiles.''')

write(R/'REPRODUCIBILITY_CHECKLIST.md', '''# Reproducibility Checklist

| Item | Status / evidence |
|---|---|
| Python and libraries | Recorded in `final_comparison_metadata.json` (Python 3.11.9; pandas, NumPy, scikit-learn, SciPy) and Stage 4 metadata (Torch 2.7.1+cu118; Transformers 5.15.0). |
| Seed/configuration | Seed 42; `configs/config.yaml`, `results/roberta/training_configuration.json`; note the config file's RoBERTa settings are stale versus executed Stage 2B settings. |
| Datasets/hashes | Raw ISOT files; FakeNewsAMT SHA-256 recorded in selection/Stage 2A/2B/4 metadata; WELFake retained but rejected. |
| Processing/splits | `src/`, `scripts/`, processed split hashes in `training_configuration.json`. |
| Models/vectorizer/checkpoint | `models/traditional/`, `results/roberta/selected_checkpoint/`; selected model hash recorded in Stage 3/4 metadata. |
| Predictions/evaluation/statistics | `results/predictions/`, `results/roberta/predictions_*`, Stage 2A/2B/3 reports and scripts. |
| Error analysis | Stage 4 script, tables, figures, report and status JSON. |
| Integrity | Stage 3 audit; Stage 4 manifest and `stage4_integrity_check.json` (225/225 unchanged). |

Missing or limited: no environment lockfile with exact package hashes, no complete machine-independent data-download script for every retained candidate, and no completed RoBERTa token-attribution output. Use executed metadata rather than stale README/config wording.''')

write(R/'APPLICATION_DECISION.md', '''# Application Decision

## Recommendation: optional, demonstration only

A small application could demonstrate frozen-model inference and disclose uncertainty, but it adds no evidence to the research findings. If included, use the saved selected RoBERTa model (or clearly selectable frozen models), accept title/body text, return predicted class/probability and a prominent limitation notice. It must state that it is a demonstration artefact, not a validated misinformation-verification system; it must not use WELFake as validation or imply real-world reliability. No application is built in Stage 5.''')

write(R/'FINAL_INTEGRITY_AUDIT.md', '''# Final Integrity Audit

## Result: PASS for frozen Stages 1–4

Stage 5 generated only derivative documentation, master tables and figures. It did not run training, tuning, evaluation, prediction generation, or any model checkpoint write. The Stage 4 integrity manifest remains the frozen-artifact baseline: 225 files were SHA-256 checked and all 225 were unchanged (0 modified, 0 missing), recorded in `stage4_integrity_check.json`. The selected RoBERTa `model.safetensors` hash is `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d`, matching Stage 3/4 metadata.

WELFake is excluded from the valid external evaluation; FakeNewsAMT was evaluation-only after model selection. Stage 1–4 integrity records remain present. No conflicting final values were found among the final comparison, RoBERTa evaluation and Stage 2A reports.

## Retained stale or invalid material

`PROJECT_AUDIT.md`, `PROJECT_COMPLETION_AUDIT.md`, `NEXT_STEPS.md` and `configs/config.yaml` contain pre-Stage-1/2B plans or superseded WELFake assumptions. Historical `results/predictions/*_welfake.csv`, `cross_dataset_results.csv`, `generalisation_gap.csv` and WELFake figures/results must be labelled **historical / invalid for independent external generalisation** and excluded from the dissertation's final performance claims. They are retained for auditability, not deleted.''')

write(ROOT/'README.md', '''# Fake-news classification under cross-corpus shift

## Research question

How do ISOT-trained TF-IDF baselines and RoBERTa perform in-domain and on an independently audited cross-corpus evaluation set?

## Final status

Stages 1–5 are complete. ISOT was used for training/validation/test; Naive Bayes, Logistic Regression, Random Forest and RoBERTa were compared. FakeNewsAMT (430 usable records) was evaluation-only. RoBERTa led ISOT Macro-F1 (0.999742), while Naive Bayes had the highest observed FakeNewsAMT Macro-F1 (0.570970); external paired differences were not significant after Holm correction.

## Critical data-integrity finding

WELFake is **rejected as an independent external evaluation dataset**: the audit detected all 39,101 unique cleaned ISOT articles in it. Retained WELFake predictions/results are historical audit evidence and must not be reported as independent generalisation.

## Reproducibility

Read `results/REPRODUCIBILITY_CHECKLIST.md`, `results/FINAL_RESEARCH_AUDIT.md`, and `results/FINAL_INTEGRITY_AUDIT.md`. Executed source data, models, configurations, predictions, reports, tables and figures are retained. Do not rerun scripts unless reproducing the documented study; no Streamlit application is included.

## Repository layout

- `News_Dataset/`, `data/`: source and processed data
- `models/`, `results/roberta/selected_checkpoint/`: frozen models
- `scripts/`, `src/`: pipeline and analysis code
- `results/`: reports, integrity records, tables and figures

## Limitations and ethics

FakeNewsAMT comprises short excerpts with crowdsourced fake items; it is not a large-scale real-world benchmark. Exact/normalised overlap checks do not rule out all residual dependence. Respect dataset licences and avoid representing model outputs as factual verification.''')

# Recheck the immutable Stage-4 baseline after all Stage-5 derivative outputs exist.
manifest=json.loads((R/'stage4_pre_analysis_manifest.json').read_text(encoding='utf-8-sig'))
missing=[]; modified=[]; unchanged=[]
for entry in manifest:
    p=ROOT/entry['Path']
    if not p.is_file(): missing.append(entry['Path']); continue
    h=hashlib.sha256()
    with p.open('rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''): h.update(block)
    digest=h.hexdigest().upper()
    if digest != entry['SHA256'].upper(): modified.append({'path':entry['Path'],'expected_sha256':entry['SHA256'],'actual_sha256':digest})
    else: unchanged.append(entry['Path'])
status='pass' if not missing and not modified else 'fail'
(R/'final_integrity_check.json').write_text(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'manifest_used':'results/stage4_pre_analysis_manifest.json','number_of_files_checked':len(manifest),'modified_files':modified,'missing_files':missing,'unchanged_files':{'count':len(unchanged),'all_manifest_entries_unchanged':not missing and not modified},'overall_status':status,'confirmation_no_frozen_experiment_artefacts_modified':not missing and not modified},indent=2),encoding='utf-8')
if status != 'pass': raise SystemExit('Frozen-artifact integrity check failed')
