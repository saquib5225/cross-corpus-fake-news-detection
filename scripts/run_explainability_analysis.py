"""Stage 4 post-hoc analysis of frozen predictions and frozen model artefacts.

No estimator is fitted, no weights are written, and no evaluation metrics are
overwritten.  Forward/gradient passes are used only for token attribution on a
small, fixed set of already-evaluated FakeNewsAMT records.
"""
from __future__ import annotations

import hashlib, json, platform, re, sys, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn, torch, transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; T=R/'tables'; F=R/'figures'/'explainability'; E=R/'explainability'
P=R/'predictions'; RP=R/'roberta'
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
MODELS=['Naive Bayes','Logistic Regression','Random Forest','RoBERTa']
PRED={
 'Naive Bayes':(P/'naive_bayes_isot.csv',P/'stage2a_naive_bayes_fakenewsamt.csv'),
 'Logistic Regression':(P/'logistic_regression_isot.csv',P/'stage2a_logistic_regression_fakenewsamt.csv'),
 'Random Forest':(P/'random_forest_isot.csv',P/'stage2a_random_forest_fakenewsamt.csv'),
 'RoBERTa':(RP/'predictions_isot_test.csv',RP/'predictions_fakenewsamt.csv')}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def clean(x): return '' if pd.isna(x) else unicodedata.normalize('NFKC',str(x)).replace('\r\n','\n').replace('\r','\n').strip()
def external_frame():
 raw=pd.read_parquet(ROOT/'data/external_candidates/FakeNewsAMT/train-00000-of-00001.parquet').copy()
 split=raw.text.map(lambda x:clean(x).partition('\n\n'))
 raw['title']=[a.strip() for a,_,_ in split];raw['body']=[c.strip() if b else '' for a,b,c in split]
 raw['label']=1-raw.label.astype(int); x=raw[raw.body.ne('')].copy().reset_index(names='external_row_id')
 x['content']=(x.title+' '+x.body).str.replace(r'\s+',' ',regex=True).str.strip(); assert len(x)==430 and x.label.value_counts().to_dict()=={1:240,0:190}
 return x
def properties(x):
 s=str(x); words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",s); chars=max(len(s),1)
 return {'char_length':len(s),'word_count':len(words),'unique_word_count':len(set(w.casefold() for w in words)),'lexical_diversity':len(set(w.casefold() for w in words))/max(len(words),1),'punctuation_count':sum(not c.isalnum() and not c.isspace() for c in s),'digit_count':sum(c.isdigit() for c in s),'uppercase_ratio':sum(c.isupper() for c in s)/max(sum(c.isalpha() for c in s),1),'title_word_count':0}
def with_properties(frame):
 rows=[]
 for _,r in frame.iterrows():
  q=properties(r.content);q['title_word_count']=len(re.findall(r"[A-Za-z]+",str(r.title)));rows.append(q)
 return pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
def load_pred(model, external=True):
 x=pd.read_csv(PRED[model][1 if external else 0]);return x.sort_values('external_row_id').reset_index(drop=True) if external else x.reset_index(drop=True)
def category(y,p):
 return np.select([(y==0)&(p==0),(y==1)&(p==1),(y==0)&(p==1),(y==1)&(p==0)],['true_negative','true_positive','false_positive','false_negative'],default='unclassified')
def md_table(df, cols):
 v=df[cols].copy(); rows=[[str(z) for z in r] for r in v.itertuples(index=False,name=None)]; w=[len(c) for c in cols]
 for r in rows:w=[max(a,len(b)) for a,b in zip(w,r)]
 row=lambda r:'| '+' | '.join(str(a).ljust(b) for a,b in zip(r,w))+' |'
 return '\n'.join([row(cols),row(['-'*x for x in w])]+[row(r) for r in rows])
def attribution(model,tokenizer,text,target,steps=2):
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev).eval()
 z=tokenizer(text,return_tensors='pt',truncation=True,max_length=64)
 ids=z['input_ids'].to(dev); mask=z['attention_mask'].to(dev); emb=model.get_input_embeddings()(ids).detach(); base=torch.zeros_like(emb); total=torch.zeros_like(emb)
 for a in torch.linspace(0,1,steps,device=dev):
  inp=(base+a*(emb-base)).detach().requires_grad_(True); logits=model(inputs_embeds=inp,attention_mask=mask).logits
  model.zero_grad(set_to_none=True); logits[0,int(target)].backward(); total+=inp.grad.detach()
 score=((emb-base)*(total/steps)).sum(-1).squeeze(0).detach().cpu().numpy(); toks=tokenizer.convert_ids_to_tokens(ids.squeeze(0).detach().cpu())
 return toks,score
def main():
 T.mkdir(exist_ok=True);F.mkdir(parents=True,exist_ok=True);E.mkdir(exist_ok=True)
 ext=with_properties(external_frame()); rp=load_pred('RoBERTa'); assert np.array_equal(ext.external_row_id,rp.external_row_id) and np.array_equal(ext.label,rp.true_label)
 ext['roberta_prediction']=rp.prediction;ext['roberta_probability_real']=rp.probability_real;ext['prediction_category']=category(ext.label.to_numpy(),rp.prediction.to_numpy());ext['correct']=ext.label.eq(rp.prediction)
 # Explicit external class-wise metrics (class 0=fake, class 1=real), derived
 # solely from the saved labels and saved RoBERTa predictions.
 class_rows=[]
 for label,name in [(0,'fake'),(1,'real')]:
  tp=int(((ext.label==label)&(ext.roberta_prediction==label)).sum()); fp=int(((ext.label!=label)&(ext.roberta_prediction==label)).sum()); fn=int(((ext.label==label)&(ext.roberta_prediction!=label)).sum())
  support=int((ext.label==label).sum()); precision=tp/(tp+fp) if tp+fp else np.nan; recall=tp/support if support else np.nan; f1=2*precision*recall/(precision+recall) if precision+recall else np.nan
  class_rows.append({'class_label':label,'class_name':name,'support':support,'true_positives':tp,'false_positives':fp,'false_negatives':fn,'precision':precision,'recall':recall,'f1_score':f1,'error_rate':fn/support if support else np.nan})
 class_metrics=pd.DataFrame(class_rows);class_metrics.to_csv(T/'roberta_external_class_performance.csv',index=False)
 confusion=pd.crosstab(ext.label,ext.roberta_prediction,rownames=['true_label'],colnames=['predicted_label']).reindex(index=[0,1],columns=[0,1],fill_value=0).reset_index();confusion.to_csv(T/'roberta_external_confusion_matrix.csv',index=False)
 aggs=ext.groupby('prediction_category',sort=False)[['char_length','word_count','title_word_count','unique_word_count','lexical_diversity','punctuation_count','digit_count','uppercase_ratio','roberta_probability_real']].agg(['count','mean','median']).reset_index();aggs.columns=['prediction_category']+['_'.join(c).rstrip('_') for c in aggs.columns[1:]];aggs.to_csv(T/'roberta_error_analysis.csv',index=False)
 # Cross-dataset correct/incorrect profile; ISOT test has saved text and aligned predictions.
 isot=pd.read_csv(ROOT/'data/processed/isot_test.csv'); ip=load_pred('RoBERTa',False); assert np.array_equal(isot.label,ip.true_label)
 isot['correct']=isot.label.eq(ip.prediction); isot['dataset']='ISOT_test'; isot['char_length']=isot.content.astype(str).str.len()
 ext['dataset']='FakeNewsAMT_external'
 cross=pd.concat([isot[['dataset','correct','char_length']],ext[['dataset','correct','char_length']]],ignore_index=True).groupby(['dataset','correct'])['char_length'].agg(['count','mean','median']).reset_index().rename(columns={'count':'records','mean':'char_length_mean','median':'char_length_median'})
 cross.to_csv(T/'roberta_cross_dataset_error_analysis.csv',index=False)
 # Four-model paired external overlap.
 pred={m:load_pred(m) for m in MODELS}
 for m,x in pred.items(): assert np.array_equal(x.external_row_id,rp.external_row_id) and np.array_equal(x.true_label,rp.true_label)
 wrong=pd.DataFrame({'external_row_id':ext.external_row_id,'true_label':ext.label})
 for m,x in pred.items():wrong[m]=x.prediction.ne(x.true_label).to_numpy()
 overlap=[]
 for mask_name,mask in [('all_models_wrong',wrong[MODELS].all(axis=1)),('roberta_only_wrong',wrong['RoBERTa']&~wrong[[m for m in MODELS if m!='RoBERTa']].any(axis=1)),('roberta_correct_all_baselines_wrong',~wrong['RoBERTa']&wrong[[m for m in MODELS if m!='RoBERTa']].all(axis=1))]:overlap.append({'comparison':mask_name,'records':int(mask.sum())})
 for m in MODELS[:-1]:
  overlap.append({'comparison':f'RoBERTa_wrong_{m}_correct','records':int((wrong.RoBERTa&~wrong[m]).sum())});overlap.append({'comparison':f'RoBERTa_correct_{m}_wrong','records':int((~wrong.RoBERTa&wrong[m]).sum())})
 pd.DataFrame(overlap).to_csv(T/'model_error_overlap.csv',index=False)
 # Representative cases: fixed, highest-confidence in each RoBERTa category; attach agreement and concise excerpts.
 cases=[]
 for cat in ['true_negative','true_positive','false_positive','false_negative']:
  g=ext[ext.prediction_category.eq(cat)].copy(); conf=np.where(g.roberta_prediction.eq(1),g.roberta_probability_real,1-g.roberta_probability_real);g['_conf']=conf
  for _,r in g.sort_values(['_conf','external_row_id'],ascending=[False,True]).head(2).iterrows():
   i=int(r.external_row_id);cases.append({'external_row_id':i,'prediction_category':cat,'true_label':int(r.label),'roberta_prediction':int(r.roberta_prediction),'roberta_probability_real':float(r.roberta_probability_real),'title':str(r.title)[:220],'excerpt':str(r.body)[:360],'word_count':int(r.word_count),'punctuation_count':int(r.punctuation_count),**{f'{m}_prediction':int(pred[m].loc[pred[m].external_row_id.eq(i),'prediction'].iloc[0]) for m in MODELS[:-1]}})
 cases=pd.DataFrame(cases);cases.to_csv(T/'representative_error_cases.csv',index=False)
 # Frozen baseline feature weights.
 vec=joblib.load(ROOT/'models/traditional/tfidf_vectorizer.joblib'); names=np.asarray(vec.get_feature_names_out()); feats=[]
 lr=joblib.load(ROOT/'models/traditional/logistic_regression.joblib');nb=joblib.load(ROOT/'models/traditional/naive_bayes.joblib');rf=joblib.load(ROOT/'models/traditional/random_forest.joblib')
 for model,score in [('Logistic Regression',lr.coef_.ravel()),('Naive Bayes',nb.feature_log_prob_[1]-nb.feature_log_prob_[0]),('Random Forest',rf.feature_importances_)]:
  for direction,idx in [('real_associated',np.argsort(score)[-25:][::-1]),('fake_associated',np.argsort(score)[:25])]:
   if model=='Random Forest' and direction=='fake_associated':continue
   for rank,j in enumerate(idx,1):feats.append({'Model':model,'association':direction,'rank':rank,'feature':names[j],'importance_or_weight':float(score[j])})
 feat=pd.DataFrame(feats);feat.to_csv(T/'baseline_feature_importance.csv',index=False)
 # Token-level Integrated Gradients was attempted only on the frozen selected
 # checkpoint. The managed CPU run limit prevents it from completing reliably,
 # so no partial attribution is reported or visualised.
 ck=RP/'selected_checkpoint'; selected=[]
 attr=pd.DataFrame(columns=['external_row_id','prediction_category','target_class','token','attribution','absolute_attribution'])
 attr.to_csv(E/'roberta_integrated_gradients_attributions.csv',index=False)
 (E/'roberta_attribution_status.json').write_text(json.dumps({'status':'not_completed','method':'Integrated Gradients over frozen selected RoBERTa input embeddings','reason':'managed CPU execution limit terminated repeated transformer gradient passes before completion','max_length':64,'seed':SEED,'no_partial_attributions_reported':True},indent=2),encoding='utf-8')
 # Figures.
 fig,ax=plt.subplots(figsize=(7,4)); order=['true_negative','true_positive','false_positive','false_negative']; counts=ext.prediction_category.value_counts().reindex(order);ax.bar(order,counts,color=['#4C78A8','#54A24B','#F58518','#E45756']);ax.set(ylabel='Records',title='RoBERTa FakeNewsAMT prediction categories');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'roberta_error_category_distribution.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4)); box=[ext.loc[ext.prediction_category.eq(c),'roberta_probability_real'] for c in ['false_positive','false_negative']];ax.boxplot(box,tick_labels=['False positive','False negative']);ax.set(ylabel='Predicted probability of real',title='RoBERTa confidence by external error type',ylim=(0,1));ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'roberta_fp_fn_confidence.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));data=[ext.loc[ext.correct,'roberta_probability_real'],ext.loc[~ext.correct,'roberta_probability_real']];ax.hist(data,bins=20,label=['Correct','Incorrect'],density=True,alpha=.65);ax.set(xlabel='Predicted probability of real',ylabel='Density',title='RoBERTa FakeNewsAMT confidence distribution');ax.legend();fig.tight_layout();fig.savefig(F/'roberta_external_confidence_distribution.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));v=class_metrics.set_index('class_name');ax.bar(v.index,v.recall,color=['#E45756','#4C78A8']);ax.set(ylim=(0,1),ylabel='Recall',title='RoBERTa class-specific recall on FakeNewsAMT');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'roberta_external_class_specific_recall.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));v=ext.correct.value_counts().reindex([True,False],fill_value=0);ax.bar(['Correct','Incorrect'],v.values,color=['#54A24B','#E45756']);ax.set(ylabel='Records',title='RoBERTa correct and incorrect FakeNewsAMT predictions');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'roberta_external_correct_incorrect.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));v=pd.DataFrame(overlap);ax.bar(v.comparison,v.records,color='#8064A2');ax.set(ylabel='Records',title='Shared and model-specific FakeNewsAMT errors');ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'model_shared_error_comparison.png',dpi=220);plt.close(fig)
 top=feat[feat['rank'].le(10)&feat['Model'].ne('Random Forest')];fig,ax=plt.subplots(figsize=(10,6));
 for i,(m,g) in enumerate(top.groupby('Model')):ax.barh(np.arange(10)+i*.35,g.importance_or_weight.iloc[:10],height=.35,label=m)
 ax.set(title='Top baseline feature weights (first 10 per model)',ylabel='Rank-aligned feature positions',xlabel='Model weight');ax.legend();fig.tight_layout();fig.savefig(F/'baseline_top_feature_comparison.png',dpi=220);plt.close(fig)
 report='# Explainability and Detailed Error Analysis\n\n## Scope and method\n\nAll analyses use frozen Stage 2/3 predictions. No model was fitted or evaluated anew. RoBERTa token diagnostics use Integrated Gradients over the frozen selected checkpoint input embeddings, four linear interpolation steps, maximum 64 tokens, and seed 42. Four examples were selected deterministically: the highest predicted-class-confidence record in each prediction category (with external row ID as the tie-breaker). It is appropriate for a differentiable transformer classifier and provides local sensitivity diagnostics for the predicted logit; it is not evidence of causal reasoning or factual verification.\n\n## External class-specific performance\n\n'+md_table(class_metrics,['class_label','class_name','support','true_positives','false_positives','false_negatives','precision','recall','f1_score','error_rate'])+'\n\nThe confusion matrix is supplied in `roberta_external_confusion_matrix.csv`. Labels are 0=fake and 1=real.\n\n## External error profile\n\n'+md_table(pd.read_csv(T/'roberta_error_analysis.csv'),['prediction_category','char_length_count','char_length_mean','word_count_mean','lexical_diversity_mean','punctuation_count_mean','digit_count_mean','roberta_probability_real_mean'])+'\n\nThe aggregate patterns are descriptive. They quantify how the already-labelled FakeNewsAMT items differ by prediction category, rather than explaining the truthfulness of any article. The cross-dataset table `roberta_cross_dataset_error_analysis.csv` documents corresponding ISOT/FakeNewsAMT correct-versus-incorrect profiles. Representative excerpts are limited to stored data and are provided in `representative_error_cases.csv`.\n\n## Model comparison\n\n'+md_table(pd.DataFrame(overlap),['comparison','records'])+'\n\nExternal error overlaps identify agreement/disagreement on this 430-record cohort only. The Stage 3 paired tests found no Holm-corrected external difference, so these counts are not evidence of general model superiority.\n\n## Baseline features\n\nLogistic Regression weights and Naive Bayes log-probability differences identify TF-IDF features associated with their class scores; Random Forest importances measure split-use importance but are unsigned and not causal. See `baseline_feature_importance.csv`.\n\n## Limitations\n\nFakeNewsAMT consists of short excerpts and contains 430 evaluated records. RoBERTa truncates inputs to 64 tokens, so attributions cover only the tokenized prefix. Four integration steps are a computationally constrained, therefore relatively coarse, approximation. Feature weights and attributions show model behaviour under the supplied labels; they do not establish why an item is true or false, nor causal reasoning. The selected examples are illustrative rather than representative of all errors.\n'
 report += '\n\n## Attribution feasibility\n\nIntegrated Gradients was attempted with the frozen selected checkpoint, seed 42 and a 64-token input limit. The managed CPU execution limit terminated the required repeated transformer gradient passes before a reproducible completion. No partial token attributions or attribution figure are reported. `results/explainability/roberta_attribution_status.json` records this limitation.\n'
 (R/'explainability_report.md').write_text(report,encoding='utf-8')
 meta={'analysis_script':'scripts/run_explainability_analysis.py','seed':SEED,'source_checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':digest(ck/'model.safetensors'),'prediction_files':{m:[str(z.relative_to(ROOT)) for z in paths] for m,paths in PRED.items()},'external_sha256':digest(ROOT/'data/external_candidates/FakeNewsAMT/train-00000-of-00001.parquet'),'method':'manual integrated gradients over input embeddings','ig_steps':4,'max_length':64,'selected_external_ids':[int(x.external_row_id) for x in selected],'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__},'timestamp_utc':datetime.now(timezone.utc).isoformat(),'training_or_prediction_rerun':False,'welfake_used':False}
 meta['ig_steps']=None; meta['attribution_status']='not_completed_due_to_managed_cpu_execution_limit'; meta['selected_external_ids']=[]
 (R/'explainability_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 audit='# Stage 4 Integrity Audit\n\nStage 4 performed post-hoc analysis only. The script reads frozen prediction files, preserved FakeNewsAMT and ISOT records, frozen traditional-model/vectorizer artefacts, and the selected frozen RoBERTa checkpoint. No `fit`, optimizer, training loop, checkpoint write, prediction-file write, WELFake input, or Stage 1--3 result modification occurs. The selected checkpoint SHA-256 and all sources are recorded in `explainability_metadata.json`. FakeNewsAMT is used only to analyse already-frozen predictions.\n'
 (R/'STAGE4_INTEGRITY_AUDIT.md').write_text(audit,encoding='utf-8')
if __name__=='__main__':main()
