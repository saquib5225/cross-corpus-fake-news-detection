"""Run audit, leakage-safe splits, TF-IDF baselines, and report-ready artefacts."""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from src.data_loader import load_isot, load_welfake

T=ROOT/'results'/'tables'; P=ROOT/'results'/'predictions'; C=ROOT/'results'/'confusion_matrices'; M=ROOT/'models'/'traditional'
for p in (T,P,C,M): p.mkdir(parents=True,exist_ok=True)
def metric(y,p,s):
    q=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    return {'Accuracy':accuracy_score(y,p),'Precision':q[0],'Recall':q[1],'F1':q[2],'ROC_AUC':roc_auc_score(y,s)}
def split(df):
    train,temp=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
    val,test=train_test_split(temp,test_size=.5,stratify=temp.label,random_state=42)
    for name,x in {'train':train,'validation':val,'test':test}.items(): x.to_csv(ROOT/'data'/'processed'/f'isot_{name}.csv',index=False)
    return train,val,test
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baselines',action='store_true'); ap.add_argument('--download-welfake',action='store_true'); a=ap.parse_args()
    if a.download_welfake:
        import urllib.request
        out=ROOT/'data/raw/WELFake/WELFake_Dataset.csv'; out.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1',out)
    d=load_isot(ROOT/'News_Dataset'); before=len(d); d=d[d.content.ne('')].drop_duplicates('content').reset_index(drop=True)
    pd.DataFrame([{'Rows_before':before,'Rows_after':len(d),'Rows_removed':before-len(d),'Fake':int((d.label==0).sum()),'Real':int((d.label==1).sum())}]).to_csv(T/'isot_dataset_statistics.csv',index=False)
    train,val,test=split(d)
    pd.DataFrame([{'split':n,'records':len(x),'fake':int((x.label==0).sum()),'real':int((x.label==1).sum())} for n,x in [('train',train),('validation',val),('test',test)]]).to_csv(T/'class_distribution.csv',index=False)
    if not a.baselines:return
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.95,max_features=50000,lowercase=True); X=v.fit_transform(train.content); Xte=v.transform(test.content)
    joblib.dump(v,M/'tfidf_vectorizer.joblib')
    models={'Naive Bayes':MultinomialNB(alpha=.5),'Logistic Regression':LogisticRegression(C=1,max_iter=2000,n_jobs=-1),'Random Forest':RandomForestClassifier(n_estimators=200,max_features='sqrt',n_jobs=-1,random_state=42)}
    rows=[]; external=None
    wp=ROOT/'data/raw/WELFake/WELFake_Dataset.csv'
    if wp.exists(): external=load_welfake(wp); external=external[external.content.ne('')].drop_duplicates('content'); Xwe=v.transform(external.content)
    for name,model in models.items():
        start=time.perf_counter(); model.fit(X,train.label); fit=time.perf_counter()-start; start=time.perf_counter(); pred=model.predict(Xte); score=model.predict_proba(Xte)[:,1]; infer=time.perf_counter()-start
        r={'Model':name,'Evaluation':'ISOT_test','Training_seconds':fit,'Inference_seconds':infer}|metric(test.label,pred,score); rows.append(r); joblib.dump(model,M/(name.lower().replace(' ','_')+'.joblib'))
        stem = name.lower().replace(' ', '_')
        pd.DataFrame({'true_label':test.label,'prediction':pred,'probability_real':score}).to_csv(P / f'{stem}_isot.csv', index=False)
        pd.DataFrame(confusion_matrix(test.label,pred),index=['actual_fake','actual_real'],columns=['pred_fake','pred_real']).to_csv(C / f'{stem}_isot.csv')
        if external is not None:
            p=model.predict(Xwe); s=model.predict_proba(Xwe)[:,1]; rows.append({'Model':name,'Evaluation':'WELFake_external','Training_seconds':0,'Inference_seconds':0}|metric(external.label,p,s)); pd.DataFrame({'true_label':external.label,'prediction':p,'probability_real':s}).to_csv(P / f'{stem}_welfake.csv',index=False)
    out=pd.DataFrame(rows); out[out.Evaluation.eq('ISOT_test')].to_csv(T/'in_domain_baseline_results.csv',index=False); out[out.Evaluation.eq('WELFake_external')].to_csv(T/'cross_dataset_results.csv',index=False); out.to_csv(T/'all_baseline_results.csv',index=False)
    if external is not None:
        i=out[out.Evaluation.eq('ISOT_test')].set_index('Model'); e=out[out.Evaluation.eq('WELFake_external')].set_index('Model'); (i[['Accuracy','Precision','Recall','F1']]-e[['Accuracy','Precision','Recall','F1']]).reset_index().to_csv(T/'generalisation_gap.csv',index=False)
    json.dump({'seed':42,'tfidf_fitted_on':'ISOT_train_only','external_used_for_training_or_tuning':False},open(ROOT/'results'/'baseline_metadata.json','w'),indent=2)
if __name__=='__main__':main()
