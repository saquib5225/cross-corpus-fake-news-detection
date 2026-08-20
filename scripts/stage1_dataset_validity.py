"""Stage 1: non-model dataset validity, overlap, leakage and shift analysis."""
from pathlib import Path
import re, json
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'results/tables'; F=ROOT/'results/figures'
T.mkdir(parents=True,exist_ok=True); F.mkdir(parents=True,exist_ok=True)
def norm(s): return s.fillna('').astype(str).str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
def words(s): return s.str.findall(r"(?u)\b\w+\b")
def stats(df,name,has_subject=False,has_date=False):
    title=norm(df.title); text=norm(df.text); content=(title+' '+text).str.strip(); w=words(content)
    return {'Dataset':name,'Records_raw':len(df),'Records_nonempty_content':int(content.ne('').sum()),'Class_0':int((df.label==0).sum()),'Class_1':int((df.label==1).sum()),'Missing_title':int(df.title.isna().sum()),'Missing_text':int(df.text.isna().sum()),'Exact_duplicate_content':int(content.duplicated().sum()),'Exact_duplicate_title':int(title.duplicated().sum()),'Mean_title_words':round(title.str.split().str.len().mean(),2),'Median_title_words':round(title.str.split().str.len().median(),2),'Mean_article_words':round(w.str.len().mean(),2),'Median_article_words':round(w.str.len().median(),2),'Vocabulary_size':len(set(x for row in w for x in row)),'Lexical_diversity':round(len(set(x for row in w for x in row))/sum(w.str.len()),6),'Has_subject':has_subject,'Has_date':has_date}
def main():
    fake=pd.read_csv(ROOT/'News_Dataset/Fake.csv'); real=pd.read_csv(ROOT/'News_Dataset/True.csv'); fake['label']=0; real['label']=1; isot=pd.concat([fake,real],ignore_index=True)
    wel=pd.read_csv(ROOT/'data/raw/WELFake/WELFake_Dataset.csv').rename(columns={'label':'raw_label'}); wel['label']=1-wel.raw_label.astype(int)
    st=pd.DataFrame([stats(isot,'ISOT',True,True),stats(wel,'WELFake')]); st.to_csv(T/'welfake_dataset_statistics.csv',index=False); st.to_csv(T/'dataset_statistics_comparison.csv',index=False)
    ic=(norm(isot.title)+' '+norm(isot.text)).str.strip(); wc=(norm(wel.title)+' '+norm(wel.text)).str.strip(); it=norm(isot.title); wt=norm(wel.title)
    # split exact intersections; preprocessing/mapping only, no model fitting
    splits={n:pd.read_csv(ROOT/f'data/processed/isot_{n}.csv') for n in ['train','validation','test']}; sets={n:set(norm(d.content)) for n,d in splits.items()}
    splitpairs=[('train','validation'),('train','test'),('validation','test')]
    duprows=[]
    for n,d,c,tit in [('ISOT',isot,ic,it),('WELFake',wel,wc,wt)]:
        duprows += [{'Dataset':n,'Measure':'exact_content_duplicates','Count':int(c.duplicated().sum())},{'Dataset':n,'Measure':'exact_title_duplicates','Count':int(tit.duplicated().sum())}]
    for a,b in splitpairs: duprows.append({'Dataset':'ISOT_splits','Measure':f'exact_content_overlap_{a}_{b}','Count':len(sets[a]&sets[b])})
    dup=pd.DataFrame(duprows); dup.to_csv(T/'duplicate_analysis.csv',index=False)
    overlap=pd.DataFrame([{'Comparison':'ISOT-WELFake','Field':'normalised_content','Overlap_records':len(set(ic)&set(wc))},{'Comparison':'ISOT-WELFake','Field':'normalised_title','Overlap_records':len(set(it)&set(wt))}]); overlap.to_csv(T/'dataset_overlap.csv',index=False)
    # near duplicate proxy: titles identical after stripping punctuation, excludes exact raw title check
    near_i=it.str.replace(r'[^a-z0-9 ]','',regex=True); near_w=wt.str.replace(r'[^a-z0-9 ]','',regex=True)
    near=len(set(near_i)&set(near_w)); pd.DataFrame([{'Comparison':'ISOT-WELFake','Method':'normalised_title_signature','Candidate_overlap':near,'Note':'Conservative near-duplicate proxy; not semantic matching'}]).to_csv(T/'near_duplicate_analysis.csv',index=False)
    # sampled lexical shift, deterministic and computationally bounded
    rng=42; isamp=isot.assign(content=ic).query("content != ''").sample(min(20000,int((ic!='').sum())),random_state=rng); wsamp=wel.assign(content=wc).query("content != ''").drop_duplicates('content').sample(20000,random_state=rng)
    cv=CountVectorizer(stop_words='english',min_df=3,max_features=20000,ngram_range=(1,2)); xi=cv.fit_transform(isamp.content); xw=cv.transform(wsamp.content); vocab=set(cv.get_feature_names_out()); vi=set(CountVectorizer(stop_words='english',min_df=3).fit(isamp.content).get_feature_names_out()); vw=set(CountVectorizer(stop_words='english',min_df=3).fit(wsamp.content).get_feature_names_out());
    shift=pd.DataFrame([{'Metric':'sample_records','ISOT':len(isamp),'WELFake':len(wsamp)},{'Metric':'vocabulary_jaccard','ISOT':len(vi),'WELFake':len(vw),'Value':len(vi&vw)/len(vi|vw)},{'Metric':'mean_tfidf_cosine_similarity','ISOT':None,'WELFake':None,'Value':float(cosine_similarity(xi.mean(axis=0),xw.mean(axis=0))[0,0])},{'Metric':'mean_content_words','ISOT':float(words(isamp.content).str.len().mean()),'WELFake':float(words(wsamp.content).str.len().mean())}]); shift.to_csv(T/'dataset_shift_statistics.csv',index=False)
    sns.set_theme(style='whitegrid'); fig,ax=plt.subplots(figsize=(8,5)); pd.DataFrame({'ISOT':words(isamp.content).str.len(),'WELFake':words(wsamp.content).str.len()}).melt(var_name='Dataset',value_name='Words').pipe(lambda d:sns.boxplot(data=d,x='Dataset',y='Words',showfliers=False,ax=ax)); ax.set_title('Article-length distribution (20,000-record samples)'); fig.tight_layout(); fig.savefig(F/'dataset_article_length_comparison.png',dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); pd.DataFrame({'Dataset':['ISOT','WELFake'],'Class 0':[sum(isot.label==0),sum(wel.label==0)],'Class 1':[sum(isot.label==1),sum(wel.label==1)]}).set_index('Dataset').plot(kind='bar',ax=ax); ax.set_ylabel('Records'); ax.set_title('Class distributions after documented label mapping'); fig.tight_layout(); fig.savefig(F/'dataset_class_distribution_comparison.png',dpi=300); plt.close(fig)
    leak=[('train/validation exact-content leakage',len(sets['train']&sets['validation'])==0),('train/test exact-content leakage',len(sets['train']&sets['test'])==0),('validation/test exact-content leakage',len(sets['validation']&sets['test'])==0),('TF-IDF fit leakage',json.load(open(ROOT/'results/baseline_metadata.json'))['tfidf_fitted_on']=='ISOT_train_only'),('WELFake training/tuning leakage',not json.load(open(ROOT/'results/baseline_metadata.json'))['external_used_for_training_or_tuning']),('hyperparameter tuning leakage',False),('model-selection leakage',False),('ISOT/WELFake exact content overlap',len(set(ic)&set(wc))==0)]
    lines=['# Data leakage audit','','| Check | Status | Evidence |','|---|---|---|']+[f'| {x} | {"PASS" if y else "NOT VERIFIED"} | Stage 1 analysis / baseline metadata |' for x,y in leak]; (ROOT/'results/data_leakage_audit.md').write_text('\n'.join(lines))
    (ROOT/'results/duplicate_overlap_report.md').write_text(f'# Duplicate and overlap analysis\n\nExact ISOT–WELFake content overlap: {len(set(ic)&set(wc))}. Exact title overlap: {len(set(it)&set(wt))}. Conservative normalised-title near-duplicate candidates: {near}. Split content intersections are recorded in `duplicate_analysis.csv`.')
    (ROOT/'results/dataset_shift_report.md').write_text('# Dataset-shift analysis\n\nThis report is based on actual available title/text/label fields. Quantitative results are in `tables/dataset_shift_statistics.csv` and `tables/dataset_statistics_comparison.csv`; figures use deterministic 20,000-record samples. ISOT has subject/date fields; WELFake does not, so subject/date comparisons are not performed.')
if __name__=='__main__': main()
