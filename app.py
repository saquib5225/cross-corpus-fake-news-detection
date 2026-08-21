"""Frozen-model MSc research demonstration; inference and presentation only."""
from __future__ import annotations
import hashlib,re
from pathlib import Path
import joblib,pandas as pd,plotly.express as px,plotly.graph_objects as go,streamlit as st
ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"; TABLES=RESULTS/"tables"; MODELS=ROOT/"models"/"traditional"
EXPECTED_SHA="6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d"; ROBERTA_REPO_ID="Cancer5225/fake-news-detection-roberta"; ROBERTA_REVISION="5bd82453a54dfa7e25e41f9323228986bb2b310e"
COLOURS={"Naive Bayes":"#40b5a4","Logistic Regression":"#6d93e9","Random Forest":"#e7a45b","RoBERTa":"#ab7ad8"}
META={"Naive Bayes":("Classical ML","Probabilistic TF-IDF baseline","A compact reference for simple word-frequency evidence."),"Logistic Regression":("Classical ML","Linear TF-IDF classifier","A strong linear sparse-feature baseline."),"Random Forest":("Classical ML","Tree-ensemble TF-IDF classifier","Tests a non-linear ensemble."),"RoBERTa":("Transformer","Fine-tuned contextual language model","Tests contextual representations against classical baselines.")}
st.set_page_config(page_title="Cross-Corpus Fake News Detection",page_icon="N",layout="wide")
def css():
 st.markdown("""<style>.stApp{background:#091321;color:#eef3fb}[data-testid="stSidebar"]{background:#0d1929;border-right:1px solid #263853}h1,h2,h3{letter-spacing:-.025em;color:#f6f8fc}.hero{padding:3rem;border:1px solid #304866;border-radius:20px;background:#10213a;margin:.25rem 0 1.35rem}.eyebrow{color:#48c0af;font-size:.78rem;font-weight:750;letter-spacing:.13em;text-transform:uppercase}.hero h1{font-size:clamp(2.2rem,4vw,3.6rem);line-height:1.04}.hero p,.card p{color:#c7d3e4;line-height:1.6}.card,.metric{background:#111d30;border:1px solid #293b56;border-radius:14px;padding:1.15rem;height:100%}.model{border-top:3px solid #48c0af}.label,.small{color:#afbdd0;font-size:.78rem}.value{font-size:1.65rem;font-weight:750}.note{color:#8ee0d1;font-size:.83rem}.callout{border-left:4px solid #48c0af;background:#102439;padding:1rem;border-radius:0 10px 10px 0;margin:1rem 0}.warning{border-left-color:#edb06b;background:#292016}.danger{border-left-color:#ee8995;background:#2b1922}.step{border:1px solid #304866;border-radius:12px;padding:.85rem;background:#0f1d31;min-height:95px;text-align:center}.step b{display:block;color:#8ee0d1;margin-bottom:.3rem}.real{color:#82e1cb;font-size:1.5rem;font-weight:800}.fake{color:#ff9fa8;font-size:1.5rem;font-weight:800}.stButton>button{border:0;border-radius:9px;background:#48c0af;color:#06181a;font-weight:750}[data-testid="stImage"]{display:flex;justify-content:center;margin:0 auto .65rem}[data-testid="stImage"] img{object-fit:contain}.intro-school{color:#8ee0d1;font-size:clamp(.9rem,1.8vw,1.05rem);font-weight:800;letter-spacing:.15em;margin:.15rem 0 .35rem;text-align:center}.intro-programme{color:#c7d3e4;font-size:clamp(1rem,2vw,1.18rem);letter-spacing:.025em;margin:0 0 2rem;text-align:center}.intro-title{font-size:clamp(1.9rem,4.1vw,3.15rem);line-height:1.14;margin:0 auto;max-width:720px;text-align:center}.intro-type{color:#8ee0d1;font-size:.84rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;margin:1.7rem 0 1.15rem;text-align:center}.intro-details{color:#dbe5f2;font-size:1rem;line-height:1.75;margin:0 auto;text-align:center}.intro-details span{color:#9fb1c7;font-weight:650;margin-right:.35rem}@media(max-width:640px){.intro-programme{margin-bottom:1.5rem}.intro-type{margin-top:1.35rem}}</style>""",unsafe_allow_html=True)
def combined_text(a,b): return re.sub(r"\s+"," ",f"{a.strip()} {b.strip()}").strip()
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
@st.cache_data
def results(): return pd.read_csv(TABLES/"final_results_master.csv")
@st.cache_data
def gap_table(): return pd.read_csv(TABLES/"final_generalisation_master.csv")
@st.cache_resource
def classical(n):
 f={"Naive Bayes":"naive_bayes.joblib","Logistic Regression":"logistic_regression.joblib","Random Forest":"random_forest.joblib"}[n]
 return joblib.load(MODELS/f),joblib.load(MODELS/"tfidf_vectorizer.joblib")
def hf_settings(): return ROBERTA_REPO_ID,ROBERTA_REVISION,None
@st.cache_resource(show_spinner="Preparing the pinned RoBERTa checkpoint...")
def roberta(repo,revision,token):
 if repo!=ROBERTA_REPO_ID or revision!=ROBERTA_REVISION: raise RuntimeError("RoBERTa must use the verified repository and immutable revision.")
 from huggingface_hub import snapshot_download
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 folder=Path(snapshot_download(repo_id=repo,revision=revision,token=token,allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json"]))
 if digest(folder/"model.safetensors")!=EXPECTED_SHA: raise RuntimeError("Downloaded checkpoint failed frozen-artifact SHA-256 verification.")
 return AutoTokenizer.from_pretrained(folder,local_files_only=True),AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True)
def metrics(n):
 d=results(); return d[(d.Model==n)&(d.Dataset=="ISOT")].iloc[0],d[(d.Model==n)&(d.Dataset=="FakeNewsAMT")].iloc[0],gap_table().query("Model == @n").iloc[0]
def cards(items):
 for c,(a,b,d) in zip(st.columns(len(items)),items):c.markdown(f'<div class="metric"><div class="label">{a}</div><div class="value">{b}</div><div class="note">{d}</div></div>',unsafe_allow_html=True)
def pipe(items):
 for c,(a,b) in zip(st.columns(len(items)),items):c.markdown(f'<div class="step"><b>{a}</b><span class="small">{b}</span></div>',unsafe_allow_html=True)
def section(title,kicker,body): st.markdown(f'<div class="eyebrow">{kicker}</div><h1>{title}</h1><p style="color:#c7d3e4;max-width:900px">{body}</p>',unsafe_allow_html=True)
def chart(col,title):
 d=results().copy();d[col]*=100; f=px.bar(d,x="Model",y=col,color="Dataset",barmode="group",text_auto=".2f",title=title,color_discrete_map={"ISOT":"#6d93e9","FakeNewsAMT":"#40b5a4"});f.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend_title=None,yaxis_title=f"{col} (%)");return f
def gaps():
 f=px.bar(gap_table(),x="Model",y="Generalisation Gap (pp)",color="Model",text_auto=".2f",title="Macro-F1 generalisation gap",color_discrete_map=COLOURS);f.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False);return f
def introduction():
 a,b,c=st.columns((3,2,3))
 with b:
  st.image(str(ROOT/"assets"/"DBS.png"),width=180)
 st.markdown('<div class="intro-school">DUBLIN BUSINESS SCHOOL</div>',unsafe_allow_html=True)
 st.markdown('<div class="intro-programme">MSc in Artificial Intelligence</div>',unsafe_allow_html=True)
 st.markdown('<h1 class="intro-title">Cross-Corpus Fake News Detection: Evaluating Model Generalisation Across News Domains</h1>',unsafe_allow_html=True)
 st.markdown('<div class="intro-type">Research Project / Dissertation</div>',unsafe_allow_html=True)
 st.markdown('<div class="intro-details"><div><span>Student:</span> Saquib Pirjade</div><div><span>Student Number:</span> 20079780</div></div>',unsafe_allow_html=True)
def home():
 st.markdown('<section class="hero"><div class="eyebrow">MSc research demonstration</div><h1>Cross-Corpus Fake News Detection: Evaluating Model Generalisation Across News Domains</h1><p>A completed comparison of classical text classifiers and RoBERTa across in-domain and independently screened external evaluation. The question is not only whether a model scores highly, but whether performance transfers when the news domain changes.</p></section>',unsafe_allow_html=True)
 cards([("RoBERTa / ISOT Macro-F1","99.97%","Best observed in-domain result"),("Naive Bayes / external Macro-F1","57.10%","Highest observed FakeNewsAMT result"),("External evaluation cohort","430","FakeNewsAMT records; evaluation only")])
 a,b=st.columns((1.2,1));a.markdown('<h2>Research motivation</h2><div class="callout">High performance on one source corpus does not establish dependable cross-corpus performance. Dataset independence and external evaluation are central research questions.</div><h2>What was built</h2><p>A leakage-aware protocol, three TF-IDF baselines, a frozen RoBERTa model, validated comparisons, external analysis and live inference separate from dissertation evaluation.</p>',unsafe_allow_html=True);b.markdown("## Research objectives\n- Audit external data for independence.\n- Compare four trained models.\n- Test in-domain and external performance.\n- Report uncertainty and limitations.")
 st.subheader("Leakage-safe research pipeline");pipe([("Datasets","ISOT + audited candidates"),("Cleaning","preprocessing"),("Development","train / validation"),("Selection","ISOT validation"),("Testing","ISOT test"),("External test","FakeNewsAMT only")]);pipe([("Analysis","generalisation gaps"),("Statistics","paired comparison"),("Findings","transparent reporting")]);st.markdown('<div class="callout"><b>Explore the research:</b> use the sidebar to inspect methodology, models, performance, generalisation and live prediction.</div>',unsafe_allow_html=True)
def why():
 section("Why this project matters","Research motivation","Fake-news detection is often judged by a single benchmark score. This study asks whether that score remains informative when the corpus changes.")
 for c,h,t in zip(st.columns(3),["The challenge","The scientific risk","The comparison"],["News language, provenance, labels and style vary across sources.","A high in-domain score alone does not establish cross-corpus performance.","Classical TF-IDF models and a transformer were evaluated under one completed protocol."]):c.markdown(f'<div class="card"><h3>{h}</h3><p>{t}</p></div>',unsafe_allow_html=True)
 st.plotly_chart(chart("Macro-F1","Frozen Macro-F1 results across evaluation settings"),use_container_width=True);st.markdown('<div class="callout">The strongest observed ISOT Macro-F1 did not correspond to the strongest observed FakeNewsAMT Macro-F1. This is cohort-specific evidence, not a causal or universal claim.</div>',unsafe_allow_html=True)
def methodology():
 section("Methodology","Leakage-safe protocol","Development, selection, in-domain testing and external evaluation remain separated so external results are not used to tune the models.")
 pipe([("Data preparation","cleaned article text"),("Preprocessing","title + body"),("Training","ISOT development"),("Validation","model selection"),("ISOT test","frozen results"),("External evaluation","FakeNewsAMT only")]);pipe([("Statistical comparison","paired differences"),("Explainability","descriptive outputs"),("Reporting","gaps and limitations")])
 a,b=st.columns(2);a.markdown('<div class="card"><h3>Classical path</h3><p>Normalised title/body -> frozen TF-IDF vectorizer -> Naive Bayes, Logistic Regression or Random Forest.</p></div>',unsafe_allow_html=True);b.markdown('<div class="card"><h3>RoBERTa path</h3><p>Title/body -> tokenizer -> frozen source-pinned checkpoint. Research execution used a 64-token maximum.</p></div>',unsafe_allow_html=True)
def datasets():
 section("Datasets and independence","Evaluation design","Dataset names alone do not demonstrate independence. Candidate external data was audited before it was used for a generalisation claim.")
 for c,h,t,cl in zip(st.columns(3),["ISOT","FakeNewsAMT","WELFake: rejected"],["Development corpus: 44,898 raw records and 39,101 unique non-empty cleaned contents. Used for training, validation and in-domain testing.","Independently screened external evaluation cohort: 430 usable excerpts, 190 fake and 240 legitimate/real. Never used for training or tuning.","All 39,101 unique cleaned ISOT articles occurred in WELFake. Historical WELFake outputs are invalid for independent external-generalisation claims."],["",""," warning"]):c.markdown(f'<div class="card{cl}"><h3>{h}</h3><p>{t}</p></div>',unsafe_allow_html=True)
 f=go.Figure(go.Bar(x=["ISOT -> FakeNewsAMT","ISOT -> WELFake"],y=[0,39101],marker_color=["#40b5a4","#ee8995"],text=["0","39,101"],textposition="outside"));f.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",title="Unique cleaned ISOT articles detected by the independence audit",yaxis_title="Articles");st.plotly_chart(f,use_container_width=True)
def models_page():
 section("Model overview","Four frozen trained models","Validated frozen metrics are presented below; the models are comparative research instruments, not factual-verification systems.")
 for n in META:
  i,e,g=metrics(n);cat,desc,reason=META[n];st.markdown(f'<div class="card model" style="border-top-color:{COLOURS[n]}"><div class="eyebrow">{cat}</div><h2>{n}</h2><p><b>{desc}</b></p><p>{reason}</p><p class="small">ISOT Macro-F1 <b>{i["Macro-F1"]*100:.3f}%</b> | FakeNewsAMT Macro-F1 <b>{e["Macro-F1"]*100:.3f}%</b> | Generalisation gap <b>{g["Generalisation Gap (pp)"]:.3f} pp</b></p></div>',unsafe_allow_html=True)
def predict_classical(n,a):
 m,v=classical(n);x=v.transform([a]);p=m.predict_proba(x)[0];return int(m.predict(x)[0]),float(p[list(m.classes_).index(1)])
def predict_transformer(a):
 repo,rev,tok=hf_settings();tokenizer,model=roberta(repo,rev,tok);import torch;x=tokenizer(a,return_tensors="pt",truncation=True,max_length=64)
 with torch.no_grad():p=torch.softmax(model(**x).logits,dim=1)[0].tolist()
 return int(max(range(2),key=lambda i:p[i])),float(p[1])
def interactive():
 section("Interactive fake-news detection","Live model prediction","Frozen inference only. This is separate from the completed dissertation evaluation and does not verify claims against authoritative sources.")
 st.markdown('<div class="callout warning"><b>Live prediction is not a frozen research result.</b> Model probabilities are not factual certainty and are not added to research evidence.</div>',unsafe_allow_html=True)
 h=st.text_input("Headline",placeholder="Optional headline");b=st.text_area("Article/body text",height=230,max_chars=50000,placeholder="Paste up to 50,000 characters.");selected=st.multiselect("Models to run",list(META),default=list(META),help="All four are selected by default. RoBERTa loads only after analysis is requested.")
 if st.button("Analyse Article",type="primary"):
  a=combined_text(h,b)
  if not a:st.error("Enter a headline or article text before analysis.");return
  if not selected:st.warning("Select at least one frozen model.");return
  out=[]
  with st.spinner("Running frozen-model inference..."):
   for n in selected:
    try:l,p=predict_transformer(a) if n=="RoBERTa" else predict_classical(n,a);out.append((n,l,p))
    except Exception as err:st.error(f"{n} could not complete inference: {err}")
  if out:
   agree=len({l for _,l,_ in out})==1;st.markdown(f'<div class="callout {" " if agree else "warning"}"><b>Model agreement:</b> {"All completed models agree." if agree else "The completed models disagree; compare the individual outputs below."}</div>',unsafe_allow_html=True)
  for c,(n,l,p) in zip(st.columns(len(out)),out):
   verdict="REAL" if l==1 else "FAKE";klass="real" if l==1 else "fake";c.markdown(f'<div class="card model" style="border-top-color:{COLOURS[n]}"><div class="small">{META[n][0]}</div><h3>{n}</h3><div class="{klass}">{verdict}</div><p>Confidence: <b>{max(p,1-p)*100:.1f}%</b><br><span class="small">Predicted real probability: {p*100:.1f}%</span></p></div>',unsafe_allow_html=True)
  st.markdown('<div class="callout danger">This demonstration evaluates learned linguistic patterns. It is not a fact-checking service and should not be the sole basis for decisions about a claim.</div>',unsafe_allow_html=True)
def dashboard():
 section("Performance dashboard","Frozen research results","Charts use completed frozen tables and project figures. No metrics are recalculated here.")
 cards([("Best ISOT Macro-F1","99.974%","RoBERTa"),("Best external Macro-F1","57.097%","Naive Bayes"),("External result","Not significant","Paired differences after Holm correction")]);st.plotly_chart(chart("Macro-F1","Macro-F1: in-domain versus external evaluation"),use_container_width=True);a,b=st.columns(2);a.plotly_chart(chart("Accuracy","Accuracy comparison"),use_container_width=True);b.plotly_chart(gaps(),use_container_width=True)
 d=results().melt(id_vars=["Model","Dataset"],value_vars=["Precision","Recall","Macro-F1"],var_name="Metric",value_name="Score");f=px.bar(d,x="Model",y="Score",color="Metric",facet_col="Dataset",barmode="group",text_auto=".2f",title="Precision, recall and Macro-F1");f.update_yaxes(tickformat=".0%");f.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)");st.plotly_chart(f,use_container_width=True);a,b=st.columns(2);a.image(str(RESULTS/"figures/final/05_fakenewsamt_confusion_comparison.png"),caption="Frozen FakeNewsAMT confusion-matrix comparison",use_container_width=True);b.image(str(RESULTS/"figures/final/06_roberta_external_error_patterns.png"),caption="Frozen RoBERTa external error pattern",use_container_width=True)
def generalisation():
 section("Generalisation under corpus shift","Central finding","Very high ISOT performance did not transfer at the same level to the independently screened FakeNewsAMT cohort.")
 pipe([("High ISOT performance","completed in-domain test"),("External evaluation","FakeNewsAMT only"),("Performance decline","all four models"),("Scientific interpretation","generalisation requires evidence")]);st.plotly_chart(chart("Macro-F1","The change from ISOT to FakeNewsAMT"),use_container_width=True);st.plotly_chart(gaps(),use_container_width=True);cards([("Best observed ISOT Macro-F1","RoBERTa","99.974%"),("Best observed external Macro-F1","Naive Bayes","57.097%"),("Largest observed gap","RoBERTa","45.568 pp")]);st.markdown('<div class="callout">RoBERTa is strongest observed in-domain, but not the strongest observed external Macro-F1 model. Its gap is largest by a very small margin. External paired differences were not statistically significant after Holm correction; these findings do not establish causality.</div>',unsafe_allow_html=True)
def explainability():
 section("Explainability and error analysis","Transparent limitation","Completed descriptive outputs are shown below. Token-level Integrated Gradients is not presented as a completed result.")
 st.markdown('<div class="callout warning"><b>Token-level Integrated Gradients is unavailable.</b> The attempted computation did not complete under the managed execution limit, so no token-attribution findings are claimed.</div>',unsafe_allow_html=True);a,b=st.columns(2);a.image(str(RESULTS/"figures/explainability/roberta_external_correct_incorrect.png"),caption="Frozen RoBERTa external correct/incorrect outcomes",use_container_width=True);b.image(str(RESULTS/"figures/explainability/model_shared_error_comparison.png"),caption="Frozen shared and model-specific external errors",use_container_width=True);st.caption("Reported descriptive external recalls: RoBERTa fake recall 0.736842; real recall 0.400000.")
def challenges():
 section("Challenges and solutions","Research delivery","Methodological, computational and deployment constraints are retained as part of the research story.")
 items=[("Dataset independence","WELFake contained all 39,101 unique cleaned ISOT articles.","A formal audit rejected it for independent external claims.","FakeNewsAMT was selected after stated checks."),("Cross-corpus decline","High in-domain performance did not transfer at the same level.","Gaps and paired external comparisons were reported.","The limitation is a central finding."),("RoBERTa execution","Training was computationally demanding and an unexpected termination required safeguards.","GPU, fp16, gradient accumulation and resumable checkpoints were used.","A validation-selected frozen checkpoint was retained."),("Checkpoint delivery","The selected weight is too large for normal Git hosting.","The unchanged checkpoint was hosted at a verified immutable Hugging Face revision.","The app checks its SHA-256 before loading."),("Token attribution","Integrated Gradients did not finish within the managed execution limit.","The limitation was documented.","No token-level attribution result is claimed.")]
 for t,h,handle,outcome in items:
  with st.expander(t):st.markdown(f"**What happened:** {h}\n\n**How it was handled:** {handle}\n\n**Final outcome:** {outcome}")
def findings():
 section("Research findings","Completed evidence","A leakage-aware comparison shows why cross-corpus evaluation and independence checks materially change benchmark interpretation.")
 cards([("1","RoBERTa","Best observed in-domain Macro-F1: 99.974%"),("2","All models declined","Substantial external Macro-F1 reduction"),("3","Naive Bayes","Highest observed external Macro-F1: 57.097%")]);cards([("4","RoBERTa","Largest observed gap: 45.568 pp"),("5","Not significant","External paired differences after Holm correction"),("6","Dataset independence","Required for credible generalisation claims")])
def limitations():
 section("Limitations","Scope and interpretation","These completed-study constraints bound the claims that can be made from the evidence.")
 for n,(h,t) in enumerate([("External cohort size","FakeNewsAMT has 430 usable records, limiting precision and statistical power."),("Dataset scope","Short excerpts and crowdsourced deceptive-news examples are not all real-world misinformation."),("Dataset construction and age","ISOT provenance, age and construction may encode source or topical patterns."),("Residual dependence","Deterministic overlap checks cannot exclude every paraphrase, event or source relationship."),("Token attribution","Integrated Gradients did not complete, so token-level attribution is incomplete.")]):st.markdown(f'<div class="card"><h3>{n+1:02d}. {h}</h3><p>{t}</p></div>',unsafe_allow_html=True)
def story():
 section("Project / dissertation story","From question to contribution","A presentation-ready account of how a model-comparison task developed into a study of data independence and cross-corpus generalisation.")
 steps=[("Why build it","Test whether benchmark strength transfers across domains."),("Research question","Compare ISOT-trained baselines and RoBERTa across corpora."),("Dataset decisions","Audit candidates; reject WELFake; retain FakeNewsAMT for evaluation."),("Models and experiments","Compare three classical models and RoBERTa."),("Problems and solutions","Address overlap, decline, computational constraints and delivery integrity."),("What was discovered","In-domain strength did not preserve ranking or performance externally."),("Contribution","A transparent leakage-aware account of results and limits.")]
 for i in range(0,len(steps),3):pipe(steps[i:i+3])
PAGES={"Introduction":introduction,"Home":home,"Why This Project Matters":why,"Methodology":methodology,"Datasets & Independence":datasets,"Model Overview":models_page,"Interactive Detection":interactive,"Performance Dashboard":dashboard,"Generalisation":generalisation,"Explainability & Error Analysis":explainability,"Challenges & Solutions":challenges,"Research Findings":findings,"Limitations":limitations,"Project / Dissertation Story":story}
def main():
 css()
 with st.sidebar:
  st.markdown("### Research Demonstration");st.caption("Frozen-model inference | cross-corpus evaluation");page=st.radio("Navigate",list(PAGES),label_visibility="collapsed");st.divider();st.caption("Live predictions are not factual verification. Submitted text is not intentionally stored.")
 PAGES[page]()
if __name__=="__main__":main()
