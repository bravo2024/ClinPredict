from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="ClinPredict | Axtria Clinical Trials", layout="wide", page_icon="\U0001f48a")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Axtria | Clinical Trial Outcome Prediction")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Success Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f9ea Biomarkers","\U0001f3af Trial Design"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Fail","Success"],[1-data["positive_rate"],data["positive_rate"]],color=["#f43f5e","#22c55e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Trial Outcome Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Biomarker Analysis")
    st.latex(r"\text{logit}(p) = \beta_0 + \beta_1 \cdot \text{Alpha} + \beta_2 \cdot \text{Beta} + \cdots")
    fig,ax=plt.subplots(figsize=((8,4))); _style()
    for bm,color in [("biomarker_alpha","#22d3ee"),("biomarker_beta","#f97316")]:
        bins=np.linspace(0,100,30)
        success_rate=[data["df"][(data["df"][bm]>=bins[i])&(data["df"][bm]<bins[i+1])][TARGET_NAME].mean() for i in range(len(bins)-1)]
        ax.plot(bins[:-1],success_rate,label=bm,color=color,lw=2)
    ax.set_xlabel("Biomarker Level"); ax.set_ylabel("Success Rate"); ax.set_title("Biomarker vs. Trial Success",color="white")
    ax.legend(fontsize=8); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.subheader("Genetic Risk Distribution")
    gen_good=data["df"][data["df"]["trial_success"]==0]["genetic_risk_score"]
    gen_bad=data["df"][data["df"]["trial_success"]==1]["genetic_risk_score"]
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    ax.hist(gen_good,bins=30,alpha=0.5,color="#22c55e",label="Success",density=True)
    ax.hist(gen_bad,bins=30,alpha=0.5,color="#f43f5e",label="Fail",density=True)
    ax.set_title("Genetic Risk Score by Outcome",color="white"); ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t4:
    st.subheader("Trial Design Optimization")
    st.latex(r"\text{Power} = \Phi\left(\frac{\Delta}{\sigma/\sqrt{n}} - z_{\alpha/2}\right)")
    st.metric("Estimated Sample Size Needed",f"{int(len(data['df'])*1.5):,}")
    st.subheader("Adverse Event Profile")
    ae_by_phase=data["df"].groupby("trial_phase")["adverse_event_count"].mean()
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(ae_by_phase.index,ae_by_phase.values,color=["#22d3ee","#f97316","#f43f5e"])
    ax.set_title("Mean Adverse Events by Phase",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
