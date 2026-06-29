from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["age","bmi","biomarker_alpha","biomarker_beta","systolic_bp","diastolic_bp","ldl_cholesterol","hdl_cholesterol","prior_treatments","genetic_risk_score","trial_phase","adverse_event_count","enrollment_duration_days"]
CATEGORICAL_FEATURES = ["trial_phase"]
NUMERICAL_FEATURES = ["age","bmi","biomarker_alpha","biomarker_beta","systolic_bp","diastolic_bp","ldl_cholesterol","hdl_cholesterol","prior_treatments","genetic_risk_score","adverse_event_count","enrollment_duration_days"]
TARGET_NAME = "trial_success"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "age": rng.normal(55,12,size=n).clip(18,85).astype(int),
        "bmi": rng.normal(28,5,size=n).clip(15,45).round(1),
        "biomarker_alpha": rng.lognormal(mean=1.5,sigma=0.5,size=n).round(2),
        "biomarker_beta": rng.uniform(0,100,size=n).round(1),
        "systolic_bp": rng.normal(130,15,size=n).clip(80,200).astype(int),
        "diastolic_bp": rng.normal(80,10,size=n).clip(50,130).astype(int),
        "ldl_cholesterol": rng.normal(120,35,size=n).clip(30,250).astype(int),
        "hdl_cholesterol": rng.normal(50,12,size=n).clip(15,100).astype(int),
        "prior_treatments": rng.poisson(lam=2,size=n).clip(0,8),
        "genetic_risk_score": rng.uniform(0,100,size=n).round(1),
        "trial_phase": rng.choice(["Phase I","Phase II","Phase III"],size=n,p=[0.25,0.40,0.35]),
        "adverse_event_count": rng.poisson(lam=3,size=n).clip(0,15),
        "enrollment_duration_days": rng.exponential(scale=180,size=n).clip(30,730).astype(int),
    })
    age=np.clip(df["age"]/85,0,1); bmi=np.clip((df["bmi"]-15)/30,0,1); ba=df["biomarker_alpha"]/(df["biomarker_alpha"].max()+1e-8)
    bb=df["biomarker_beta"]/100; sbp=np.clip(df["systolic_bp"]/200,0,1)
    ldl=np.clip(df["ldl_cholesterol"]/250,0,1); hdl=np.clip(df["hdl_cholesterol"]/100,0,1)
    prior=np.clip(df["prior_treatments"]/8,0,1); gen=df["genetic_risk_score"]/100
    ae=np.clip(df["adverse_event_count"]/15,0,1); enroll=np.clip(df["enrollment_duration_days"]/730,0,1)
    phase_map={"Phase I":0,"Phase II":0.5,"Phase III":1}; phase=df["trial_phase"].map(phase_map).values
    log_odds = 2.0 - 0.3*age + 0.2*bmi + 0.6*ba - 0.2*bb - 0.1*sbp - 0.15*ldl + 0.2*hdl - 0.3*prior - 0.4*gen - 0.5*ae + 0.3*phase - 0.1*enroll + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,75)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(trial_success=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
