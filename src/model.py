"""model.py — Cox proportional hazards model for ClinPredict (J&J).

Implements survival analysis using the Cox partial likelihood, NOT
generic binary classification. The Cox model estimates hazard ratios
without specifying the baseline hazard.

The partial likelihood for the Cox model:
  L(beta) = product_i [ exp(beta * x_i) / sum_{j in R(t_i)} exp(beta * x_j) ]
where R(t_i) is the risk set at time t_i.

References
----------
Cox (1972), "Regression Models and Life-Tables." JRSS-B.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core import concordance_index, kaplan_meier, logrank_test


def cox_partial_likelihood(beta, X, times, events):
    """Negative log partial likelihood for the Cox model.

    L = sum_{i: event=1} [ beta * x_i - log(sum_{j in R(t_i)} exp(beta * x_j)) ]
    """
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = X @ beta
    log_lik = 0.0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        risk_set = times >= times[i]
        denom = np.log(np.sum(np.exp(risk_scores[risk_set])))
        log_lik += risk_scores[i] - denom
    return -log_lik


def fit_cox_model(df, features, time_col, event_col, seed=42):
    """Fit Cox proportional hazards model via partial likelihood optimization."""
    X = df[features].values.astype(float)
    times = df[time_col].values.astype(float)
    events = df[event_col].values.astype(int)
    n_features = X.shape[1]
    result = minimize(
        cox_partial_likelihood, np.zeros(n_features),
        args=(X, times, events), method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6},
    )
    beta = result.x
    risk_scores = X @ beta
    c_index = concordance_index(times, events, risk_scores)
    hazard_ratios = np.exp(beta)
    return {
        "beta": beta, "hazard_ratios": hazard_ratios,
        "feature_names": features, "c_index": c_index,
        "risk_scores": risk_scores, "converged": result.success,
    }


def fit_and_evaluate(data, seed=42):
    """Fit Cox model and evaluate with C-index, KM curves, and log-rank test."""
    df = data["df"]
    features = data["features"]
    time_col = data["time_col"]
    event_col = data["event_col"]

    cox = fit_cox_model(df, features, time_col, event_col, seed=seed)

    # Kaplan-Meier by treatment group
    control = df[df["treatment"] == 0]
    treatment = df[df["treatment"] == 1]
    km_control = kaplan_meier(control[time_col].values, control[event_col].values)
    km_treatment = kaplan_meier(treatment[time_col].values, treatment[event_col].values)

    # Log-rank test between treatment groups
    lr = logrank_test(
        control[time_col].values, control[event_col].values,
        treatment[time_col].values, treatment[event_col].values,
    )

    model = {"cox_model": cox, "km_control": km_control, "km_treatment": km_treatment}
    metrics = {
        "n_samples": data["n_samples"],
        "event_rate": data["event_rate"],
        "c_index": cox["c_index"],
        "hazard_ratios": dict(zip(features, cox["hazard_ratios"].tolist())),
        "logrank_chi2": lr["chi2"],
        "logrank_p_value": lr["p_value"],
        "median_survival": data["median_survival"],
        "treatment_hazard_ratio": float(np.exp(cox["beta"][features.index("treatment")])),
    }
    return model, metrics