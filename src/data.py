"""data.py — Synthetic clinical trial data for ClinPredict (Johnson & Johnson).

Patient-level data with time-to-event, event indicator (1=event, 0=censored),
and clinical covariates (age, biomarker, treatment arm, disease stage).
This mirrors the structure of real oncology clinical trial data.

The outcome is time-to-event (survival), NOT a binary classification target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def make_synthetic(n: int = 500, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic clinical trial data with survival outcomes.

    The treatment arm reduces hazard by ~40%. Higher biomarker values and
    advanced stage increase hazard. Age has a moderate effect.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(30, 80, n).astype(float)
    biomarker = rng.normal(50, 15, n).clip(0, 100).round(1)
    treatment = rng.choice([0, 1], n, p=[0.5, 0.5])  # 0=control, 1=treatment
    stage = rng.choice([1, 2, 3, 4], n, p=[0.30, 0.30, 0.25, 0.15])

    # Hazard (exponential): h = h0 * exp(beta * X)
    log_hazard = (
        -4.0
        + 0.03 * age
        + 0.04 * biomarker
        - 0.50 * treatment  # treatment reduces hazard
        + 0.60 * stage
        + rng.normal(0, 0.3, n)
    )
    hazard = np.exp(log_hazard)
    # Exponential survival: T ~ Exponential(hazard)
    time_to_event = rng.exponential(1.0 / hazard).round(1)
    # Clip to reasonable range (days)
    time_to_event = np.clip(time_to_event, 1, 365 * 5)

    # Random censoring (20% censored)
    censor_time = rng.uniform(0, time_to_event.max() * 0.8, n)
    observed_time = np.minimum(time_to_event, censor_time)
    event = (time_to_event <= censor_time).astype(int)

    df = pd.DataFrame({
        "age": age, "biomarker": biomarker, "treatment": treatment,
        "stage": stage, "time": observed_time, "event": event,
    })

    return {
        "df": df,
        "features": ["age", "biomarker", "treatment", "stage"],
        "categorical_features": ["treatment", "stage"],
        "numerical_features": ["age", "biomarker"],
        "time_col": "time",
        "event_col": "event",
        "n_samples": n,
        "event_rate": float(event.mean()),
        "median_survival": float(np.median(observed_time)),
    }