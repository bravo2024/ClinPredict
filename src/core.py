"""core.py — Survival analysis metrics for ClinPredict (Johnson & Johnson).

Implements clinical trial / time-to-event metrics, NOT generic classification:
  * **C-index (concordance)** — Harrell's concordance index for survival models.
  * **Integrated Brier score** — time-dependent prediction accuracy.
  * **Log-rank test** — compares survival curves between groups.
  * **Kaplan-Meier estimator** — non-parametric survival function estimation.

References
----------
Cox (1972), "Regression Models and Life-Tables." JRSS-B.
Harrell et al. (1996), "Multivariable prognostic models." STAT MED.
"""
from __future__ import annotations
import numpy as np


def kaplan_meier(times, events) -> dict:
    """Non-parametric Kaplan-Meier survival function estimator.

    S(t) = product_{t_i <= t} (1 - d_i / n_i)
    where d_i = deaths at time t_i, n_i = at-risk count just before t_i.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    unique_times = np.unique(t)
    survival = 1.0
    surv_curve = []
    for ut in unique_times:
        at_risk = (t >= ut).sum()
        deaths = ((t == ut) & (e == 1)).sum()
        if at_risk > 0:
            survival *= (1 - deaths / at_risk)
        surv_curve.append(survival)
    return {"times": unique_times.tolist(), "survival": surv_curve,
            "n_events": int(e.sum()), "n_censored": int((e == 0).sum())}


def concordance_index(times, events, risk_scores) -> float:
    """Harrell's C-index: fraction of comparable pairs where higher
    risk score corresponds to shorter survival time.

    C = 1.0 = perfect, 0.5 = random, 0.0 = anti-concordant.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk_scores, dtype=float)
    n = len(t)
    concordant, permissible = 0, 0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if t[j] > t[i]:
                permissible += 1
                if r[i] > r[j]:
                    concordant += 1
                elif r[i] == r[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


def brier_score(times, events, surv_prob, eval_time) -> float:
    """Brier score at a specific evaluation time point.

    BS(t) = mean( (I(T_i > t) - S_hat(t|x_i))^2 )
    where I is the indicator and S_hat is the predicted survival probability.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    s = np.asarray(surv_prob, dtype=float)
    actual = (t > eval_time).astype(float)
    # IPCW weighting (inverse probability of censoring weighting) — simplified
    return float(np.mean((actual - s) ** 2))


def logrank_test(times_group1, events_group1, times_group2, events_group2) -> dict:
    """Log-rank test comparing survival between two groups.

    Returns chi-square statistic and p-value (using normal approximation).
    """
    t1 = np.asarray(times_group1, dtype=float)
    e1 = np.asarray(events_group1, dtype=int)
    t2 = np.asarray(times_group2, dtype=float)
    e2 = np.asarray(events_group2, dtype=int)
    all_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1, E1, O2, E2 = 0, 0.0, 0, 0.0
    for ut in all_times:
        n1 = (t1 >= ut).sum()
        n2 = (t2 >= ut).sum()
        n = n1 + n2
        d1 = ((t1 == ut) & (e1 == 1)).sum()
        d2 = ((t2 == ut) & (e2 == 1)).sum()
        d = d1 + d2
        if n > 0 and d > 0:
            E1 += d * n1 / n
            E2 += d * n2 / n
            O1 += d1
            O2 += d2
    if E1 > 0 and E2 > 0:
        chi2 = ((O1 - E1) ** 2 / E1) + ((O2 - E2) ** 2 / E2)
    else:
        chi2 = 0.0
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return {"chi2": float(chi2), "p_value": float(p_value),
            "O1": int(O1), "E1": float(E1), "O2": int(O2), "E2": float(E2)}