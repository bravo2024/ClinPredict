# ClinPredict

> Clinical trial outcome prediction using biomarker and trial design features.

Trains four classifiers on synthetic clinical trial data to predict trial success based on biomarker expression, patient demographics, phase, trial duration, and adverse event rates. Dashboard explores trial design trade-offs and biomarker significance.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.731 |
| Gini | 0.463 |
| KS Statistic | 0.477 |
| F1 Score | 0.493 |
| Accuracy | 0.704 |

5-fold CV AUC: 0.725 ± 0.029. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Trial records overview, success/failure distribution, feature descriptions |
| **Model Lab** | Multi-model comparison table, ROC curves, calibration plots, CV results |
| **Biomarkers** | Biomarker expression distributions by outcome, feature importance analysis |
| **Trial Design** | Phase-level success rates, duration vs outcome analysis, adverse event patterns |

## Repo Structure

```
ClinPredict/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic clinical trial dataset: biomarker expression levels, patient demographics, trial phase, duration, enrolment size, adverse event rate, and endpoint achievement.

## License

MIT
