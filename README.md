# Analysing Insurance Claims for SafeDrive Insurance Ltd.

Predicting which motor policyholders will file a claim, and turning that prediction into an underwriting decision.

**[→ Open the interactive dashboard](https://YOUR-USERNAME.github.io/safedrive-claim-analytics/)**

![Python](https://img.shields.io/badge/Python-3.10+-1B3A55)
![scikit--learn](https://img.shields.io/badge/scikit--learn-pipeline-1B3A55)
![Optuna](https://img.shields.io/badge/Optuna-tuned-1B3A55)
![MLflow](https://img.shields.io/badge/MLflow-tracked-1B3A55)
![SHAP](https://img.shields.io/badge/SHAP-explained-1B3A55)
![Evidently](https://img.shields.io/badge/Evidently-drift%20monitored-1B3A55)

---

## The problem

SafeDrive assesses claim risk manually. The process is slow, varies between underwriters, and leaves no record of *why* a policy was treated as risky. This project replaces that judgement with a consistent, auditable risk score built on 58,592 policies.

Only 6.40% of policies result in a claim — a 14.6 : 1 imbalance. Predicting "no claim" for everyone scores 93.6% accuracy and is worthless. The real task is **ranking**: finding the small share of the book that carries a disproportionate share of the claims.

## Headline result

The model sorts held-out policies into four bands. Claim rates below are **actual outcomes** on 11,719 policies the model never saw during training.

| Risk band | Policies | Share | Actual claim rate | Lift vs portfolio |
|---|---:|---:|---:|---:|
| Low | 4,258 | 36.3% | 3.05% | 0.48× |
| Medium | 3,049 | 26.0% | 6.07% | 0.95× |
| High | 3,651 | 31.2% | 8.79% | 1.37× |
| **Very High** | **761** | **6.5%** | **14.98%** | **2.34×** |

Claim rates increase monotonically across every band — a 4.9× spread from Low to Very High. That property is asserted programmatically as a release gate: if the ordering breaks, the model is not ranking risk and is not shipped.

## What the data said

- **Policy tenure is the dominant driver.** Claim rate climbs from 3.6% to 8.7% across quintiles, and tenure carries the strongest linear signal in the data (r = +0.079), roughly three times the next feature.
- **Who the customer is beats what they drive.** Policyholder age moves claim rate from 5.6% to 7.1%. Every vehicle specification sits at |r| < 0.014.
- **Crash-safety rating carries no signal.** Claim rates are flat at 6.2%–6.7% across the whole NCAP scale, confirmed by four independent methods — flat claim rates, near-zero correlations, negligible regression coefficients, and mean |SHAP| of exactly zero.
- **Older vehicles claim less, not more.** The oldest quintile claims at 5.0% against 6.9% for the newest.
- **A textbook underwriting rule runs backwards.** "Young driver in a poorly-rated vehicle" claims at 5.7% against 6.6% for the rest of the book.

## Models

All three were fitted on identical rows, with preprocessing fitted on training data only. Selection was on test ROC-AUC, fixed in advance, because the deliverable is a ranking rather than a yes/no call.

| Model | ROC-AUC | Recall | Precision | F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression (baseline) | 0.5850 | 55.2% | 8.0% | 0.139 | 0.610 ± 0.005 |
| **Decision tree, Optuna-tuned** | **0.6464** | **65.3%** | **8.9%** | **0.157** | **0.637 ± 0.009** |
| Decision tree + Featuretools | 0.6336 | 70.1% | 8.8% | 0.156 | 0.635 ± 0.009 |

The Box-Tidwell test flagged two of six continuous predictors as non-linear in the log-odds. Rather than patching the baseline with polynomial terms, that became the rationale for the tree — which makes no linearity assumption and duly gained six points of AUC.

### Did AutoML help?

Featuretools generated **342** candidate interaction features. Each had to beat the strongest correlation any existing feature had with the target. **Three survived** — all recombinations of `policy_tenure`, all beating it in the third decimal place. SHAP shows only one is used at all; the other two rank 108th and 109th of 109 with zero contribution.

The resulting model scored slightly *worse* than the hand-engineered tree. That is a useful negative result: brute-force generation of every pairwise interaction could not beat hand-built features, which is evidence the accuracy ceiling is set by **which attributes were collected**, not by how cleverly they are combined.

## Repository

```
├── index.html                                       # interactive dashboard (GitHub Pages)
├── SafeDrive_Business_Report.docx                   # management report
├── notebooks/
│   ├── data_preprocessing_eda_feature_engineering.ipynb
│   ├── ml_model_building_and_validation.ipynb
│   └── automl_model_comparison_and_versioning.ipynb
├── data/
│   ├── cleaned_dataset.csv                          # 58,592 × 108, cleaned and encoded
│   └── prediction_output.csv                        # 11,719 scored policies
└── artifacts/
    ├── safedrive_claim_model_*.joblib               # versioned pipeline
    ├── model_metadata.json                          # training reference, metrics, MLflow run ID
    ├── shap_feature_summary.csv
    └── evidently_report.html
```

## Method

**Preparation** — IQR winsorisation rather than row removal (the outliers carried signal: the oldest policyholders claim at double the base rate); log transform on population density; engine text like `88.50bhp@6000rpm` parsed into power, torque and rpm; 17 binary flags detected by value rather than by name; 9 columns one-hot encoded.

**Feature engineering** — six derived features, each validated against the claim rate before being kept. `tenure × policyholder age` was the strongest at a 2.5× spread across quintiles.

**Feature selection** — 14 column pairs exceeded |r| > 0.9, almost all inside the vehicle-specification block (these are model-level constants, so a handful of distinct vehicles reproduce one another). Near-duplicates removed, then iterative VIF elimination to VIF < 10.

**Validation** — 80/20 stratified split with the claim rate preserved to four decimals; 5-fold stratified CV with the preprocessor inside the pipeline so leakage prevention applies fold by fold; train–test AUC gap of 0.035.

**Governance** — MLflow experiment tracking, joblib artifacts with JSON metadata, Evidently drift baseline, SHAP explanations for any individual prediction, `random_state = 42` throughout.

## Honest limitations

- **ROC-AUC 0.646 is modest.** It supports triage. It does not support automated decline or individual pricing.
- **Precision is low by design.** Roughly 11 policies are flagged for every genuine claim caught — intrinsic to a 6.4% base rate at this level of discrimination.
- **Tenure confounds risk with exposure.** A policy held longer has had more time to claim. The direction is not in doubt; the magnitude as a pure risk signal is.
- **Small clusters are noisy.** The extreme geographic rates rest on 109–492 policies and should not drive regional pricing.
- **Severity is out of scope.** The model predicts whether a claim occurs, never what it costs.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels \
            optuna mlflow shap evidently featuretools joblib
```

Run the notebooks in order — each writes the inputs the next one reads. Notebook 1 produces `cleaned_dataset.csv`; Notebook 2 produces `prediction_output.csv` and the versioned model; Notebook 3 produces the AutoML comparison and the dashboard KPIs.

The dashboard is a single self-contained `index.html` with no build step and no JavaScript dependencies. Open it directly, or enable GitHub Pages under **Settings → Pages → Deploy from branch → main / root**.

---

*Capstone project — Data Science and Generative AI. All figures are reproducible from the notebooks with the random seed fixed at 42.*
