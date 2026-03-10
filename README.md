# Hospital Readmission Prediction
MSIN0097 Predictive Analytics Coursework

## Research Question
Can we predict, at the point of discharge, whether a diabetic patient will be readmitted within 30 days?

## Dataset
UCI Diabetes 130-US Hospitals (1999–2008) — 101,766 encounters, 50 features. Analytical cohort after population filter: 99,319 encounters.

**To access the data:**
1. Go to: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Download `diabetic_data.csv` and `IDS_mapping.csv`
3. Place both files in the `/data` folder

## How to Run

```bash
git clone <repo-url>
cd readmission
pip install -r requirements.txt
jupyter lab readmission.ipynb
```

Run all cells top to bottom. Fully reproducible with `RANDOM_STATE=42`.

## Skills

The `/skills` folder contains two reusable SKILL.md files encoding the methodology decisions
made in this project. To use in a new Claude session, upload the relevant file and say:
*"Follow the instructions in this SKILL.md for this project."*

## Repository Structure

```
readmission/
├── data/                         # Place dataset files here (not tracked by git)
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── skills/                       # Reusable SKILL.md files for future Claude sessions
│   ├── leakage-audit.md
│   └── ablation-reproducibility.md
├── readmission.ipynb             # Main notebook — full pipeline
├── requirements.txt
└── README.md
```

## Pipeline Overview
| Step | Description |
|---|---|
| 1 | Problem framing — leakage investigation, population filter, class weights, metric definitions |
| 2 | EDA — missingness, class imbalance, feature distributions, NZV analysis, subgroup analysis, correlation |
| 3 | Data preparation — feature definitions, grouped split, CCS diagnosis grouping, imputation, encoding, log transform, scaling, OHE, validation checks |
| 4 | Modelling — Logistic Regression, XGBoost, MLP; comparison table, McNemar significance tests, ROC curves, learning curve |
| 5 | Threshold optimisation & Final evaluation — calibration, PR curve, threshold sweep, error analysis, subgroup failure analysis|
| 6 | Final Solution|

## Key Decisions
- **Grouped split**: 80/10/10 by `patient_nbr` — no patient appears across splits
- **Class weighting**: inverse-frequency weights applied to all models
- **NZV**: 16 medication columns dropped (dominant frequency ≥98%)
- **Diagnosis grouping**: ICD-9 mapped to 18 CCS categories
- **Log transform**: applied to `number_outpatient`, `number_emergency`, `number_inpatient`
- **Lab encoding**: `A1Cresult` and `max_glu_serum` NaN encoded as `Not_Tested`
- **Leakage controls**: scaler, imputer, encoder fitted on X_train only

## Results

**Best model: Logistic Regression at t=0.40**

| Metric | Score |
|---|---|
| F2-Score | 0.4230 |
| ROC-AUC | 0.6628 (95% CI: 0.6455–0.6793) |
| Recall | 0.7939 |
| Missed Readmissions | 20.6% |

Threshold optimised on validation set only. Test set evaluated once.

## Splits

| Split | Rows | Features |
|---|---|---|
| Training | 79,568 | 136 |
| Validation | 9,888 | 136 |
| Test | 9,863 | 136 |

## Environment
Python 3.10+ — see `requirements.txt` for full dependencies.
