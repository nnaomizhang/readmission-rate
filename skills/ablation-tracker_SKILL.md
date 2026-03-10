---
name: ablation-tracker
description: |
  Structured ablation study and reproducibility tracker for ML experiments. Use this
  skill whenever you need to compare models, justify model selection with evidence,
  track the incremental effect of pipeline changes, or prove an improvement is
  statistically real rather than noise. Trigger on phrases like: "ablation study",
  "compare models", "track experiments", "is this improvement real?", "justify model
  choice", "reproduce results", "model selection", "compare configurations",
  "which model is best", "did feature X help?". Also trigger proactively before
  any model selection decision is finalised — every model choice must be backed by
  an ablation entry, not just narrative claims. Designed for tabular/clinical ML
  projects including hospital readmission prediction.
---

# Ablation Tracker Skill

## Purpose
Enforce disciplined, reproducible model comparison. Every configuration change —
a new feature set, a different model class, a hyperparameter change — must be
registered in the ablation table before it is accepted or rejected.

The key principle: **an improvement is only real if it (a) moves a pre-specified
primary metric by more than its confidence interval, and (b) is logged with a
fixed random seed so it can be reproduced.**

---

## What Is an Ablation Study?
An ablation study systematically removes or adds one component at a time to measure
its isolated effect. In ML: start from a baseline, change one thing, measure the
delta, record it. This prevents the "cumulative luck" problem — where many small
random improvements compound into a result that doesn't replicate.

---

## Ablation Table Schema

Each experiment run gets one row. The table lives in `ablation_register.csv`
in the project root.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Short unique identifier, e.g., `"run_003"` |
| `timestamp` | str | ISO-8601, auto-filled |
| `description` | str | One sentence: what changed vs. previous run |
| `model_class` | str | e.g., `"LogisticRegression"`, `"XGBClassifier"` |
| `features` | str | Feature set label, e.g., `"baseline"`, `"+ CCS codes"` |
| `hyperparams` | str | JSON string of key hyperparameters |
| `random_seed` | int | Fixed seed used for split + model init |
| `cv_roc_auc_mean` | float | Cross-val ROC-AUC (mean) |
| `cv_roc_auc_std` | float | Cross-val ROC-AUC (std — used to assess noise) |
| `val_roc_auc` | float | Hold-out validation ROC-AUC |
| `val_f1_positive` | float | F1 for the positive class (readmitted) |
| `val_precision` | float | Precision on positive class |
| `val_recall` | float | Recall on positive class |
| `brier_score` | float | Calibration quality (lower = better) |
| `accepted` | bool | Was this run accepted into the final pipeline? |
| `rejection_reason` | str | If rejected: why (e.g., "no sig. improvement", "leakage risk") |
| `agent_suggested` | bool | True if this configuration was proposed by an AI agent |
| `human_verified` | bool | True if human reviewed and validated the result |

---

## Core Functions

### Log a Run
```python
import json, csv, os
from datetime import datetime

ABLATION_PATH = "ablation_register.csv"
COLUMNS = [
    "run_id", "timestamp", "description", "model_class", "features",
    "hyperparams", "random_seed", "cv_roc_auc_mean", "cv_roc_auc_std",
    "val_roc_auc", "val_f1_positive", "val_precision", "val_recall",
    "brier_score", "accepted", "rejection_reason", "agent_suggested", "human_verified"
]

def log_run(run_id, description, model_class, features, hyperparams,
            random_seed, cv_roc_auc_mean, cv_roc_auc_std,
            val_roc_auc, val_f1_positive, val_precision, val_recall,
            brier_score, accepted=None, rejection_reason="",
            agent_suggested=False, human_verified=False):
    """
    Append one row to ablation_register.csv.
    accepted=None means 'pending human review'.
    """
    file_exists = os.path.exists(ABLATION_PATH)
    row = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "description": description,
        "model_class": model_class,
        "features": features,
        "hyperparams": json.dumps(hyperparams) if isinstance(hyperparams, dict) else hyperparams,
        "random_seed": random_seed,
        "cv_roc_auc_mean": round(cv_roc_auc_mean, 4),
        "cv_roc_auc_std": round(cv_roc_auc_std, 4),
        "val_roc_auc": round(val_roc_auc, 4),
        "val_f1_positive": round(val_f1_positive, 4),
        "val_precision": round(val_precision, 4),
        "val_recall": round(val_recall, 4),
        "brier_score": round(brier_score, 4),
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "agent_suggested": agent_suggested,
        "human_verified": human_verified,
    }
    with open(ABLATION_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[LOGGED] {run_id}: val_roc_auc={row['val_roc_auc']}, "
          f"cv_mean={row['cv_roc_auc_mean']} ± {row['cv_roc_auc_std']}")
```

---

### Check If Improvement Is Real (Noise Test)
**Statistical logic:** A new model's CV ROC-AUC must exceed the baseline by more than
one standard deviation of the baseline's CV distribution to be considered a real
improvement. If delta < 1 std, the result is within noise and should not be accepted.

```python
def is_improvement_significant(baseline_mean, baseline_std,
                                 new_mean, multiplier=1.0):
    """
    Returns True if the new model's mean AUC exceeds the baseline mean
    by at least `multiplier` * baseline_std.

    Conservative default: multiplier=1.0 (one std). 
    For high-stakes decisions, use multiplier=2.0.
    """
    threshold = baseline_mean + multiplier * baseline_std
    delta     = new_mean - baseline_mean
    result    = new_mean > threshold
    print(f"Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"New:      {new_mean:.4f}")
    print(f"Delta:    {delta:+.4f} | Threshold: {multiplier}×std = {multiplier*baseline_std:.4f}")
    print(f"Decision: {'SIGNIFICANT IMPROVEMENT' if result else 'WITHIN NOISE — do not accept'}")
    return result
```

---

### Print Ablation Summary Table
```python
def print_ablation_summary(path=ABLATION_PATH):
    """
    Pretty-print the ablation register sorted by val_roc_auc descending.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = ["run_id", "description", "model_class", "cv_roc_auc_mean",
            "cv_roc_auc_std", "val_roc_auc", "brier_score", "accepted", "agent_suggested"]
    print(df[cols].sort_values("val_roc_auc", ascending=False).to_string(index=False))
    return df
```

---

### Reproducibility Seed Fixture
**Statistical logic:** Without a fixed random seed, two runs of the same code can
produce different results due to non-determinism in train/test shuffling, model
weight initialisation, or XGBoost internal parallelism. Always set seeds globally.

```python
def set_global_seeds(seed=42):
    """
    Set seeds for Python, NumPy, and common ML libraries.
    Call this at the very top of your notebook/script.
    """
    import random, numpy as np, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    print(f"[SEEDS] Global random seeds set to {seed}")
```

---

## Required Ablation Sequence for This Project

Before selecting a final model, the ablation register **must** contain entries for:

1. `run_001` — Baseline: Logistic Regression, raw features, no tuning
2. `run_002` — Logistic Regression + CCS ICD mapping applied
3. `run_003` — XGBoost baseline (same feature set as best LR)
4. `run_004` — MLP baseline (same feature set)
5. `run_005` — Best model with threshold optimisation (F1/recall tuned)
6. `run_006` — Best model with calibration (Platt or isotonic)

Each step must clear the noise test (`is_improvement_significant`) before proceeding
to the next. **Do not skip steps.** This is the audit trail.

---

## Usage in Notebook

```python
from ablation_tracker import log_run, is_improvement_significant, set_global_seeds

# At the top of the notebook:
set_global_seeds(42)

# After each model evaluation:
log_run(
    run_id            = "run_001",
    description       = "Baseline LogReg, raw features, no class weighting",
    model_class       = "LogisticRegression",
    features          = "baseline_v1",
    hyperparams       = {"C": 1.0, "max_iter": 1000, "class_weight": None},
    random_seed       = 42,
    cv_roc_auc_mean   = 0.6412,
    cv_roc_auc_std    = 0.0083,
    val_roc_auc       = 0.6389,
    val_f1_positive   = 0.2941,
    val_precision     = 0.3100,
    val_recall        = 0.2800,
    brier_score       = 0.1723,
    accepted          = True,
    agent_suggested   = False,
    human_verified    = True,
)
```

---

## Agent Transparency Rule
If an experiment was proposed by an AI agent (e.g., "try C=0.01" or "add dropout"),
set `agent_suggested=True`. The entry is only accepted when `human_verified=True`.
An agent-suggested run that has not been human-verified **must not be used** as
justification for model selection in the final report.

---

## References
- Melis, G. et al. (2018). On the State of the Art of Evaluation in NLP. *ICLR*.
  (Canonical argument that many claimed improvements are within noise.)
- Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
- Scikit-learn: `cross_val_score`, `StratifiedKFold`.
