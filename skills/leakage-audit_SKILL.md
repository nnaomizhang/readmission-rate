---
name: leakage-audit
description: |
  Systematic data leakage audit for ML pipelines. Use this skill whenever you are
  building or reviewing a machine learning pipeline and need to check for data leakage,
  target leakage, temporal leakage, or preprocessing leakage. Trigger this skill for
  any of these phrases or contexts: "check for leakage", "leakage audit", "verify my
  split", "preprocessing pipeline review", "train/test contamination", "feature leakage",
  "is this leaking?", "validate my pipeline". Also trigger proactively whenever a
  pipeline is about to be finalised or a new feature is proposed, even if leakage
  is not explicitly mentioned. This skill is critical for hospital readmission and any
  medical/tabular ML project where patient-level grouping, ICD code mapping, or
  imputation is involved.
---

# Leakage Audit Skill

## Purpose
Provide a structured, repeatable checklist and code scaffold to detect and document
all major forms of data leakage in a supervised ML pipeline. Outputs a human-readable
audit report and raises exceptions on hard failures.

---

## Leakage Taxonomy (know these before auditing)

| Type | Definition | Common Cause |
|------|-----------|--------------|
| **Target leakage** | Feature computed using or correlated with the label after the fact | Including discharge diagnosis or readmission flag as a feature |
| **Temporal leakage** | Future information bleeds into training data | Sorting by date before splitting; using a running aggregate across the split boundary |
| **Group leakage** | Same entity (patient, customer) appears in both train and test | Row-level split when data has repeated IDs |
| **Preprocessing leakage** | Statistics from test set influence train preprocessing | Fitting a scaler or imputer on the full dataset before splitting |
| **Label encoding leakage** | Encoding derived from full-dataset label frequencies | Target encoding fitted on all rows |

---

## Audit Protocol (run in this order)

### Step 1 — Group Integrity Check
**Statistical logic:** If the same patient ID appears in both train and test, the model
can memorise patient-specific patterns. This inflates validation metrics and produces
a model that cannot generalise to new patients.

```python
def audit_group_leakage(train_ids, test_ids, id_col="patient_nbr"):
    """
    Verify zero overlap between train and test patient IDs.
    Raises AssertionError if overlap exists.
    """
    train_set = set(train_ids)
    test_set  = set(test_ids)
    overlap   = train_set & test_set
    assert len(overlap) == 0, (
        f"GROUP LEAKAGE: {len(overlap)} patient IDs appear in both splits. "
        f"Example IDs: {list(overlap)[:5]}"
    )
    print(f"[PASS] Group leakage: 0 overlapping IDs | "
          f"{len(train_set)} train patients / {len(test_set)} test patients.")
    return {"status": "PASS", "overlap_count": 0}
```

**Verification:** Run assertion. If it raises, fix by using `GroupShuffleSplit`
(sklearn) with `groups=df[id_col]`.

---

### Step 2 — Preprocessing Fit-on-Train-Only Check
**Statistical logic:** Scalers, imputers, and encoders estimate distribution parameters
(mean, std, mode, quantiles). If these are estimated on the full dataset (including test),
the model has indirect access to test-set statistics — a subtle but real form of leakage.

```python
def audit_preprocessing_leakage(pipeline, X_train, X_test):
    """
    Confirm the pipeline was fitted only on X_train by checking that
    test-set statistics do not match train+test combined statistics.
    Heuristic: if scaler mean ~ combined mean (delta < 1e-6), leakage is likely.
    """
    import numpy as np
    fitted_means = {}
    for name, step in pipeline.named_steps.items():
        if hasattr(step, "mean_"):  # StandardScaler, SimpleImputer (mean strategy)
            fitted_means[name] = step.mean_.copy()

    if not fitted_means:
        print("[INFO] No mean-based transformers found to audit.")
        return

    X_combined = np.vstack([X_train, X_test])
    for name, train_mean in fitted_means.items():
        combined_mean = X_combined.mean(axis=0)
        delta = np.abs(train_mean - combined_mean).mean()
        if delta < 1e-6:
            raise ValueError(
                f"PREPROCESSING LEAKAGE RISK: Transformer '{name}' mean is nearly "
                f"identical to the combined train+test mean. Was it fitted on all data?"
            )
        print(f"[PASS] '{name}': mean delta vs combined = {delta:.6f} (non-zero — fitted on train only)")
```

**Verification:** Non-zero delta confirms the scaler/imputer saw only train data.

---

### Step 3 — Target Leakage Feature Scan
**Statistical logic:** Features that are near-perfect univariate predictors (AUC > 0.85
alone) are almost certainly post-outcome features and must be excluded. We detect this
via single-feature AUC screening.

```python
def audit_target_leakage(X_train, y_train, feature_names, auc_threshold=0.85):
    """
    Flag any feature whose individual AUC against the target exceeds auc_threshold.
    High individual AUC is a strong signal of target leakage.
    """
    from sklearn.metrics import roc_auc_score
    leaky = []
    for i, col in enumerate(feature_names):
        try:
            auc = roc_auc_score(y_train, X_train[:, i])
            auc = max(auc, 1 - auc)  # handle inverse predictors
            if auc >= auc_threshold:
                leaky.append((col, round(auc, 4)))
        except Exception:
            pass
    if leaky:
        print(f"[WARN] TARGET LEAKAGE CANDIDATES (single-feature AUC >= {auc_threshold}):")
        for col, auc in sorted(leaky, key=lambda x: -x[1]):
            print(f"  - {col}: AUC = {auc}")
    else:
        print(f"[PASS] No single feature exceeds AUC threshold of {auc_threshold}.")
    return leaky
```

**Diabetes-dataset watch-list:**
- `discharge_disposition_id` — encodes death/hospice (post-outcome information)
- `num_lab_procedures`, `num_medications` — may be correlated but are pre-outcome
- Any column that is a direct encoding of the readmission label itself

---

### Step 4 — ICD Code Mapping Leakage Check (Diabetes-dataset specific)
**Statistical logic:** When mapping ICD-9 codes to CCS categories, the mapping must
be applied identically to train and test without any label-informed grouping.
If CCS categories were selected by their association with the readmission label,
that constitutes target leakage.

```python
def audit_icd_mapping(mapping_df, train_df, test_df, icd_col="diag_1"):
    """
    Verify:
    1. All ICD codes in test are covered by the mapping (no silent NaN injection).
    2. Mapping was not derived from label frequencies.
    """
    train_codes  = set(train_df[icd_col].dropna().astype(str).unique())
    test_codes   = set(test_df[icd_col].dropna().astype(str).unique())
    mapped_codes = set(mapping_df.iloc[:, 0].astype(str).unique())

    unseen_in_train = test_codes - train_codes
    unmapped_test   = test_codes - mapped_codes

    print(f"[INFO] ICD codes in test not seen in train: {len(unseen_in_train)}")
    if unmapped_test:
        print(f"[WARN] {len(unmapped_test)} test ICD codes not in mapping — will become NaN.")
        print(f"       Examples: {list(unmapped_test)[:5]}")
    else:
        print("[PASS] All test ICD codes are covered by the mapping table.")

    return {"unseen_in_train": len(unseen_in_train), "unmapped_test": len(unmapped_test)}
```

---

### Step 5 — Full Audit Runner
```python
def run_full_leakage_audit(pipeline, X_train, X_test, y_train,
                            train_ids, test_ids, feature_names,
                            mapping_df=None, train_df=None, test_df=None):
    """
    Master function. Run all leakage checks and return a summary dict.
    """
    results = {}
    print("=" * 60)
    print("LEAKAGE AUDIT REPORT")
    print("=" * 60)

    print("\n[1/4] Group Leakage Check")
    results["group"] = audit_group_leakage(train_ids, test_ids)

    print("\n[2/4] Preprocessing Leakage Check")
    audit_preprocessing_leakage(pipeline, X_train, X_test)
    results["preprocessing"] = "checked"

    print("\n[3/4] Target Leakage Feature Scan")
    results["target_leaky"] = audit_target_leakage(X_train, y_train, feature_names)

    if mapping_df is not None:
        print("\n[4/4] ICD Mapping Leakage Check")
        results["icd_mapping"] = audit_icd_mapping(mapping_df, train_df, test_df)
    else:
        print("\n[4/4] ICD Mapping Check: skipped (no mapping_df provided)")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE — Review all [WARN] entries before proceeding.")
    return results
```

---

## Usage in Notebook
```python
# At the end of your preprocessing cell, call:
from leakage_audit import run_full_leakage_audit

audit_results = run_full_leakage_audit(
    pipeline      = preprocessor,       # sklearn Pipeline, fitted on X_train only
    X_train       = X_train_arr,
    X_test        = X_test_arr,
    y_train       = y_train,
    train_ids     = train_df["patient_nbr"],
    test_ids      = test_df["patient_nbr"],
    feature_names = feature_names,
    mapping_df    = icd_mapping_df,     # from IDS_mapping.csv
    train_df      = train_df,
    test_df       = test_df,
)
```

---

## Decision Register Notes
When this audit is run, log the following in the Decision Register table:
- Date and pipeline version audited
- Result of each check (PASS / WARN / FAIL)
- Action taken on any WARN/FAIL (e.g., "Removed discharge_disposition_id — post-outcome")
- Human validator sign-off (not agent)

---

## References
- Kaufman, S. et al. (2012). Leakage in Data Mining. *ACM TKDD*, 6(4).
- Scikit-learn documentation: `GroupShuffleSplit`, `Pipeline`.
- Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*.
