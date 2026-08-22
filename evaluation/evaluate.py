"""
Independent, leak-free evaluation of the DEPLOYED credit-risk model
(~/Desktop/AI/projects/ml-project-credit-risk-modelling/artifacts/model_data.joblib)
against the exact original held-out test split (train_test_split(..., stratify=y,
test_size=0.25, random_state=42) on the merged customers+loans+bureau data),
reproduced from raw CSVs since the notebook's own recent output is stale/uncommitted.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Raw customers.csv / loans.csv / bureau_data.csv (codebasics course dataset) are not
# bundled in this repo. Point RAW at a local copy to reproduce this evaluation.
RAW = "./raw_data"
ARTIFACT = "../artifacts/model_data.joblib"

customers = pd.read_csv(f"{RAW}/customers.csv")
loans = pd.read_csv(f"{RAW}/loans.csv")
bureau = pd.read_csv(f"{RAW}/bureau_data.csv")

df = pd.merge(customers, loans, on="cust_id")
df = pd.merge(df, bureau, on="cust_id")

X = df.drop("default", axis="columns")
y = df["default"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)
df_train = pd.concat([X_train, y_train], axis="columns")
df_test = pd.concat([X_test, y_test], axis="columns")

# --- cleaning, replicated exactly from the training notebook ---
mode_residence = df_train["residence_type"].mode()[0]
df_test["residence_type"] = df_test["residence_type"].fillna(mode_residence)
df_test["loan_purpose"] = df_test["loan_purpose"].replace("Personaal", "Personal")
df_test = df_test[df_test.processing_fee / df_test.loan_amount < 0.03].copy()

# --- derived features, replicated exactly from the training notebook ---
df_test["loan_to_income"] = round(df_test["loan_amount"] / df_test["income"], 2)
df_test["delinquency_ratio"] = (
    df_test["delinquent_months"] * 100 / df_test["total_loan_months"]
).round(1)
df_test["avg_dpd_per_delinquency"] = np.where(
    df_test["delinquent_months"] != 0,
    (df_test["total_dpd"] / df_test["delinquent_months"]).round(1),
    0,
)

model_data = joblib.load(ARTIFACT)
model = model_data["model"]
scaler = model_data["scaler"]
features = list(model_data["features"])
cols_to_scale = list(model_data["cols_to_scale"])

encoded = pd.get_dummies(
    df_test[["residence_type", "loan_purpose", "loan_type"]], drop_first=True, dtype=int
)
df_eval = pd.concat([df_test.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)

missing_scale_cols = [c for c in cols_to_scale if c not in df_eval.columns]
assert not missing_scale_cols, f"missing cols_to_scale columns: {missing_scale_cols}"
df_eval[cols_to_scale] = scaler.transform(df_eval[cols_to_scale])

missing_features = [c for c in features if c not in df_eval.columns]
assert not missing_features, f"missing feature columns: {missing_features}"
X_eval = df_eval[features]
y_eval = df_eval["default"].astype(int)

y_pred = model.predict(X_eval)
y_prob = model.predict_proba(X_eval)[:, 1]

print(f"held-out test rows: {len(y_eval)} (positive rate: {y_eval.mean():.3f})")
print()
print(classification_report(y_eval, y_pred, digits=3))
print("confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_eval, y_pred))
auc = roc_auc_score(y_eval, y_prob)
print(f"\nROC-AUC: {auc:.4f}")
print(f"Gini: {2*auc-1:.4f}")
