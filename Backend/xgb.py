import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from xgboost import XGBClassifier


DATA_FILE = "selected_features_dataset.csv"


print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ------------------------------------------------
# Clean feature names (OpenSMILE compatibility)
# ------------------------------------------------
df.columns = df.columns.str.replace('[\\[\\]<>]', '', regex=True)
df.columns = df.columns.str.replace(" ", "_")


# ------------------------------------------------
# Target variable
# ------------------------------------------------
y = df["label"]


# ------------------------------------------------
# Patient groups for splitting
# ------------------------------------------------
groups = df["patient_id"]


# ------------------------------------------------
# Feature matrix (remove non-numeric columns)
# ------------------------------------------------
drop_cols = ["label", "patient_id", "task"]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

print("Feature count:", X.shape[1])


# ------------------------------------------------
# Standardize features
# ------------------------------------------------
scaler = StandardScaler()

X = scaler.fit_transform(X)


# ------------------------------------------------
# Handle class imbalance
# ------------------------------------------------
pos = np.sum(y == 1)
neg = np.sum(y == 0)

scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)


# ------------------------------------------------
# XGBoost model (tuned parameters)
# ------------------------------------------------
model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)


# ------------------------------------------------
# Group K-Fold cross validation
# ------------------------------------------------
gkf = GroupKFold(n_splits=5)

accuracies = []
f1_scores = []
auc_scores = []


for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

    print("\nFold:", fold + 1)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    y_pred = (y_prob > 0.45).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    accuracies.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------
# Final results
# ------------------------------------------------
print("\n========== FINAL RESULTS ==========")

print("Average Accuracy:", np.mean(accuracies))
print("Average F1 Score:", np.mean(f1_scores))
print("Average ROC-AUC:", np.mean(auc_scores))