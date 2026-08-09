# logistic_corrected.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# ============================
# LOAD DATA
# ============================
print("Loading processed features...")
df = pd.read_csv("processed_audio_features.csv")

# Remove non-numeric columns except label
X = df.drop(columns=["label", "AudioPath"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["label"]

print("Feature count:", X.shape[1])
print("Total samples:", X.shape[0])

# Check NaN count
print("Total NaN values in dataset:", X.isna().sum().sum())


# ============================
# TRAIN-TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# HANDLE MISSING VALUES (IMPORTANT FIX)
# ============================
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ============================
# SCALING (IMPORTANT FOR LR)
# ============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================
# BASELINE LOGISTIC REGRESSION
# ============================
print("\nTraining Baseline Logistic Regression...")

log_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = log_model.predict(X_train_scaled)
y_test_pred = log_model.predict(X_test_scaled)
y_test_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Baseline Results ---")
print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
print("Test ROC-AUC  :", round(roc_auc_score(y_test, y_test_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# ============================
# HYPERPARAMETER TUNING
# ============================
print("\nTuning Logistic Regression...")

param_grid = {
    "C": [0.01, 0.1, 1, 5, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    LogisticRegression(max_iter=3000, class_weight="balanced"),
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

best_model = grid.best_estimator_

y_test_pred = best_model.predict(X_test_scaled)
y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Tuned Logistic Regression ---")
print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
print("Test ROC-AUC  :", round(roc_auc_score(y_test, y_test_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))