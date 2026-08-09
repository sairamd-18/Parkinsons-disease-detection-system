# rf_top50_baseline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# ============================
# LOAD DATA
# ============================
print("Loading processed features...")
df = pd.read_csv("processed_audio_features.csv")

X = df.drop(columns=["label", "AudioPath"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["label"]

print("Original feature count:", X.shape[1])
print("Total samples:", X.shape[0])


# ============================
# TRAIN–TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# HANDLE NaN
# ============================
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ============================
# SCALING (not mandatory for RF, but keeps consistency)
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================
# FEATURE SELECTION (TOP 50 using Random Forest importance)
# IMPORTANT: Fit on TRAIN only
# ============================
print("\nSelecting top 50 features using RF importance...")

temp_rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

temp_rf.fit(X_train, y_train)

importances = temp_rf.feature_importances_
top_k = 50
indices = np.argsort(importances)[::-1][:top_k]

X_train_selected = X_train[:, indices]
X_test_selected = X_test[:, indices]

print("Selected feature count:", X_train_selected.shape[1])


# ============================
# BASELINE RANDOM FOREST
# ============================
print("\nTraining Baseline Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_selected, y_train)

# TRAIN results
y_train_pred = rf.predict(X_train_selected)
y_train_prob = rf.predict_proba(X_train_selected)[:, 1]

# TEST results
y_test_pred = rf.predict(X_test_selected)
y_test_prob = rf.predict_proba(X_test_selected)[:, 1]

print("\n--- RANDOM FOREST BASELINE RESULTS ---")

print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
print("Train ROC-AUC :", round(roc_auc_score(y_train, y_train_prob), 4))

print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
print("Test ROC-AUC  :", round(roc_auc_score(y_test, y_test_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))