# rf_50_features_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():

    print("Loading dataset...")
    df = pd.read_csv("../audio_features.csv")

    # -----------------------------------------
    # STEP 1: Create Label
    # -----------------------------------------
    def extract_label(path):
        return 1 if "PD" in path else 0

    df["label"] = df["AudioPath"].apply(extract_label)

    X = df.drop(columns=["AudioPath", "label"])
    y = df["label"]

    print("Original Features:", X.shape[1])

    # -----------------------------------------
    # STEP 2: Train-Test Split FIRST
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------------------
    # STEP 3: Scale (fit only on train)
    # -----------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------
    # STEP 4: Select Top 50 Features using XGBoost Importance
    # -----------------------------------------
    print("Selecting Top 50 Features...")

    temp_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05
    )

    temp_model.fit(X_train_scaled, y_train)

    importances = temp_model.feature_importances_
    top_k = 50
    indices = np.argsort(importances)[::-1][:top_k]

    X_train_selected = X_train_scaled[:, indices]
    X_test_selected = X_test_scaled[:, indices]

    print("Selected Features:", X_train_selected.shape[1])

    # =========================================
    # PART 1: BASIC RANDOM FOREST
    # =========================================
    print("\nTraining Basic Random Forest...")

    rf_basic = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    rf_basic.fit(X_train_selected, y_train)

    y_train_pred = rf_basic.predict(X_train_selected)
    y_test_pred = rf_basic.predict(X_test_selected)

    print("\n--- BASIC RANDOM FOREST RESULTS ---")
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
    print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
    print("Test ROC AUC  :", round(roc_auc_score(y_test, y_test_pred), 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_test_pred))

    # =========================================
    # PART 2: HYPERPARAMETER TUNING
    # =========================================
    print("\nStarting Random Forest Hyperparameter Tuning...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": [200, 300],
        "max_depth": [5, 8, 10],          # No None to avoid overfitting
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [3, 5, 8],
        "max_features": ["sqrt", 0.5],
        "bootstrap": [True]
    }

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train_selected, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("Best CV ROC-AUC:", round(random_search.best_score_, 4))

    final_model = random_search.best_estimator_

    y_train_pred = final_model.predict(X_train_selected)
    y_test_pred = final_model.predict(X_test_selected)

    print("\n--- TUNED RANDOM FOREST RESULTS ---")
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
    print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
    print("Test ROC AUC  :", round(roc_auc_score(y_test, y_test_pred), 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    main()