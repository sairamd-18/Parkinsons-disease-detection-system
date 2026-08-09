# preprocess_and_select_shap50.py

import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier


def main():

    print("Loading dataset...")
    df = pd.read_csv("../audio_features.csv")

    # ----------------------------------
    # STEP 1: Create Label
    # ----------------------------------
    def extract_label(path):
        return 1 if "PD" in path else 0

    df["label"] = df["AudioPath"].apply(extract_label)

    X = df.drop(columns=["AudioPath", "label"])
    y = df["label"]

    print("Initial Feature Count:", X.shape[1])

    # ----------------------------------
    # STEP 2: Remove Low Variance Features
    # ----------------------------------
    print("\nRemoving low variance features...")

    selector = VarianceThreshold(threshold=0.0001)
    X = selector.fit_transform(X)

    X = pd.DataFrame(X)

    print("After Variance Threshold:", X.shape[1])

    # ----------------------------------
    # STEP 3: Remove Highly Correlated Features
    # ----------------------------------
    print("\nRemoving highly correlated features...")

    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    X = X.drop(columns=to_drop)

    print("After Correlation Filtering:", X.shape[1])

    # ----------------------------------
    # STEP 4: Train Test Split
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    # ----------------------------------
    # STEP 5: Standard Scaling
    # ----------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------
    # STEP 6: Train Temporary XGBoost
    # ----------------------------------
    print("\nTraining temporary XGBoost model...")

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

    # ----------------------------------
    # STEP 7: SHAP Feature Importance
    # ----------------------------------
    print("\nCalculating SHAP values...")

    booster = temp_model.get_booster()

    explainer = shap.TreeExplainer(booster)

    # Use sample to speed up SHAP
    sample = X_train_scaled[:1000]

    shap_values = explainer.shap_values(sample)

    shap_importance = np.abs(shap_values).mean(axis=0)

    # ----------------------------------
    # STEP 8: Select Top 50 Features
    # ----------------------------------
    top_k = 50

    indices = np.argsort(shap_importance)[::-1][:top_k]

    X_train_selected = X_train_scaled[:, indices]
    X_test_selected = X_test_scaled[:, indices]

    print("\nFinal Selected Feature Count:", X_train_selected.shape[1])

    # ----------------------------------
    # STEP 9: Save Dataset
    # ----------------------------------
    train_df = pd.DataFrame(X_train_selected)
    train_df["label"] = y_train.values

    test_df = pd.DataFrame(X_test_selected)
    test_df["label"] = y_test.values

    train_df.to_csv("train_shap_50.csv", index=False)
    test_df.to_csv("test_shap_50.csv", index=False)

    print("\nDatasets saved:")
    print("train_shap_50.csv")
    print("test_shap_50.csv")


if __name__ == "__main__":
    main()