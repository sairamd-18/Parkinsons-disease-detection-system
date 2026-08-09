# stacked_model_training.py

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def main():

    print("Loading datasets...")

    train_df = pd.read_csv("train_shap_50.csv")
    test_df = pd.read_csv("test_shap_50.csv")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    # -----------------------------
    # Base Models
    # -----------------------------

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    lgbm = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=-1,
        random_state=42
    )

    # -----------------------------
    # Stacking Model
    # -----------------------------

    stack_model = StackingClassifier(

        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("lgbm", lgbm)
        ],

        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
        passthrough=False

    )

    print("\nTraining Stacked Model...")

    stack_model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------

    train_preds = stack_model.predict(X_train)
    test_preds = stack_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    probs = stack_model.predict_proba(X_test)[:,1]

    roc_auc = roc_auc_score(y_test, probs)

    print("\nTrain Accuracy:", round(train_acc,4))
    print("Test Accuracy :", round(test_acc,4))
    print("ROC-AUC       :", round(roc_auc,4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, test_preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    main()