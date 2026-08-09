import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

from xgboost import XGBClassifier


def main():

    print("Loading selected feature datasets...")

    train_df = pd.read_csv("train_selected_100.csv")
    test_df  = pd.read_csv("test_selected_100.csv")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    # ---------------------------------------------------
    # Cross Validation Setup
    # ---------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Base model (no early stopping here)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    # Safer parameter space (reduce overfitting)
    param_dist = {
        "max_depth": [2, 3],
        "min_child_weight": [5, 7],
        "gamma": [0.3, 0.5, 0.7],
        "subsample": [0.6, 0.7],
        "colsample_bytree": [0.5, 0.6],
        "learning_rate": [0.01, 0.02],
        "n_estimators": [300, 400, 500],
        "reg_alpha": [1, 5, 10],
        "reg_lambda": [5, 10, 20]
    }

    print("\nStarting Hyperparameter Tuning...")

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest CV ROC-AUC:")
    print(round(random_search.best_score_, 4))

    # ---------------------------------------------------
    # Validation split (IMPORTANT: no test leakage)
    # ---------------------------------------------------
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # ---------------------------------------------------
    # Final Model (XGBoost 2.x compatible early stopping)
    # ---------------------------------------------------
    final_model = XGBClassifier(
        **random_search.best_params_,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    final_model.fit(
        X_train_part,
        y_train_part,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------
    y_train_pred = final_model.predict(X_train)
    y_test_pred  = final_model.predict(X_test)

    y_train_prob = final_model.predict_proba(X_train)[:, 1]
    y_val_prob   = final_model.predict_proba(X_val)[:, 1]
    y_test_prob  = final_model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    print("\n--- Final Model Performance ---")

    print("\nTrain Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
    print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))

    print("\nTrain ROC AUC :", round(roc_auc_score(y_train, y_train_prob), 4))
    print("Val   ROC AUC :", round(roc_auc_score(y_val, y_val_prob), 4))
    print("Test  ROC AUC :", round(roc_auc_score(y_test, y_test_prob), 4))

    print("\nClassification Report (Test):\n")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion Matrix (Test):\n")
    print(confusion_matrix(y_test, y_test_pred))

    # ---------------------------------------------------
    # Threshold Tuning (Optional but powerful)
    # ---------------------------------------------------
    print("\nSearching Best Threshold for F1 on Validation Set...")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.3, 0.7, 0.01):
        temp_pred = (y_val_prob > threshold).astype(int)
        score = f1_score(y_val, temp_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print("Best Threshold:", round(best_threshold, 3))
    print("Best Validation F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()