# random_forest_tuning.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def main():

    print("Loading selected feature datasets...")

    train_df = pd.read_csv("train_selected_100.csv")
    test_df = pd.read_csv("test_selected_100.csv")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # Controlled parameter space to avoid overfitting
    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [5, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
        "bootstrap": [True]
    }

    print("\nStarting Random Forest Hyperparameter Tuning...")

    random_search = RandomizedSearchCV(
        estimator=rf,
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

    final_model = random_search.best_estimator_

    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)

    print("\nTrain Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
    print("Test Accuracy :", round(accuracy_score(y_test, y_test_pred), 4))
    print("Test ROC AUC  :", round(roc_auc_score(y_test, y_test_pred), 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    main()