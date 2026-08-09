import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


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

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist"
    )

    param_dist = {
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.1, 0.2, 0.3],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "learning_rate": [0.01, 0.02, 0.03],
        "n_estimators": [500, 700, 900],
        "reg_alpha": [0.1, 0.5, 1],
        "reg_lambda": [1, 2, 5, 10]
    }

    print("Starting Hyperparameter Tuning...")

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

    final_model = random_search.best_estimator_

    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))

    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(final_model, "xgboost_parkinson_model.pkl")

    print("Model saved: xgboost_parkinson_model.pkl")


if __name__ == "__main__":
    main()