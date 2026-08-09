import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import numpy as np

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

    print("Original Feature Count:", X.shape[1])

    # ----------------------------------
    # STEP 2: Train-Test Split FIRST
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------------------------------
    # STEP 3: Scale Features (fit ONLY on train)
    # ----------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------
    # STEP 4: Use XGBoost for Feature Importance
    # ----------------------------------
    print("Training temporary XGBoost for feature importance...")

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

    # ----------------------------------
    # STEP 5: Select Top 100 Features
    # ----------------------------------
    top_k = 100
    indices = np.argsort(importances)[::-1][:top_k]

    X_train_selected = X_train_scaled[:, indices]
    X_test_selected = X_test_scaled[:, indices]

    print("Selected Feature Count:", X_train_selected.shape[1])

    # ----------------------------------
    # STEP 6: Save New Dataset
    # ----------------------------------
    train_df = pd.DataFrame(X_train_selected)
    train_df["label"] = y_train.values

    test_df = pd.DataFrame(X_test_selected)
    test_df["label"] = y_test.values


    train_df.to_csv("train_selected_100.csv", index=False)
    test_df.to_csv("test_selected_100.csv", index=False)

# Save scaler
    joblib.dump(scaler, "scaler.pkl")

# Save selected feature indices
    np.save("feature_indices.npy", indices)

    print("Saved:")
    print(" - train_selected_100.csv")
    print(" - test_selected_100.csv")
    print(" - scaler.pkl")
    print(" - feature_indices.npy")

if __name__ == "__main__":
    main()