# preprocess_l1_pca.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def main():

    print("Loading dataset...")
    df = pd.read_csv("../audio_features.csv")

    # Create label
    def extract_label(path):
        return 1 if "PD" in path else 0

    df["label"] = df["AudioPath"].apply(extract_label)

    X = df.drop(columns=["AudioPath", "label"])
    y = df["label"]

    print("Initial feature count:", X.shape[1])

    # -----------------------------
    # Variance Threshold
    # -----------------------------
    vt = VarianceThreshold(threshold=0.0001)
    X = vt.fit_transform(X)
    X = pd.DataFrame(X)

    print("After variance filter:", X.shape[1])

    # -----------------------------
    # Correlation Filtering
    # -----------------------------
    corr = X.corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=drop_cols)

    print("After correlation filter:", X.shape[1])

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # L1 Feature Selection
    # -----------------------------
    print("Running L1 feature selection...")

    l1_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.05,
        max_iter=2000
    )

    selector = SelectFromModel(l1_model)
    selector.fit(X_train, y_train)

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    print("After L1 selection:", X_train.shape[1])

    # -----------------------------
    # PCA
    # -----------------------------
    print("Applying PCA...")

    pca = PCA(n_components=150)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print("Final PCA features:", X_train.shape[1])

    # -----------------------------
    # Save datasets
    # -----------------------------
    train = pd.DataFrame(X_train)
    train["label"] = y_train.values

    test = pd.DataFrame(X_test)
    test["label"] = y_test.values

    train.to_csv("train_final.csv", index=False)
    test.to_csv("test_final.csv", index=False)

    print("Saved train_final.csv and test_final.csv")


if __name__ == "__main__":
    main()