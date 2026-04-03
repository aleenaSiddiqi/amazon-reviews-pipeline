import argparse
import os
import time
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    # Hyperparameters for sweep
    parser.add_argument("--C",        type=float, default=1.0)
    parser.add_argument("--max_iter", type=int,   default=1000)
    return parser.parse_args()

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Path does not exist: {folder_path}")
    parquet_path = os.path.join(folder_path, "data.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"data.parquet not found in {folder_path}")
    return pd.read_parquet(parquet_path)

# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# --------------------------------------------------
# Features
# --------------------------------------------------
def build_features(df):
    # Get all bert embedding columns
    bert_cols = [c for c in df.columns if c.startswith("bert_embedding_")]

    # Numeric feature columns (exclude non-features and bert cols)
    exclude_cols = [
        "overall", "label", "reviewText", "summary",
        "reviewerID", "asin", "reviewerName", "title", "brand",
        "reviewTime", "unixReviewTime", "helpful",
    ] + bert_cols  # exclude bert cols from numeric — handled separately

    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_numeric = df[numeric_cols].fillna(0).values      # sentiment, length, helpfulness
    X_sbert   = df[bert_cols].fillna(0).values         # bert_embedding_0 ... 383

    X = np.hstack([X_numeric, X_sbert])

    if len(X) == 0:
        raise RuntimeError("Feature matrix is empty. Impressive.")

    print(f"Features: {X_numeric.shape[1]} numeric + {X_sbert.shape[1]} bert = {X.shape[1]} total")

    return X

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split):
    preds      = model.predict(X)
    proba      = model.predict_proba(X)[:, 1]

    acc        = accuracy_score(y, preds)
    f1         = f1_score(y, preds, average="weighted")
    precision  = precision_score(y, preds, average="weighted")
    recall     = recall_score(y, preds, average="weighted")
    auc        = roc_auc_score(y, proba)

    mlflow.log_metric(f"{split}_accuracy",  acc)
    mlflow.log_metric(f"{split}_f1",        f1)
    mlflow.log_metric(f"{split}_precision", precision)
    mlflow.log_metric(f"{split}_recall",    recall)
    mlflow.log_metric(f"{split}_auc",       auc)

    print(f"[{split}] accuracy={acc:.4f} f1={f1:.4f} "
          f"precision={precision:.4f} recall={recall:.4f} auc={auc:.4f}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    with mlflow.start_run():

        print("Loading data...")
        train_df = load_data(args.train_data)
        val_df   = load_data(args.val_data)
        test_df  = load_data(args.test_data)

        print("Creating labels...")
        train_df = create_labels(train_df)
        val_df   = create_labels(val_df)
        test_df  = create_labels(test_df)

        print("Building features...")
        X_train = build_features(train_df)
        X_val   = build_features(val_df)
        X_test  = build_features(test_df)

        y_train = train_df["label"]
        y_val   = val_df["label"]
        y_test  = test_df["label"]

        if len(X_train) == 0:
            raise RuntimeError("Training data is empty. That's concerning.")

        # Log hyperparameters
        mlflow.log_param("C",        args.C)
        mlflow.log_param("max_iter", args.max_iter)

        # Model definition using hyperparameters
        print("Training model...")
        model = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        print("Evaluating...")
        evaluate(model, X_train, y_train, "train")
        evaluate(model, X_val,   y_val,   "val")
        evaluate(model, X_test,  y_test,  "test")

        print("Saving model...")
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        runtime = time.time() - start_time
        mlflow.log_metric("training_runtime_seconds", runtime)
        print(f"Done in {runtime:.1f}s")

if __name__ == "__main__":
    main()
