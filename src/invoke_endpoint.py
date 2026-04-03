import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

# --------------------------------------------------
# Endpoint details
# --------------------------------------------------
ENDPOINT_URL = "https://amazon-review-endpoint-60308963.qatarcentral.inference.ml.azure.com/score"
API_KEY = os.environ.get("AZURE_ML_API_KEY", "")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------
# Feature construction — must match train.py exactly
# --------------------------------------------------
def build_features(df):
    bert_cols  = [c for c in df.columns if c.startswith("bert_embedding_")]
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]

    exclude_cols = bert_cols + tfidf_cols + [
        "overall", "label", "reviewText", "summary",
        "reviewerID", "asin", "reviewerName", "title",
        "brand", "reviewTime", "unixReviewTime", "helpful"
    ]

    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_numeric = df[numeric_cols].fillna(0).values
    X_sbert   = df[bert_cols].fillna(0).values
    X_tfidf   = df[tfidf_cols].fillna(0).values

    return np.hstack([X_numeric, X_sbert, X_tfidf])

# --------------------------------------------------
# Labels — must match train.py exactly
# --------------------------------------------------
def create_labels(df):
    return (df["overall"] >= 4).astype(int).values

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("=" * 50)

    # 1. Load deployment dataset
    print("Loading deployment dataset...")
    df = pd.read_parquet("data/data.parquet")
    
    print(f"Loaded {len(df)} rows")

    # 2. Build feature matrix
    print("Building feature matrix...")
    X = build_features(df)
    print(f"Feature matrix shape: {X.shape}")

    # 3. Get true labels
    y_true = create_labels(df)
    print(f"Label distribution — Positive: {y_true.sum()}, Negative: {len(y_true) - y_true.sum()}")

    # 4. Send requests in batches
    batch_size = 100
    all_predictions   = []
    all_probabilities = []

    print(f"\nSending {len(X)} rows to endpoint in batches of {batch_size}...")

    for i in range(0, len(X), batch_size):
        batch   = X[i:i + batch_size]
        payload = json.dumps(batch.tolist())

        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            data=payload
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, str):
                result = json.loads(result)
            all_predictions.extend(result.get("predictions", []))
            all_probabilities.extend(result.get("probabilities", []))
            print(f"  Batch {i // batch_size + 1} ✓ ({len(batch)} rows)")
        else:
            print(f"  Batch {i // batch_size + 1} FAILED: {response.status_code} — {response.text}")
            return

    # 5. Compute evaluation metrics
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    print("\n" + "=" * 50)
    print("DEPLOYMENT EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total predictions : {len(y_pred)}")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score          : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Precision         : {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall            : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"AUC               : {roc_auc_score(y_true, y_prob):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
