import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

def init():
    global model
    # AZUREML_MODEL_DIR is set automatically by Azure ML when the endpoint starts
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully from:", model_path)


def build_features(df):
    """
    Must match EXACTLY the same logic as train.py build_features()
    """
    bert_cols = [c for c in df.columns if c.startswith("bert_embedding_")]
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


def run(raw_data):
    try:
        data = json.loads(raw_data)

        # data is a list of lists (feature rows) not dicts
        # convert directly to numpy array
        X = np.array(data)

        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        return json.dumps({
            "predictions": preds.tolist(),
            "probabilities": proba.tolist()
        })

    except Exception as e:
        return json.dumps({"error": str(e)})
