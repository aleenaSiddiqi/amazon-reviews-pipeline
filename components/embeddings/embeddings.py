import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    model = SentenceTransformer(args.model_name)

    texts = df["reviewText"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Store each embedding dimension as bert_embedding_0, bert_embedding_1, etc.
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"bert_embedding_{i}" for i in range(embeddings.shape[1])]
    )

    df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Embedding dimensions:", embeddings.shape[1])
    print("Rows processed:", len(df))


if __name__ == "__main__":
    main()
