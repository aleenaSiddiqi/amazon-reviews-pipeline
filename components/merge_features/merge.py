import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_length_data", type=str, required=True)
    parser.add_argument("--sentiment_data", type=str, required=True)
    parser.add_argument("--tfidf_train_data", type=str, required=True)
    parser.add_argument("--embeddings_data", type=str, required=True)
    parser.add_argument("--helpfulness_data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_parquet(path):
    return pd.read_parquet(os.path.join(path, "data.parquet"))


def main():
    args = parse_args()

    # Load all feature datasets
    review_length_df = load_parquet(args.review_length_data)
    sentiment_df = load_parquet(args.sentiment_data)
    tfidf_df = load_parquet(args.tfidf_train_data)
    embeddings_df = load_parquet(args.embeddings_data)
    helpfulness_df = load_parquet(args.helpfulness_data)

    # Define entity keys
    keys = ["asin", "reviewerID"]

    # Get only the new feature columns from each df (avoid duplicating shared columns)
    def get_feature_cols(df, existing_cols, keys):
        new_cols = [c for c in df.columns if c not in existing_cols or c in keys]
        return df[new_cols]

    # Start with review_length as base
    merged_df = review_length_df
    existing_cols = set(merged_df.columns)

    # Merge each feature set on entity keys
    for df in [sentiment_df, tfidf_df, embeddings_df, helpfulness_df]:
        feature_cols = get_feature_cols(df, existing_cols, keys)
        merged_df = merged_df.merge(feature_cols, on=keys, how="left")
        existing_cols.update(merged_df.columns)

    os.makedirs(args.out, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Final merged dataset shape:", merged_df.shape)
    print("Columns:", list(merged_df.columns))


if __name__ == "__main__":
    main()
