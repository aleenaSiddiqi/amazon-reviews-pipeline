import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)
    test_df = pd.read_parquet(args.test_data)

    # Fit TF-IDF ONLY on training
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    train_tfidf = vectorizer.fit_transform(train_df["reviewText"].fillna(""))
    val_tfidf = vectorizer.transform(val_df["reviewText"].fillna(""))
    test_tfidf = vectorizer.transform(test_df["reviewText"].fillna(""))

    # Create output folders
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)

    # Save ORIGINAL datasets only (no TF-IDF merged)
    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))

    # Save TF-IDF matrices separately
    sparse.save_npz(os.path.join(args.train_out, "tfidf.npz"), train_tfidf)
    sparse.save_npz(os.path.join(args.val_out, "tfidf.npz"), val_tfidf)
    sparse.save_npz(os.path.join(args.test_out, "tfidf.npz"), test_tfidf)

    # Save feature names
    feature_names = vectorizer.get_feature_names_out()
    with open(os.path.join(args.train_out, "tfidf_features.txt"), "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print("TF-IDF pipeline completed successfully.")
    print("Features:", len(feature_names))


if __name__ == "__main__":
    main()
