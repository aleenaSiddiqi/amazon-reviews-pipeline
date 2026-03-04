import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def normalize_text(text):
    if not isinstance(text, str):
        return None
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Trim whitespace
    text = text.strip()
    
    return text


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # Apply normalization
    df["reviewText"] = df["reviewText"].apply(normalize_text)

    # Filter out empty or very short reviews
    df = df[df["reviewText"].str.len() >= 10]

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows after normalization:", len(df))


if __name__ == "__main__":
    main()
