import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def extract_sentiment(text, sia):
    if not isinstance(text, str):
        return 0.0, 0.0, 0.0, 0.0
    scores = sia.polarity_scores(text)
    return scores["pos"], scores["neg"], scores["neu"], scores["compound"]


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    sia = SentimentIntensityAnalyzer()

    df[["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]] = df["reviewText"].apply(
        lambda x: pd.Series(extract_sentiment(x, sia))
    )

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows processed:", len(df))
    print(df[["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]].describe())


if __name__ == "__main__":
    main()
