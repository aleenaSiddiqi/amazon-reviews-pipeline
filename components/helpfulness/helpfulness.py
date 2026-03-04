import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    df["helpfulness_ratio"] = df["helpful"].apply(
    lambda x: x[0] / x[1] if x is not None and len(x) == 2 and x[1] > 0 else 0.0
)

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows processed:", len(df))
    print(df["helpfulness_ratio"].describe())


if __name__ == "__main__":
    main()
