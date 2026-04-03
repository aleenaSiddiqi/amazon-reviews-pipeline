import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # ratios
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    # outputs
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # Step 1: split train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )

    # Remaining = val + test + deploy = 40%
    remaining_ratio = 1 - args.train_ratio

    # Step 2: split val from temp
    val_relative = args.val_ratio / remaining_ratio
    val_df, temp_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative),
        random_state=args.seed,
        shuffle=True
    )

    # Step 3: split test vs deploy
    test_relative = args.test_ratio / (args.test_ratio + (remaining_ratio - args.val_ratio - args.test_ratio))
    test_df, deploy_df = train_test_split(
        temp_df,
        test_size=(1 - test_relative),
        random_state=args.seed,
        shuffle=True
    )

    # Create folders
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)

    # Save files
    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"))

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))
    print("Deploy:", len(deploy_df))


if __name__ == "__main__":
    main()
