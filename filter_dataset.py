import argparse
from pathlib import Path

import pandas as pd

# TODO: We installed a whole `pandas` lib just to do this
# hand write this without pandas in the future
parser = argparse.ArgumentParser(description="Filter dataset for training")
parser.add_argument(
    "-d", "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
)
args = parser.parse_args()

# Paths to the dataset files
dataset_dir = Path(args.dataset_dir)
validated_path = dataset_dir / "validated.tsv"
dev_path = dataset_dir / "dev.tsv"
test_path = dataset_dir / "test.tsv"
train_path = dataset_dir / "train.tsv"

# Load TSV files
validated_df = pd.read_csv(validated_path, sep="\t")
dev_df = pd.read_csv(dev_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")
train_df = pd.read_csv(train_path, sep="\t")

# Collect all used paths (those already in train/dev/test)
used_paths = set(dev_df["path"]) | set(test_df["path"]) | set(train_df["path"])

# Filter out those entries from validated.tsv
filtered_df = validated_df[~validated_df["path"].isin(used_paths)]

# Save to new training TSV
output_path = dataset_dir / "filtered_validated.tsv"
filtered_df.to_csv(output_path, sep="\t", index=False)

print(f"Filtered training set saved to: {output_path}")
print(f"Original validated.tsv size: {len(validated_df)}")
print(f"Filtered training set size: {len(filtered_df)}")
