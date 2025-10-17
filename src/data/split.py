import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_behaviors(behaviors_path, out_dir):
    print("Splitting behaviors...")
    behaviors = pd.read_csv(behaviors_path)
    train, val = train_test_split(behaviors, test_size=0.2, random_state=42)
    train.to_csv(out_dir / "train_behaviors.csv", index=False)
    val.to_csv(out_dir / "val_behaviors.csv", index=False)
    print("Saved train/val behaviors")

if __name__ == "__main__":
    split_behaviors("Data/processed/behaviors_raw.csv", Path("Data/processed"))
