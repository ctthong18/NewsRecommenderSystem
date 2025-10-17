from src.data.make_dataset import main as make_dataset
from src.data.preprocess import preprocess_news
from src.features.build_features import build_features
from src.data.split import split_behaviors
from pathlib import Path

def main():
    print("Starting full data pipeline...")

    make_dataset()
    preprocess_news(
        news_path="Data/processed/news_raw.csv",
        entity_vec_path="Data/raw/MINDsmall_train/entity_embedding.vec",
        output_path="data/processed/news_with_entity.pkl"
    )
    build_features(
        input_path="Data/processed/news_with_entity.pkl",
        output_path="Data/processed/news_final_embeddings.npy"
    )
    split_behaviors("Data/processed/behaviors_raw.csv", Path("Data/processed"))

    print("Data pipeline completed!")

if __name__ == "__main__":
    main()
