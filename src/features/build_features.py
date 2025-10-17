import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.models.once_encoder import ONCEEncoder


def build_features(input_path, output_path):
    print("Loading preprocessed data")
    news = pd.read_pickle(input_path)

    print("Generating text embeddings with ONCEEncoder")
    encoder = ONCEEncoder(
        text_model_name="all-MiniLM-L6-v2",
        entity_dim=len(news["entity_vec"][0])
    )

    texts = news["title"].tolist()
    entity_vecs = np.vstack(news["entity_vec"])

    # Encode batch safely
    with torch.no_grad():
        fused_tensor = encoder.encode_batch(texts, entity_vecs)
        fused_np = fused_tensor.cpu().numpy()

    output_path = Path(output_path)
    np.save(output_path, fused_np)
    print(f"Saved combined embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    build_features(
        "Data/processed/news_with_entity.pkl",
        "Data/processed/news_final_embeddings.npy"
    )
