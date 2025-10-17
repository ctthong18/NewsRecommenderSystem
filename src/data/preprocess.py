import pandas as pd
import ast
import numpy as np
from src.features.feature_utils import load_vec, avg_entity_vec
from pathlib import Path

def extract_entities(entity_str):
    try:
        entities = ast.literal_eval(entity_str)
        return [e['Label'] for e in entities] if isinstance(entities, list) else []
    except:
        return []
    
def preprocess_news(news_path, entity_vec_path, output_path):
    print("Loading news")
    news = pd.read_csv(news_path)
    
    print("Extracting entities")
    news["entities"] = news["title_entities"].apply(extract_entities)
    
    print("Averaging entity embedding")
    entity_emb = load_vec(entity_vec_path)
    
    print("Averaging entity vectors")
    news["entity_vec"] = news["entities"].apply(lambda x: avg_entity_vec(x, entity_emb))
    
    news.to_pickle(output_path)
    print(f"Preprocessed news saved to {output_path}")

if __name__ == "__main__":
     preprocess_news(
        news_path="Data/processed/news_raw.csv",
        entity_vec_path="Data/raw/MINDsmall_train/entity_embedding.vec",
        output_path="Data/processed/news_with_entity.pkl"
    )