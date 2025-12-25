# scripts/preprocess.py
import pandas as pd
import os
from tqdm import tqdm
import json
import argparse

def load_news(news_path):
    # news.tsv: newsID \t category \t subCategory \t title \t abstract \t ...
    df = pd.read_table(news_path, header=None, names=["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"])
    return df

def load_behaviors(beh_path):
    # behaviors.tsv: impression_id \t user_id \t time \t history \t impressions
    df = pd.read_table(beh_path, header=None, names=["impression_id","user_id","time","history","impressions"])
    return df

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    news = load_news(args.news)
    behaviors = load_behaviors(args.behaviors)

    # build news metadata
    news_meta = {}
    for _, row in news.iterrows():
        news_meta[row.news_id] = {
            "title": str(row.title),
            "abstract": str(row.abstract),
            "category": str(row.category),
            "title_entities": str(row.title_entities),
            "abstract_entities": str(row.abstract_entities)
        }

    save_json(news_meta, os.path.join(args.out_dir, "news_meta.json"))

    # build impressions list preserving order
    impressions = []
    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
        imp_id = str(row.impression_id)
        candidate_ids = [x.split("-")[0] for x in str(row.impressions).split()]  # MIND format: Nxxx-0 Nyyy-1 ...
        impressions.append({"impression_id": imp_id, "user_id": str(row.user_id), "history": str(row.history), "candidates": candidate_ids})
    save_json(impressions, os.path.join(args.out_dir, "impressions.json"))
    print("Saved preprocessed files to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", required=True)
    parser.add_argument("--behaviors", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    main(args)
