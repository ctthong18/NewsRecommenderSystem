import pandas as pd
from pathlib import Path 

def load_mind_data(raw_dir: Path):
    news_path = raw_dir / "news.tsv"
    behaviors_path = raw_dir / "behaviors.tsv"
    
    news_cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
    
    news = pd.read_csv(news_path, sep='\t', names=news_cols)
    behaviors = pd.read_csv(behaviors_path, sep='\t', names=behaviors_cols)
    
    return news, behaviors

def main():
    raw_dir = Path("Data/raw/MINDsmall_train")
    processed_dir = Path("Data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    news, behaviors = load_mind_data(raw_dir)
    news.to_csv(processed_dir / "news_raw.csv", index=False)
    behaviors.to_csv(processed_dir / "behaviors_raw.csv", index=False)
    
    print("Saved raw CSV to Data/processed")
    
if __name__ == "__main__":
    main()