import json
import pandas as pd

def merge_llm_descriptions(news_df: pd.DataFrame, llm_desc_path: str):
    """
    Ghép mô tả sinh bởi LLM (ví dụ GPT-4) vào news.tsv
    File JSON có dạng: {"N123": "LLM summary ...", "N456": "..." }
    """
    with open(llm_desc_path, 'r', encoding='utf-8') as f:
        llm_desc = json.load(f)

    news_df["llm_description"] = news_df["news_id"].map(llm_desc).fillna("")
    news_df["abstract"] = news_df["llm_description"].where(news_df["llm_description"] != "", news_df["abstract"])
    return news_df
