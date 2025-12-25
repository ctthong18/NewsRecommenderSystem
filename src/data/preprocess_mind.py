import os
import pandas as pd
from tqdm import tqdm
from .utils.llm_description import merge_llm_descriptions

def preprocess_mind(dataset_dir, output_dir, llm_desc_path=None):
    """
    Tiền xử lý dữ liệu MIND:
      - Đọc news.tsv, behaviors.tsv
      - Hợp nhất mô tả sinh bởi LLM (nếu có)
      - Chuẩn hóa văn bản, loại bỏ trùng, lưu ra csv
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Đọc news.tsv ...")
    news_df = pd.read_csv(os.path.join(dataset_dir, "news.tsv"), sep='\t',
                          names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])

    # Hợp nhất mô tả từ LLM (nếu có)
    if llm_desc_path is not None and os.path.exists(llm_desc_path):
        news_df = merge_llm_descriptions(news_df, llm_desc_path)

    # Gộp title + description làm input cho DeBERTa
    news_df["text"] = news_df["title"].fillna('') + " [SEP] " + news_df["abstract"].fillna('')

    # Ghi ra file processed
    news_df.to_csv(os.path.join(output_dir, "news_preprocessed.csv"), index=False)

    print("Đọc behaviors.tsv ...")
    behaviors_df = pd.read_csv(os.path.join(dataset_dir, "behaviors.tsv"), sep='\t',
                               names=["impression_id", "user_id", "time", "history", "impressions"])

    behaviors_df.to_csv(os.path.join(output_dir, "behaviors_preprocessed.csv"), index=False)
    print(f"✅ Tiền xử lý hoàn tất. Lưu tại {output_dir}")
