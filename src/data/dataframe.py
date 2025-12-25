"""
Helper functions to read MIND dataset files.
Based on gpt-augmented-news-recommendation implementation.
"""
import pandas as pd
import polars as pl
from pathlib import Path
from src.const.mind import UNKNOWN_USER_IDX


def read_news_df(path_to_tsv: Path) -> pl.DataFrame:
    """
    Read news.tsv file.
    
    Args:
        path_to_tsv: Path to news.tsv file
    
    Returns:
        polars DataFrame with columns: news_id, category, subcategory, title, abstract, url
    """
    # Use pandas first, then convert to polars (more reliable for TSV)
    news_df = pd.read_csv(path_to_tsv, sep="\t", encoding="utf8", header=None)
    news_df.columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    news_df = pl.from_pandas(news_df)
    # Drop entities columns if not needed
    return news_df.drop("title_entities", "abstract_entities")


def read_behavior_df(path_to_tsv: Path) -> pl.DataFrame:
    """
    Read behaviors.tsv file.
    
    Args:
        path_to_tsv: Path to behaviors.tsv file
    
    Returns:
        polars DataFrame with parsed impressions and history
    """
    behavior_df = pl.read_csv(path_to_tsv, separator="\t", encoding="utf8-lossy", has_header=False)
    behavior_df = behavior_df.rename(
        {
            "column_1": "impression_id",
            "column_2": "user_id",
            "column_3": "time",
            "column_4": "history_str",
            "column_5": "impressions_str",
        }
    )
    behavior_df = (
        behavior_df.with_columns((pl.col("impressions_str").str.split(" ")).alias("impression_news_list"))
        .with_columns(
            [
                pl.col("impression_news_list")
                .map_elements(lambda v: [
                    {
                        "news_id": item.split("-")[0],
                        "clicked": int(item.split("-")[1])
                    }
                    for item in v if item and "-" in item
                ], return_dtype=pl.List(pl.Struct([
                    pl.Field("news_id", pl.Utf8),
                    pl.Field("clicked", pl.Int64)
                ])))
                .alias("impressions")
            ]
        )
        .with_columns([pl.col("history_str").str.split(" ").alias("history")])
        .select(["impression_id", "user_id", "time", "history", "impressions"])
    )
    return behavior_df


def create_user_ids_to_idx_map(train_behavior_df: pl.DataFrame, val_behavior_df: pl.DataFrame) -> dict[str, int]:
    """
    Create mapping from user_id to index.
    
    Args:
        train_behavior_df: Training behavior dataframe
        val_behavior_df: Validation behavior dataframe
    
    Returns:
        Dictionary mapping user_id to index
    """
    user_ids_in_train_set = list(set(train_behavior_df["user_id"].to_list()))

    d: dict[str, int] = {}
    for i, user_id in enumerate(user_ids_in_train_set):
        d[user_id] = i + 1  # idx = 0 is reserved for unknown users

    user_ids_in_val_set = list(set(val_behavior_df["user_id"].to_list()))

    for user_id in user_ids_in_val_set:
        if user_id not in d:
            d[user_id] = UNKNOWN_USER_IDX

    return d

