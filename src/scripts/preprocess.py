# scripts/preprocess.py
"""
Preprocess MIND dataset with structured logging and progress tracking.

This script processes raw MIND dataset files (news.tsv, behaviors.tsv) and
converts them into JSON format for easier consumption by the training pipeline.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger, setup_logger


def load_news(news_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load news data from TSV file.
    
    Args:
        news_path: Path to news.tsv file
        logger: Logger instance
        
    Returns:
        DataFrame with news data
        
    Raises:
        Exception: If file loading fails
    """
    logger.info(f"Loading news data from {news_path}")
    
    try:
        # news.tsv: newsID \t category \t subCategory \t title \t abstract \t ...
        df = pd.read_table(
            news_path, 
            header=None, 
            names=["news_id", "category", "subcategory", "title", "abstract", 
                   "url", "title_entities", "abstract_entities"]
        )
        
        logger.info(
            f"Successfully loaded {len(df)} news articles",
            extra = {
                "num_news": len(df),
                "num_categories": df['category'].nunique()
            }
        )
        
        # Log category distribution
        category_counts = df['category'].value_counts()
        logger.debug(f"Category distribution: {category_counts.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load news data: {str(e)}")
        raise


def load_behaviors(beh_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load user behavior data from TSV file.
    
    Args:
        beh_path: Path to behaviors.tsv file
        logger: Logger instance
        
    Returns:
        DataFrame with behavior data
        
    Raises:
        Exception: If file loading fails
    """
    logger.info(f"Loading behavior data from {beh_path}")
    
    try:
        # behaviors.tsv: impression_id \t user_id \t time \t history \t impressions
        df = pd.read_table(
            beh_path, 
            header=None, 
            names=["impression_id", "user_id", "time", "history", "impressions"]
        )
        
        num_users = df['user_id'].nunique()
        num_impressions = len(df)
        
        logger.info(
            f"Successfully loaded {num_impressions} impressions from {num_users} users",
            extra = {
                "num_users": num_users,
                "num_impressions": num_impressions
            }
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load behavior data: {str(e)}")
        raise


def save_json(obj: Any, path: str, logger: logging.Logger) -> None:
    """
    Save object to JSON file.
    
    Args:
        obj: Object to save (must be JSON serializable)
        path: Output file path
        logger: Logger instance
        
    Raises:
        Exception: If file saving fails
    """
    logger.debug(f"Saving JSON to {path}")
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        logger.info(f"Saved {path} ({file_size:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {str(e)}")
        raise


def main(args: argparse.Namespace) -> None:
    """
    Main preprocessing function.
    
    Args:
        args: Command-line arguments containing paths and options
    """
    # Setup logger
    logger_instance = setup_logger(
        name="preprocess",
        log_dir=args.log_dir if hasattr(args, 'log_dir') and args.log_dir else None,
        log_level=args.log_level if hasattr(args, 'log_level') else "INFO",
        console_output=True
    )
    logger = logger_instance.get_logger("preprocess")
    
    logger.info("=" * 60)
    logger.info("Starting MIND dataset preprocessing")
    logger.info("=" * 60)
    logger.info(f"News file: {args.news}")
    logger.info(f"Behaviors file: {args.behaviors}")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Created output directory: {args.out_dir}")
    
    # Load data
    logger.info("Step 1/3: Loading data files")
    news = load_news(args.news, logger)
    behaviors = load_behaviors(args.behaviors, logger)
    
    # Build news metadata
    logger.info("Step 2/3: Building news metadata")
    news_meta = {}
    
    for idx, row in tqdm(news.iterrows(), total=len(news), desc="Processing news"):
        news_meta[row.news_id] = {
            "title": str(row.title),
            "abstract": str(row.abstract),
            "category": str(row.category),
            "title_entities": str(row.title_entities),
            "abstract_entities": str(row.abstract_entities)
        }
    
    logger.info(f"Built metadata for {len(news_meta)} news articles")
    
    # Save news metadata
    news_meta_path = os.path.join(args.out_dir, "news_meta.json")
    save_json(news_meta, news_meta_path, logger)
    
    # Build impressions list
    logger.info("Step 3/3: Building impressions list")
    impressions = []
    total_clicks = 0
    
    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Processing behaviors"):
        imp_id = str(row.impression_id)
        
        # Parse impressions: MIND format is "Nxxx-0 Nyyy-1 ..." where 0/1 is click label
        impression_items = str(row.impressions).split()
        candidate_ids = [x.split("-")[0] for x in impression_items]
        
        # Count clicks for statistics
        clicks = sum(1 for x in impression_items if x.endswith("-1"))
        total_clicks += clicks
        
        impressions.append({
            "impression_id": imp_id,
            "user_id": str(row.user_id),
            "history": str(row.history),
            "candidates": candidate_ids
        })
    
    logger.info(
        f"Built {len(impressions)} impressions with {total_clicks} total clicks",
        
        extra = {
            "num_impressions": len(impressions),
            "total_clicks": total_clicks,
            "avg_clicks_per_impression": total_clicks / len(impressions) if impressions else 0
        }
    )
    
    # Save impressions
    impressions_path = os.path.join(args.out_dir, "impressions.json")
    save_json(impressions, impressions_path, logger)
    
    # Log final statistics
    logger.info("=" * 60)
    logger.info("Preprocessing completed successfully!")
    logger.info("=" * 60)
    logger.info("Statistics:")
    logger.info(f"  - Total news articles: {len(news_meta)}")
    logger.info(f"  - Total users: {behaviors['user_id'].nunique()}")
    logger.info(f"  - Total impressions: {len(impressions)}")
    logger.info(f"  - Total clicks: {total_clicks}")
    logger.info(f"  - Click-through rate: {total_clicks / (len(impressions) * len(candidate_ids)) * 100:.2f}%")
    logger.info(f"Output files saved to: {args.out_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", required=True)
    parser.add_argument("--behaviors", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    main(args)
