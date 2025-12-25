"""
Shared utilities for news processing across datasets.

This module provides common functionality for:
- Loading LLM descriptions
- Building news ID to text mappings
- Processing news metadata
"""
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import polars as pl

from src.const.mind import EMPTY_NEWS_ID

logger = logging.getLogger(__name__)


def load_llm_descriptions(llm_description_path: Optional[Path]) -> Dict[str, str]:
    """
    Load LLM-generated descriptions from JSON file.
    
    Args:
        llm_description_path: Path to LLM descriptions JSON file
        
    Returns:
        Dictionary mapping news_id to LLM description
    """
    if not llm_description_path or not llm_description_path.exists():
        logger.info("No LLM descriptions file provided or file does not exist")
        return {}
    
    try:
        with open(llm_description_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        logger.info(f"Loaded {len(descriptions)} LLM descriptions from {llm_description_path}")
        return descriptions
    except Exception as e:
        logger.error(f"Failed to load LLM descriptions: {e}")
        return {}


def build_news_id_to_text_map(
    news_df: pl.DataFrame,
    llm_descriptions: Optional[Dict[str, str]] = None
) -> Dict[str, Tuple[str, str]]:
    """
    Build mapping from news_id to (title, description) tuple.
    
    Uses LLM descriptions if available, otherwise falls back to abstract.
    Optimized for memory efficiency by using tuples instead of dicts.
    
    Args:
        news_df: Polars DataFrame containing news data
        llm_descriptions: Optional dictionary of LLM-generated descriptions
        
    Returns:
        Dictionary mapping news_id to (title, description) tuple
    """
    llm_descriptions = llm_descriptions or {}
    news_map: Dict[str, Tuple[str, str]] = {}
    
    for i in range(len(news_df)):
        news_id = news_df[i]["news_id"].item()
        title = news_df[i]["title"].item()
        
        # Use LLM description if available, otherwise use abstract
        description = llm_descriptions.get(news_id, "")
        if not description and "abstract" in news_df.columns:
            abstract_value = news_df[i]["abstract"].item()
            description = abstract_value if abstract_value is not None else ""
        
        # Store as tuple to reduce memory overhead
        news_map[news_id] = (title, description)
    
    # Add empty news placeholder
    news_map[EMPTY_NEWS_ID] = ("", "")
    
    logger.info(f"Built news ID to text map with {len(news_map)} entries")
    return news_map


def get_news_texts(
    news_ids: list[str],
    news_map: Dict[str, Tuple[str, str]]
) -> list[list[str]]:
    """
    Get news texts (title and description) for a list of news IDs.
    
    Args:
        news_ids: List of news identifiers
        news_map: Mapping from news_id to (title, description)
        
    Returns:
        List of [title, description] pairs
    """
    return [list(news_map.get(nid, ("", ""))) for nid in news_ids]


class NewsTextProcessor:
    """
    Processor for news text data with caching and optimization.
    
    This class encapsulates common news processing logic used across
    training and validation datasets.
    """
    
    def __init__(
        self,
        news_df: pl.DataFrame,
        llm_description_path: Optional[Path] = None
    ):
        """
        Initialize news text processor.
        
        Args:
            news_df: Polars DataFrame containing news data
            llm_description_path: Optional path to LLM descriptions
        """
        self.news_df = news_df
        self.llm_descriptions = load_llm_descriptions(llm_description_path)
        self.news_map = build_news_id_to_text_map(news_df, self.llm_descriptions)
        
        logger.info(
            f"Initialized NewsTextProcessor with {len(self.news_map)} news items, "
            f"{len(self.llm_descriptions)} LLM descriptions"
        )
    
    def get_news_texts(self, news_ids: list[str]) -> list[list[str]]:
        """
        Get news texts for a list of news IDs.
        
        Args:
            news_ids: List of news identifiers
            
        Returns:
            List of [title, description] pairs
        """
        return get_news_texts(news_ids, self.news_map)
    
    def get_single_news_text(self, news_id: str) -> Tuple[str, str]:
        """
        Get news text for a single news ID.
        
        Args:
            news_id: News identifier
            
        Returns:
            Tuple of (title, description)
        """
        return self.news_map.get(news_id, ("", ""))
    
    def has_llm_description(self, news_id: str) -> bool:
        """
        Check if a news item has an LLM-generated description.
        
        Args:
            news_id: News identifier
            
        Returns:
            True if LLM description exists, False otherwise
        """
        return news_id in self.llm_descriptions
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the news processor.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_news": float(len(self.news_map)),
            "llm_descriptions": float(len(self.llm_descriptions)),
            "coverage": len(self.llm_descriptions) / max(len(self.news_map) - 1, 1)  # -1 for EMPTY_NEWS_ID
        }
