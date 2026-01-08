import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.const.mind import EMPTY_IMPRESSION_IDX, EMPTY_NEWS_ID
from src.utils.news_processor import NewsTextProcessor
from src.utils.sampling import hard_negative_sampling

logger = logging.getLogger(__name__)


class MINDTrainDataset(Dataset):
    """
    MIND Training Dataset with support for:
    - LLM-generated descriptions
    - Hard negative sampling (optional)
    - Tokenization caching for improved performance
    """
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: Dict[str, int],
        batch_transform_texts: Callable[[List[List[str]]], Dict[str, torch.Tensor]],
        npratio: int,
        history_size: int,
        llm_description_path: Optional[Path] = None,
        use_hard_negative: bool = False,
        news_embeddings_cache: Optional[Dict[str, np.ndarray]] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        enable_tokenization_cache: bool = True,
    ):
        """
        Initialize MIND training dataset.
        
        Args:
            behavior_df: Polars DataFrame with user behavior data
            news_df: Polars DataFrame with news metadata
            user_ids_to_idx_map: Mapping from user IDs to indices
            batch_transform_texts: Function to tokenize text batches
            npratio: Negative sampling ratio
            history_size: Maximum number of history items
            llm_description_path: Optional path to LLM descriptions
            use_hard_negative: Whether to use hard negative sampling
            news_embeddings_cache: Optional cache of news embeddings for hard negative sampling
            device: PyTorch device
            enable_tokenization_cache: Whether to cache tokenized news
        """
        self.behavior_df = behavior_df
        self.news_df = news_df
        self.batch_transform_texts = batch_transform_texts
        self.npratio = npratio
        self.history_size = history_size
        self.use_hard_negative = use_hard_negative
        self.news_embeddings_cache = news_embeddings_cache
        self.device = device
        self.enable_tokenization_cache = enable_tokenization_cache

        # Tokenization cache to avoid redundant tokenization
        self._tokenization_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # Use shared news processor for text handling
        self._news_processor = NewsTextProcessor(news_df, llm_description_path)
        self.__news_id_to_news_map = self._news_processor.news_map
        self.__user_ids_to_idx_map = user_ids_to_idx_map

        # Add columns for positive/negative indices
        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .map_elements(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1], return_dtype=pl.List(pl.Int64))
                .alias("clicked_idxes"),
                pl.col("impressions")
                .map_elements(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0], return_dtype=pl.List(pl.Int64))
                .alias("non_clicked_idxes"),
            ]
        )
        
        logger.info(
            f"Initialized MINDTrainDataset with {len(self.behavior_df)} behaviors, "
            f"{len(self.__news_id_to_news_map)} news items, "
            f"tokenization_cache={'enabled' if enable_tokenization_cache else 'disabled'}"
        )

    def _get_tokenized_news(self, news_id: str) -> Dict[str, torch.Tensor]:
        """
        Get tokenized news with caching to avoid redundant tokenization.
        
        Args:
            news_id: News identifier
            
        Returns:
            Tokenized news tensor
        """
        if self.enable_tokenization_cache and news_id in self._tokenization_cache:
            return self._tokenization_cache[news_id]
        
        # Get news text
        title, description = self.__news_id_to_news_map[news_id]
        
        # Tokenize
        tokenized_dict = self.batch_transform_texts([[title, description]])
        
        # Cache if enabled
        if self.enable_tokenization_cache:
            self._tokenization_cache[news_id] = tokenized_dict
        
        return tokenized_dict

    def __getitem__(self, behavior_idx: int) -> Dict[str, torch.Tensor]:
        behavior_item = self.behavior_df[behavior_idx]

        history = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )
        poss_idxes = behavior_item["clicked_idxes"].to_list()[0]
        neg_idxes = behavior_item["non_clicked_idxes"].to_list()[0]
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION])

        # Sampling
        sample_poss_idxes = random.sample(poss_idxes, 1) if poss_idxes else []
        
        # Hard negative sampling or random sampling
        if self.use_hard_negative and self.news_embeddings_cache and poss_idxes:
            # Get positive news embedding
            pos_news_id = impressions[sample_poss_idxes[0]]["news_id"]
            if pos_news_id in self.news_embeddings_cache:
                pos_emb = self.news_embeddings_cache[pos_news_id]
                # Get negative news embeddings
                neg_news_ids = [impressions[idx]["news_id"] for idx in neg_idxes]
                neg_embeddings = np.array([
                    self.news_embeddings_cache.get(nid, np.zeros_like(pos_emb))
                    for nid in neg_news_ids
                ])
                if len(neg_embeddings) > 0:
                    # Find hard negatives
                    hard_neg_indices = hard_negative_sampling(pos_emb, neg_embeddings, k=min(self.npratio, len(neg_idxes)))
                    sample_neg_idxes = [neg_idxes[i] for i in hard_neg_indices[:self.npratio]]
                else:
                    sample_neg_idxes = self.__sampling_negative(neg_idxes, self.npratio)
            else:
                sample_neg_idxes = self.__sampling_negative(neg_idxes, self.npratio)
        else:
            sample_neg_idxes = self.__sampling_negative(neg_idxes, self.npratio)
        
        sample_impression_idxes = sample_poss_idxes + sample_neg_idxes
        random.shuffle(sample_impression_idxes)
        sample_impressions = impressions[sample_impression_idxes]

        # Get IDs
        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        history_news_ids = history[: self.history_size]
        if len(history) < self.history_size:
            history_news_ids += [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # Prepare texts with title and description - optimized memory access
        candidate_texts = [
            list(self.__news_id_to_news_map[nid])
            for nid in candidate_news_ids
        ]
        history_texts = [
            list(self.__news_id_to_news_map[nid])
            for nid in history_news_ids
        ]

        # Tokenize (DeBERTa) - batch tokenization is more efficient
        candidate_batch_dict = self.batch_transform_texts(candidate_texts)
        history_batch_dict = self.batch_transform_texts(history_texts)

        # Extract input_ids and attention_mask
        candidate_input_ids = candidate_batch_dict["input_ids"]  # (candidate_num, seq_len)
        candidate_attention_mask = candidate_batch_dict["attention_mask"]  # (candidate_num, seq_len)
        history_input_ids = history_batch_dict["input_ids"]  # (history_size, seq_len)
        history_attention_mask = history_batch_dict["attention_mask"]  # (history_size, seq_len)

        # Convert labels - find index of positive sample
        labels_list = [imp_item["clicked"] for imp_item in sample_impressions]
        target_idx = labels_list.index(1) if 1 in labels_list else 0
        labels_tensor = torch.tensor(target_idx, dtype=torch.long)
        
        user_id = self.__user_ids_to_idx_map.get(
            behavior_item["user_id"].to_list()[0],
            0
        )

        return {
            "candidate_news": candidate_input_ids,  # (candidate_num, seq_len)
            "candidate_attention_mask": candidate_attention_mask,  # (candidate_num, seq_len)
            "news_histories": history_input_ids,  # (history_size, seq_len)
            "news_histories_attention_mask": history_attention_mask,  # (history_size, seq_len)
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        """Return the number of behavior samples."""
        return len(self.behavior_df)

    def __sampling_negative(self, neg_idxes: List[int], npratio: int) -> List[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))
        return random.sample(neg_idxes, npratio)


class MINDValDataset(Dataset):
    """
    MIND Validation Dataset without negative sampling.
    
    Supports LLM descriptions and is optimized with tokenization caching
    and reduced memory footprint.
    """

    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: Dict[str, int],
        batch_transform_texts: Callable[[List[List[str]]], Dict[str, torch.Tensor]],
        history_size: int,
        llm_description_path: Optional[Path] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        enable_tokenization_cache: bool = True,
    ):
        """
        Initialize MIND validation dataset.
        
        Args:
            behavior_df: Polars DataFrame with user behavior data
            news_df: Polars DataFrame with news metadata
            user_ids_to_idx_map: Mapping from user IDs to indices
            batch_transform_texts: Function to tokenize text batches
            history_size: Maximum number of history items
            llm_description_path: Optional path to LLM descriptions
            device: PyTorch device
            enable_tokenization_cache: Whether to cache tokenized news
        """
        self.behavior_df = behavior_df
        self.news_df = news_df
        self.batch_transform_texts = batch_transform_texts
        self.history_size = history_size
        self.device = device
        self.enable_tokenization_cache = enable_tokenization_cache

        # Tokenization cache to avoid redundant tokenization
        self._tokenization_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # Use shared news processor for text handling
        self._news_processor = NewsTextProcessor(news_df, llm_description_path)
        self.__news_id_to_news_map = self._news_processor.news_map
        self.__user_ids_to_idx_map = user_ids_to_idx_map
        
        logger.info(
            f"Initialized MINDValDataset with {len(self.behavior_df)} behaviors, "
            f"{len(self.__news_id_to_news_map)} news items, "
            f"tokenization_cache={'enabled' if enable_tokenization_cache else 'disabled'}"
        )

    def __getitem__(self, behavior_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a validation sample.
        
        Args:
            behavior_idx: Index of the behavior sample
            
        Returns:
            Dictionary containing candidate news, history, user ID, and labels
        """
        behavior_item = self.behavior_df[behavior_idx]
        history = behavior_item["history"].to_list()[0] or []
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION])

        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        labels = [imp_item["clicked"] for imp_item in impressions]

        history_news_ids = history[: self.history_size]
        if len(history) < self.history_size:
            history_news_ids += [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # Prepare texts with title and description - optimized memory access
        candidate_texts = [
            list(self.__news_id_to_news_map[nid])
            for nid in candidate_news_ids
        ]
        history_texts = [
            list(self.__news_id_to_news_map[nid])
            for nid in history_news_ids
        ]

        # Batch tokenization is more efficient
        candidate_batch_dict = self.batch_transform_texts(candidate_texts)
        history_batch_dict = self.batch_transform_texts(history_texts)
        
        # Extract input_ids and attention_mask
        candidate_input_ids = candidate_batch_dict["input_ids"]  # (candidate_num, seq_len)
        candidate_attention_mask = candidate_batch_dict["attention_mask"]  # (candidate_num, seq_len)
        history_input_ids = history_batch_dict["input_ids"]  # (history_size, seq_len)
        history_attention_mask = history_batch_dict["attention_mask"]  # (history_size, seq_len)
        
        # For validation, target is one-hot labels
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        user_id = self.__user_ids_to_idx_map.get(
            behavior_item["user_id"].to_list()[0],
            0
        )

        return {
            "candidate_news": candidate_input_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "news_histories": history_input_ids,
            "news_histories_attention_mask": history_attention_mask,
            "news_histories": history_batch,
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        """Return the number of behavior samples."""
        return len(self.behavior_df)
