"""
Simple Python API wrapper for news recommendation inference.

Provides a clean, easy-to-use interface for making predictions with
comprehensive input validation and error handling.
"""
import torch
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from src.models.DeBERTaNewsEncoder import DeBERTaNewsEncoder
from src.models.UserEncoder import UserEncoder
from src.models.NAML import NAML
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.const.mind import EMPTY_NEWS_ID
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """
    Result of a recommendation prediction.
    
    Attributes:
        news_ids: List of news IDs ranked by relevance
        scores: Prediction scores for each news item
        top_k: Number of top recommendations returned
    """
    news_ids: List[str]
    scores: List[float]
    top_k: int
    
    def get_top_k(self, k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """
        Get top-k recommendations.
        
        Args:
            k: Number of recommendations to return (default: self.top_k)
            
        Returns:
            Tuple of (news_ids, scores)
        """
        k = k or self.top_k
        return self.news_ids[:k], self.scores[:k]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "news_ids": self.news_ids,
            "scores": self.scores,
            "top_k": self.top_k
        }


class InferenceAPIError(Exception):
    """Base exception for inference API errors."""
    pass


class ModelLoadError(InferenceAPIError):
    """Error loading model checkpoint."""
    pass


class InputValidationError(InferenceAPIError):
    """Error validating input data."""
    pass


class NewsRecommendationAPI:
    """
    Simple Python API for news recommendation inference.
    
    Example usage:
        ```python
        # Initialize API
        api = NewsRecommendationAPI(
            checkpoint_path="output/checkpoints/best_model.pt",
            config_path="configs/base_config.yaml"
        )
        
        # Load news data
        api.load_news_data(
            news_path="Data/raw/MINDsmall_dev/news.tsv",
            llm_description_path="Data/generated/llm_descriptions.json"
        )
        
        # Get recommendations for a user
        result = api.recommend(
            candidate_news_ids=["N12345", "N67890", "N11111"],
            history_news_ids=["N99999", "N88888"],
            top_k=10
        )
        
        # Get top recommendations
        top_news, top_scores = result.get_top_k(5)
        ```
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the inference API.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            config: Configuration dictionary (optional)
            config_path: Path to config file (optional, overrides config)
            device: Device to use (cuda/cpu, default: auto-detect)
            verbose: Whether to print initialization messages
            
        Raises:
            ModelLoadError: If model loading fails
            FileNotFoundError: If checkpoint file not found
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load configuration
        if config_path:
            from src.utils.config_loader import load_config
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = {}
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = self.config.get("training.device",
                                         "cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            logger.info(f"Initializing NewsRecommendationAPI on device: {self.device}")
        
        # Load model
        try:
            self.model = self._load_model()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
        
        # Load tokenizer
        pretrained_model = self.config.get("model.pretrained", "microsoft/deberta-v3-base")
        max_length = self.config.get("model.max_length", 64)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True, use_safetensors=True)
        self.transform_fn = create_transform_fn_from_pretrained_tokenizer(
            self.tokenizer, max_length
        )
        
        # Initialize news data storage
        self.news_map = {}
        self.news_df = None
        
        # Configuration
        self.history_size = self.config.get("training.history_size", 50)
        
        if self.verbose:
            logger.info("NewsRecommendationAPI initialized successfully")
    
    def _load_model(self) -> NAML:
        """Load model from checkpoint."""
        # Extract model configuration
        pretrained_model = self.config.get("model.pretrained", "microsoft/deberta-v3-base")
        conv_kernel_num = self.config.get("model.conv_kernel_num", 400)
        query_dim = self.config.get("model.query_dim", 200)
        
        # Initialize model architecture
        news_encoder = DeBERTaNewsEncoder(
            pretrained=pretrained_model,
            conv_kernel_num=conv_kernel_num,
            kernel_size=3,
            query_dim=query_dim
        )
        user_encoder = UserEncoder(conv_kernel_num=conv_kernel_num, query_dim=query_dim)
        model = NAML(news_encoder=news_encoder, user_encoder=user_encoder)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if self.verbose:
                epoch = checkpoint.get("epoch", -1)
                metrics = checkpoint.get("metrics", {})
                logger.info(f"Loaded checkpoint from epoch {epoch}, metrics: {metrics}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_news_data(
        self,
        news_path: str,
        llm_description_path: Optional[str] = None
    ):
        """
        Load news data for inference.
        
        Args:
            news_path: Path to news.tsv file
            llm_description_path: Path to LLM descriptions JSON (optional)
            
        Raises:
            FileNotFoundError: If news file not found
        """
        from src.data.dataframe import read_news_df
        
        news_path = Path(news_path)
        if not news_path.exists():
            raise FileNotFoundError(f"News file not found: {news_path}")
        
        if self.verbose:
            logger.info(f"Loading news data from: {news_path}")
        
        self.news_df = read_news_df(news_path)
        
        # Load LLM descriptions if provided
        llm_descriptions = {}
        if llm_description_path:
            llm_desc_path = Path(llm_description_path)
            if llm_desc_path.exists():
                with open(llm_desc_path, 'r') as f:
                    llm_descriptions = json.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(llm_descriptions)} LLM descriptions")
        
        # Build news map
        self.news_map = self._build_news_map(self.news_df, llm_descriptions)
        
        if self.verbose:
            logger.info(f"Loaded {len(self.news_map)} news items")
    
    def _build_news_map(
        self,
        news_df: pl.DataFrame,
        llm_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """Build news ID to (title, description) mapping."""
        news_map = {}
        llm_desc = llm_descriptions or {}
        
        for i in range(len(news_df)):
            news_id = news_df[i]["news_id"].item()
            title = news_df[i]["title"].item()
            description = llm_desc.get(news_id, "")
            if not description:
                description = news_df[i].get("abstract", pl.Series([""]))[0] if "abstract" in news_df.columns else ""
            news_map[news_id] = (title, description)
        
        news_map[EMPTY_NEWS_ID] = ("", "")
        
        return news_map
    
    def recommend(
        self,
        candidate_news_ids: List[str],
        history_news_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> RecommendationResult:
        """
        Get news recommendations for a user.
        
        Args:
            candidate_news_ids: List of candidate news IDs to rank
            history_news_ids: List of user's history news IDs (optional)
            top_k: Number of top recommendations to return
            
        Returns:
            RecommendationResult with ranked news and scores
            
        Raises:
            InputValidationError: If input validation fails
            InferenceAPIError: If inference fails
        """
        # Validate inputs
        self._validate_inputs(candidate_news_ids, history_news_ids)
        
        # Prepare history
        history_news_ids = history_news_ids or []
        if len(history_news_ids) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history_news_ids))
        else:
            history_news_ids = history_news_ids[:self.history_size]
        
        # Prepare texts
        try:
            candidate_texts = [list(self.news_map.get(nid, ("", ""))) for nid in candidate_news_ids]
            history_texts = [list(self.news_map.get(nid, ("", ""))) for nid in history_news_ids]
        except Exception as e:
            raise InferenceAPIError(f"Error preparing news texts: {e}")
        
        # Tokenize
        try:
            candidate_batch = self.transform_fn(candidate_texts)  # (candidate_num, seq_len)
            history_batch = self.transform_fn(history_texts)  # (history_size, seq_len)
        except Exception as e:
            raise InferenceAPIError(f"Error tokenizing texts: {e}")
        
        # Add batch dimension
        candidate_batch = candidate_batch.unsqueeze(0)  # (1, candidate_num, seq_len)
        history_batch = history_batch.unsqueeze(0)  # (1, history_size, seq_len)
        user_id = torch.tensor([0], dtype=torch.long)  # Dummy user ID
        
        # Predict
        try:
            scores = self._predict(candidate_batch, history_batch, user_id)
        except Exception as e:
            raise InferenceAPIError(f"Error during model inference: {e}")
        
        # Rank news by scores
        ranked_indices = np.argsort(scores)[::-1]  # Descending order
        ranked_news_ids = [candidate_news_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        
        return RecommendationResult(
            news_ids=ranked_news_ids,
            scores=ranked_scores,
            top_k=top_k
        )
    
    @torch.no_grad()
    def _predict(
        self,
        candidate_news: torch.Tensor,
        news_histories: torch.Tensor,
        user_ids: torch.Tensor
    ) -> np.ndarray:
        """Perform model prediction."""
        # Move to device
        candidate_news = candidate_news.to(self.device)
        news_histories = news_histories.to(self.device)
        user_ids = user_ids.to(self.device)
        
        # Create dummy target
        batch_size, candidate_num = candidate_news.shape[0], candidate_news.shape[1]
        target = torch.zeros(batch_size, candidate_num).to(self.device)
        
        # Forward pass
        output = self.model(
            candidate_news=candidate_news,
            news_histories=news_histories,
            user_id=user_ids,
            target=target
        )
        
        logits = output.logits  # (batch_size, candidate_num)
        scores = torch.softmax(logits, dim=1)  # (batch_size, candidate_num)
        
        return scores[0].cpu().numpy()  # Return scores for single user
    
    def _validate_inputs(
        self,
        candidate_news_ids: List[str],
        history_news_ids: Optional[List[str]]
    ):
        """
        Validate input data.
        
        Raises:
            InputValidationError: If validation fails
        """
        # Check if news data is loaded
        if not self.news_map:
            raise InputValidationError(
                "News data not loaded. Call load_news_data() first."
            )
        
        # Validate candidate news IDs
        if not candidate_news_ids:
            raise InputValidationError("candidate_news_ids cannot be empty")
        
        if not isinstance(candidate_news_ids, list):
            raise InputValidationError("candidate_news_ids must be a list")
        
        # Check for missing news IDs
        missing_candidates = [nid for nid in candidate_news_ids if nid not in self.news_map]
        if missing_candidates:
            raise InputValidationError(
                f"Missing candidate news IDs in news data: {missing_candidates[:5]}"
            )
        
        # Validate history news IDs if provided
        if history_news_ids is not None:
            if not isinstance(history_news_ids, list):
                raise InputValidationError("history_news_ids must be a list")
            
            missing_history = [nid for nid in history_news_ids 
                             if nid not in self.news_map and nid != EMPTY_NEWS_ID]
            if missing_history:
                logger.warning(
                    f"Missing history news IDs in news data: {missing_history[:5]}"
                )
    
    def batch_recommend(
        self,
        requests: List[Dict[str, Union[List[str], int]]],
        top_k: int = 10
    ) -> List[RecommendationResult]:
        """
        Process multiple recommendation requests in batch.
        
        Args:
            requests: List of request dictionaries with keys:
                - candidate_news_ids: List of candidate news IDs
                - history_news_ids: List of history news IDs (optional)
                - top_k: Number of recommendations (optional)
            top_k: Default number of top recommendations
            
        Returns:
            List of RecommendationResult objects
            
        Raises:
            InputValidationError: If any request validation fails
        """
        results = []
        
        for i, request in enumerate(requests):
            try:
                candidate_news_ids = request.get("candidate_news_ids")
                history_news_ids = request.get("history_news_ids")
                request_top_k = request.get("top_k", top_k)
                
                result = self.recommend(
                    candidate_news_ids=candidate_news_ids,
                    history_news_ids=history_news_ids,
                    top_k=request_top_k
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing request {i}: {e}")
                raise InputValidationError(f"Error in request {i}: {e}")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "history_size": self.history_size,
            "num_news_loaded": len(self.news_map),
            "model_config": {
                "pretrained": self.config.get("model.pretrained", "microsoft/deberta-v3-base"),
                "conv_kernel_num": self.config.get("model.conv_kernel_num", 400),
                "query_dim": self.config.get("model.query_dim", 200),
                "max_length": self.config.get("model.max_length", 64)
            }
        }
