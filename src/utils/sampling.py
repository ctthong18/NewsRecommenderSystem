import random
import numpy as np
import torch
from typing import List, Union, Tuple, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Sampling strategy options for hard negative sampling."""
    HARDEST = "hardest"  # Select only the hardest negatives
    MIXED = "mixed"  # Mix of hard and random negatives
    SEMI_HARD = "semi_hard"  # Select semi-hard negatives (medium difficulty)


class NegativeSampler:
    """
    Optimized negative sampler with multiple strategies and monitoring.
    
    Features:
    - Efficient batch similarity computation
    - Multiple sampling strategies (hardest, mixed, semi-hard)
    - Memory-efficient processing for large candidate sets
    - Statistics tracking for monitoring
    """
    
    def __init__(
        self,
        strategy: Union[str, SamplingStrategy] = SamplingStrategy.HARDEST,
        batch_size: int = 1000,
        mixed_ratio: float = 0.7,
        track_stats: bool = True,
        monitor: Optional[Any] = None
    ):
        """
        Initialize negative sampler.
        
        Args:
            strategy: Sampling strategy to use
            batch_size: Batch size for processing large candidate sets
            mixed_ratio: Ratio of hard negatives in mixed strategy (0.0-1.0)
            track_stats: Whether to track sampling statistics
            monitor: Optional SamplingMonitor instance for detailed tracking
        """
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        self.strategy = strategy
        self.batch_size = batch_size
        self.mixed_ratio = max(0.0, min(1.0, mixed_ratio))
        self.track_stats = track_stats
        self.monitor = monitor
        
        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "avg_similarity": [],
            "min_similarity": [],
            "max_similarity": [],
            "strategy_counts": {s.value: 0 for s in SamplingStrategy}
        }
    
    def sample(
        self,
        pos_emb: Union[np.ndarray, torch.Tensor],
        neg_pool_emb: Union[np.ndarray, torch.Tensor],
        k: int = 5,
        return_scores: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Sample negative examples based on configured strategy.
        
        Args:
            pos_emb: (d,) - positive news embedding
            neg_pool_emb: (N, d) - negative news embeddings pool
            k: number of negatives to select
            return_scores: whether to return similarity scores
        
        Returns:
            indices: list of indices of sampled negatives
            scores: (optional) similarity scores of sampled negatives
        """
        # Convert to numpy for efficient computation
        pos_emb = self._to_numpy(pos_emb)
        neg_pool_emb = self._to_numpy(neg_pool_emb)
        
        # Validate inputs
        if len(neg_pool_emb) == 0:
            logger.warning("Empty negative pool provided")
            return ([], np.array([])) if return_scores else []
        
        k = min(k, len(neg_pool_emb))
        
        # Compute similarities efficiently
        similarities = self._compute_similarities_batched(pos_emb, neg_pool_emb)
        
        # Sample based on strategy
        if self.strategy == SamplingStrategy.HARDEST:
            indices = self._sample_hardest(similarities, k)
        elif self.strategy == SamplingStrategy.MIXED:
            indices = self._sample_mixed(similarities, k)
        elif self.strategy == SamplingStrategy.SEMI_HARD:
            indices = self._sample_semi_hard(similarities, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Track statistics
        if self.track_stats:
            self._update_stats(similarities, indices)
        
        # Log to monitor if available
        if self.monitor is not None:
            self.monitor.log_sampling(
                similarities=similarities,
                sampled_indices=indices,
                strategy=self.strategy.value
            )
        
        if return_scores:
            return indices, similarities[indices]
        return indices
    
    def _to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _compute_similarities_batched(
        self,
        pos_emb: np.ndarray,
        neg_pool_emb: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities efficiently in batches.
        
        Args:
            pos_emb: (d,) - positive embedding
            neg_pool_emb: (N, d) - negative embeddings
        
        Returns:
            similarities: (N,) - cosine similarities
        """
        # Normalize embeddings for efficient cosine similarity
        pos_norm = np.linalg.norm(pos_emb) + 1e-9
        pos_emb_normalized = pos_emb / pos_norm
        
        n_negatives = len(neg_pool_emb)
        similarities = np.zeros(n_negatives, dtype=np.float32)
        
        # Process in batches to manage memory
        for i in range(0, n_negatives, self.batch_size):
            end_idx = min(i + self.batch_size, n_negatives)
            batch = neg_pool_emb[i:end_idx]
            
            # Normalize batch
            batch_norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-9
            batch_normalized = batch / batch_norms
            
            # Compute cosine similarity: dot product of normalized vectors
            similarities[i:end_idx] = np.dot(batch_normalized, pos_emb_normalized)
        
        return similarities
    
    def _sample_hardest(self, similarities: np.ndarray, k: int) -> List[int]:
        """
        Sample the k hardest (most similar) negatives.
        
        Args:
            similarities: (N,) - similarity scores
            k: number of samples
        
        Returns:
            indices: list of selected indices
        """
        # Use argpartition for efficient top-k selection (O(n) vs O(n log n))
        if k < len(similarities):
            # Get indices of k largest values
            indices = np.argpartition(similarities, -k)[-k:]
            # Sort these k indices by similarity (descending)
            indices = indices[np.argsort(similarities[indices])[::-1]]
        else:
            indices = np.argsort(similarities)[::-1]
        
        return indices.tolist()
    
    def _sample_mixed(self, similarities: np.ndarray, k: int) -> List[int]:
        """
        Sample a mix of hard and random negatives.
        
        Args:
            similarities: (N,) - similarity scores
            k: number of samples
        
        Returns:
            indices: list of selected indices
        """
        k_hard = int(k * self.mixed_ratio)
        k_random = k - k_hard
        
        # Sample hard negatives
        hard_indices = self._sample_hardest(similarities, k_hard)
        
        # Sample random negatives from remaining pool
        remaining_indices = list(set(range(len(similarities))) - set(hard_indices))
        if remaining_indices and k_random > 0:
            k_random = min(k_random, len(remaining_indices))
            random_indices = random.sample(remaining_indices, k_random)
            return hard_indices + random_indices
        
        return hard_indices
    
    def _sample_semi_hard(self, similarities: np.ndarray, k: int) -> List[int]:
        """
        Sample semi-hard negatives (medium difficulty).
        Semi-hard negatives are those with similarity in the middle range.
        
        Args:
            similarities: (N,) - similarity scores
            k: number of samples
        
        Returns:
            indices: list of selected indices
        """
        # Sort by similarity
        sorted_indices = np.argsort(similarities)
        
        # Select from middle range (25th to 75th percentile)
        n = len(sorted_indices)
        start_idx = n // 4
        end_idx = 3 * n // 4
        
        semi_hard_pool = sorted_indices[start_idx:end_idx]
        
        if len(semi_hard_pool) >= k:
            # Randomly sample from semi-hard pool
            selected = np.random.choice(semi_hard_pool, size=k, replace=False)
        else:
            # If not enough semi-hard, take all and fill with hardest
            selected = semi_hard_pool.tolist()
            remaining = k - len(selected)
            hardest = sorted_indices[-remaining:]
            selected = np.concatenate([selected, hardest])
        
        return selected.tolist()
    
    def _update_stats(self, similarities: np.ndarray, indices: List[int]):
        """Update sampling statistics."""
        if len(indices) == 0:
            return
        
        sampled_sims = similarities[indices]
        self.stats["total_samples"] += len(indices)
        self.stats["avg_similarity"].append(float(np.mean(sampled_sims)))
        self.stats["min_similarity"].append(float(np.min(sampled_sims)))
        self.stats["max_similarity"].append(float(np.max(sampled_sims)))
        self.stats["strategy_counts"][self.strategy.value] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get sampling statistics.
        
        Returns:
            Dictionary containing sampling statistics
        """
        if not self.track_stats or self.stats["total_samples"] == 0:
            return {}
        
        return {
            "total_samples": self.stats["total_samples"],
            "avg_similarity_mean": float(np.mean(self.stats["avg_similarity"])),
            "avg_similarity_std": float(np.std(self.stats["avg_similarity"])),
            "min_similarity": float(np.min(self.stats["min_similarity"])),
            "max_similarity": float(np.max(self.stats["max_similarity"])),
            "strategy_distribution": self.stats["strategy_counts"]
        }
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.stats = {
            "total_samples": 0,
            "avg_similarity": [],
            "min_similarity": [],
            "max_similarity": [],
            "strategy_counts": {s.value: 0 for s in SamplingStrategy}
        }


# Backward compatibility functions
def hard_negative_sampling(
    pos_emb: Union[np.ndarray, torch.Tensor],
    neg_pool_emb: Union[np.ndarray, torch.Tensor],
    k: int = 5,
    strategy: str = "hardest"
) -> List[int]:
    """
    Sample hard negatives using the optimized sampler.
    
    Args:
        pos_emb: (d,) - positive news embedding
        neg_pool_emb: (N, d) - negative news embeddings pool
        k: number of hard negatives to select
        strategy: sampling strategy ("hardest", "mixed", "semi_hard")
    
    Returns:
        indices: list of indices of hard negatives
    """
    sampler = NegativeSampler(strategy=strategy, track_stats=False)
    return sampler.sample(pos_emb, neg_pool_emb, k)


def random_negative_sampling(neg_candidates: List[int], k: int = 5) -> List[int]:
    """
    Random negative sampling.
    
    Args:
        neg_candidates: list of negative candidate indices
        k: number of negatives to sample
    
    Returns:
        sampled indices
    """
    return random.sample(neg_candidates, min(k, len(neg_candidates)))
