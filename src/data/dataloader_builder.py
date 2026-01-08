# src/data/dataloader_builder.py
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def collate_fn_optimized(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Optimized collate function for batching MIND dataset samples.
    
    Efficiently stacks tensors and handles variable-length sequences.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if not batch:
        return {}
    
    # Pre-allocate lists for better performance
    candidate_news_list = []
    news_histories_list = []
    user_id_list = []
    target_list = []
    
    # Collect all tensors
    for sample in batch:
        candidate_news_list.append(sample["candidate_news"])
        news_histories_list.append(sample["news_histories"])
        user_id_list.append(sample["user_id"])
        target_list.append(sample["target"])
    
    # Stack tensors efficiently
    # Use torch.stack for better performance than manual concatenation
    batched = {
        "candidate_news": torch.stack(candidate_news_list, dim=0),
        "news_histories": torch.stack(news_histories_list, dim=0),
        "user_id": torch.stack(user_id_list, dim=0),
        "target": torch.stack(target_list, dim=0),
    }
    
    return batched


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    collate_fn: Optional[Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]] = None,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build an optimized DataLoader with prefetching and memory optimizations.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function (uses optimized default if None)
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Configured DataLoader instance
    """
    # Use optimized collate_fn by default
    if collate_fn is None:
        collate_fn = collate_fn_optimized
    
    # Adjust num_workers based on batch_size and dataset size
    if len(dataset) < batch_size * num_workers:
        # For small datasets, reduce workers to avoid overhead
        num_workers = max(1, len(dataset) // batch_size)
        logger.info(f"Reduced num_workers to {num_workers} for small dataset")
    
    # Disable persistent_workers if num_workers is 0
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    
    logger.info(
        f"DataLoader created: batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, prefetch_factor={prefetch_factor}, "
        f"persistent_workers={persistent_workers}"
    )
    
    return dataloader


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build optimized training DataLoader with shuffling and prefetching.
    
    Args:
        dataset: Training dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Use pinned memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        Configured training DataLoader
    """
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=True,  # Drop last incomplete batch for stable training
    )


def build_val_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build optimized validation DataLoader without shuffling.
    
    Args:
        dataset: Validation dataset
        batch_size: Batch size for validation (typically 1 for proper evaluation)
        num_workers: Number of data loading workers
        pin_memory: Use pinned memory for faster GPU transfer
        
    Returns:
        Configured validation DataLoader
    """
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=False,  # Don't need persistent workers for validation
        drop_last=False,
    )
