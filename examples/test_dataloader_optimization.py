"""
Example script to test and benchmark dataloader optimizations.

This script demonstrates the performance improvements from:
- Prefetching with multiple workers
- Pin memory optimization
- Optimized collate function
- Dataset memory optimizations
"""
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer

from src.data.dataset_mind import MINDTrainDataset, MINDValDataset
from src.data.dataloader_builder import build_train_dataloader, build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.const.path import MIND_SMALL_TRAIN_DATASET_DIR


def benchmark_dataloader(dataloader, num_batches=50, description="DataLoader"):
    """
    Benchmark dataloader performance.
    
    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to iterate
        description: Description for logging
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {description}")
    print(f"{'='*60}")
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Simulate some processing
        _ = batch["candidate_news"].shape
        _ = batch["news_histories"].shape
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Results
    throughput = num_batches / elapsed
    avg_batch_time = elapsed / num_batches
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Avg batch time: {avg_batch_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} batches/s")
    
    return {
        "elapsed": elapsed,
        "avg_batch_time": avg_batch_time,
        "throughput": throughput,
    }


def main():
    """Main benchmark function."""
    print("Data Loading Optimization Benchmark")
    print("="*60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_news_path = MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv"
    train_behavior_path = MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv"
    
    train_news_df = read_news_df(train_news_path)
    train_behavior_df = read_behavior_df(train_behavior_path)
    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df)
    
    # Create tokenizer
    pretrained_model = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(
        tokenizer=tokenizer,
        max_title_length=30,
        max_abstract_length=100,
        device=device,
    )
    
    # Create dataset
    print("Creating dataset...")
    dataset = MINDTrainDataset(
        behavior_df=train_behavior_df,
        news_df=train_news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        npratio=4,
        history_size=50,
        device=device,
        enable_tokenization_cache=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Benchmark 1: Optimized dataloader (default)
    print("\n" + "="*60)
    print("Test 1: Optimized DataLoader (with prefetching)")
    print("="*60)
    
    optimized_loader = build_train_dataloader(
        dataset=dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=(device == "cuda"),
        prefetch_factor=2,
    )
    
    results_optimized = benchmark_dataloader(
        optimized_loader,
        num_batches=50,
        description="Optimized (workers=4, prefetch=2, pin_memory=True)"
    )
    
    # Benchmark 2: Basic dataloader (no prefetching)
    print("\n" + "="*60)
    print("Test 2: Basic DataLoader (no prefetching)")
    print("="*60)
    
    from torch.utils.data import DataLoader
    basic_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    results_basic = benchmark_dataloader(
        basic_loader,
        num_batches=50,
        description="Basic (workers=0, no prefetch, pin_memory=False)"
    )
    
    # Benchmark 3: With workers but no prefetch
    print("\n" + "="*60)
    print("Test 3: DataLoader with workers (no prefetch)")
    print("="*60)
    
    workers_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    
    results_workers = benchmark_dataloader(
        workers_loader,
        num_batches=50,
        description="Workers only (workers=4, no prefetch)"
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    speedup_vs_basic = results_basic["elapsed"] / results_optimized["elapsed"]
    speedup_vs_workers = results_workers["elapsed"] / results_optimized["elapsed"]
    
    print(f"\nOptimized vs Basic:")
    print(f"  Speedup: {speedup_vs_basic:.2f}x faster")
    print(f"  Time saved: {results_basic['elapsed'] - results_optimized['elapsed']:.2f}s")
    
    print(f"\nOptimized vs Workers-only:")
    print(f"  Speedup: {speedup_vs_workers:.2f}x faster")
    print(f"  Time saved: {results_workers['elapsed'] - results_optimized['elapsed']:.2f}s")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
