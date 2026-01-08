import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the code block being timed
        logger: Optional logger for output
        
    Example:
        >>> with timer("data loading"):
        ...     data = load_data()
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        message = f"{name} took {elapsed:.4f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)


def profile_function(output_file: Optional[str] = None):
    """
    Decorator to profile a function's execution.
    
    Args:
        output_file: Optional file to save profiling results
        
    Example:
        >>> @profile_function("profile_results.txt")
        ... def my_function():
        ...     # code here
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                
                # Print stats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                
                print(f"\n=== Profile for {func.__name__} ===")
                print(s.getvalue())
                
                # Save to file if specified
                if output_file:
                    with open(output_file, 'w') as f:
                        ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                        ps.print_stats()
                    print(f"Full profile saved to {output_file}")
            
            return result
        return wrapper
    return decorator


class MemoryTracker:
    """
    Track memory usage during execution.
    
    Useful for identifying memory leaks and optimizing memory consumption.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize memory tracker.
        
        Args:
            device: Device to track ("cuda" or "cpu")
        """
        self.device = device
        self.snapshots: Dict[str, Dict[str, float]] = {}
        
    def snapshot(self, name: str) -> None:
        """
        Take a memory snapshot.
        
        Args:
            name: Name for this snapshot
        """
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            self.snapshots[name] = {
                "allocated": allocated,
                "reserved": reserved,
                "max_allocated": max_allocated
            }
            
            logger.info(
                f"Memory snapshot '{name}': "
                f"allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max={max_allocated:.2f}GB"
            )
        else:
            # For CPU, we could use psutil here
            logger.warning(f"Memory tracking not available for device: {self.device}")
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("Reset peak memory statistics")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all snapshots.
        
        Returns:
            Dictionary of snapshot names to memory stats
        """
        return self.snapshots.copy()
    
    def print_summary(self) -> None:
        """Print a formatted summary of memory usage."""
        if not self.snapshots:
            print("No memory snapshots recorded")
            return
        
        print("\n=== Memory Usage Summary ===")
        print(f"{'Snapshot':<30} {'Allocated (GB)':<15} {'Reserved (GB)':<15} {'Max (GB)':<15}")
        print("-" * 75)
        
        for name, stats in self.snapshots.items():
            print(
                f"{name:<30} "
                f"{stats['allocated']:<15.2f} "
                f"{stats['reserved']:<15.2f} "
                f"{stats['max_allocated']:<15.2f}"
            )


class PerformanceMonitor:
    """
    Monitor performance metrics during training or inference.
    
    Tracks throughput, latency, and resource utilization.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list] = {
            "throughput": [],
            "latency": [],
            "batch_size": []
        }
        self.start_time: Optional[float] = None
        self.total_samples = 0
    
    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()
        self.total_samples = 0
    
    def record_batch(self, batch_size: int, batch_time: float) -> None:
        """
        Record metrics for a batch.
        
        Args:
            batch_size: Number of samples in the batch
            batch_time: Time taken to process the batch (seconds)
        """
        throughput = batch_size / batch_time if batch_time > 0 else 0
        latency = batch_time / batch_size if batch_size > 0 else 0
        
        self.metrics["throughput"].append(throughput)
        self.metrics["latency"].append(latency)
        self.metrics["batch_size"].append(batch_size)
        self.total_samples += batch_size
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics["throughput"]:
            return {}
        
        import numpy as np
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "total_samples": self.total_samples,
            "total_time": elapsed,
            "avg_throughput": np.mean(self.metrics["throughput"]),
            "avg_latency": np.mean(self.metrics["latency"]),
            "p50_latency": np.percentile(self.metrics["latency"], 50),
            "p95_latency": np.percentile(self.metrics["latency"], 95),
            "p99_latency": np.percentile(self.metrics["latency"], 99),
        }
    
    def print_summary(self) -> None:
        """Print formatted performance summary."""
        summary = self.get_summary()
        
        if not summary:
            print("No performance data recorded")
            return
        
        print("\n=== Performance Summary ===")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        print(f"Average throughput: {summary['avg_throughput']:.2f} samples/s")
        print(f"Average latency: {summary['avg_latency']*1000:.2f}ms")
        print(f"P50 latency: {summary['p50_latency']*1000:.2f}ms")
        print(f"P95 latency: {summary['p95_latency']*1000:.2f}ms")
        print(f"P99 latency: {summary['p99_latency']*1000:.2f}ms")


def optimize_dataloader_workers(
    dataset_size: int,
    batch_size: int,
    max_workers: int = 8
) -> int:
    """
    Calculate optimal number of dataloader workers.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Batch size
        max_workers: Maximum number of workers to use
        
    Returns:
        Recommended number of workers
    """
    # Rule of thumb: 4 * num_gpus, but cap at max_workers
    import torch
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    recommended = min(4 * num_gpus, max_workers)
    
    # For small datasets, fewer workers may be better
    num_batches = dataset_size // batch_size
    if num_batches < recommended:
        recommended = max(1, num_batches // 2)
    
    logger.info(f"Recommended dataloader workers: {recommended}")
    return recommended
