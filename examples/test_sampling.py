"""
Example script demonstrating the improved hard negative sampling functionality.

This script shows how to:
1. Use different sampling strategies
2. Track sampling statistics
3. Monitor sampling effectiveness
4. Generate visualizations
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.sampling import NegativeSampler, SamplingStrategy
from src.utils.sampling_monitor import SamplingMonitor, create_sampling_visualizations


def generate_sample_embeddings(n_samples: int = 100, dim: int = 768):
    """Generate sample embeddings for testing."""
    # Generate random embeddings
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    # Normalize to unit length
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    return embeddings


def demo_sampling_strategies():
    """Demonstrate different sampling strategies."""
    print("=" * 60)
    print("DEMO: Sampling Strategies")
    print("=" * 60)
    
    # Generate sample data
    pos_emb = generate_sample_embeddings(1, 768)[0]
    neg_pool = generate_sample_embeddings(100, 768)
    
    strategies = [
        SamplingStrategy.HARDEST,
        SamplingStrategy.MIXED,
        SamplingStrategy.SEMI_HARD
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.value}")
        sampler = NegativeSampler(strategy=strategy, track_stats=True)
        
        # Sample negatives
        indices, scores = sampler.sample(pos_emb, neg_pool, k=10, return_scores=True)
        
        print(f"  Sampled {len(indices)} negatives")
        print(f"  Similarity range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Mean similarity: {scores.mean():.4f}")


def demo_monitoring():
    """Demonstrate sampling monitoring and statistics."""
    print("\n" + "=" * 60)
    print("DEMO: Sampling Monitoring")
    print("=" * 60)
    
    # Create monitor
    monitor = SamplingMonitor(log_dir="output/sampling_logs", log_interval=10)
    
    # Create sampler with monitor
    sampler = NegativeSampler(
        strategy=SamplingStrategy.HARDEST,
        track_stats=True,
        monitor=monitor
    )
    
    # Simulate multiple sampling calls
    print("\nSimulating 50 sampling calls...")
    for i in range(50):
        pos_emb = generate_sample_embeddings(1, 768)[0]
        neg_pool = generate_sample_embeddings(100, 768)
        
        indices = sampler.sample(pos_emb, neg_pool, k=5)
    
    # Get statistics
    print("\nSampler Statistics:")
    stats = sampler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get monitor statistics
    print("\nMonitor Statistics:")
    dist_stats = monitor.get_distribution_stats()
    print(f"  Total samples: {dist_stats['total_samples']}")
    print(f"  Mean similarity: {dist_stats['mean']:.4f}")
    print(f"  Std similarity: {dist_stats['std']:.4f}")
    
    # Get effectiveness metrics
    print("\nEffectiveness Metrics:")
    eff_metrics = monitor.get_effectiveness_metrics()
    for key, value in eff_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Generate report
    print("\n" + monitor.generate_report())
    
    # Save summary
    monitor.save_summary()
    print("\nSummary saved to output/sampling_logs/sampling_summary.json")
    
    return monitor


def demo_visualization(monitor: SamplingMonitor):
    """Demonstrate visualization generation."""
    print("\n" + "=" * 60)
    print("DEMO: Visualization Generation")
    print("=" * 60)
    
    try:
        create_sampling_visualizations(monitor, "output/sampling_logs")
        print("\nVisualizations created successfully!")
        print("Check output/sampling_logs/ for:")
        print("  - similarity_distribution.png")
        print("  - similarity_over_time.png")
        print("  - similarity_boxplot.png")
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        print("Note: Install matplotlib to generate visualizations:")
        print("  pip install matplotlib")


def demo_memory_efficiency():
    """Demonstrate memory-efficient processing of large candidate sets."""
    print("\n" + "=" * 60)
    print("DEMO: Memory-Efficient Large-Scale Sampling")
    print("=" * 60)
    
    # Test with large candidate set
    print("\nTesting with 10,000 candidates...")
    pos_emb = generate_sample_embeddings(1, 768)[0]
    neg_pool = generate_sample_embeddings(10000, 768)
    
    # Use small batch size for memory efficiency
    sampler = NegativeSampler(
        strategy=SamplingStrategy.HARDEST,
        batch_size=500,  # Process in batches of 500
        track_stats=True
    )
    
    import time
    start_time = time.time()
    indices, scores = sampler.sample(pos_emb, neg_pool, k=20, return_scores=True)
    elapsed = time.time() - start_time
    
    print(f"  Sampled {len(indices)} negatives from 10,000 candidates")
    print(f"  Time elapsed: {elapsed:.4f} seconds")
    print(f"  Similarity range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean similarity: {scores.mean():.4f}")


if __name__ == "__main__":
    print("Hard Negative Sampling - Feature Demonstration\n")
    
    # Run demos
    demo_sampling_strategies()
    monitor = demo_monitoring()
    demo_visualization(monitor)
    demo_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
