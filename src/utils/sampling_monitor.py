"""
Sampling statistics monitoring and visualization utilities.

This module provides tools to track, log, and visualize hard negative sampling
effectiveness during training.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SamplingMonitor:
    """
    Monitor and track sampling statistics over time.
    
    Features:
    - Track sampling distribution over training
    - Log sampling effectiveness metrics
    - Export statistics for analysis
    - Generate summary reports
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 100
    ):
        """
        Initialize sampling monitor.
        
        Args:
            log_dir: Directory to save monitoring logs
            log_interval: Log statistics every N sampling calls
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_interval = log_interval
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking data
        self.history = {
            "timestamps": [],
            "similarities": [],
            "sample_counts": [],
            "strategies": []
        }
        self.call_count = 0
    
    def log_sampling(
        self,
        similarities: np.ndarray,
        sampled_indices: List[int],
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a sampling event.
        
        Args:
            similarities: All similarity scores
            sampled_indices: Indices of sampled negatives
            strategy: Sampling strategy used
            metadata: Additional metadata to log
        """
        self.call_count += 1
        
        # Extract sampled similarities
        sampled_sims = similarities[sampled_indices] if len(sampled_indices) > 0 else np.array([])
        
        # Store in history
        self.history["timestamps"].append(datetime.now().isoformat())
        self.history["similarities"].append(sampled_sims.tolist())
        self.history["sample_counts"].append(len(sampled_indices))
        self.history["strategies"].append(strategy)
        
        # Log at intervals
        if self.call_count % self.log_interval == 0:
            stats = self._compute_recent_stats()
            logger.info(
                f"Sampling stats (last {self.log_interval} calls): "
                f"avg_similarity={stats['avg_similarity']:.4f}, "
                f"std_similarity={stats['std_similarity']:.4f}, "
                f"min={stats['min_similarity']:.4f}, "
                f"max={stats['max_similarity']:.4f}"
            )
            
            # Save checkpoint
            if self.log_dir:
                self._save_checkpoint()
    
    def _compute_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Compute statistics for recent sampling events.
        
        Args:
            window: Number of recent events to consider
        
        Returns:
            Dictionary of statistics
        """
        recent_sims = self.history["similarities"][-window:]
        all_sims = [sim for batch in recent_sims for sim in batch]
        
        if not all_sims:
            return {
                "avg_similarity": 0.0,
                "std_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0
            }
        
        return {
            "avg_similarity": float(np.mean(all_sims)),
            "std_similarity": float(np.std(all_sims)),
            "min_similarity": float(np.min(all_sims)),
            "max_similarity": float(np.max(all_sims))
        }
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive distribution statistics.
        
        Returns:
            Dictionary containing distribution statistics
        """
        all_sims = [sim for batch in self.history["similarities"] for sim in batch]
        
        if not all_sims:
            return {}
        
        all_sims = np.array(all_sims)
        
        return {
            "total_samples": len(all_sims),
            "mean": float(np.mean(all_sims)),
            "std": float(np.std(all_sims)),
            "min": float(np.min(all_sims)),
            "max": float(np.max(all_sims)),
            "percentiles": {
                "25th": float(np.percentile(all_sims, 25)),
                "50th": float(np.percentile(all_sims, 50)),
                "75th": float(np.percentile(all_sims, 75)),
                "90th": float(np.percentile(all_sims, 90)),
                "95th": float(np.percentile(all_sims, 95))
            },
            "strategy_counts": self._count_strategies()
        }
    
    def _count_strategies(self) -> Dict[str, int]:
        """Count usage of each strategy."""
        from collections import Counter
        return dict(Counter(self.history["strategies"]))
    
    def get_effectiveness_metrics(self) -> Dict[str, float]:
        """
        Compute metrics to assess sampling effectiveness.
        
        Returns:
            Dictionary of effectiveness metrics
        """
        all_sims = [sim for batch in self.history["similarities"] for sim in batch]
        
        if not all_sims:
            return {}
        
        all_sims = np.array(all_sims)
        
        # Higher similarity means harder negatives (more effective)
        # Compute trend over time to see if model is learning
        batch_means = [np.mean(batch) if len(batch) > 0 else 0 
                      for batch in self.history["similarities"]]
        
        effectiveness = {
            "avg_hardness": float(np.mean(all_sims)),
            "hardness_std": float(np.std(all_sims)),
            "consistency": float(1.0 - np.std(batch_means) / (np.mean(batch_means) + 1e-9))
        }
        
        # Compute trend (are negatives getting easier over time? = model learning)
        if len(batch_means) > 10:
            # Simple linear trend
            x = np.arange(len(batch_means))
            y = np.array(batch_means)
            trend = np.polyfit(x, y, 1)[0]  # Slope
            effectiveness["hardness_trend"] = float(trend)
        
        return effectiveness
    
    def _save_checkpoint(self):
        """Save current statistics to disk."""
        if not self.log_dir:
            return
        
        checkpoint_file = self.log_dir / f"sampling_stats_{self.call_count}.json"
        
        stats = {
            "call_count": self.call_count,
            "distribution": self.get_distribution_stats(),
            "effectiveness": self.get_effectiveness_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def save_summary(self, filepath: Optional[str] = None):
        """
        Save comprehensive summary of sampling statistics.
        
        Args:
            filepath: Path to save summary. If None, uses log_dir
        """
        if filepath is None:
            if self.log_dir is None:
                logger.warning("No log directory specified, cannot save summary")
                return
            filepath = self.log_dir / "sampling_summary.json"
        else:
            filepath = Path(filepath)
        
        summary = {
            "total_calls": self.call_count,
            "distribution_stats": self.get_distribution_stats(),
            "effectiveness_metrics": self.get_effectiveness_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sampling summary saved to {filepath}")
    
    def generate_report(self) -> str:
        """
        Generate a human-readable text report.
        
        Returns:
            Formatted report string
        """
        dist_stats = self.get_distribution_stats()
        eff_metrics = self.get_effectiveness_metrics()
        
        if not dist_stats:
            return "No sampling data available"
        
        report = []
        report.append("=" * 60)
        report.append("SAMPLING STATISTICS REPORT")
        report.append("=" * 60)
        report.append(f"Total sampling calls: {self.call_count}")
        report.append(f"Total samples: {dist_stats['total_samples']}")
        report.append("")
        
        report.append("Distribution Statistics:")
        report.append(f"  Mean similarity: {dist_stats['mean']:.4f}")
        report.append(f"  Std deviation: {dist_stats['std']:.4f}")
        report.append(f"  Min similarity: {dist_stats['min']:.4f}")
        report.append(f"  Max similarity: {dist_stats['max']:.4f}")
        report.append("")
        
        report.append("Percentiles:")
        for pct, val in dist_stats['percentiles'].items():
            report.append(f"  {pct}: {val:.4f}")
        report.append("")
        
        report.append("Strategy Usage:")
        for strategy, count in dist_stats['strategy_counts'].items():
            report.append(f"  {strategy}: {count}")
        report.append("")
        
        if eff_metrics:
            report.append("Effectiveness Metrics:")
            report.append(f"  Average hardness: {eff_metrics['avg_hardness']:.4f}")
            report.append(f"  Hardness std: {eff_metrics['hardness_std']:.4f}")
            report.append(f"  Consistency: {eff_metrics['consistency']:.4f}")
            if 'hardness_trend' in eff_metrics:
                trend_dir = "decreasing" if eff_metrics['hardness_trend'] < 0 else "increasing"
                report.append(f"  Hardness trend: {trend_dir} ({eff_metrics['hardness_trend']:.6f})")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def reset(self):
        """Reset all tracking data."""
        self.history = {
            "timestamps": [],
            "similarities": [],
            "sample_counts": [],
            "strategies": []
        }
        self.call_count = 0


def create_sampling_visualizations(
    monitor: SamplingMonitor,
    output_dir: str
):
    """
    Create visualizations of sampling statistics.
    
    Note: This function requires matplotlib. If not available,
    it will log a warning and skip visualization.
    
    Args:
        monitor: SamplingMonitor instance with data
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not available. Install it to generate visualizations: "
            "pip install matplotlib"
        )
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_sims = [sim for batch in monitor.history["similarities"] for sim in batch]
    
    if not all_sims:
        logger.warning("No sampling data available for visualization")
        return
    
    # 1. Histogram of similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_sims, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sampled Negative Similarities')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'similarity_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Similarity over time (batch means)
    batch_means = [np.mean(batch) if len(batch) > 0 else 0 
                  for batch in monitor.history["similarities"]]
    
    plt.figure(figsize=(12, 6))
    plt.plot(batch_means, alpha=0.7, linewidth=1)
    plt.xlabel('Sampling Call')
    plt.ylabel('Mean Similarity')
    plt.title('Average Similarity of Sampled Negatives Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(batch_means) > 10:
        x = np.arange(len(batch_means))
        z = np.polyfit(x, batch_means, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.6f})')
        plt.legend()
    
    plt.savefig(output_path / 'similarity_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot of similarities
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_sims, vert=True)
    plt.ylabel('Similarity Score')
    plt.title('Box Plot of Sampled Negative Similarities')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'similarity_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_path}")
