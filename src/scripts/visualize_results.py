"""
Visualization and reporting tools for news recommendation model results.

Features:
- Plot metrics over training epochs
- Visualize attention weights  
- Create confusion matrix for categories
- Generate HTML report
"""
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.checkpoint_manager import CheckpointManager
from src.utils.logger import get_logger

# Optional imports for model-specific features
try:
    from src.models.NAML import NAML
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    MODEL_IMPORTS_AVAILABLE = False
    print(f"Warning: Model imports failed: {e}")

logger = get_logger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultVisualizer:
    """Visualizer for model training and evaluation results."""
    
    def __init__(self, output_dir: str = "output/figures"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer initialized. Output directory: {self.output_dir}")
    
    def plot_training_metrics(self, checkpoint_dir: str, save_path: Optional[str] = None) -> str:
        """Plot metrics over training epochs from checkpoint metadata."""
        logger.info(f"Loading checkpoint metadata from {checkpoint_dir}")
        
        checkpoint_dir = Path(checkpoint_dir)
        metadata_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*_metadata.json"))
        
        if not metadata_files:
            logger.warning("No checkpoint metadata files found")
            return None
        
        # Extract metrics from metadata
        epochs = []
        metrics_data = {"ndcg_at_5": [], "ndcg_at_10": [], "auc": [], "mrr": []}
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                epochs.append(metadata["epoch"])
                for metric_name in metrics_data.keys():
                    value = metadata.get("metrics", {}).get(metric_name, None)
                    metrics_data[metric_name].append(value)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
        
        metric_labels = {"ndcg_at_5": "nDCG@5", "ndcg_at_10": "nDCG@10", "auc": "AUC", "mrr": "MRR"}
        
        for ax, metric_name in zip(axes.flat, metrics_data.keys()):
            values = metrics_data[metric_name]
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if not valid_values:
                ax.text(0.5, 0.5, f"No data for {metric_labels[metric_name]}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            ax.plot(valid_epochs, valid_values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(metric_labels[metric_name], fontweight='bold')
            ax.set_title(f'{metric_labels[metric_name]} over Training', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            best_idx = np.argmax(valid_values)
            ax.plot(valid_epochs[best_idx], valid_values[best_idx], 
                   'r*', markersize=15, label=f'Best: {valid_values[best_idx]:.4f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "training_metrics.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training metrics plot saved to {save_path}")
        plt.close()
        
        return str(save_path)

    def create_category_confusion_matrix(self, category_metrics: Dict[str, Dict[str, float]], 
                                         save_path: Optional[str] = None) -> str:
        """Create a visualization showing performance across categories."""
        logger.info("Creating category performance visualization")
        
        if not category_metrics:
            logger.warning("No category metrics provided")
            return None
        
        categories = list(category_metrics.keys())
        metrics_names = ["ndcg_at_5", "ndcg_at_10", "auc", "mrr"]
        
        data_matrix = []
        for metric_name in metrics_names:
            row = [category_metrics[cat].get(metric_name, 0) for cat in categories]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 6))
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(metrics_names)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticklabels(['nDCG@5', 'nDCG@10', 'AUC', 'MRR'])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Metric Value', rotation=270, labelpad=20, fontweight='bold')
        
        for i in range(len(metrics_names)):
            for j in range(len(categories)):
                ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Performance by Category', fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "category_performance.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Category performance plot saved to {save_path}")
        plt.close()
        
        return str(save_path)
    
    def plot_category_comparison(self, category_metrics: Dict[str, Dict[str, float]], 
                                 save_path: Optional[str] = None) -> str:
        """Create bar chart comparing categories by nDCG@10."""
        logger.info("Creating category comparison chart")
        
        if not category_metrics:
            logger.warning("No category metrics provided")
            return None
        
        sorted_categories = sorted(category_metrics.items(),
                                  key=lambda x: x[1].get("ndcg_at_10", 0), reverse=True)
        
        categories = [cat for cat, _ in sorted_categories]
        ndcg_values = [metrics.get("ndcg_at_10", 0) for _, metrics in sorted_categories]
        ndcg_stds = [metrics.get("ndcg_at_10_std", 0) for _, metrics in sorted_categories]
        sample_counts = [metrics.get("num_samples", 0) for _, metrics in sorted_categories]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        x_pos = np.arange(len(categories))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
        
        bars = ax1.bar(x_pos, ndcg_values, yerr=ndcg_stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Category', fontweight='bold', fontsize=12)
        ax1.set_ylabel('nDCG@10', fontweight='bold', fontsize=12)
        ax1.set_title('Category Performance Comparison (nDCG@10)', fontweight='bold', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(ndcg_values) * 1.2 if ndcg_values else 1)
        
        for bar, val in zip(bars, ndcg_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        bars2 = ax2.bar(x_pos, sample_counts, color='steelblue', alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Category', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax2.set_title('Sample Distribution by Category', fontweight='bold', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars2, sample_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "category_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Category comparison plot saved to {save_path}")
        plt.close()
        
        return str(save_path)

    def generate_html_report(self, checkpoint_dir: str, 
                            category_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                            overall_metrics: Optional[Dict[str, float]] = None,
                            save_path: Optional[str] = None) -> str:
        """Generate comprehensive HTML report with all visualizations."""
        logger.info("Generating HTML report")
        
        training_plot = self.plot_training_metrics(checkpoint_dir)
        
        category_performance_plot = None
        category_comparison_plot = None
        if category_metrics:
            category_performance_plot = self.create_category_confusion_matrix(category_metrics)
            category_comparison_plot = self.plot_category_comparison(category_metrics)
        
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        checkpoints = checkpoint_manager.list_checkpoints()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Recommendation Model - Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .metric-label {{ font-size: 14px; opacity: 0.9; margin-bottom: 5px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .plot-container {{ margin: 30px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; margin-top: 30px; text-align: center; }}
        .best-badge {{ background-color: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 News Recommendation Model - Evaluation Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        if overall_metrics:
            html_content += '<h2>Overall Performance Metrics</h2><div class="metric-grid">'
            metric_labels = {"ndcg_at_5": "nDCG@5", "ndcg_at_10": "nDCG@10", "auc": "AUC", "mrr": "MRR"}
            for metric_key, metric_label in metric_labels.items():
                if metric_key in overall_metrics:
                    value = overall_metrics[metric_key]
                    html_content += f'<div class="metric-card"><div class="metric-label">{metric_label}</div><div class="metric-value">{value:.4f}</div></div>'
            html_content += '</div>'
        
        if training_plot:
            html_content += f'<h2>Training Progress</h2><div class="plot-container"><img src="{Path(training_plot).name}" alt="Training Metrics"></div>'
        
        if category_performance_plot:
            html_content += f'<h2>Performance by Category</h2><div class="plot-container"><img src="{Path(category_performance_plot).name}" alt="Category Performance"></div>'
        
        if category_comparison_plot:
            html_content += f'<div class="plot-container"><img src="{Path(category_comparison_plot).name}" alt="Category Comparison"></div>'
        
        if checkpoints:
            html_content += '<h2>Training Checkpoints</h2><table><thead><tr><th>Epoch</th><th>nDCG@10</th><th>nDCG@5</th><th>AUC</th><th>MRR</th><th>Timestamp</th><th>Status</th></tr></thead><tbody>'
            for ckpt in checkpoints:
                metrics = ckpt.get("metrics", {})
                is_best = ckpt.get("is_best", False)
                best_badge = '<span class="best-badge">BEST</span>' if is_best else ''
                html_content += f'<tr><td>{ckpt.get("epoch", "N/A")}</td><td>{metrics.get("ndcg_at_10", 0):.4f}</td><td>{metrics.get("ndcg_at_5", 0):.4f}</td><td>{metrics.get("auc", 0):.4f}</td><td>{metrics.get("mrr", 0):.4f}</td><td>{ckpt.get("timestamp", "N/A")}</td><td>{best_badge}</td></tr>'
            html_content += '</tbody></table>'
        
        html_content += '</div></body></html>'
        
        if save_path is None:
            save_path = self.output_dir / "evaluation_report.html"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {save_path}")
        return str(save_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize news recommendation model results")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", type=str, default="output/figures", help="Directory to save visualizations")
    parser.add_argument("--category-metrics", type=str, default=None, help="Path to JSON file with per-category metrics")
    parser.add_argument("--overall-metrics", type=str, default=None, help="Path to JSON file with overall metrics")
    parser.add_argument("--generate-html", action="store_true", help="Generate HTML report")
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(output_dir=args.output_dir)
    
    category_metrics = None
    if args.category_metrics:
        with open(args.category_metrics, 'r') as f:
            category_metrics = json.load(f)
        logger.info(f"Loaded category metrics from {args.category_metrics}")
    
    overall_metrics = None
    if args.overall_metrics:
        with open(args.overall_metrics, 'r') as f:
            overall_metrics = json.load(f)
        logger.info(f"Loaded overall metrics from {args.overall_metrics}")
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    print("📈 Plotting training metrics...")
    training_plot = visualizer.plot_training_metrics(args.checkpoint_dir)
    if training_plot:
        print(f"✓ Training metrics plot saved: {training_plot}")
    
    if category_metrics:
        print("\n📊 Creating category visualizations...")
        category_perf = visualizer.create_category_confusion_matrix(category_metrics)
        if category_perf:
            print(f"✓ Category performance heatmap saved: {category_perf}")
        
        category_comp = visualizer.plot_category_comparison(category_metrics)
        if category_comp:
            print(f"✓ Category comparison chart saved: {category_comp}")
    
    if args.generate_html:
        print("\n📄 Generating HTML report...")
        html_report = visualizer.generate_html_report(
            checkpoint_dir=args.checkpoint_dir,
            category_metrics=category_metrics,
            overall_metrics=overall_metrics
        )
        print(f"✓ HTML report saved: {html_report}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
