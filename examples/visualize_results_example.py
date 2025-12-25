"""
Example script demonstrating how to use the visualization tools.

This script shows how to:
1. Generate training metrics plots from checkpoints
2. Visualize attention weights for news articles
3. Create category performance visualizations
4. Generate comprehensive HTML reports
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scripts.visualize_results import ResultVisualizer
import json


def example_1_plot_training_metrics():
    """
    Example 1: Plot training metrics from checkpoint directory.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Plot Training Metrics")
    print("="*80 + "\n")
    
    # Initialize visualizer
    visualizer = ResultVisualizer(output_dir="output/figures")
    
    # Plot metrics from checkpoint directory
    checkpoint_dir = "output/examples/checkpoints_basic"
    
    print(f"Loading checkpoints from: {checkpoint_dir}")
    plot_path = visualizer.plot_training_metrics(checkpoint_dir)
    
    if plot_path:
        print(f"✓ Training metrics plot saved to: {plot_path}")
    else:
        print("✗ No checkpoint metadata found")


def example_2_category_visualization():
    """
    Example 2: Create category performance visualizations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Category Performance Visualization")
    print("="*80 + "\n")
    
    # Sample category metrics (in practice, load from evaluation)
    category_metrics = {
        "sports": {
            "ndcg_at_10": 0.452,
            "ndcg_at_5": 0.389,
            "auc": 0.721,
            "mrr": 0.341,
            "ndcg_at_10_std": 0.023,
            "ndcg_at_5_std": 0.019,
            "auc_std": 0.015,
            "mrr_std": 0.021,
            "num_samples": 1250
        },
        "news": {
            "ndcg_at_10": 0.418,
            "ndcg_at_5": 0.356,
            "auc": 0.698,
            "mrr": 0.312,
            "ndcg_at_10_std": 0.028,
            "ndcg_at_5_std": 0.024,
            "auc_std": 0.018,
            "mrr_std": 0.025,
            "num_samples": 1580
        },
        "entertainment": {
            "ndcg_at_10": 0.395,
            "ndcg_at_5": 0.334,
            "auc": 0.682,
            "mrr": 0.298,
            "ndcg_at_10_std": 0.031,
            "ndcg_at_5_std": 0.027,
            "auc_std": 0.021,
            "mrr_std": 0.028,
            "num_samples": 980
        },
        "finance": {
            "ndcg_at_10": 0.438,
            "ndcg_at_5": 0.372,
            "auc": 0.709,
            "mrr": 0.328,
            "ndcg_at_10_std": 0.025,
            "ndcg_at_5_std": 0.021,
            "auc_std": 0.016,
            "mrr_std": 0.023,
            "num_samples": 1120
        }
    }
    
    # Initialize visualizer
    visualizer = ResultVisualizer(output_dir="output/figures")
    
    # Create heatmap
    print("Creating category performance heatmap...")
    heatmap_path = visualizer.create_category_confusion_matrix(category_metrics)
    if heatmap_path:
        print(f"✓ Heatmap saved to: {heatmap_path}")
    
    # Create comparison chart
    print("\nCreating category comparison chart...")
    comparison_path = visualizer.plot_category_comparison(category_metrics)
    if comparison_path:
        print(f"✓ Comparison chart saved to: {comparison_path}")


def example_3_generate_html_report():
    """
    Example 3: Generate comprehensive HTML report.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Generate HTML Report")
    print("="*80 + "\n")
    
    # Sample data
    checkpoint_dir = "output/examples/checkpoints_basic"
    
    category_metrics = {
        "sports": {
            "ndcg_at_10": 0.452,
            "ndcg_at_5": 0.389,
            "auc": 0.721,
            "mrr": 0.341,
            "ndcg_at_10_std": 0.023,
            "ndcg_at_5_std": 0.019,
            "auc_std": 0.015,
            "mrr_std": 0.021,
            "num_samples": 1250
        },
        "news": {
            "ndcg_at_10": 0.418,
            "ndcg_at_5": 0.356,
            "auc": 0.698,
            "mrr": 0.312,
            "ndcg_at_10_std": 0.028,
            "ndcg_at_5_std": 0.024,
            "auc_std": 0.018,
            "mrr_std": 0.025,
            "num_samples": 1580
        }
    }
    
    overall_metrics = {
        "ndcg_at_10": 0.425,
        "ndcg_at_5": 0.363,
        "auc": 0.713,
        "mrr": 0.326
    }
    
    # Initialize visualizer
    visualizer = ResultVisualizer(output_dir="output/figures")
    
    # Generate HTML report
    print("Generating comprehensive HTML report...")
    report_path = visualizer.generate_html_report(
        checkpoint_dir=checkpoint_dir,
        category_metrics=category_metrics,
        overall_metrics=overall_metrics
    )
    
    if report_path:
        print(f"✓ HTML report saved to: {report_path}")
        print(f"\nOpen the report in your browser:")
        print(f"  file://{Path(report_path).absolute()}")


def example_4_attention_visualization():
    """
    Example 4: Visualize attention weights (requires trained model).
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Attention Weights Visualization")
    print("="*80 + "\n")
    
    print("Note: This example requires a trained model and tokenizer.")
    print("To use attention visualization:")
    print("  1. Load your trained NAML model")
    print("  2. Load the tokenizer")
    print("  3. Call visualizer.visualize_attention_weights(model, news_text, tokenizer)")
    print("\nExample code:")
    print("""
    from transformers import AutoTokenizer
    from src.models.NAML import NAML
    from src.utils.checkpoint_manager import CheckpointManager
    
    # Load model
    checkpoint_manager = CheckpointManager("output/checkpoints")
    model = NAML(...)  # Initialize your model
    checkpoint_manager.load_best_checkpoint(model, device="cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    # Visualize attention
    visualizer = ResultVisualizer()
    news_text = "Breaking news: AI technology advances rapidly..."
    visualizer.visualize_attention_weights(model, news_text, tokenizer)
    """)


def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("VISUALIZATION TOOLS - EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        example_1_plot_training_metrics()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_category_visualization()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_generate_html_report()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    example_4_attention_visualization()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nCheck the output/figures directory for generated visualizations.")


if __name__ == "__main__":
    main()
