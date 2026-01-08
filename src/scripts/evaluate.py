import torch
import argparse
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import scipy.stats as stats

from src.models.DeBERTaNewsEncoder import DeBERTaNewsEncoder
from src.models.UserEncoder import UserEncoder
from src.models.NAML import NAML
from src.utils.metrics import RecEvaluator, RecMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.data.dataset_mind import MINDValDataset
from src.data.dataloader_builder import build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.utils.config_loader import load_config
from src.evalutation.category_evaluator import evaluate_model_per_category
from src.evalutation.statistical_tests import StatisticalComparator, perform_bootstrap_test
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate news recommendation model")
    
    # Model and checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file (default: configs/base_config.yaml)"
    )
    
    # Data paths
    parser.add_argument(
        "--news-path",
        type=str,
        help="Path to news.tsv file (overrides config)"
    )
    parser.add_argument(
        "--behaviors-path",
        type=str,
        help="Path to behaviors.tsv file (overrides config)"
    )
    parser.add_argument(
        "--llm-description-path",
        type=str,
        help="Path to LLM descriptions JSON file (overrides config)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Compute per-category metrics"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    # Comparison mode
    parser.add_argument(
        "--compare-with",
        type=str,
        help="Path to another checkpoint for statistical comparison"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: str
) -> Tuple[NAML, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model to
        
    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Extract model configuration
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v-base")
    conv_kernel_num = config.get("model.conv_kernel_num", 400)
    query_dim = config.get("model.query_dim", 200)
    
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "epoch": checkpoint.get("epoch", -1),
            "metrics": checkpoint.get("metrics", {}),
            "timestamp": checkpoint.get("timestamp", "unknown")
        }
    else:
        # Assume it's a raw state dict
        model.load_state_dict(checkpoint)
        metadata = {"epoch": -1, "metrics": {}, "timestamp": "unknown"}
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    if metadata["epoch"] >= 0:
        print(f"  Epoch: {metadata['epoch']}")
        print(f"  Training metrics: {metadata['metrics']}")
    
    return model, metadata


def evaluate_model(
    model: NAML,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    save_predictions: bool = False
) -> Tuple[Dict[str, float], Optional[List[Dict]]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: NAML model
        dataloader: Data loader
        device: Device to run evaluation on
        save_predictions: Whether to save predictions
        
    Returns:
        Tuple of (metrics_dict, predictions_list)
    """
    model.eval()
    all_labels = []
    all_scores = []
    predictions_list = [] if save_predictions else None
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Forward pass
            output = model(
                candidate_news=batch["candidate_news"],
                news_histories=batch["news_histories"],
                user_id=batch["user_id"],
                target=batch["target"]
            )
            
            # Extract logits and convert to scores
            logits = output.logits  # (batch_size, candidate_num)
            scores = torch.softmax(logits, dim=1)  # (batch_size, candidate_num)
            
            # Get target labels
            target = batch["target"]  # (batch_size, candidate_num)
            
            # Convert to numpy
            scores_np = scores.cpu().numpy()
            labels_np = target.cpu().numpy()
            
            all_scores.append(scores_np)
            all_labels.append(labels_np)
            
            # Save predictions if requested
            if save_predictions:
                for i in range(len(scores_np)):
                    predictions_list.append({
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "scores": scores_np[i].tolist(),
                        "labels": labels_np[i].tolist()
                    })
    
    # Concatenate all batches
    all_scores_np = np.concatenate(all_scores, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    # Evaluate each impression separately (MIND evaluation style)
    metrics_list = []
    for y_true, y_score in zip(all_labels_np, all_scores_np):
        metrics = RecEvaluator.evaluate_all(y_true, y_score)
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        "ndcg_at_10": np.mean([m.ndcg_at_10 for m in metrics_list]),
        "ndcg_at_5": np.mean([m.ndcg_at_5 for m in metrics_list]),
        "auc": np.mean([m.auc for m in metrics_list]),
        "mrr": np.mean([m.mrr for m in metrics_list]),
        "num_samples": len(metrics_list)
    }
    
    # Add standard deviations
    avg_metrics["ndcg_at_10_std"] = np.std([m.ndcg_at_10 for m in metrics_list])
    avg_metrics["ndcg_at_5_std"] = np.std([m.ndcg_at_5 for m in metrics_list])
    avg_metrics["auc_std"] = np.std([m.auc for m in metrics_list])
    avg_metrics["mrr_std"] = np.std([m.mrr for m in metrics_list])
    
    return avg_metrics, predictions_list








def save_evaluation_report(
    output_dir: Path,
    checkpoint_path: str,
    metrics: Dict[str, float],
    checkpoint_metadata: Dict,
    per_category_metrics: Optional[Dict] = None,
    comparison_results: Optional[Dict] = None,
    predictions: Optional[List[Dict]] = None,
    category_report: Optional[str] = None,
    comparison_report: Optional[str] = None
):
    """
    Save comprehensive evaluation report.
    
    Args:
        output_dir: Output directory
        checkpoint_path: Path to evaluated checkpoint
        metrics: Overall metrics
        checkpoint_metadata: Checkpoint metadata
        per_category_metrics: Per-category metrics (optional)
        comparison_results: Statistical comparison results (optional)
        predictions: Model predictions (optional)
        category_report: Category comparison report text (optional)
        comparison_report: Statistical comparison report text (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "overall_metrics": metrics,
    }
    
    if per_category_metrics:
        report["per_category_metrics"] = per_category_metrics
    
    if comparison_results:
        report["statistical_comparison"] = comparison_results
    
    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nEvaluation report saved to: {report_path}")
    
    # Save predictions if provided
    if predictions:
        predictions_path = output_dir / "predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to: {predictions_path}")
    
    # Save text reports
    if category_report:
        category_report_path = output_dir / "category_report.txt"
        with open(category_report_path, 'w') as f:
            f.write(category_report)
        print(f"Category report saved to: {category_report_path}")
    
    if comparison_report:
        comparison_report_path = output_dir / "comparison_report.txt"
        with open(comparison_report_path, 'w') as f:
            f.write(comparison_report)
        print(f"Comparison report saved to: {comparison_report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    if checkpoint_metadata.get("epoch", -1) >= 0:
        print(f"Epoch: {checkpoint_metadata['epoch']}")
    print(f"\nOverall Metrics:")
    print(f"  nDCG@5:  {metrics['ndcg_at_5']:.4f} ± {metrics.get('ndcg_at_5_std', 0):.4f}")
    print(f"  nDCG@10: {metrics['ndcg_at_10']:.4f} ± {metrics.get('ndcg_at_10_std', 0):.4f}")
    print(f"  AUC:     {metrics['auc']:.4f} ± {metrics.get('auc_std', 0):.4f}")
    print(f"  MRR:     {metrics['mrr']:.4f} ± {metrics.get('mrr_std', 0):.4f}")
    print(f"  Samples: {metrics['num_samples']}")
    
    if per_category_metrics:
        print(f"\nPer-Category Metrics:")
        for category, cat_metrics in sorted(per_category_metrics.items(), 
                                           key=lambda x: x[1]['ndcg_at_10'], 
                                           reverse=True)[:5]:  # Top 5 categories
            print(f"  {category}:")
            print(f"    nDCG@10: {cat_metrics['ndcg_at_10']:.4f} (n={cat_metrics['num_samples']})")
    
    if comparison_results:
        print(f"\nStatistical Comparison:")
        for metric_name, results in comparison_results.items():
            sig_marker = "***" if results.get('significant_at_0.01', False) else ("**" if results['significant_at_0.05'] else "")
            print(f"  {metric_name}:")
            print(f"    Model 1: {results['model1_mean']:.4f}")
            print(f"    Model 2: {results['model2_mean']:.4f}")
            print(f"    Difference: {results['difference_mean']:.4f} {sig_marker}")
            print(f"    p-value: {results['p_value']:.4f}")
            print(f"    Better model: {results['better_model']}")
    
    print("="*60)


def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = load_config(None)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = config.get("training.device", "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    model, checkpoint_metadata = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device
    )
    
    # Prepare data paths
    if args.news_path:
        news_path = Path(args.news_path)
    else:
        news_path = Path(config.get("data.val_news", "Data/raw/MINDsmall_dev/news.tsv"))
    
    if args.behaviors_path:
        behaviors_path = Path(args.behaviors_path)
    else:
        behaviors_path = Path(config.get("data.val_behaviors", "Data/raw/MINDsmall_dev/behaviors.tsv"))
    
    if args.llm_description_path:
        llm_description_path = Path(args.llm_description_path)
    else:
        llm_desc_str = config.get("data.llm_description", 
                                  "gpt-augmented-news-recommendation/dataset/generated/category_description_gpt4.json")
        llm_description_path = Path(llm_desc_str) if llm_desc_str else None
    
    # Load tokenizer
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True, use_safetensors=True)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)
    
    # Load data
    print(f"\nLoading data from:")
    print(f"  News: {news_path}")
    print(f"  Behaviors: {behaviors_path}")
    
    news_df = read_news_df(news_path)
    behavior_df = read_behavior_df(behaviors_path)
    
    # Create user mapping (for evaluation, we can use a simple mapping)
    user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
    
    # Create dataset
    history_size = config.get("training.history_size", 50)
    dataset = MINDValDataset(
        behavior_df=behavior_df,
        news_df=news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        llm_description_path=llm_description_path if llm_description_path and llm_description_path.exists() else None,
        device=device,
    )
    
    # Create dataloader
    dataloader = build_val_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Run evaluation
    metrics, predictions = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_predictions=args.save_predictions or args.compare_with is not None
    )
    
    # Per-category evaluation
    per_category_metrics = None
    category_report = None
    if args.per_category:
        per_category_metrics, category_report = evaluate_model_per_category(
            model=model,
            dataloader=dataloader,
            news_df=news_df,
            behavior_df=behavior_df,
            device=device
        )
        
        # Print category report
        if category_report:
            print("\n" + category_report)
    
    # Statistical comparison
    comparison_results = None
    comparison_report = None
    if args.compare_with:
        print(f"\nLoading comparison model from: {args.compare_with}")
        model2, _ = load_model_from_checkpoint(
            checkpoint_path=args.compare_with,
            config=config,
            device=device
        )
        
        metrics2, predictions2 = evaluate_model(
            model=model2,
            dataloader=dataloader,
            device=device,
            save_predictions=True
        )
        
        # Perform statistical comparison
        comparator = StatisticalComparator(confidence_level=0.95)
        comparison_results = comparator.compare_models(
            predictions1=predictions,
            predictions2=predictions2
        )
        
        # Generate comparison report
        comparison_report = comparator.generate_comparison_report(
            comparison_results=comparison_results,
            model1_name=args.checkpoint,
            model2_name=args.compare_with
        )
        
        # Print comparison report
        print("\n" + comparison_report)
    
    # Save report
    output_dir = Path(args.output_dir)
    save_evaluation_report(
        output_dir=output_dir,
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        checkpoint_metadata=checkpoint_metadata,
        per_category_metrics=per_category_metrics,
        comparison_results=comparison_results,
        predictions=predictions if args.save_predictions else None,
        category_report=category_report,
        comparison_report=comparison_report
    )


if __name__ == "__main__":
    main()
