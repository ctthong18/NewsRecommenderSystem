"""
Example script demonstrating model evaluation usage.

This script shows how to:
1. Evaluate a trained model
2. Perform per-category analysis
3. Compare two models statistically
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from src.scripts.evaluate import (
    load_model_from_checkpoint,
    evaluate_model,
    save_evaluation_report
)
from src.utils.config_loader import load_config
from src.data.dataset_mind import MINDValDataset
from src.data.dataloader_builder import build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.evalutation.category_evaluator import evaluate_model_per_category
from src.evalutation.statistical_tests import StatisticalComparator
from transformers import AutoTokenizer


def example_basic_evaluation():
    """Example 1: Basic model evaluation."""
    print("="*80)
    print("EXAMPLE 1: Basic Model Evaluation")
    print("="*80)
    
    # Configuration
    checkpoint_path = "output/checkpoints/best_model.pt"
    config_path = "configs/base_config.yaml"
    output_dir = Path("output/evaluation/example_basic")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using: python train.py")
        return
    
    # Load config
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Load model
    model, checkpoint_metadata = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Prepare data
    news_path = Path(config.get("data.val_news", "Data/raw/MINDsmall_dev/news.tsv"))
    behaviors_path = Path(config.get("data.val_behaviors", "Data/raw/MINDsmall_dev/behaviors.tsv"))
    
    if not news_path.exists() or not behaviors_path.exists():
        print(f"Data not found. Please download MIND dataset first.")
        return
    
    # Load tokenizer
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True, use_safetensors=True)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)
    
    # Load data
    print(f"\nLoading data...")
    news_df = read_news_df(news_path)
    behavior_df = read_behavior_df(behaviors_path)
    user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
    
    # Create dataset
    history_size = config.get("training.history_size", 50)
    dataset = MINDValDataset(
        behavior_df=behavior_df,
        news_df=news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        device=device,
    )
    
    # Create dataloader
    dataloader = build_val_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics, predictions = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_predictions=False
    )
    
    # Save report
    save_evaluation_report(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        checkpoint_metadata=checkpoint_metadata
    )
    
    print(f"\nResults saved to: {output_dir}")


def example_category_evaluation():
    """Example 2: Per-category evaluation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Per-Category Evaluation")
    print("="*80)
    
    checkpoint_path = "output/checkpoints/best_model.pt"
    config_path = "configs/base_config.yaml"
    output_dir = Path("output/evaluation/example_category")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load config and model (similar to example 1)
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, checkpoint_metadata = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Prepare data (similar to example 1)
    news_path = Path(config.get("data.val_news", "Data/raw/MINDsmall_dev/news.tsv"))
    behaviors_path = Path(config.get("data.val_behaviors", "Data/raw/MINDsmall_dev/behaviors.tsv"))
    
    if not news_path.exists() or not behaviors_path.exists():
        print(f"Data not found.")
        return
    
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True, use_safetensors=True)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)
    
    news_df = read_news_df(news_path)
    behavior_df = read_behavior_df(behaviors_path)
    user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
    
    history_size = config.get("training.history_size", 50)
    dataset = MINDValDataset(
        behavior_df=behavior_df,
        news_df=news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        device=device,
    )
    
    dataloader = build_val_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    
    # Per-category evaluation
    print("\nRunning per-category evaluation...")
    category_metrics, category_report = evaluate_model_per_category(
        model=model,
        dataloader=dataloader,
        news_df=news_df,
        behavior_df=behavior_df,
        device=device
    )
    
    # Print report
    print("\n" + category_report)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "category_report.txt", 'w') as f:
        f.write(category_report)
    
    print(f"\nCategory report saved to: {output_dir / 'category_report.txt'}")


def example_model_comparison():
    """Example 3: Statistical comparison of two models."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Model Comparison with Statistical Testing")
    print("="*80)
    
    checkpoint1 = "output/checkpoints/checkpoint_epoch_0.pt"
    checkpoint2 = "output/checkpoints/checkpoint_epoch_1.pt"
    config_path = "configs/base_config.yaml"
    output_dir = Path("output/evaluation/example_comparison")
    
    if not Path(checkpoint1).exists() or not Path(checkpoint2).exists():
        print(f"Checkpoints not found. Need at least 2 checkpoints for comparison.")
        return
    
    # Load config
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load both models
    print(f"\nLoading Model 1: {checkpoint1}")
    model1, metadata1 = load_model_from_checkpoint(checkpoint1, config, device)
    
    print(f"Loading Model 2: {checkpoint2}")
    model2, metadata2 = load_model_from_checkpoint(checkpoint2, config, device)
    
    # Prepare data
    news_path = Path(config.get("data.val_news", "Data/raw/MINDsmall_dev/news.tsv"))
    behaviors_path = Path(config.get("data.val_behaviors", "Data/raw/MINDsmall_dev/behaviors.tsv"))
    
    if not news_path.exists() or not behaviors_path.exists():
        print(f"Data not found.")
        return
    
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True, use_safetensors=True)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)
    
    news_df = read_news_df(news_path)
    behavior_df = read_behavior_df(behaviors_path)
    user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
    
    history_size = config.get("training.history_size", 50)
    dataset = MINDValDataset(
        behavior_df=behavior_df,
        news_df=news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        device=device,
    )
    
    dataloader = build_val_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    
    # Evaluate both models
    print("\nEvaluating Model 1...")
    metrics1, predictions1 = evaluate_model(model1, dataloader, device, save_predictions=True)
    
    print("\nEvaluating Model 2...")
    metrics2, predictions2 = evaluate_model(model2, dataloader, device, save_predictions=True)
    
    # Statistical comparison
    print("\nPerforming statistical comparison...")
    comparator = StatisticalComparator(confidence_level=0.95)
    comparison_results = comparator.compare_models(
        predictions1=predictions1,
        predictions2=predictions2
    )
    
    # Generate report
    comparison_report = comparator.generate_comparison_report(
        comparison_results=comparison_results,
        model1_name=f"Model 1 (epoch {metadata1['epoch']})",
        model2_name=f"Model 2 (epoch {metadata2['epoch']})"
    )
    
    # Print report
    print("\n" + comparison_report)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write(comparison_report)
    
    print(f"\nComparison report saved to: {output_dir / 'comparison_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which example to run (1: basic, 2: category, 3: comparison)"
    )
    args = parser.parse_args()
    
    if args.example == 1:
        example_basic_evaluation()
    elif args.example == 2:
        example_category_evaluation()
    elif args.example == 3:
        example_model_comparison()


if __name__ == "__main__":
    main()
