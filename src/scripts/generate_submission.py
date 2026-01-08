import torch
import argparse
import json
import zipfile
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime

from src.models.DeBERTaNewsEncoder import DeBERTaNewsEncoder
from src.models.UserEncoder import UserEncoder
from src.models.NAML import NAML
from src.data.dataset_mind import MINDValDataset
from src.data.dataloader_builder import build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.utils.config_loader import load_config
from src.evalutation.submit_formatter import SubmissionFormatter
from transformers import AutoTokenizer


def load_model(checkpoint_path: str, config: Dict, device: str) -> NAML:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded NAML model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Extract model configuration
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
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
        epoch = checkpoint.get("epoch", -1)
        metrics = checkpoint.get("metrics", {})
        print(f"Loaded checkpoint from epoch {epoch}")
        if metrics:
            print(f"Checkpoint metrics: {metrics}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


@torch.no_grad()
def run_inference(
    model: NAML,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> List[np.ndarray]:
    """
    Run inference on test set.
    
    Args:
        model: Trained NAML model
        dataloader: Test dataloader
        device: Device to run inference on
        
    Returns:
        List of prediction scores for each batch
    """
    print("\nRunning inference on test set...")
    all_scores = []
    
    for batch in tqdm(dataloader, desc="Inference"):
        # Move batch to device
        candidate_news = batch["candidate_news"].to(device)
        news_histories = batch["news_histories"].to(device)
        user_ids = batch["user_id"].to(device)
        
        # Create dummy target for inference
        batch_size, candidate_num = candidate_news.shape[0], candidate_news.shape[1]
        target = torch.zeros(batch_size, candidate_num).to(device)
        
        # Forward pass
        output = model(
            candidate_news=candidate_news,
            news_histories=news_histories,
            user_id=user_ids,
            target=target
        )
        
        # Get scores
        logits = output.logits  # (batch_size, candidate_num)
        scores = torch.softmax(logits, dim=1)  # (batch_size, candidate_num)
        
        all_scores.append(scores.cpu().numpy())
    
    print(f"Inference complete: {len(all_scores)} batches processed")
    return all_scores


def extract_impression_ids(behavior_df: pl.DataFrame) -> List[str]:
    """
    Extract impression IDs from behavior dataframe.
    
    Args:
        behavior_df: Behavior dataframe
        
    Returns:
        List of impression IDs
    """
    impression_ids = []
    for i in range(len(behavior_df)):
        imp_id = behavior_df[i]["impression_id"].item()
        impression_ids.append(str(imp_id))
    return impression_ids


def create_submission_package(
    scores: List[np.ndarray],
    impression_ids: List[str],
    output_dir: Path,
    checkpoint_path: str,
    config: Dict,
    create_zip: bool = True
) -> Dict[str, str]:
    """
    Create submission package with predictions and metadata.
    
    Args:
        scores: List of prediction scores
        impression_ids: List of impression IDs
        output_dir: Output directory
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        create_zip: Whether to create a zip file
        
    Returns:
        Dictionary with paths to created files
    """
    print("\nCreating submission package...")
    
    # Flatten scores
    all_scores = np.concatenate(scores, axis=0)
    
    # Verify lengths match
    if len(all_scores) != len(impression_ids):
        raise ValueError(
            f"Mismatch: {len(all_scores)} predictions vs {len(impression_ids)} impressions"
        )
    
    # Initialize formatter with metadata
    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_config": {
            "pretrained": config.get("model.pretrained", "microsoft/deberta-v3-base"),
            "conv_kernel_num": config.get("model.conv_kernel_num", 400),
            "query_dim": config.get("model.query_dim", 200)
        },
        "num_impressions": len(impression_ids),
        "created_at": datetime.now().isoformat()
    }
    
    formatter = SubmissionFormatter(metadata=metadata)
    
    # Format predictions
    print("Formatting predictions...")
    ranked_predictions = formatter.format_predictions(
        scores=all_scores,
        impression_ids=impression_ids
    )
    
    # Validate submission
    print("Validating submission format...")
    validation_result = formatter.validate_submission(
        predictions=ranked_predictions,
        impression_ids=impression_ids
    )
    
    if not validation_result["valid"]:
        print("ERROR: Submission validation failed!")
        for error in validation_result["errors"]:
            print(f"  - {error}")
        raise ValueError("Submission validation failed")
    
    print("✓ Submission format validated successfully")
    
    # Create submission package
    output_dir.mkdir(parents=True, exist_ok=True)
    
    package_result = formatter.create_submission_package(
        predictions=ranked_predictions,
        impression_ids=impression_ids,
        output_dir=output_dir,
        package_name="prediction",
        include_metadata=True
    )
    
    if not package_result["success"]:
        raise ValueError(f"Failed to create submission package: {package_result['errors']}")
    
    print(f"✓ Prediction file created: {package_result['prediction_file']}")
    print(f"✓ Metadata file created: {package_result['metadata_file']}")
    
    result = {
        "prediction_file": package_result["prediction_file"],
        "metadata_file": package_result["metadata_file"]
    }
    
    # Create zip file if requested
    if create_zip:
        zip_path = output_dir / "submission.zip"
        print(f"\nCreating submission.zip...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add prediction file
            pred_file = Path(package_result["prediction_file"])
            zipf.write(pred_file, pred_file.name)
            
            # Add metadata file
            if package_result["metadata_file"]:
                meta_file = Path(package_result["metadata_file"])
                zipf.write(meta_file, meta_file.name)
        
        print(f"✓ Submission package created: {zip_path}")
        result["zip_file"] = str(zip_path)
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate submission file for MIND leaderboard"
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--test-news",
        type=str,
        required=True,
        help="Path to test news.tsv file"
    )
    parser.add_argument(
        "--test-behaviors",
        type=str,
        required=True,
        help="Path to test behaviors.tsv file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file (default: configs/base_config.yaml)"
    )
    parser.add_argument(
        "--llm-description-path",
        type=str,
        help="Path to LLM descriptions JSON file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/submission",
        help="Directory to save submission files (default: output/submission)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Don't create submission.zip file"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("MIND Leaderboard Submission Generator")
    print("="*60)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded from: {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file not found: {args.config}")
        print("Using default configuration...")
        config = load_config(None)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = config.get("training.device", 
                           "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"✓ Using device: {device}")
    
    # Validate input files
    checkpoint_path = Path(args.checkpoint)
    test_news_path = Path(args.test_news)
    test_behaviors_path = Path(args.test_behaviors)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_news_path.exists():
        raise FileNotFoundError(f"Test news file not found: {test_news_path}")
    if not test_behaviors_path.exists():
        raise FileNotFoundError(f"Test behaviors file not found: {test_behaviors_path}")
    
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Test news: {test_news_path}")
    print(f"✓ Test behaviors: {test_behaviors_path}")
    
    # Load model
    model = load_model(str(checkpoint_path), config, device)
    
    # Load tokenizer
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)
    
    # Load test data
    print("\nLoading test data...")
    news_df = read_news_df(test_news_path)
    behavior_df = read_behavior_df(test_behaviors_path)
    user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
    
    print(f"✓ Loaded {len(news_df)} news items")
    print(f"✓ Loaded {len(behavior_df)} test impressions")
    
    # Load LLM descriptions if provided
    llm_description_path = None
    if args.llm_description_path:
        llm_desc_path = Path(args.llm_description_path)
        if llm_desc_path.exists():
            llm_description_path = llm_desc_path
            print(f"✓ LLM descriptions: {llm_description_path}")
    
    # Create dataset
    history_size = config.get("training.history_size", 50)
    dataset = MINDValDataset(
        behavior_df=behavior_df,
        news_df=news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        llm_description_path=llm_description_path,
        device=device,
    )
    
    # Create dataloader
    dataloader = build_val_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Run inference
    scores = run_inference(model, dataloader, device)
    
    # Extract impression IDs
    impression_ids = extract_impression_ids(behavior_df)
    
    # Create submission package
    output_dir = Path(args.output_dir)
    result = create_submission_package(
        scores=scores,
        impression_ids=impression_ids,
        output_dir=output_dir,
        checkpoint_path=str(checkpoint_path),
        config=config,
        create_zip=not args.no_zip
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUBMISSION GENERATION COMPLETE")
    print("="*60)
    print(f"Prediction file: {result['prediction_file']}")
    print(f"Metadata file: {result['metadata_file']}")
    if "zip_file" in result:
        print(f"Submission package: {result['zip_file']}")
    print("\nYou can now submit the prediction.txt file (or submission.zip)")
    print("to the MIND leaderboard for evaluation.")
    print("="*60)


if __name__ == "__main__":
    main()