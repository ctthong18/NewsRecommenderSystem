"""
Inference pipeline for news recommendation models.

Features:
- Batch inference API for processing multiple users
- Single-user inference mode for real-time recommendations
- Optimized inference speed (no gradient computation)
- Support for model ensemble inference
- Input validation and error handling
"""
import torch
import argparse
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from datetime import datetime

from src.models.DeBERTaNewsEncoder import DeBERTaNewsEncoder
from src.models.UserEncoder import UserEncoder
from src.models.NAML import NAML
from src.utils.checkpoint_manager import CheckpointManager
from src.data.dataset_mind import MINDValDataset
from src.data.dataloader_builder import build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.utils.config_loader import load_config
from transformers import AutoTokenizer
from src.const.mind import EMPTY_NEWS_ID


class NewsRecommendationInference:
    """
    Inference wrapper for news recommendation models.
    
    Provides both batch and single-user inference capabilities with
    optimized performance and error handling.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        ensemble_checkpoints: Optional[List[str]] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary (optional)
            device: Device to use (cuda/cpu, default: auto-detect)
            ensemble_checkpoints: List of checkpoint paths for ensemble (optional)
        """
        self.checkpoint_path = checkpoint_path
        self.config = config or {}
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = self.config.get("training.device", 
                                         "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing inference pipeline on device: {self.device}")
        
        # Load main model
        self.model = self._load_model(checkpoint_path)
        
        # Load ensemble models if provided
        self.ensemble_models = []
        if ensemble_checkpoints:
            print(f"Loading {len(ensemble_checkpoints)} ensemble models...")
            for ckpt_path in ensemble_checkpoints:
                model = self._load_model(ckpt_path)
                self.ensemble_models.append(model)
            print(f"Loaded {len(self.ensemble_models)} ensemble models")
        
        # Load tokenizer
        pretrained_model = self.config.get("model.pretrained", "microsoft/deberta-v3-base")
        max_length = self.config.get("model.max_length", 64)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.transform_fn = create_transform_fn_from_pretrained_tokenizer(
            self.tokenizer, max_length
        )
        
        # Cache for news embeddings (for faster inference)
        self.news_embedding_cache = {}
        
        print("Inference pipeline initialized successfully")
    
    def _load_model(self, checkpoint_path: str) -> NAML:
        """Load model from checkpoint."""
        # Extract model configuration
        pretrained_model = self.config.get("model.pretrained", "microsoft/deberta-v3-base")
        conv_kernel_num = self.config.get("model.conv_kernel_num", 400)
        query_dim = self.config.get("model.query_dim", 200)
        
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def predict_batch(
        self,
        candidate_news: torch.Tensor,
        news_histories: torch.Tensor,
        user_ids: torch.Tensor,
        use_ensemble: bool = False
    ) -> np.ndarray:
        """
        Perform batch inference.
        
        Args:
            candidate_news: Candidate news tensor (batch_size, candidate_num, seq_len)
            news_histories: User history tensor (batch_size, history_size, seq_len)
            user_ids: User ID tensor (batch_size,)
            use_ensemble: Whether to use ensemble models
            
        Returns:
            Prediction scores (batch_size, candidate_num)
        """
        # Move to device
        candidate_news = candidate_news.to(self.device)
        news_histories = news_histories.to(self.device)
        user_ids = user_ids.to(self.device)
        
        # Create dummy target for inference
        batch_size, candidate_num = candidate_news.shape[0], candidate_news.shape[1]
        target = torch.zeros(batch_size, candidate_num).to(self.device)
        
        # Main model prediction
        output = self.model(
            candidate_news=candidate_news,
            news_histories=news_histories,
            user_id=user_ids,
            target=target
        )
        
        logits = output.logits  # (batch_size, candidate_num)
        scores = torch.softmax(logits, dim=1)  # (batch_size, candidate_num)
        
        # Ensemble prediction if enabled
        if use_ensemble and self.ensemble_models:
            ensemble_scores = [scores]
            
            for model in self.ensemble_models:
                output = model(
                    candidate_news=candidate_news,
                    news_histories=news_histories,
                    user_id=user_ids,
                    target=target
                )
                logits = output.logits
                model_scores = torch.softmax(logits, dim=1)
                ensemble_scores.append(model_scores)
            
            # Average ensemble scores
            scores = torch.stack(ensemble_scores).mean(dim=0)
        
        return scores.cpu().numpy()
    
    @torch.no_grad()
    def predict_single_user(
        self,
        candidate_news_ids: List[str],
        history_news_ids: List[str],
        news_df: pl.DataFrame,
        llm_descriptions: Optional[Dict[str, str]] = None,
        use_ensemble: bool = False
    ) -> Dict[str, Union[List[float], List[str]]]:
        """
        Perform single-user inference for real-time recommendations.
        
        Args:
            candidate_news_ids: List of candidate news IDs
            history_news_ids: List of user's history news IDs
            news_df: News dataframe
            llm_descriptions: Optional LLM descriptions dictionary
            use_ensemble: Whether to use ensemble models
            
        Returns:
            Dictionary with scores and ranked news IDs
        """
        # Validate inputs
        if not candidate_news_ids:
            raise ValueError("candidate_news_ids cannot be empty")
        
        # Build news ID to news map
        news_map = self._build_news_map(news_df, llm_descriptions)
        
        # Validate news IDs exist
        missing_candidates = [nid for nid in candidate_news_ids if nid not in news_map]
        if missing_candidates:
            raise ValueError(f"Missing candidate news IDs: {missing_candidates[:5]}")
        
        # Prepare history (pad or truncate)
        history_size = self.config.get("training.history_size", 50)
        if len(history_news_ids) < history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (history_size - len(history_news_ids))
        else:
            history_news_ids = history_news_ids[:history_size]
        
        # Prepare texts
        candidate_texts = [list(news_map.get(nid, ("", ""))) for nid in candidate_news_ids]
        history_texts = [list(news_map.get(nid, ("", ""))) for nid in history_news_ids]
        
        # Tokenize
        candidate_batch = self.transform_fn(candidate_texts)  # (candidate_num, seq_len)
        history_batch = self.transform_fn(history_texts)  # (history_size, seq_len)
        
        # Add batch dimension
        candidate_batch = candidate_batch.unsqueeze(0)  # (1, candidate_num, seq_len)
        history_batch = history_batch.unsqueeze(0)  # (1, history_size, seq_len)
        user_id = torch.tensor([0], dtype=torch.long)  # Dummy user ID
        
        # Predict
        scores = self.predict_batch(
            candidate_news=candidate_batch,
            news_histories=history_batch,
            user_ids=user_id,
            use_ensemble=use_ensemble
        )
        
        # Get scores for single user
        scores = scores[0]  # (candidate_num,)
        
        # Rank news by scores
        ranked_indices = np.argsort(scores)[::-1]  # Descending order
        ranked_news_ids = [candidate_news_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        
        return {
            "news_ids": ranked_news_ids,
            "scores": ranked_scores,
            "top_k_news_ids": ranked_news_ids[:10],  # Top 10 recommendations
            "top_k_scores": ranked_scores[:10]
        }
    
    def _build_news_map(
        self,
        news_df: pl.DataFrame,
        llm_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """Build news ID to (title, description) mapping."""
        news_map = {}
        llm_desc = llm_descriptions or {}
        
        for i in range(len(news_df)):
            news_id = news_df[i]["news_id"].item()
            title = news_df[i]["title"].item()
            description = llm_desc.get(news_id, "")
            if not description:
                description = news_df[i].get("abstract", pl.Series([""]))[0] if "abstract" in news_df.columns else ""
            news_map[news_id] = (title, description)
        
        news_map[EMPTY_NEWS_ID] = ("", "")
        
        return news_map
    
    @torch.no_grad()
    def batch_inference_from_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_ensemble: bool = False,
        save_predictions: bool = False
    ) -> Tuple[List[np.ndarray], Optional[List[Dict]]]:
        """
        Perform batch inference from a dataloader.
        
        Args:
            dataloader: PyTorch dataloader
            use_ensemble: Whether to use ensemble models
            save_predictions: Whether to save detailed predictions
            
        Returns:
            Tuple of (scores_list, predictions_list)
        """
        all_scores = []
        predictions_list = [] if save_predictions else None
        
        print("Running batch inference...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Predict
            scores = self.predict_batch(
                candidate_news=batch["candidate_news"],
                news_histories=batch["news_histories"],
                user_ids=batch["user_id"],
                use_ensemble=use_ensemble
            )
            
            all_scores.append(scores)
            
            # Save predictions if requested
            if save_predictions:
                target = batch["target"].cpu().numpy()
                for i in range(len(scores)):
                    predictions_list.append({
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "scores": scores[i].tolist(),
                        "labels": target[i].tolist() if len(target.shape) > 1 else [target[i].item()]
                    })
        
        return all_scores, predictions_list


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="News recommendation inference")
    
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
        help="Path to config file"
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        type=str,
        nargs="+",
        help="Paths to ensemble model checkpoints"
    )
    
    # Inference mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="batch",
        help="Inference mode: batch or single-user"
    )
    
    # Data paths (for batch mode)
    parser.add_argument(
        "--news-path",
        type=str,
        help="Path to news.tsv file"
    )
    parser.add_argument(
        "--behaviors-path",
        type=str,
        help="Path to behaviors.tsv file"
    )
    parser.add_argument(
        "--llm-description-path",
        type=str,
        help="Path to LLM descriptions JSON file"
    )
    
    # Single-user mode parameters
    parser.add_argument(
        "--candidate-news-ids",
        type=str,
        nargs="+",
        help="Candidate news IDs for single-user mode"
    )
    parser.add_argument(
        "--history-news-ids",
        type=str,
        nargs="+",
        help="History news IDs for single-user mode"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/inference",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save detailed predictions"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = load_config(None)
    
    # Initialize inference pipeline
    inference = NewsRecommendationInference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
        ensemble_checkpoints=args.ensemble_checkpoints
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "single":
        # Single-user inference mode
        if not args.candidate_news_ids or not args.news_path:
            raise ValueError("Single-user mode requires --candidate-news-ids and --news-path")
        
        # Load news data
        news_path = Path(args.news_path)
        news_df = read_news_df(news_path)
        
        # Load LLM descriptions if provided
        llm_descriptions = None
        if args.llm_description_path:
            llm_desc_path = Path(args.llm_description_path)
            if llm_desc_path.exists():
                with open(llm_desc_path, 'r') as f:
                    llm_descriptions = json.load(f)
        
        # Perform inference
        history_news_ids = args.history_news_ids or []
        results = inference.predict_single_user(
            candidate_news_ids=args.candidate_news_ids,
            history_news_ids=history_news_ids,
            news_df=news_df,
            llm_descriptions=llm_descriptions,
            use_ensemble=bool(args.ensemble_checkpoints)
        )
        
        # Print results
        print("\n" + "="*60)
        print("SINGLE-USER INFERENCE RESULTS")
        print("="*60)
        print(f"Top 10 Recommendations:")
        for i, (news_id, score) in enumerate(zip(results["top_k_news_ids"], results["top_k_scores"]), 1):
            print(f"  {i}. {news_id}: {score:.4f}")
        print("="*60)
        
        # Save results
        output_path = output_dir / "single_user_predictions.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    else:
        # Batch inference mode
        if not args.news_path or not args.behaviors_path:
            raise ValueError("Batch mode requires --news-path and --behaviors-path")
        
        # Load data
        news_path = Path(args.news_path)
        behaviors_path = Path(args.behaviors_path)
        
        print(f"Loading data from:")
        print(f"  News: {news_path}")
        print(f"  Behaviors: {behaviors_path}")
        
        news_df = read_news_df(news_path)
        behavior_df = read_behavior_df(behaviors_path)
        user_ids_to_idx_map = create_user_ids_to_idx_map(behavior_df)
        
        # Load LLM descriptions
        llm_description_path = None
        if args.llm_description_path:
            llm_description_path = Path(args.llm_description_path)
        
        # Create dataset
        history_size = config.get("training.history_size", 50)
        dataset = MINDValDataset(
            behavior_df=behavior_df,
            news_df=news_df,
            user_ids_to_idx_map=user_ids_to_idx_map,
            batch_transform_texts=inference.transform_fn,
            history_size=history_size,
            llm_description_path=llm_description_path if llm_description_path and llm_description_path.exists() else None,
            device=inference.device,
        )
        
        # Create dataloader
        dataloader = build_val_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=(inference.device == "cuda"),
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Perform batch inference
        all_scores, predictions = inference.batch_inference_from_dataloader(
            dataloader=dataloader,
            use_ensemble=bool(args.ensemble_checkpoints),
            save_predictions=args.save_predictions
        )
        
        # Save results
        print(f"\nProcessed {len(all_scores)} batches")
        
        if predictions:
            predictions_path = output_dir / "batch_predictions.json"
            with open(predictions_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Predictions saved to: {predictions_path}")
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "ensemble_checkpoints": args.ensemble_checkpoints,
            "num_samples": len(dataset),
            "num_batches": len(all_scores),
            "batch_size": args.batch_size
        }
        summary_path = output_dir / "inference_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
