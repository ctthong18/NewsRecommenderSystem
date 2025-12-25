"""
Per-category evaluation for news recommendation models.

Provides functionality to:
- Group predictions by news category
- Calculate metrics per category
- Generate category comparison reports
"""
import numpy as np
import polars as pl
from typing import Dict, List, Tuple
from collections import defaultdict
from src.utils.metrics import RecEvaluator


class CategoryEvaluator:
    """
    Evaluator for computing per-category metrics.
    """
    
    def __init__(self, news_df: pl.DataFrame):
        """
        Initialize category evaluator.
        
        Args:
            news_df: News dataframe with category information
        """
        self.news_df = news_df
        
        # Create news_id to category mapping
        self.news_to_category = {}
        for i in range(len(news_df)):
            news_id = news_df[i]["news_id"].item()
            category = news_df[i].get("category", pl.Series(["unknown"]))[0] if "category" in news_df.columns else "unknown"
            self.news_to_category[news_id] = category
        
        # Storage for per-category metrics
        self.category_predictions = defaultdict(list)
    
    def add_sample(
        self,
        candidate_news_ids: List[str],
        labels: np.ndarray,
        scores: np.ndarray
    ):
        """
        Add a sample for per-category evaluation.
        
        Args:
            candidate_news_ids: List of candidate news IDs
            labels: Ground truth labels (binary array)
            scores: Predicted scores (probability array)
        """
        # Find clicked news
        clicked_indices = np.where(labels == 1)[0]
        
        if len(clicked_indices) == 0:
            # No clicked news, skip
            return
        
        # Get categories of clicked news
        clicked_categories = set()
        for idx in clicked_indices:
            if idx < len(candidate_news_ids):
                news_id = candidate_news_ids[idx]
                category = self.news_to_category.get(news_id, "unknown")
                clicked_categories.add(category)
        
        # Store predictions for each category
        for category in clicked_categories:
            self.category_predictions[category].append({
                "labels": labels,
                "scores": scores
            })
    
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each category.
        
        Returns:
            Dictionary mapping category to metrics
        """
        category_metrics = {}
        
        for category, predictions in self.category_predictions.items():
            if len(predictions) == 0:
                continue
            
            # Compute metrics for each sample
            metrics_list = []
            for pred in predictions:
                y_true = pred["labels"]
                y_score = pred["scores"]
                metrics = RecEvaluator.evaluate_all(y_true, y_score)
                metrics_list.append(metrics)
            
            # Average metrics
            category_metrics[category] = {
                "ndcg_at_10": np.mean([m.ndcg_at_10 for m in metrics_list]),
                "ndcg_at_5": np.mean([m.ndcg_at_5 for m in metrics_list]),
                "auc": np.mean([m.auc for m in metrics_list]),
                "mrr": np.mean([m.mrr for m in metrics_list]),
                "num_samples": len(metrics_list),
                # Add standard deviations
                "ndcg_at_10_std": np.std([m.ndcg_at_10 for m in metrics_list]),
                "ndcg_at_5_std": np.std([m.ndcg_at_5 for m in metrics_list]),
                "auc_std": np.std([m.auc for m in metrics_list]),
                "mrr_std": np.std([m.mrr for m in metrics_list]),
            }
        
        return category_metrics
    
    def generate_comparison_report(self) -> str:
        """
        Generate a text report comparing categories.
        
        Returns:
            Formatted comparison report
        """
        metrics = self.compute_metrics()
        
        if not metrics:
            return "No category metrics available."
        
        # Sort categories by nDCG@10
        sorted_categories = sorted(
            metrics.items(),
            key=lambda x: x[1]["ndcg_at_10"],
            reverse=True
        )
        
        report_lines = [
            "="*80,
            "PER-CATEGORY EVALUATION REPORT",
            "="*80,
            ""
        ]
        
        # Summary table
        report_lines.append(f"{'Category':<20} {'nDCG@5':<12} {'nDCG@10':<12} {'AUC':<12} {'MRR':<12} {'Samples':<10}")
        report_lines.append("-"*80)
        
        for category, cat_metrics in sorted_categories:
            report_lines.append(
                f"{category:<20} "
                f"{cat_metrics['ndcg_at_5']:>6.4f}±{cat_metrics['ndcg_at_5_std']:>4.3f} "
                f"{cat_metrics['ndcg_at_10']:>6.4f}±{cat_metrics['ndcg_at_10_std']:>4.3f} "
                f"{cat_metrics['auc']:>6.4f}±{cat_metrics['auc_std']:>4.3f} "
                f"{cat_metrics['mrr']:>6.4f}±{cat_metrics['mrr_std']:>4.3f} "
                f"{cat_metrics['num_samples']:>10}"
            )
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Best and worst categories
        if len(sorted_categories) > 0:
            best_cat, best_metrics = sorted_categories[0]
            worst_cat, worst_metrics = sorted_categories[-1]
            
            report_lines.append(f"\nBest performing category: {best_cat}")
            report_lines.append(f"  nDCG@10: {best_metrics['ndcg_at_10']:.4f}")
            report_lines.append(f"  Samples: {best_metrics['num_samples']}")
            
            report_lines.append(f"\nWorst performing category: {worst_cat}")
            report_lines.append(f"  nDCG@10: {worst_metrics['ndcg_at_10']:.4f}")
            report_lines.append(f"  Samples: {worst_metrics['num_samples']}")
        
        return "\n".join(report_lines)


def evaluate_model_per_category(
    model,
    dataloader,
    news_df: pl.DataFrame,
    behavior_df: pl.DataFrame,
    device: str
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Evaluate model with per-category metrics.
    
    Args:
        model: NAML model
        dataloader: Data loader
        news_df: News dataframe
        behavior_df: Behavior dataframe
        device: Device to run on
        
    Returns:
        Tuple of (category_metrics_dict, comparison_report)
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    evaluator = CategoryEvaluator(news_df)
    
    print("Running per-category evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating by category")):
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
            
            # Extract scores and labels
            logits = output.logits
            scores = torch.softmax(logits, dim=1)
            target = batch["target"]
            
            scores_np = scores.cpu().numpy()
            labels_np = target.cpu().numpy()
            
            # Get candidate news IDs from behavior dataframe
            # Note: This requires tracking which behavior we're evaluating
            # For now, we'll extract from the dataloader's dataset
            behavior_item = behavior_df[batch_idx]
            impressions = behavior_item["impressions"].to_list()[0]
            candidate_news_ids = [imp["news_id"] for imp in impressions]
            
            # Add each sample in batch
            for i in range(len(scores_np)):
                evaluator.add_sample(
                    candidate_news_ids=candidate_news_ids,
                    labels=labels_np[i],
                    scores=scores_np[i]
                )
    
    # Compute metrics
    category_metrics = evaluator.compute_metrics()
    comparison_report = evaluator.generate_comparison_report()
    
    return category_metrics, comparison_report
