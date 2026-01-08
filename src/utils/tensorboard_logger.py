import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path


class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter with convenience methods for logging.
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            enabled: Whether logging is enabled (useful for disabling in tests)
        """
        self.enabled = enabled
        self.log_dir = log_dir
        
        if self.enabled:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled. Logs will be saved to: {log_dir}")
            print(f"To view logs, run: tensorboard --logdir={log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the scalar (e.g., 'train/loss')
            value: Scalar value to log
            step: Global step (iteration or epoch number)
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars in one chart.
        
        Args:
            main_tag: Parent name for the group (e.g., 'losses')
            tag_scalar_dict: Dictionary of {tag: value}
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step
            prefix: Prefix for metric names (e.g., 'train/' or 'val/')
        """
        if not self.enabled or self.writer is None:
            return
        
        for name, value in metrics.items():
            tag = f"{prefix}{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """
        Log learning rate.
        
        Args:
            lr: Current learning rate
            step: Global step
        """
        self.log_scalar("train/learning_rate", lr, step)
    
    def log_gradient_norm(self, model: torch.nn.Module, step: int):
        """
        Log gradient norms for monitoring gradient flow.
        
        Args:
            model: PyTorch model
            step: Global step
        """
        if not self.enabled or self.writer is None:
            return
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.writer.add_scalar("train/gradient_norm", total_norm, step)
    
    def log_model_weights(self, model: torch.nn.Module, step: int):
        """
        Log model weight histograms.
        
        Args:
            model: PyTorch model
            step: Global step
        """
        if not self.enabled or self.writer is None:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad.data, step)
    
    def log_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        step: int,
        num_samples: int = 5
    ):
        """
        Log sample predictions for visualization.
        
        Args:
            predictions: Model predictions (scores)
            labels: Ground truth labels
            step: Global step
            num_samples: Number of samples to log
        """
        if not self.enabled or self.writer is None:
            return
        
        # Log first few samples as text
        num_samples = min(num_samples, len(predictions))
        
        for i in range(num_samples):
            pred_str = ", ".join([f"{p:.3f}" for p in predictions[i][:10]])  # First 10 candidates
            label_str = ", ".join([f"{int(l)}" for l in labels[i][:10]])
            
            text = f"Sample {i}:\n"
            text += f"Predictions: [{pred_str}...]\n"
            text += f"Labels:      [{label_str}...]\n"
            
            self.writer.add_text(f"predictions/sample_{i}", text, step)
    
    def log_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        step: int,
        class_names: Optional[List[str]] = None
    ):
        """
        Log confusion matrix (for classification tasks).
        
        Args:
            predictions: Predicted class indices
            labels: True class indices
            step: Global step
            class_names: Optional list of class names
        """
        if not self.enabled or self.writer is not None:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(labels, predictions)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            self.writer.add_figure("confusion_matrix", fig, step)
            plt.close(fig)
    
    def log_attention_weights(
        self,
        attention_weights: torch.Tensor,
        step: int,
        tag: str = "attention"
    ):
        """
        Log attention weight heatmaps.
        
        Args:
            attention_weights: Attention weights tensor
            step: Global step
            tag: Tag for the attention visualization
        """
        if not self.enabled or self.writer is None:
            return
        
        import matplotlib.pyplot as plt
        
        # Convert to numpy and take first sample if batch
        if attention_weights.dim() > 2:
            attention_weights = attention_weights[0]
        
        attn_np = attention_weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Attention Weights')
        plt.colorbar(im, ax=ax)
        
        self.writer.add_figure(f"{tag}/heatmap", fig, step)
        plt.close(fig)
    
    def log_text(self, tag: str, text: str, step: int):
        """
        Log text data.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        if self.enabled and self.writer is not None:
            self.writer.add_hparams(hparams, metrics)
    
    def flush(self):
        """Flush pending logs to disk."""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print(f"TensorBoard logs saved to: {self.log_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
