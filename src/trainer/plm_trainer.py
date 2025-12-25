import torch
from tqdm import tqdm
from src.evalutation.evaluator import evaluate_model
from src.utils.metrics import RecEvaluator
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.tensorboard_logger import TensorBoardLogger
import numpy as np
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PLMTrainer:
    """
    Trainer for PLM-based news recommendation models (NAML with DeBERTa).
    Handles ModelOutput format from NAML model.
    
    Features:
    - Checkpoint management with automatic saving
    - Resume training from checkpoint
    - Early stopping based on validation metrics
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        loss_fn, 
        device, 
        evaluator=None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        early_stopping_patience: int = 3,
        scheduler: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = False,
        tensorboard_logger: Optional[TensorBoardLogger] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.evaluator = evaluator if evaluator else RecEvaluator()
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.tensorboard_logger = tensorboard_logger
        
        # Mixed precision training setup
        self.scaler = None
        if use_mixed_precision:
            if device == "cuda":
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision training enabled with GradScaler")
            else:
                logger.warning("Mixed precision requested but device is not CUDA. Disabling AMP.")
                self.use_mixed_precision = False
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.best_metric_value = float('-inf')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Log gradient accumulation info
        if gradient_accumulation_steps > 1:
            logger.info(f"Gradient accumulation enabled: {gradient_accumulation_steps} steps")
            logger.info(f"Effective batch size will be: batch_size × {gradient_accumulation_steps}")

    def train_epoch(self, dataloader, epoch=0, log_interval=100):
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.amp.autocast(device_type="cuda"):
                    output = self.model(
                        candidate_news=batch["candidate_news"],
                        news_histories=batch["news_histories"],
                        user_id=batch["user_id"],
                        target=batch["target"],
                        candidate_attention_mask=batch.get("candidate_attention_mask"),
                        news_histories_attention_mask=batch.get("news_histories_attention_mask")
                    )
                    loss = output.loss
            else:
                output = self.model(
                    candidate_news=batch["candidate_news"],
                    news_histories=batch["news_histories"],
                    user_id=batch["user_id"],
                    target=batch["target"],
                    candidate_attention_mask=batch.get("candidate_attention_mask"),
                    news_histories_attention_mask=batch.get("news_histories_attention_mask")
                )
                loss = output.loss
            
            # Scale loss by accumulation steps
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate loss for logging
            accumulated_loss += loss.item()
            total_loss += loss.item() * self.gradient_accumulation_steps  # Unscaled loss
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Log gradient norm before optimizer step
                if self.tensorboard_logger is not None and self.global_step % 100 == 0:
                    self.tensorboard_logger.log_gradient_norm(self.model, self.global_step)
                
                # Optimizer step with gradient scaling
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate scheduler (after optimizer step)
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update global step counter (counts optimizer updates)
                self.global_step += 1
                
                # Log to TensorBoard
                if self.tensorboard_logger is not None:
                    # Log training loss
                    self.tensorboard_logger.log_scalar(
                        "train/loss", 
                        accumulated_loss * self.gradient_accumulation_steps, 
                        self.global_step
                    )
                    
                    # Log learning rate
                    if self.scheduler is not None:
                        current_lr = self.scheduler.get_last_lr()[0]
                        self.tensorboard_logger.log_learning_rate(current_lr, self.global_step)
                    
                    # Log gradient scale for mixed precision
                    if self.use_mixed_precision and self.scaler is not None:
                        self.tensorboard_logger.log_scalar(
                            "train/gradient_scale",
                            self.scaler.get_scale(),
                            self.global_step
                        )
                
                # Reset accumulated loss
                accumulated_loss = 0.0
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / (step + 1)
                log_msg = f"Step {step+1}: loss={avg_loss:.4f}"
                
                # Log current learning rate
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_msg += f", lr={current_lr:.2e}"
                
                # Log effective batch size
                if self.gradient_accumulation_steps > 1:
                    log_msg += f" (accum_steps={self.gradient_accumulation_steps})"
                
                # Log gradient scale if using mixed precision
                if self.use_mixed_precision:
                    log_msg += f", scale={self.scaler.get_scale():.0f}"
                
                print(log_msg)
                logger.info(log_msg)
        
        # Handle remaining gradients if dataloader length is not divisible by accumulation steps
        if (step + 1) % self.gradient_accumulation_steps != 0:
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
        
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Move batch to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                
                # Forward pass
                output = self.model(
                    candidate_news=batch["candidate_news"],
                    news_histories=batch["news_histories"],
                    user_id=batch["user_id"],
                    target=batch["target"],
                    candidate_attention_mask=batch.get("candidate_attention_mask"),
                    news_histories_attention_mask=batch.get("news_histories_attention_mask")
                )
                
                # Extract logits and convert to scores
                logits = output.logits  # (batch_size, candidate_num)
                scores = torch.softmax(logits, dim=1)  # (batch_size, candidate_num)
                
                # Get target labels
                target = batch["target"]  # For validation, this is one-hot labels (batch_size, candidate_num)
                
                # Convert to numpy for evaluation
                all_scores.append(scores.cpu().numpy())
                all_labels.append(target.cpu().numpy())
        
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
        }
        
        print("Validation metrics:", avg_metrics)
        
        # Log sample predictions to TensorBoard
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_predictions(
                predictions=all_scores_np,
                labels=all_labels_np,
                step=self.global_step,
                num_samples=5
            )
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save checkpoint using checkpoint manager.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
        """
        if self.checkpoint_manager is None:
            print("Warning: No checkpoint manager configured. Skipping checkpoint save.")
            return
        
        # Check if this is the best model
        is_best = self.checkpoint_manager.is_best_checkpoint(metrics)
        
        # Prepare additional state
        additional_state = {
            "global_step": self.global_step,
            "patience_counter": self.patience_counter,
            "best_metric_value": self.best_metric_value
        }
        
        # Add scheduler state if available
        if self.scheduler is not None:
            additional_state["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Add scaler state if using mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            additional_state["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            additional_state=additional_state,
            is_best=is_best
        )
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest checkpoint.
        """
        if self.checkpoint_manager is None:
            raise ValueError("Cannot resume: No checkpoint manager configured")
        
        # Get checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                print("No checkpoint found. Starting training from scratch.")
                return
        
        # Load checkpoint
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # Restore training state
        self.current_epoch = metadata["epoch"]
        self.global_step = metadata.get("global_step", 0)
        self.patience_counter = metadata.get("patience_counter", 0)
        self.best_metric_value = metadata.get("best_metric_value", float('-inf'))
        
        # Restore scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in metadata:
            self.scheduler.load_state_dict(metadata["scheduler_state_dict"])
            print("Restored scheduler state")
        
        # Restore scaler state if using mixed precision
        if self.use_mixed_precision and self.scaler is not None and "scaler_state_dict" in metadata:
            self.scaler.load_state_dict(metadata["scaler_state_dict"])
            print("Restored gradient scaler state")
        
        print(f"Resumed training from epoch {self.current_epoch}")
        print(f"Best metric so far: {self.best_metric_value:.4f}")
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early based on validation metrics.
        
        Args:
            metrics: Validation metrics
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.checkpoint_manager is None:
            return False
        
        metric_name = self.checkpoint_manager.metric_name
        if metric_name not in metrics:
            print(f"Warning: Metric '{metric_name}' not found for early stopping")
            return False
        
        current_value = metrics[metric_name]
        
        # Check if improved
        if current_value > self.best_metric_value:
            self.best_metric_value = current_value
            self.patience_counter = 0
            print(f"Metric improved: {metric_name}={current_value:.4f}")
        else:
            self.patience_counter += 1
            print(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
        
        # Check if should stop
        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
        
        return False
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        num_epochs: int,
        start_epoch: int = 0,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Complete training loop with checkpoint management and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Total number of epochs to train
            start_epoch: Starting epoch (for resuming)
            log_interval: Logging interval for training steps
            
        Returns:
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_metrics": []
        }
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch=epoch, log_interval=log_interval)
            history["train_loss"].append(train_loss)
            print(f"Train loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            history["val_metrics"].append(val_metrics)
            
            # Log validation metrics to TensorBoard
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.log_metrics(
                    metrics=val_metrics,
                    step=self.global_step,
                    prefix="val/"
                )
                
                # Log epoch number
                self.tensorboard_logger.log_scalar("epoch", epoch, self.global_step)
            
            # Log current learning rate at end of epoch
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                lr_msg = f"Learning rate at end of epoch {epoch + 1}: {current_lr:.2e}"
                print(lr_msg)
                logger.info(lr_msg)
            
            # Save checkpoint
            self.save_checkpoint(epoch=epoch, metrics=val_metrics)
            
            # Check early stopping
            if self.check_early_stopping(val_metrics):
                print(f"Training stopped early at epoch {epoch + 1}")
                break
        
        print("\nTraining completed!")
        
        # Load best model if checkpoint manager is available
        if self.checkpoint_manager is not None:
            try:
                print("\nLoading best model...")
                self.checkpoint_manager.load_best_checkpoint(
                    model=self.model,
                    device=self.device
                )
            except FileNotFoundError:
                print("Best model not found. Using final model.")
        
        # Flush TensorBoard logs
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.flush()
        
        return history
