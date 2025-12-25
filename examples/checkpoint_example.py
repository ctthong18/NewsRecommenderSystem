"""
Example script demonstrating checkpoint management usage.

This script shows how to:
1. Initialize a checkpoint manager
2. Save checkpoints during training
3. Resume training from a checkpoint
4. Load the best model
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.checkpoint_manager import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def simulate_training_epoch(model, optimizer, epoch):
    """Simulate one training epoch."""
    # Dummy training
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
    
    # Simulate validation metrics (gradually improving)
    metrics = {
        "ndcg_at_10": 0.35 + epoch * 0.03 + torch.rand(1).item() * 0.02,
        "auc": 0.65 + epoch * 0.02 + torch.rand(1).item() * 0.01,
        "mrr": 0.30 + epoch * 0.025 + torch.rand(1).item() * 0.015
    }
    
    return loss.item(), metrics


def example_basic_usage():
    """Example 1: Basic checkpoint saving and loading."""
    print("=" * 60)
    print("Example 1: Basic Checkpoint Saving and Loading")
    print("=" * 60)
    
    # Setup
    checkpoint_dir = "output/examples/checkpoints_basic"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=3,
        metric_name="ndcg_at_10",
        mode="max"
    )
    
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Tracking metric: {checkpoint_manager.metric_name}")
    print(f"Keeping last {checkpoint_manager.keep_last_n} checkpoints\n")
    
    # Simulate training for 5 epochs
    for epoch in range(5):
        loss, metrics = simulate_training_epoch(model, optimizer, epoch)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, metrics={metrics}")
        
        # Check if best
        is_best = checkpoint_manager.is_best_checkpoint(metrics)
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
        print()
    
    # List all checkpoints
    print("\nAll checkpoints:")
    checkpoints = checkpoint_manager.list_checkpoints()
    for ckpt in checkpoints:
        best_marker = " (BEST)" if ckpt.get("is_best") else ""
        print(f"  Epoch {ckpt['epoch']}: {ckpt['metrics']}{best_marker}")
    
    # Load best checkpoint
    print("\nLoading best checkpoint...")
    new_model = SimpleModel()
    metadata = checkpoint_manager.load_best_checkpoint(
        model=new_model,
        device="cpu"
    )
    print(f"Loaded best model from epoch {metadata['epoch']}")
    print(f"Best metrics: {metadata['metrics']}")


def example_resume_training():
    """Example 2: Resume training from checkpoint."""
    print("\n" + "=" * 60)
    print("Example 2: Resume Training from Checkpoint")
    print("=" * 60)
    
    # Setup
    checkpoint_dir = "output/examples/checkpoints_resume"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=2,
        metric_name="ndcg_at_10",
        mode="max"
    )
    
    print(f"\nCheckpoint directory: {checkpoint_dir}\n")
    
    # First training session (3 epochs)
    print("=== First Training Session ===")
    for epoch in range(3):
        loss, metrics = simulate_training_epoch(model, optimizer, epoch)
        print(f"Epoch {epoch}: loss={loss:.4f}, ndcg={metrics['ndcg_at_10']:.4f}")
        
        is_best = checkpoint_manager.is_best_checkpoint(metrics)
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            additional_state={"training_session": 1},
            is_best=is_best
        )
    
    print("\nTraining interrupted! Saving checkpoint...")
    
    # Simulate resuming training
    print("\n=== Resuming Training ===")
    
    # Create new model and optimizer (simulating fresh start)
    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    # Load latest checkpoint
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    metadata = checkpoint_manager.load_checkpoint(
        checkpoint_path=str(latest_checkpoint),
        model=new_model,
        optimizer=new_optimizer,
        device="cpu"
    )
    
    start_epoch = metadata["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous metrics: {metadata['metrics']}\n")
    
    # Continue training for 2 more epochs
    for epoch in range(start_epoch, start_epoch + 2):
        loss, metrics = simulate_training_epoch(new_model, new_optimizer, epoch)
        print(f"Epoch {epoch}: loss={loss:.4f}, ndcg={metrics['ndcg_at_10']:.4f}")
        
        is_best = checkpoint_manager.is_best_checkpoint(metrics)
        checkpoint_manager.save_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            epoch=epoch,
            metrics=metrics,
            additional_state={"training_session": 2},
            is_best=is_best
        )
    
    print("\nTraining completed!")


def example_early_stopping():
    """Example 3: Early stopping simulation."""
    print("\n" + "=" * 60)
    print("Example 3: Early Stopping Simulation")
    print("=" * 60)
    
    # Setup
    checkpoint_dir = "output/examples/checkpoints_early_stop"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=3,
        metric_name="ndcg_at_10",
        mode="max"
    )
    
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print("Early stopping patience: 3 epochs\n")
    
    # Early stopping parameters
    patience = 3
    patience_counter = 0
    best_metric = float('-inf')
    
    # Simulate training with plateau
    for epoch in range(10):
        # Simulate metrics that plateau after epoch 4
        if epoch < 4:
            base_metric = 0.35 + epoch * 0.05
        else:
            base_metric = 0.50 + torch.rand(1).item() * 0.01 - 0.015
        
        loss = 2.0 - epoch * 0.15
        metrics = {
            "ndcg_at_10": base_metric,
            "auc": 0.65 + epoch * 0.01,
            "mrr": 0.30 + epoch * 0.02
        }
        
        print(f"Epoch {epoch}: loss={loss:.4f}, ndcg={metrics['ndcg_at_10']:.4f}")
        
        # Check if improved
        is_best = checkpoint_manager.is_best_checkpoint(metrics)
        
        if metrics["ndcg_at_10"] > best_metric:
            best_metric = metrics["ndcg_at_10"]
            patience_counter = 0
            print(f"  → Metric improved! New best: {best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{patience}")
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            additional_state={"patience_counter": patience_counter},
            is_best=is_best
        )
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch}!")
            print(f"Best metric: {best_metric:.4f}")
            break
        
        print()
    
    # Load best model
    print("\nLoading best model...")
    best_metadata = checkpoint_manager.load_best_checkpoint(
        model=model,
        device="cpu"
    )
    print(f"Best model from epoch {best_metadata['epoch']}")
    print(f"Best metrics: {best_metadata['metrics']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Checkpoint Management Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_resume_training()
    example_early_stopping()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nCheckpoint directories created:")
    print("  - output/examples/checkpoints_basic")
    print("  - output/examples/checkpoints_resume")
    print("  - output/examples/checkpoints_early_stop")
    print("\nYou can inspect the checkpoint files and metadata in these directories.")


if __name__ == "__main__":
    main()
