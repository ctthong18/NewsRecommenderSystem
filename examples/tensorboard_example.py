"""
Example: Using TensorBoard for Real-Time Training Monitoring

This example demonstrates how to:
1. Enable TensorBoard logging in training
2. View metrics in real-time
3. Compare different training runs
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tensorboard_logger import TensorBoardLogger
import torch
import numpy as np


def example_basic_logging():
    """Example 1: Basic TensorBoard logging"""
    print("=" * 60)
    print("Example 1: Basic TensorBoard Logging")
    print("=" * 60)
    
    # Create logger
    logger = TensorBoardLogger(
        log_dir="output/tensorboard/example_basic",
        enabled=True
    )
    
    # Simulate training loop
    print("\nSimulating training loop...")
    for step in range(100):
        # Simulate loss
        loss = 2.0 * np.exp(-step / 50) + 0.5 + np.random.normal(0, 0.1)
        
        # Log training loss
        logger.log_scalar("train/loss", loss, step)
        
        # Simulate learning rate schedule
        lr = 2e-5 * (1 - step / 100)
        logger.log_learning_rate(lr, step)
        
        if step % 20 == 0:
            print(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
    
    # Log validation metrics
    print("\nLogging validation metrics...")
    val_metrics = {
        "ndcg_at_10": 0.425,
        "ndcg_at_5": 0.363,
        "auc": 0.713,
        "mrr": 0.326
    }
    logger.log_metrics(val_metrics, step=100, prefix="val/")
    
    # Close logger
    logger.close()
    print("\n✓ Logs saved! View with: tensorboard --logdir=output/tensorboard/example_basic")


def example_gradient_monitoring():
    """Example 2: Monitoring gradients"""
    print("\n" + "=" * 60)
    print("Example 2: Gradient Monitoring")
    print("=" * 60)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    
    # Create logger
    logger = TensorBoardLogger(
        log_dir="output/tensorboard/example_gradients",
        enabled=True
    )
    
    # Simulate training with gradient logging
    print("\nSimulating training with gradient monitoring...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(50):
        # Forward pass
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradient norm
        logger.log_gradient_norm(model, step)
        
        # Log model weights (every 10 steps)
        if step % 10 == 0:
            logger.log_model_weights(model, step)
            print(f"Step {step}: loss={loss.item():.4f}")
        
        optimizer.step()
    
    logger.close()
    print("\n✓ Logs saved! View with: tensorboard --logdir=output/tensorboard/example_gradients")


def example_prediction_visualization():
    """Example 3: Visualizing predictions"""
    print("\n" + "=" * 60)
    print("Example 3: Prediction Visualization")
    print("=" * 60)
    
    # Create logger
    logger = TensorBoardLogger(
        log_dir="output/tensorboard/example_predictions",
        enabled=True
    )
    
    # Simulate predictions
    print("\nLogging sample predictions...")
    
    # Generate fake predictions and labels
    num_samples = 10
    num_candidates = 20
    
    predictions = np.random.rand(num_samples, num_candidates)
    labels = np.zeros((num_samples, num_candidates))
    
    # Set one positive label per sample
    for i in range(num_samples):
        pos_idx = np.random.randint(0, num_candidates)
        labels[i, pos_idx] = 1
    
    # Log predictions
    logger.log_predictions(
        predictions=predictions,
        labels=labels,
        step=0,
        num_samples=5
    )
    
    print("✓ Sample predictions logged")
    
    logger.close()
    print("\n✓ Logs saved! View with: tensorboard --logdir=output/tensorboard/example_predictions")


def example_comparing_runs():
    """Example 4: Comparing multiple training runs"""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Multiple Runs")
    print("=" * 60)
    
    # Simulate two different training configurations
    configs = [
        {"name": "baseline", "lr": 1e-4, "color": "blue"},
        {"name": "high_lr", "lr": 5e-4, "color": "red"}
    ]
    
    for config in configs:
        print(f"\nSimulating run: {config['name']}")
        
        logger = TensorBoardLogger(
            log_dir=f"output/tensorboard/comparison/{config['name']}",
            enabled=True
        )
        
        # Simulate training with different learning rates
        for step in range(100):
            # Loss depends on learning rate
            base_loss = 2.0 * np.exp(-step / 50)
            noise = np.random.normal(0, 0.1)
            
            # Higher LR converges faster but with more noise
            if config['name'] == "high_lr":
                loss = base_loss * 0.8 + noise * 1.5
            else:
                loss = base_loss + noise
            
            logger.log_scalar("train/loss", loss, step)
            logger.log_learning_rate(config['lr'], step)
        
        # Log final metrics
        if config['name'] == "high_lr":
            metrics = {"ndcg_at_10": 0.410, "auc": 0.695}
        else:
            metrics = {"ndcg_at_10": 0.425, "auc": 0.713}
        
        logger.log_metrics(metrics, step=100, prefix="val/")
        logger.close()
    
    print("\n✓ Multiple runs saved!")
    print("View comparison with: tensorboard --logdir=output/tensorboard/comparison")
    print("TensorBoard will show both runs on the same charts")


def example_with_context_manager():
    """Example 5: Using context manager"""
    print("\n" + "=" * 60)
    print("Example 5: Context Manager Usage")
    print("=" * 60)
    
    print("\nUsing TensorBoard logger with context manager...")
    
    # Use context manager for automatic cleanup
    with TensorBoardLogger("output/tensorboard/example_context", enabled=True) as logger:
        for step in range(50):
            loss = 1.5 * np.exp(-step / 30) + 0.3
            logger.log_scalar("train/loss", loss, step)
            
            if step % 10 == 0:
                print(f"Step {step}: loss={loss:.4f}")
    
    print("\n✓ Logger automatically closed")
    print("View with: tensorboard --logdir=output/tensorboard/example_context")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("TensorBoard Logging Examples")
    print("=" * 60)
    print("\nThese examples demonstrate TensorBoard integration features.")
    print("After running, start TensorBoard to view the results:")
    print("  tensorboard --logdir=output/tensorboard")
    print("\n")
    
    # Run examples
    example_basic_logging()
    example_gradient_monitoring()
    example_prediction_visualization()
    example_comparing_runs()
    example_with_context_manager()
    
    print("\n" + "=" * 60)
    print("All Examples Complete!")
    print("=" * 60)
    print("\nTo view all examples in TensorBoard:")
    print("  tensorboard --logdir=output/tensorboard")
    print("\nThen open: http://localhost:6006")
    print("\nYou can:")
    print("  - Compare different runs")
    print("  - View loss curves")
    print("  - Inspect gradient norms")
    print("  - See sample predictions")
    print("=" * 60)


if __name__ == "__main__":
    main()
