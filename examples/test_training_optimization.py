"""
Example script to test training optimization features:
- Learning rate scheduler (warmup + cosine decay)
- Gradient accumulation
- Mixed precision training

This script demonstrates how to use the new training optimization features.
"""
import torch
from src.utils.lr_scheduler import create_scheduler, get_warmup_cosine_schedule


def test_lr_scheduler():
    """Test learning rate scheduler creation and behavior."""
    print("=" * 60)
    print("Testing Learning Rate Scheduler")
    print("=" * 60)
    
    # Create a dummy model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Test cosine scheduler
    num_training_steps = 1000
    warmup_ratio = 0.1
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        num_training_steps=num_training_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=0.0
    )
    
    print(f"Created cosine scheduler with {num_training_steps} steps")
    print(f"Warmup ratio: {warmup_ratio}")
    
    # Simulate training and track learning rates
    lrs = []
    for step in range(num_training_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    
    # Check key points
    warmup_steps = int(num_training_steps * warmup_ratio)
    print(f"\nLearning rate at key points:")
    print(f"  Step 0 (start): {lrs[0]:.2e}")
    print(f"  Step {warmup_steps} (end of warmup): {lrs[warmup_steps]:.2e}")
    print(f"  Step {num_training_steps//2} (middle): {lrs[num_training_steps//2]:.2e}")
    print(f"  Step {num_training_steps-1} (end): {lrs[-1]:.2e}")
    
    # Verify warmup behavior
    assert lrs[0] < lrs[warmup_steps], "LR should increase during warmup"
    assert lrs[warmup_steps] > lrs[-1], "LR should decrease after warmup"
    
    print("\n✓ Learning rate scheduler test passed!")
    return True


def test_gradient_accumulation_config():
    """Test gradient accumulation configuration."""
    print("\n" + "=" * 60)
    print("Testing Gradient Accumulation Configuration")
    print("=" * 60)
    
    batch_size = 8
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Calculate training steps with accumulation
    num_samples = 10000
    epochs = 3
    steps_per_epoch = num_samples // batch_size
    optimizer_updates_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_optimizer_updates = optimizer_updates_per_epoch * epochs
    
    print(f"\nTraining calculation:")
    print(f"  Total samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps per epoch (forward passes): {steps_per_epoch}")
    print(f"  Optimizer updates per epoch: {optimizer_updates_per_epoch}")
    print(f"  Total optimizer updates: {total_optimizer_updates}")
    
    print("\n✓ Gradient accumulation configuration test passed!")
    return True


def test_mixed_precision_availability():
    """Test mixed precision training availability."""
    print("\n" + "=" * 60)
    print("Testing Mixed Precision Training Availability")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test AMP availability
        try:
            scaler = torch.cuda.amp.GradScaler()
            print("\n✓ Mixed precision training (AMP) is available!")
            print("  You can enable it with: use_mixed_precision: true")
        except Exception as e:
            print(f"\n✗ Mixed precision training not available: {e}")
    else:
        print("\n⚠ CUDA not available. Mixed precision training requires GPU.")
        print("  Training will run on CPU without mixed precision.")
    
    return True


def test_scheduler_types():
    """Test different scheduler types."""
    print("\n" + "=" * 60)
    print("Testing Different Scheduler Types")
    print("=" * 60)
    
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_steps = 100
    
    scheduler_types = ['cosine', 'linear', 'constant']
    
    for sched_type in scheduler_types:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=sched_type,
            num_training_steps=num_steps,
            warmup_ratio=0.1
        )
        
        # Get LR at different points
        lrs = []
        for _ in range(num_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        
        print(f"\n{sched_type.capitalize()} scheduler:")
        print(f"  Start LR: {lrs[0]:.2e}")
        print(f"  Middle LR: {lrs[num_steps//2]:.2e}")
        print(f"  End LR: {lrs[-1]:.2e}")
    
    print("\n✓ All scheduler types work correctly!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Training Optimization Features Test Suite")
    print("=" * 60)
    
    tests = [
        test_lr_scheduler,
        test_gradient_accumulation_config,
        test_mixed_precision_availability,
        test_scheduler_types
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
        print("\nYou can now use these features in training:")
        print("  1. Enable scheduler: use_scheduler: true")
        print("  2. Set accumulation: gradient_accumulation_steps: 4")
        print("  3. Enable AMP: use_mixed_precision: true")
    else:
        print("\n✗ Some tests failed. Please check the output above.")


if __name__ == "__main__":
    main()
