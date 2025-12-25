"""
Learning rate scheduler utilities for training optimization.

Provides warmup + cosine decay scheduler for better training convergence.
"""
import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def get_warmup_cosine_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    The learning rate increases linearly from 0 to the initial lr during warmup,
    then decreases following a cosine curve to min_lr_ratio * initial_lr.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half cycle)
        min_lr_ratio: Minimum learning rate as ratio of initial lr (default: 0.0)
        last_epoch: The index of last epoch (for resuming)
        
    Returns:
        LambdaLR scheduler
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        >>> scheduler = get_warmup_cosine_schedule(
        ...     optimizer, 
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000
        ... )
        >>> for epoch in range(num_epochs):
        ...     train_epoch(...)
        ...     scheduler.step()
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_warmup_linear_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr (default: 0.0)
        last_epoch: The index of last epoch (for resuming)
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        decay = 1.0 - progress
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with linear warmup then constant lr.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of steps for warmup phase
        last_epoch: The index of last epoch (for resuming)
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_scheduler(
    optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    **kwargs
):
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps (if None, calculated from warmup_ratio)
        warmup_ratio: Ratio of warmup steps to total steps (default: 0.1)
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Learning rate scheduler
        
    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='cosine',
        ...     num_training_steps=1000,
        ...     warmup_ratio=0.1
        ... )
    """
    # Calculate warmup steps if not provided
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == 'cosine':
        return get_warmup_cosine_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_type == 'linear':
        return get_warmup_linear_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_type == 'constant':
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from: 'cosine', 'linear', 'constant'"
        )
