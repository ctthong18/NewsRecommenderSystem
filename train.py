"""
Training script for DeBERTa-v3-base NAML model with:
- LLM-generated news descriptions
- Hard negative sampling (optional)
- Configuration management system
"""
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from src.models.DeBERTaNewsEncoder import DeBERTaNewsEncoder
from src.models.UserEncoder import UserEncoder
from src.models.NAML import NAML
from src.utils.metrics import RecEvaluator
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.tensorboard_logger import TensorBoardLogger
from src.trainer.plm_trainer import PLMTrainer
from src.data.dataset_mind import MINDTrainDataset, MINDValDataset
from src.data.dataloader_builder import build_train_dataloader, build_val_dataloader
from src.data.dataframe import read_news_df, read_behavior_df, create_user_ids_to_idx_map
from src.utils.tokenization import create_transform_fn_from_pretrained_tokenizer
from src.utils.config_loader import load_config
from src.utils.lr_scheduler import create_scheduler
from src.const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR
from src.const.mind import UNKNOWN_USER_IDX


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train news recommendation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file (default: configs/base_config.yaml)"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (e.g., --override model.lr=0.001 training.batch_size=16)"
    )
    return parser.parse_args()


def parse_overrides(override_args):
    """
    Parse command-line override arguments into a nested dictionary.
    
    Args:
        override_args: List of strings in format "key.path=value"
        
    Returns:
        Dictionary with nested structure
    """
    if not override_args:
        return {}
    
    overrides = {}
    for arg in override_args:
        if "=" not in arg:
            print(f"Warning: Ignoring invalid override format: {arg}")
            continue
        
        key_path, value = arg.split("=", 1)
        keys = key_path.split(".")
        
        # Navigate/create nested structure
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value type
        final_key = keys[-1]
        try:
            # Try numeric conversion
            if "." in value:
                current[final_key] = float(value)
            else:
                current[final_key] = int(value)
        except ValueError:
            # Try boolean
            if value.lower() in ("true", "false"):
                current[final_key] = value.lower() == "true"
            else:
                # Keep as string
                current[final_key] = value
    
    return overrides


def main():
    # Load Configuration
    args = parse_args()
    
    # Parse command-line overrides
    overrides = parse_overrides(args.override)
    
    # Load config with overrides
    try:
        config = load_config(args.config, overrides=overrides)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = load_config(None)  # Empty config, will use defaults below
    
    # Extract configuration with defaults
    device = config.get("training.device", "cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = config.get("model.pretrained", "microsoft/deberta-v3-base")
    max_length = config.get("model.max_length", 64)
    batch_size = config.get("training.batch_size", 8)
    lr = config.get("training.lr", 2e-5)
    num_workers = config.get("training.num_workers", 4)
    epochs = config.get("training.epochs", 3)
    npratio = config.get("training.npratio", 4)
    history_size = config.get("training.history_size", 50)
    conv_kernel_num = config.get("model.conv_kernel_num", 400)
    query_dim = config.get("model.query_dim", 200)
    
    # Paths with defaults
    train_news_path = Path(config.get("data.train_news", "Data/raw/MINDlarge_train/news.tsv"))
    train_behavior_path = Path(config.get("data.train_behaviors", "Data/raw/MINDlarge_train/behaviors.tsv"))
    val_news_path = Path(config.get("data.val_news", "Data/raw/MINDlarge_dev/news.tsv"))
    val_behavior_path = Path(config.get("data.val_behaviors", "Data/raw/MINDlarge_dev/behaviors.tsv"))
    
    # Optional features
    llm_description_path_str = config.get("data.llm_description", 
                                          "Data/generated/news_descriptions.json")
    llm_description_path = Path(llm_description_path_str) if llm_description_path_str else None
    use_hard_negative = config.get("training.use_hard_negative", False)
    news_embeddings_cache = None  # Optional: dict of {news_id: embedding}
    
    # Output paths
    output_dir = Path(config.get("paths.output_dir", "output/models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config.get("paths.checkpoint_dir", "output/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint and early stopping configuration
    keep_last_n_checkpoints = config.get("training.keep_last_n_checkpoints", 3)
    early_stopping_patience = config.get("training.early_stopping_patience", 3)
    metric_for_best_model = config.get("training.metric_for_best_model", "ndcg_at_10")
    resume_from_checkpoint = config.get("training.resume_from_checkpoint", False)
    
    # Learning rate scheduler configuration
    use_scheduler = config.get("training.use_scheduler", True)
    scheduler_type = config.get("training.scheduler_type", "cosine")
    warmup_ratio = config.get("training.warmup_ratio", 0.1)
    min_lr_ratio = config.get("training.min_lr_ratio", 0.0)
    
    # Gradient accumulation configuration
    gradient_accumulation_steps = config.get("training.gradient_accumulation_steps", 1)
    
    # Mixed precision configuration
    use_mixed_precision = config.get("training.use_mixed_precision", False)
    
    # TensorBoard configuration
    use_tensorboard = config.get("training.use_tensorboard", True)
    tensorboard_log_dir = Path(config.get("paths.tensorboard_dir", "output/tensorboard"))
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Device: {device}")
    print(f"Model: {pretrained_model}")
    print(f"Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Learning rate scheduler: {scheduler_type if use_scheduler else 'None'}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Mixed precision training: {use_mixed_precision}")
    print(f"TensorBoard logging: {use_tensorboard}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model,
        use_fast=False,
        trust_remote_code=True
    )
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_length)

    # Load Data
    print("Loading data...")
    train_news_df = read_news_df(train_news_path)
    train_behavior_df = read_behavior_df(train_behavior_path)
    val_news_df = read_news_df(val_news_path)
    val_behavior_df = read_behavior_df(val_behavior_path)
    
    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df, val_behavior_df)

    # Dataloader
    train_dataset = MINDTrainDataset(
        behavior_df=train_behavior_df,
        news_df=train_news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        npratio=npratio,
        history_size=history_size,
        llm_description_path=llm_description_path if llm_description_path.exists() else None,
        use_hard_negative=use_hard_negative,
        news_embeddings_cache=news_embeddings_cache,
        device=device,
    )
    
    val_dataset = MINDValDataset(
        behavior_df=val_behavior_df,
        news_df=val_news_df,
        user_ids_to_idx_map=user_ids_to_idx_map,
        batch_transform_texts=transform_fn,
        history_size=history_size,
        llm_description_path=llm_description_path if llm_description_path.exists() else None,
        device=device,
    )

    # Use optimized dataloader builder with prefetching and pin_memory
    train_loader = build_train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2,
    )
    
    val_loader = build_val_dataloader(
        dataset=val_dataset,
        batch_size=1,  # Validation uses batch_size=1 for proper evaluation
        num_workers=max(1, num_workers // 2),  # Use fewer workers for validation
        pin_memory=(device == "cuda"),
    )

    # Khởi tạo mô hình
    print("Initializing model...")
    news_encoder = DeBERTaNewsEncoder(
        pretrained=pretrained_model,
        conv_kernel_num=conv_kernel_num,
        kernel_size=3,
        query_dim=query_dim
    )
    user_encoder = UserEncoder(conv_kernel_num=conv_kernel_num, query_dim=query_dim)
    model = NAML(news_encoder=news_encoder, user_encoder=user_encoder).to(device)

    # Optimizer + Loss + Evaluator
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()  # chuẩn của NAML
    evaluator = RecEvaluator()

    # Learning Rate Scheduler
    scheduler = None
    if use_scheduler:
        # Calculate total training steps (accounting for gradient accumulation)
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        if len(train_loader) % gradient_accumulation_steps != 0:
            steps_per_epoch += 1
        num_training_steps = steps_per_epoch * epochs
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio
        )
        print(f"Learning rate scheduler created: {scheduler_type}")
        print(f"Total training steps (optimizer updates): {num_training_steps}")
        print(f"Warmup steps: {int(num_training_steps * warmup_ratio)}")
        if gradient_accumulation_steps > 1:
            print(f"Effective batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")

    # Checkpoint Manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n=keep_last_n_checkpoints,
        metric_name=metric_for_best_model,
        mode="max"  # Higher is better for ndcg, auc, mrr
    )

    # TensorBoard Logger
    tensorboard_logger = None
    if use_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            log_dir=str(tensorboard_log_dir),
            enabled=True
        )

    # Trainer
    trainer = PLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        evaluator=evaluator,
        checkpoint_manager=checkpoint_manager,
        early_stopping_patience=early_stopping_patience,
        scheduler=scheduler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_mixed_precision=use_mixed_precision,
        tensorboard_logger=tensorboard_logger
    )

    # Train + Eval
    print("Starting training...")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if resume_from_checkpoint:
        try:
            trainer.resume_from_checkpoint()
            start_epoch = trainer.current_epoch + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not resume from checkpoint: {e}")
            print("Starting training from scratch...")
    
    # Use the new train method with checkpoint management
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        start_epoch=start_epoch,
        log_interval=100
    )
    
    # Save final model to output directory (for backward compatibility)
    final_model_path = output_dir / "deberta_naml_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model also saved to {final_model_path} (for backward compatibility)")
    
    # Print checkpoint information
    print("\n=== Checkpoint Summary ===")
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Total checkpoints saved: {len(checkpoints)}")
    if checkpoints:
        best_ckpt = [c for c in checkpoints if c.get("is_best", False)]
        if best_ckpt:
            print(f"Best checkpoint: Epoch {best_ckpt[0]['epoch']}, "
                  f"{metric_for_best_model}={best_ckpt[0]['metrics'].get(metric_for_best_model, 0):.4f}")
    
    print(f"\nTraining completed! Checkpoints saved to {checkpoint_dir}")
    
    # Close TensorBoard logger
    if tensorboard_logger is not None:
        tensorboard_logger.close()


if __name__ == "__main__":
    main()
