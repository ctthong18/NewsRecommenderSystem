import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        metric_name: str = "ndcg_at_10",
        mode: str = "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best metric value
        self.best_metric_value = float('-inf') if mode == "max" else float('inf')
        self.best_checkpoint_path = None
        
        # Load existing best metric if available
        self._load_best_metric_info()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        additional_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "metric_name": self.metric_name,
            "best_metric_value": self.best_metric_value
        }
        
        # Add additional state if provided
        if additional_state:
            checkpoint.update(additional_state)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_metadata.json"
        metadata = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": checkpoint["timestamp"],
            "is_best": is_best
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint_path = best_path
            print(f"Saved best model: {best_path} ({self.metric_name}={metrics.get(self.metric_name, 0):.4f})")
            
            # Save best model info
            self._save_best_metric_info(metrics.get(self.metric_name, 0), epoch)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        strict: bool = True
    ) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint structure
        required_keys = ["model_state_dict", "epoch"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint format. Missing keys: {missing_keys}")
        
        # Load model state
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            print(f"Loaded model state from epoch {checkpoint['epoch']}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}")
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
        
        # Return metadata
        metadata = {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "timestamp": checkpoint.get("timestamp", "unknown"),
        }
        
        # Include any additional state
        for key in checkpoint:
            if key not in ["model_state_dict", "optimizer_state_dict", "epoch", "metrics", "timestamp"]:
                metadata[key] = checkpoint[key]
        
        print(f"Checkpoint loaded successfully. Epoch: {metadata['epoch']}, Metrics: {metadata['metrics']}")
        
        return metadata
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        return self.load_checkpoint(best_path, model, optimizer, device)
    
    def is_best_checkpoint(self, metrics: Dict[str, float]) -> bool:
        if self.metric_name not in metrics:
            print(f"Warning: Metric '{self.metric_name}' not found in metrics. Available: {list(metrics.keys())}")
            return False
        
        current_value = metrics[self.metric_name]
        
        if self.mode == "max":
            is_best = current_value > self.best_metric_value
        else:  # mode == "min"
            is_best = current_value < self.best_metric_value
        
        if is_best:
            self.best_metric_value = current_value
        
        return is_best
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = self._get_checkpoint_files()
        
        if not checkpoints:
            return None
        
        # Sort by epoch number (extracted from filename)
        checkpoints.sort(key=lambda p: self._extract_epoch_from_filename(p), reverse=True)
        
        return checkpoints[0]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        checkpoints = []
        
        for ckpt_path in self._get_checkpoint_files():
            epoch = self._extract_epoch_from_filename(ckpt_path)
            metadata_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_metadata.json"
            
            info = {
                "path": str(ckpt_path),
                "epoch": epoch,
                "is_best": ckpt_path.name == "best_model.pt"
            }
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    info.update(metadata)
            
            checkpoints.append(info)
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
        
        return checkpoints
    
    def _get_checkpoint_files(self) -> List[Path]:
        """Get list of checkpoint files in checkpoint directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        # Include best model if it exists
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists() and best_path not in checkpoints:
            checkpoints.append(best_path)
        
        return checkpoints
    
    def _extract_epoch_from_filename(self, path: Path) -> int:
        """Extract epoch number from checkpoint filename."""
        if path.name == "best_model.pt":
            # For best model, read from metadata
            info_path = self.checkpoint_dir / "best_model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    return info.get("epoch", -1)
            return -1
        
        # Extract from filename: checkpoint_epoch_N.pt
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = [p for p in self._get_checkpoint_files() if p.name != "best_model.pt"]
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by epoch
        checkpoints.sort(key=lambda p: self._extract_epoch_from_filename(p))
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.keep_last_n]
        
        for ckpt_path in to_remove:
            epoch = self._extract_epoch_from_filename(ckpt_path)
            metadata_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_metadata.json"
            
            # Remove checkpoint and metadata
            ckpt_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            print(f"Removed old checkpoint: {ckpt_path}")
    
    def _save_best_metric_info(self, metric_value: float, epoch: int):
        """Save information about the best model."""
        info_path = self.checkpoint_dir / "best_model_info.json"
        info = {
            "metric_name": self.metric_name,
            "metric_value": metric_value,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat()
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_best_metric_info(self):
        """Load information about the best model if it exists."""
        info_path = self.checkpoint_dir / "best_model_info.json"
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.best_metric_value = info.get("metric_value", self.best_metric_value)
                print(f"Loaded best metric info: {self.metric_name}={self.best_metric_value:.4f}")
