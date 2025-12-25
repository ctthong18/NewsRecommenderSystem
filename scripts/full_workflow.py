#!/usr/bin/env python3
"""
Full workflow automation script: Training → Inference → Submission

This script automates the complete pipeline from training to submission generation.

Usage:
    python scripts/full_workflow.py --mode train_and_submit
    python scripts/full_workflow.py --mode submit_only --checkpoint output/checkpoints/best_model.pt
"""
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class WorkflowManager:
    """Manages the complete training to submission workflow."""
    
    def __init__(self, config_path: str = "configs/gpu_48gb_large.yaml"):
        self.config_path = config_path
        self.start_time = datetime.now()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Run a command and return success status.
        
        Args:
            cmd: Command to run as list
            description: Description for logging
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"Starting: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log(f"✅ Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed: {description}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Unexpected error in {description}: {e}", "ERROR")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available."""
        self.log("Checking prerequisites...")
        
        # Check config file
        if not Path(self.config_path).exists():
            self.log(f"❌ Config file not found: {self.config_path}", "ERROR")
            return False
        
        # Check training data
        train_paths = [
            "Data/raw/MINDlarge_train/news.tsv",
            "Data/raw/MINDlarge_train/behaviors.tsv",
            "Data/raw/MINDlarge_dev/news.tsv", 
            "Data/raw/MINDlarge_dev/behaviors.tsv"
        ]
        
        for path in train_paths:
            if not Path(path).exists():
                self.log(f"❌ Training data not found: {path}", "ERROR")
                self.log("Please download MINDlarge dataset first:", "ERROR")
                self.log("  python -m src.scripts.download_mind --size large", "ERROR")
                return False
        
        # Check test data
        test_paths = [
            "Data/raw/MINDlarge_test/news.tsv",
            "Data/raw/MINDlarge_test/behaviors.tsv"
        ]
        
        for path in test_paths:
            if not Path(path).exists():
                self.log(f"⚠️  Test data not found: {path}", "WARNING")
                self.log("Will need test data for submission generation", "WARNING")
        
        self.log("✅ Prerequisites check completed")
        return True
    
    def run_training(self) -> Optional[str]:
        """
        Run model training.
        
        Returns:
            Path to best checkpoint if successful, None otherwise
        """
        self.log("="*60)
        self.log("STARTING MODEL TRAINING")
        self.log("="*60)
        
        # Training command
        cmd = [
            sys.executable, "train.py",
            "--config", self.config_path
        ]
        
        success = self.run_command(cmd, "Model Training")
        
        if not success:
            return None
        
        # Find best checkpoint
        checkpoint_dir = Path("output/checkpoints")
        best_checkpoint = checkpoint_dir / "best_model.pt"
        
        if best_checkpoint.exists():
            self.log(f"✅ Best checkpoint found: {best_checkpoint}")
            return str(best_checkpoint)
        else:
            # Look for any checkpoint
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                self.log(f"⚠️  Using latest checkpoint: {latest_checkpoint}", "WARNING")
                return str(latest_checkpoint)
            else:
                self.log("❌ No checkpoints found after training", "ERROR")
                return None
    
    def generate_submission(self, checkpoint_path: str, output_dir: str = None) -> Optional[str]:
        """
        Generate submission file.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_dir: Output directory for submission
            
        Returns:
            Path to submission directory if successful, None otherwise
        """
        self.log("="*60)
        self.log("GENERATING SUBMISSION")
        self.log("="*60)
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/submission_{timestamp}"
        
        # Check test data
        test_news = "Data/raw/MINDlarge_test/news.tsv"
        test_behaviors = "Data/raw/MINDlarge_test/behaviors.tsv"
        
        if not Path(test_news).exists() or not Path(test_behaviors).exists():
            self.log("❌ Test data not found. Cannot generate submission.", "ERROR")
            self.log("Please download test data first:", "ERROR")
            self.log("  python -m src.scripts.download_mind --size large --split test", "ERROR")
            return None
        
        # Submission command
        cmd = [
            sys.executable, "-m", "src.scripts.generate_submission",
            "--checkpoint", checkpoint_path,
            "--test-news", test_news,
            "--test-behaviors", test_behaviors,
            "--config", self.config_path,
            "--output-dir", output_dir,
            "--batch-size", "8"
        ]
        
        # Add LLM descriptions if available
        llm_desc_path = "Data/generated/news_descriptions.json"
        if Path(llm_desc_path).exists():
            cmd.extend(["--llm-description-path", llm_desc_path])
            self.log("✅ Using LLM descriptions")
        
        success = self.run_command(cmd, "Submission Generation")
        
        if success:
            self.log(f"✅ Submission generated in: {output_dir}")
            return output_dir
        else:
            return None
    
    def validate_submission(self, submission_dir: str) -> bool:
        """
        Validate submission files.
        
        Args:
            submission_dir: Directory containing submission files
            
        Returns:
            True if valid, False otherwise
        """
        self.log("Validating submission files...")
        
        submission_path = Path(submission_dir)
        
        # Check required files
        prediction_file = submission_path / "prediction.txt"
        metadata_file = submission_path / "prediction_metadata.json"
        
        if not prediction_file.exists():
            self.log("❌ prediction.txt not found", "ERROR")
            return False
        
        if not metadata_file.exists():
            self.log("❌ prediction_metadata.json not found", "ERROR")
            return False
        
        # Validate prediction file format
        try:
            with open(prediction_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                self.log("❌ prediction.txt is empty", "ERROR")
                return False
            
            # Check first line format
            first_line = lines[0].strip()
            parts = first_line.split()
            if len(parts) < 2:
                self.log("❌ Invalid prediction format", "ERROR")
                return False
            
            self.log(f"✅ Prediction file: {len(lines)} impressions")
            
        except Exception as e:
            self.log(f"❌ Error reading prediction file: {e}", "ERROR")
            return False
        
        # Validate metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            required_fields = ["checkpoint", "num_impressions", "created_at"]
            for field in required_fields:
                if field not in metadata:
                    self.log(f"❌ Missing metadata field: {field}", "ERROR")
                    return False
            
            self.log(f"✅ Metadata: {metadata['num_impressions']} impressions")
            
        except Exception as e:
            self.log(f"❌ Error reading metadata: {e}", "ERROR")
            return False
        
        self.log("✅ Submission validation passed")
        return True
    
    def print_summary(self, checkpoint_path: str = None, submission_dir: str = None):
        """Print workflow summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.log("="*60)
        self.log("WORKFLOW SUMMARY")
        self.log("="*60)
        self.log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Total duration: {duration}")
        
        if checkpoint_path:
            self.log(f"Model checkpoint: {checkpoint_path}")
        
        if submission_dir:
            self.log(f"Submission directory: {submission_dir}")
            self.log("Files ready for submission:")
            self.log(f"  - {submission_dir}/prediction.txt")
            self.log(f"  - {submission_dir}/submission.zip (if created)")
        
        self.log("="*60)
    
    def run_full_workflow(self) -> bool:
        """
        Run the complete workflow: training → submission.
        
        Returns:
            True if successful, False otherwise
        """
        self.log("🚀 Starting full workflow: Training → Submission")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run training
        checkpoint_path = self.run_training()
        if not checkpoint_path:
            self.log("❌ Training failed. Stopping workflow.", "ERROR")
            return False
        
        # Generate submission
        submission_dir = self.generate_submission(checkpoint_path)
        if not submission_dir:
            self.log("❌ Submission generation failed.", "ERROR")
            return False
        
        # Validate submission
        if not self.validate_submission(submission_dir):
            self.log("❌ Submission validation failed.", "ERROR")
            return False
        
        # Print summary
        self.print_summary(checkpoint_path, submission_dir)
        
        self.log("🎉 Full workflow completed successfully!")
        return True
    
    def run_submission_only(self, checkpoint_path: str) -> bool:
        """
        Run submission generation only.
        
        Args:
            checkpoint_path: Path to existing checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        self.log("🚀 Starting submission-only workflow")
        
        # Check checkpoint exists
        if not Path(checkpoint_path).exists():
            self.log(f"❌ Checkpoint not found: {checkpoint_path}", "ERROR")
            return False
        
        # Generate submission
        submission_dir = self.generate_submission(checkpoint_path)
        if not submission_dir:
            self.log("❌ Submission generation failed.", "ERROR")
            return False
        
        # Validate submission
        if not self.validate_submission(submission_dir):
            self.log("❌ Submission validation failed.", "ERROR")
            return False
        
        # Print summary
        self.print_summary(checkpoint_path, submission_dir)
        
        self.log("🎉 Submission workflow completed successfully!")
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full workflow automation: Training → Submission"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_and_submit", "submit_only"],
        default="train_and_submit",
        help="Workflow mode (default: train_and_submit)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpu_48gb_large.yaml",
        help="Config file path (default: configs/gpu_48gb_large.yaml)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path (required for submit_only mode)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("🤖 MIND TRAINING & SUBMISSION WORKFLOW")
    print("="*60)
    
    # Initialize workflow manager
    workflow = WorkflowManager(config_path=args.config)
    
    # Run workflow based on mode
    if args.mode == "train_and_submit":
        success = workflow.run_full_workflow()
    elif args.mode == "submit_only":
        if not args.checkpoint:
            print("❌ --checkpoint is required for submit_only mode")
            return 1
        success = workflow.run_submission_only(args.checkpoint)
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return 1
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())