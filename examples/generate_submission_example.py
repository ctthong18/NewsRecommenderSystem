"""
Example: Generate submission file for MIND leaderboard

This example demonstrates how to use the submission generation script
to create a submission package for the MIND leaderboard.

Usage:
    python examples/generate_submission_example.py
"""
import subprocess
import sys
from pathlib import Path


def generate_submission_small_dataset():
    """
    Generate submission for MINDsmall dataset.
    
    This is useful for testing the submission generation pipeline
    before running on the full MINDlarge test set.
    """
    print("="*60)
    print("Example: Generate Submission for MINDsmall")
    print("="*60)
    
    # Check if checkpoint exists
    checkpoint_path = Path("output/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train.py")
        return
    
    # Check if test data exists
    test_news = Path("Data/raw/MINDsmall_dev/news.tsv")
    test_behaviors = Path("Data/raw/MINDsmall_dev/behaviors.tsv")
    
    if not test_news.exists() or not test_behaviors.exists():
        print("Error: Test data not found")
        print("Please download MINDsmall dataset first:")
        print("  python -m src.scripts.download_mind --size small")
        return
    
    # Generate submission
    cmd = [
        sys.executable, "-m", "src.scripts.generate_submission",
        "--checkpoint", str(checkpoint_path),
        "--test-news", str(test_news),
        "--test-behaviors", str(test_behaviors),
        "--output-dir", "output/submission_small",
        "--batch-size", "4"
    ]
    
    print("\nRunning submission generation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Submission generated successfully!")
        print("Output files:")
        print("  - output/submission_small/prediction.txt")
        print("  - output/submission_small/prediction_metadata.json")
        print("  - output/submission_small/submission.zip")
    else:
        print("\n✗ Submission generation failed")


def generate_submission_large_dataset():
    """
    Generate submission for MINDlarge dataset.
    
    This is for the actual leaderboard submission.
    """
    print("="*60)
    print("Example: Generate Submission for MINDlarge")
    print("="*60)
    
    # Check if checkpoint exists
    checkpoint_path = Path("output/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train.py")
        return
    
    # Check if test data exists
    test_news = Path("Data/raw/MINDlarge_test/news.tsv")
    test_behaviors = Path("Data/raw/MINDlarge_test/behaviors.tsv")
    
    if not test_news.exists() or not test_behaviors.exists():
        print("Error: Test data not found")
        print("Please download MINDlarge dataset first:")
        print("  python -m src.scripts.download_mind --size large")
        return
    
    # Check for LLM descriptions
    llm_desc_path = Path("Data/generated/llm_descriptions.json")
    
    # Generate submission
    cmd = [
        sys.executable, "-m", "src.scripts.generate_submission",
        "--checkpoint", str(checkpoint_path),
        "--test-news", str(test_news),
        "--test-behaviors", str(test_behaviors),
        "--config", "configs/large.yaml",
        "--output-dir", "output/submission_large",
        "--batch-size", "8"
    ]
    
    if llm_desc_path.exists():
        cmd.extend(["--llm-description-path", str(llm_desc_path)])
        print("✓ Using LLM descriptions")
    
    print("\nRunning submission generation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Submission generated successfully!")
        print("Output files:")
        print("  - output/submission_large/prediction.txt")
        print("  - output/submission_large/prediction_metadata.json")
        print("  - output/submission_large/submission.zip")
        print("\nYou can now submit submission.zip to the MIND leaderboard!")
    else:
        print("\n✗ Submission generation failed")


def generate_submission_with_custom_checkpoint():
    """
    Generate submission using a specific checkpoint.
    
    This is useful when you have multiple checkpoints and want to
    generate submissions for comparison.
    """
    print("="*60)
    print("Example: Generate Submission with Custom Checkpoint")
    print("="*60)
    
    # Specify custom checkpoint
    checkpoint_path = Path("output/checkpoints/epoch_2.pt")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = Path("output/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pt"):
                print(f"  - {ckpt}")
        return
    
    # Test data
    test_news = Path("Data/raw/MINDsmall_dev/news.tsv")
    test_behaviors = Path("Data/raw/MINDsmall_dev/behaviors.tsv")
    
    if not test_news.exists() or not test_behaviors.exists():
        print("Error: Test data not found")
        return
    
    # Generate submission
    cmd = [
        sys.executable, "-m", "src.scripts.generate_submission",
        "--checkpoint", str(checkpoint_path),
        "--test-news", str(test_news),
        "--test-behaviors", str(test_behaviors),
        "--output-dir", f"output/submission_{checkpoint_path.stem}",
        "--batch-size", "4"
    ]
    
    print("\nRunning submission generation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Submission generated successfully!")
    else:
        print("\n✗ Submission generation failed")


def main():
    """Main function."""
    print("\nSubmission Generation Examples")
    print("="*60)
    print("1. Generate submission for MINDsmall (testing)")
    print("2. Generate submission for MINDlarge (leaderboard)")
    print("3. Generate submission with custom checkpoint")
    print("="*60)
    
    choice = input("\nSelect an example (1-3): ").strip()
    
    if choice == "1":
        generate_submission_small_dataset()
    elif choice == "2":
        generate_submission_large_dataset()
    elif choice == "3":
        generate_submission_with_custom_checkpoint()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
