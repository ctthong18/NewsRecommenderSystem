"""
Example usage of the News Recommendation Inference API.

This script demonstrates:
1. Basic single-user inference
2. Batch inference for multiple users
3. Error handling
4. Using the inference API wrapper
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation.inference_api import (
    NewsRecommendationAPI,
    InputValidationError,
    ModelLoadError
)


def example_single_user_inference():
    """Example: Single-user inference."""
    print("="*60)
    print("Example 1: Single-User Inference")
    print("="*60)
    
    try:
        # Initialize API
        api = NewsRecommendationAPI(
            checkpoint_path="output/checkpoints/best_model.pt",
            config_path="configs/base_config.yaml",
            verbose=True
        )
        
        # Load news data
        api.load_news_data(
            news_path="Data/raw/MINDsmall_dev/news.tsv",
            llm_description_path="Data/generated/llm_descriptions.json"
        )
        
        # Example candidate news IDs (replace with actual IDs from your dataset)
        candidate_news_ids = ["N24510", "N39237", "N9721", "N13905", "N50214"]
        history_news_ids = ["N12345", "N67890"]
        
        # Get recommendations
        result = api.recommend(
            candidate_news_ids=candidate_news_ids,
            history_news_ids=history_news_ids,
            top_k=10
        )
        
        # Display results
        print("\nTop 5 Recommendations:")
        top_news, top_scores = result.get_top_k(5)
        for i, (news_id, score) in enumerate(zip(top_news, top_scores), 1):
            print(f"  {i}. {news_id}: {score:.4f}")
        
        # Get model info
        print("\nModel Info:")
        info = api.get_model_info()
        print(f"  Device: {info['device']}")
        print(f"  News loaded: {info['num_news_loaded']}")
        print(f"  History size: {info['history_size']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure checkpoint and data files exist.")
    except ModelLoadError as e:
        print(f"Error loading model: {e}")
    except InputValidationError as e:
        print(f"Input validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_batch_inference():
    """Example: Batch inference for multiple users."""
    print("\n" + "="*60)
    print("Example 2: Batch Inference")
    print("="*60)
    
    try:
        # Initialize API
        api = NewsRecommendationAPI(
            checkpoint_path="output/checkpoints/best_model.pt",
            config_path="configs/base_config.yaml",
            verbose=False
        )
        
        # Load news data
        api.load_news_data(
            news_path="Data/raw/MINDsmall_dev/news.tsv"
        )
        
        # Multiple user requests
        requests = [
            {
                "candidate_news_ids": ["N24510", "N39237", "N9721"],
                "history_news_ids": ["N12345", "N67890"],
                "top_k": 3
            },
            {
                "candidate_news_ids": ["N50214", "N13905", "N24510"],
                "history_news_ids": ["N11111", "N22222", "N33333"],
                "top_k": 3
            }
        ]
        
        # Process batch
        results = api.batch_recommend(requests, top_k=5)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\nUser {i} Recommendations:")
            top_news, top_scores = result.get_top_k()
            for j, (news_id, score) in enumerate(zip(top_news, top_scores), 1):
                print(f"  {j}. {news_id}: {score:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_error_handling():
    """Example: Error handling."""
    print("\n" + "="*60)
    print("Example 3: Error Handling")
    print("="*60)
    
    try:
        api = NewsRecommendationAPI(
            checkpoint_path="output/checkpoints/best_model.pt",
            config_path="configs/base_config.yaml",
            verbose=False
        )
        
        # Try to recommend without loading news data
        print("\nTest 1: Recommend without loading news data")
        try:
            result = api.recommend(
                candidate_news_ids=["N12345"],
                history_news_ids=[]
            )
        except InputValidationError as e:
            print(f"  ✓ Caught expected error: {e}")
        
        # Load news data
        api.load_news_data(
            news_path="Data/raw/MINDsmall_dev/news.tsv"
        )
        
        # Try with empty candidate list
        print("\nTest 2: Empty candidate list")
        try:
            result = api.recommend(
                candidate_news_ids=[],
                history_news_ids=[]
            )
        except InputValidationError as e:
            print(f"  ✓ Caught expected error: {e}")
        
        # Try with invalid news ID
        print("\nTest 3: Invalid news ID")
        try:
            result = api.recommend(
                candidate_news_ids=["INVALID_NEWS_ID"],
                history_news_ids=[]
            )
        except InputValidationError as e:
            print(f"  ✓ Caught expected error: {e}")
        
        print("\n✓ All error handling tests passed!")
        
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_using_inference_script():
    """Example: Using the inference script directly."""
    print("\n" + "="*60)
    print("Example 4: Using Inference Script")
    print("="*60)
    
    print("\nSingle-user inference:")
    print("  python src/scripts/inference.py \\")
    print("    --checkpoint output/checkpoints/best_model.pt \\")
    print("    --config configs/base_config.yaml \\")
    print("    --mode single \\")
    print("    --news-path Data/raw/MINDsmall_dev/news.tsv \\")
    print("    --candidate-news-ids N24510 N39237 N9721 \\")
    print("    --history-news-ids N12345 N67890 \\")
    print("    --output-dir output/inference")
    
    print("\nBatch inference:")
    print("  python src/scripts/inference.py \\")
    print("    --checkpoint output/checkpoints/best_model.pt \\")
    print("    --config configs/base_config.yaml \\")
    print("    --mode batch \\")
    print("    --news-path Data/raw/MINDsmall_dev/news.tsv \\")
    print("    --behaviors-path Data/raw/MINDsmall_dev/behaviors.tsv \\")
    print("    --batch-size 8 \\")
    print("    --save-predictions \\")
    print("    --output-dir output/inference")
    
    print("\nEnsemble inference:")
    print("  python src/scripts/inference.py \\")
    print("    --checkpoint output/checkpoints/best_model.pt \\")
    print("    --ensemble-checkpoints \\")
    print("      output/checkpoints/checkpoint_epoch_1.pt \\")
    print("      output/checkpoints/checkpoint_epoch_2.pt \\")
    print("    --mode batch \\")
    print("    --news-path Data/raw/MINDsmall_dev/news.tsv \\")
    print("    --behaviors-path Data/raw/MINDsmall_dev/behaviors.tsv \\")
    print("    --output-dir output/inference")


if __name__ == "__main__":
    print("News Recommendation Inference API Examples")
    print("="*60)
    
    # Run examples
    example_single_user_inference()
    example_batch_inference()
    example_error_handling()
    example_using_inference_script()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
