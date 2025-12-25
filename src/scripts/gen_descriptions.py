# scripts/gen_descriptions.py
"""
Generate news descriptions using local or LLM-based methods with structured logging.
Supports OpenAI, Anthropic, Ollama, and LM Studio providers.
"""
import json
import argparse
import os
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger, setup_logger
from src.utils.llm_providers import LLMProviderFactory, LLMConfig
from src.utils.prompt_templates import get_prompt_template


def local_generate(title, abstract, max_chars=200):
    """
    Generate description by concatenating title and abstract.
    
    Args:
        title: News title
        abstract: News abstract
        max_chars: Maximum characters to include
        
    Returns:
        Generated description
    """
    text = (title or "") + ". " + (abstract or "")
    return text[:max_chars]


def load_config_from_yaml(config_path: Optional[str]) -> dict:
    """Load LLM configuration from YAML file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('llm', {})
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


def estimate_and_confirm_cost(provider, num_articles: int, logger) -> bool:
    """
    Estimate cost and ask for user confirmation.
    
    Args:
        provider: LLM provider instance
        num_articles: Number of articles to process
        logger: Logger instance
        
    Returns:
        True if user confirms, False otherwise
    """
    # Estimate tokens (rough approximation)
    avg_prompt_tokens = 150  # Title + abstract + prompt template
    avg_completion_tokens = 50  # Short description
    
    cost_estimate = provider.estimate_cost(
        num_articles,
        avg_prompt_tokens,
        avg_completion_tokens
    )
    
    logger.info("=" * 60)
    logger.info("COST ESTIMATION")
    logger.info("=" * 60)
    logger.info(f"Number of articles: {num_articles}")
    logger.info(f"Estimated input cost: ${cost_estimate['input_cost']:.4f}")
    logger.info(f"Estimated output cost: ${cost_estimate['output_cost']:.4f}")
    logger.info(f"Estimated total cost: ${cost_estimate['total_cost']:.4f} {cost_estimate['currency']}")
    
    if 'note' in cost_estimate:
        logger.info(f"Note: {cost_estimate['note']}")
    
    logger.info("=" * 60)
    
    # For local models, auto-confirm
    if cost_estimate['total_cost'] == 0:
        logger.info("Using local model - no API costs. Proceeding automatically.")
        return True
    
    # Ask for confirmation
    response = input(f"\nEstimated cost: ${cost_estimate['total_cost']:.4f}. Continue? (yes/no): ")
    return response.lower() in ['yes', 'y']


def main(args):
    """
    Main function to generate news descriptions.
    
    Args:
        args: Command-line arguments
    """
    # Setup logger
    logger_instance = setup_logger(
        name="gen_descriptions",
        log_dir=args.log_dir if hasattr(args, 'log_dir') and args.log_dir else None,
        log_level=args.log_level if hasattr(args, 'log_level') else "INFO",
        console_output=True
    )
    logger = logger_instance.get_logger("generate")
    
    logger.info("=" * 60)
    logger.info("Starting news description generation")
    logger.info("=" * 60)
    logger.info(f"Provider: {args.provider}")
    logger.info(f"News metadata: {args.news_meta}")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Created output directory: {args.out_dir}")
    
    # Load news metadata
    logger.info(f"Loading news metadata from {args.news_meta}")
    try:
        with open(args.news_meta, encoding="utf-8") as f:
            news_meta = json.load(f)
        logger.info(
            f"Loaded metadata for {len(news_meta)} news articles",
            extra={"num_news": len(news_meta)}
        )    
    except Exception as e:
        logger.error(f"Failed to load news metadata: {str(e)}")
        raise
    
    # Initialize LLM provider if not using local mode
    provider = None
    prompt_template = None
    
    if args.provider != "local":
        # Load config from YAML if provided
        yaml_config = load_config_from_yaml(args.config)
        
        # Create LLM config
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model or yaml_config.get('model', 'gpt-4o-mini'),
            api_key=args.api_key or yaml_config.get('api_key'),
            api_base=args.api_base or yaml_config.get('api_base'),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            rate_limit_rpm=args.rate_limit_rpm or yaml_config.get('rate_limit_rpm')
        )
        
        logger.info(f"Initializing {args.provider} provider with model: {llm_config.model}")
        
        try:
            provider = LLMProviderFactory.create(llm_config)
            logger.info("Provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize provider: {str(e)}")
            raise
        
        # Initialize prompt template
        prompt_template = get_prompt_template(args.prompt_template)
        logger.info(f"Using prompt template: {args.prompt_template}")
        
        # Estimate cost and get confirmation
        if not args.skip_cost_confirm:
            if not estimate_and_confirm_cost(provider, len(news_meta), logger):
                logger.info("Generation cancelled by user")
                return
    
    # Load existing descriptions to skip already processed news
    existing_descriptions = {}
    output_path = os.path.join(args.out_dir, "news_descriptions.json")
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_descriptions = json.load(f)
            logger.info(f"Found existing file with {len(existing_descriptions)} descriptions")
        except Exception as e:
            logger.warning(f"Could not load existing file: {str(e)}")
    
    # Filter out news that already have descriptions
    news_to_process = {nid: meta for nid, meta in news_meta.items() 
                       if nid not in existing_descriptions}
    
    logger.info(f"Total news in input: {len(news_meta)}")
    logger.info(f"Already have descriptions: {len(existing_descriptions)}")
    logger.info(f"Need to generate: {len(news_to_process)}")
    
    if len(news_to_process) == 0:
        logger.info("All news already have descriptions. Nothing to generate.")
        return
    
    # Generate descriptions
    out = {}
    failed = []
    logger.info(f"Generating descriptions using {args.provider} provider")
    
    for nid, meta in tqdm(news_to_process.items(), desc="Generating descriptions"):
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        category = meta.get("category", "")
        
        try:
            if args.provider == "local":
                desc = local_generate(title, abstract)
            else:
                # Generate prompt
                prompt = prompt_template.format_prompt(
                    title=title,
                    abstract=abstract,
                    category=category if args.use_category else None,
                    use_few_shot=args.use_few_shot
                )
                
                # Generate description
                desc = provider.generate(prompt)
            
            out[nid] = desc
            
        except Exception as e:
            logger.warning(f"Failed to generate description for {nid}: {str(e)}")
            failed.append(nid)
            # Fallback to local generation
            out[nid] = local_generate(title, abstract)
    
    # Save descriptions (merge with existing file)
    # existing_descriptions already loaded above
    
    # Merge: new descriptions added to existing ones
    existing_descriptions.update(out)
    total_descriptions = existing_descriptions
    
    logger.info(f"Saving {len(total_descriptions)} total descriptions to {output_path}")
    logger.info(f"  - New descriptions: {len(out)}")
    logger.info(f"  - Total after merge: {len(total_descriptions)}")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(total_descriptions, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(
            f"Successfully saved descriptions ({file_size:.2f} MB)",
            extra={
                "num_descriptions": len(out),
                "file_size_mb": file_size
            }
        )
    except Exception as e:
        logger.error(f"Failed to save descriptions: {str(e)}")
        raise
    
    # Log statistics
    avg_length = sum(len(desc) for desc in out.values()) / len(out) if out else 0
    logger.info("=" * 60)
    logger.info("Description generation completed!")
    logger.info("=" * 60)
    logger.info(f"Statistics:")
    logger.info(f"  - Total descriptions: {len(out)}")
    logger.info(f"  - Successful: {len(out) - len(failed)}")
    logger.info(f"  - Failed (used fallback): {len(failed)}")
    logger.info(f"  - Average length: {avg_length:.1f} characters")
    logger.info(f"  - Output file: {output_path}")
    
    if provider:
        logger.info(f"  - Total API requests: {provider.request_count}")
    
    logger.info("=" * 60)
    
    print(f"\nGenerated descriptions saved to: {output_path}")
    if failed:
        print(f"Warning: {len(failed)} descriptions failed and used fallback generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate news descriptions using various LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local generation (no API)
  python gen_descriptions.py --news_meta data.json --provider local
  
  # OpenAI GPT-4
  python gen_descriptions.py --news_meta data.json --provider openai --model gpt-4o-mini
  
  # Anthropic Claude
  python gen_descriptions.py --news_meta data.json --provider anthropic --model claude-3-haiku
  
  # Ollama (local)
  python gen_descriptions.py --news_meta data.json --provider ollama --model llama2
  
  # LM Studio (local)
  python gen_descriptions.py --news_meta data.json --provider lmstudio --model local-model
        """
    )
    
    # Required arguments
    parser.add_argument("--news_meta", required=True, 
                       help="Path to news metadata JSON file")
    
    # Provider configuration
    parser.add_argument("--provider", default="local",
                       choices=["local", "openai", "anthropic", "ollama", "lmstudio"],
                       help="LLM provider to use")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (e.g., gpt-4o-mini, claude-3-haiku, llama2)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the provider (or set via environment variable)")
    parser.add_argument("--api-base", type=str, default=None,
                       help="API base URL for local providers (Ollama/LM Studio)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file with LLM settings")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (0.0-1.0)")
    parser.add_argument("--max-tokens", type=int, default=250,
                       help="Maximum tokens to generate (default: 250, increased for category-aware descriptions)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds")
    
    # Retry and rate limiting
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries on failure")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                       help="Initial delay between retries (exponential backoff)")
    parser.add_argument("--rate-limit-rpm", type=int, default=None,
                       help="Rate limit in requests per minute")
    
    # Prompt configuration
    parser.add_argument("--prompt-template", default="default",
                       choices=["default", "simple", "minimal"],
                       help="Prompt template to use")
    parser.add_argument("--use-category", action="store_true",
                       help="Use category-specific prompts")
    parser.add_argument("--use-few-shot", action="store_true",
                       help="Include few-shot examples in prompts")
    
    # Output configuration
    parser.add_argument("--out_dir", default="Data/generated",
                       help="Output directory for descriptions")
    parser.add_argument("--skip-cost-confirm", action="store_true",
                       help="Skip cost confirmation prompt")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files")
    
    args = parser.parse_args()
    main(args)
