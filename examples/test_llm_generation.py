"""
Example script demonstrating LLM description generation.
This can be used to test the system without processing the full dataset.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_providers import LLMProviderFactory, LLMConfig
from src.utils.prompt_templates import get_prompt_template


def test_local_generation():
    """Test local generation (no API required)."""
    print("=" * 60)
    print("Testing Local Generation")
    print("=" * 60)
    
    # Sample news article
    title = "Lakers Beat Warriors in Overtime Thriller"
    abstract = "LeBron James scored 35 points as the Los Angeles Lakers defeated the Golden State Warriors 128-125 in overtime."
    category = "sports"
    
    # Simple concatenation
    description = (title + ". " + abstract)[:200]
    
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print(f"Category: {category}")
    print(f"\nGenerated Description:\n{description}")
    print("=" * 60)


def test_prompt_templates():
    """Test different prompt templates."""
    print("\n" + "=" * 60)
    print("Testing Prompt Templates")
    print("=" * 60)
    
    title = "New AI Model Achieves Human-Level Understanding"
    abstract = "Researchers developed a breakthrough AI model that demonstrates human-level natural language understanding capabilities."
    category = "technology"
    
    # Test different templates
    templates = ["default", "simple", "minimal"]
    
    for template_type in templates:
        print(f"\n--- {template_type.upper()} Template ---")
        template = get_prompt_template(template_type)
        prompt = template.format_prompt(
            title=title,
            abstract=abstract,
            category=category,
            use_few_shot=(template_type == "default")
        )
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("=" * 60)


def test_provider_factory():
    """Test provider factory and list available providers."""
    print("\n" + "=" * 60)
    print("Testing Provider Factory")
    print("=" * 60)
    
    providers = LLMProviderFactory.list_providers()
    print(f"Available providers: {', '.join(providers)}")
    
    # Test cost estimation for different providers
    print("\n--- Cost Estimation Examples ---")
    
    configs = [
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        LLMConfig(provider="anthropic", model="claude-3-haiku-20240307"),
        LLMConfig(provider="ollama", model="llama2"),
    ]
    
    num_articles = 1000
    
    for config in configs:
        try:
            provider = LLMProviderFactory.create(config)
            cost = provider.estimate_cost(num_articles)
            print(f"\n{config.provider.upper()} ({config.model}):")
            print(f"  Total cost for {num_articles} articles: ${cost['total_cost']:.4f}")
            if 'note' in cost:
                print(f"  Note: {cost['note']}")
        except ImportError as e:
            print(f"\n{config.provider.upper()}: {str(e)}")
    
    print("=" * 60)


def test_openai_generation():
    """Test OpenAI generation (requires API key)."""
    print("\n" + "=" * 60)
    print("Testing OpenAI Generation (Optional)")
    print("=" * 60)
    
    try:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("Skipped: OPENAI_API_KEY not set")
            print("To test: export OPENAI_API_KEY='your-key'")
            return
        
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=200
        )
        
        provider = LLMProviderFactory.create(config)
        template = get_prompt_template("simple")
        
        title = "Climate Summit Reaches Historic Agreement"
        abstract = "World leaders agreed on ambitious carbon emission reduction targets at the international climate summit."
        
        prompt = template.format_prompt(title, abstract, category="news")
        
        print("Generating description...")
        description = provider.generate(prompt)
        
        print(f"\nTitle: {title}")
        print(f"Abstract: {abstract}")
        print(f"\nGenerated Description:\n{description}")
        
    except ImportError as e:
        print(f"Skipped: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n🚀 LLM Description Generation Test Suite\n")
    
    # Run tests
    test_local_generation()
    test_prompt_templates()
    test_provider_factory()
    test_openai_generation()
    
    print("\n✅ Test suite completed!\n")
    print("Next steps:")
    print("1. Install provider packages: pip install openai anthropic")
    print("2. Set API keys: export OPENAI_API_KEY='your-key'")
    print("3. Run full generation: python src/scripts/gen_descriptions.py --help")
