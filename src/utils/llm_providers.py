import os
import time
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str  # "openai", "anthropic", "ollama", "lmstudio"
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 200
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: Optional[int] = None  # Requests per minute


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_count = 0
        self.last_request_time = 0.0
        
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def estimate_cost(self, num_requests: int, avg_prompt_tokens: int = 100, 
                     avg_completion_tokens: int = 200) -> Dict[str, float]:
        """Estimate cost for generation."""
        pass
    
    def _apply_rate_limit(self):
        """Apply rate limiting if configured."""
        if self.config.rate_limit_rpm:
            min_interval = 60.0 / self.config.rate_limit_rpm
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self._apply_rate_limit()
                result = func(*args, **kwargs)
                self.request_count += 1
                return result
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed after {self.config.max_retries} attempts: {str(e)}")
                    raise
                
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                time.sleep(delay)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        def _call_api():
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content.strip()
        
        return self._retry_with_backoff(_call_api)
    
    def estimate_cost(self, num_requests: int, avg_prompt_tokens: int = 100,
                     avg_completion_tokens: int = 200) -> Dict[str, float]:
        """Estimate cost based on OpenAI pricing."""
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }
        
        model_key = self.config.model
        if model_key not in pricing:
            # Find closest match
            for key in pricing:
                if key in model_key:
                    model_key = key
                    break
            else:
                model_key = "gpt-4o-mini"  # Default to cheapest
        
        rates = pricing[model_key]
        input_cost = (avg_prompt_tokens * num_requests / 1_000_000) * rates["input"]
        output_cost = (avg_completion_tokens * num_requests / 1_000_000) * rates["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": "USD"
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def generate(self, prompt: str) -> str:
        """Generate text using Anthropic API."""
        def _call_api():
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        return self._retry_with_backoff(_call_api)
    
    def estimate_cost(self, num_requests: int, avg_prompt_tokens: int = 100,
                     avg_completion_tokens: int = 200) -> Dict[str, float]:
        """Estimate cost based on Anthropic pricing."""
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        }
        
        model_key = self.config.model
        if model_key not in pricing:
            for key in pricing:
                if key in model_key:
                    model_key = key
                    break
            else:
                model_key = "claude-3-haiku"  # Default to cheapest
        
        rates = pricing[model_key]
        input_cost = (avg_prompt_tokens * num_requests / 1_000_000) * rates["input"]
        output_cost = (avg_completion_tokens * num_requests / 1_000_000) * rates["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": "USD"
        }


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests
            self.requests = requests
            self.api_base = config.api_base or "http://localhost:11434"
        except ImportError:
            raise ImportError("requests package not installed. Install with: pip install requests")
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama API."""
        def _call_api():
            response = self.requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        
        return self._retry_with_backoff(_call_api)
    
    def estimate_cost(self, num_requests: int, avg_prompt_tokens: int = 100,
                     avg_completion_tokens: int = 200) -> Dict[str, float]:
        """Local models are free."""
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "currency": "USD",
            "note": "Local model - no API costs"
        }


class LMStudioProvider(LLMProvider):
    """LM Studio local model provider (OpenAI-compatible API)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="lm-studio",  # LM Studio doesn't require real key
                base_url=config.api_base or "http://localhost:1234/v1",
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """Generate text using LM Studio API."""
        def _call_api():
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content.strip()
        
        return self._retry_with_backoff(_call_api)
    
    def estimate_cost(self, num_requests: int, avg_prompt_tokens: int = 100,
                     avg_completion_tokens: int = 200) -> Dict[str, float]:
        """Local models are free."""
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "currency": "USD",
            "note": "Local model - no API costs"
        }


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Create an LLM provider based on config."""
        provider_class = cls._providers.get(config.provider.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {config.provider}. "
                f"Available: {list(cls._providers.keys())}"
            )
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())
