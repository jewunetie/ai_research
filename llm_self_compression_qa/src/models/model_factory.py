"""Model factory for easy instantiation of different LLM models."""

from typing import Dict, Any, Optional
import logging
from .base import BaseLLM
from .openai_model import OpenAIModel


class ModelFactory:
    """
    Factory for creating model instances.

    Supports OpenAI models with plans for Anthropic and HuggingFace models.
    """

    SUPPORTED_MODELS = {
        # OpenAI models
        'gpt-5.1-chat-latest': 'openai',
        'gpt-5.1-thinking': 'openai',
        'gpt-5.1': 'openai',
        'gpt-5': 'openai',
        'gpt-4o': 'openai',
        'gpt-4-turbo': 'openai',
        'gpt-4': 'openai',
        'gpt-3.5-turbo': 'openai',

        # Anthropic models (not yet implemented)
        'claude-3-opus': 'anthropic',
        'claude-3-sonnet': 'anthropic',
        'claude-3-haiku': 'anthropic',

        # HuggingFace models (not yet implemented)
        'llama-2-7b': 'huggingface',
        'llama-2-13b': 'huggingface',
        'mistral-7b': 'huggingface',
    }

    @staticmethod
    def create(
        model_name: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create a model instance.

        Args:
            model_name: Name of the model
            provider: Provider name (openai, anthropic, huggingface).
                     If None, auto-detected from model_name.
            **kwargs: Additional arguments for model initialization

        Returns:
            Model instance

        Raises:
            ValueError: If model or provider not supported
        """
        logger = logging.getLogger(__name__)

        # Auto-detect provider if not specified
        if provider is None:
            if model_name in ModelFactory.SUPPORTED_MODELS:
                provider = ModelFactory.SUPPORTED_MODELS[model_name]
            else:
                # Try to infer from name
                if model_name.startswith('gpt'):
                    provider = 'openai'
                elif model_name.startswith('claude'):
                    provider = 'anthropic'
                elif 'llama' in model_name.lower() or 'mistral' in model_name.lower():
                    provider = 'huggingface'
                else:
                    raise ValueError(
                        f"Unknown model: {model_name}. "
                        f"Supported models: {list(ModelFactory.SUPPORTED_MODELS.keys())}"
                    )

        logger.info(f"Creating {provider} model: {model_name}")

        # Create model based on provider
        if provider == 'openai':
            return ModelFactory._create_openai(model_name, **kwargs)

        elif provider == 'anthropic':
            raise NotImplementedError(
                "Anthropic models not yet implemented. "
                "Use OpenAI models or implement AnthropicModel class."
            )

        elif provider == 'huggingface':
            raise NotImplementedError(
                "HuggingFace models not yet implemented. "
                "Use OpenAI models or implement HuggingFaceModel class."
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _create_openai(model_name: str, **kwargs) -> OpenAIModel:
        """
        Create OpenAI model instance.

        Args:
            model_name: OpenAI model name
            **kwargs: Model parameters

        Returns:
            OpenAIModel instance
        """
        # Default parameters
        defaults = {
            'temperature': 0.0,
            'seed': 42,
            'reasoning_effort': 'none',
        }

        # Merge with provided kwargs
        params = {**defaults, **kwargs}

        # Create model
        try:
            model = OpenAIModel(model_name=model_name, **params)
            return model

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create OpenAI model: {e}")
            raise

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseLLM:
        """
        Create model from configuration dictionary.

        Args:
            config: Configuration dict with fields:
                - name: str (required)
                - provider: str (optional)
                - temperature: float (optional)
                - seed: int (optional)
                - reasoning_effort: str (optional)
                - api_key: str (optional)
                - other model-specific parameters

        Returns:
            Model instance

        Example config:
            {
                "name": "gpt-5.1-chat-latest",
                "temperature": 0.0,
                "seed": 42,
                "reasoning_effort": "none"
            }
        """
        if 'name' not in config:
            raise ValueError("Config must contain 'name' field")

        model_name = config['name']
        provider = config.get('provider')

        # Extract other parameters (exclude name and provider)
        params = {k: v for k, v in config.items() if k not in ['name', 'provider']}

        return ModelFactory.create(model_name, provider=provider, **params)

    @staticmethod
    def list_supported_models() -> Dict[str, str]:
        """
        Get list of supported models.

        Returns:
            Dictionary mapping model_name -> provider
        """
        return ModelFactory.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_supported(model_name: str) -> bool:
        """
        Check if a model is supported.

        Args:
            model_name: Name of the model

        Returns:
            True if supported, False otherwise
        """
        return model_name in ModelFactory.SUPPORTED_MODELS


def create_model(model_name: str, **kwargs) -> BaseLLM:
    """
    Convenience function to create a model.

    Args:
        model_name: Name of the model
        **kwargs: Additional model parameters

    Returns:
        Model instance
    """
    return ModelFactory.create(model_name, **kwargs)
