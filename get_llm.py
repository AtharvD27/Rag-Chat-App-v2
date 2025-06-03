import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time

from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import from enhanced utils
from utils import (
    setup_logging, ModelError, handle_errors, retry_with_backoff,
    time_operation, performance_monitor
)
from monitoring import get_monitoring_instance


class ModelProvider(str, Enum):
    """Supported model providers"""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelProfile:
    """Configuration profile for a model"""
    name: str
    provider: ModelProvider
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    streaming: bool = False
    cost_per_1k_tokens: float = 0.0
    supports_functions: bool = False
    context_window: int = 4096
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class ModelResponse:
    """Standardized model response"""
    text: str
    model_name: str
    provider: str
    tokens_used: int
    response_time: float
    cost: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModelProvider(ABC):
    """Abstract base class for model providers"""
    
    def __init__(self, profile: ModelProfile, config: Dict[str, Any]):
        self.profile = profile
        self.config = config
        self.logger = setup_logging(f"rag_chat.{self.__class__.__name__}")
        self.metrics = get_monitoring_instance(config)
        
    @abstractmethod
    def get_model(self) -> Union[BaseLLM, BaseChatModel]:
        """Get the configured model instance"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider-specific configuration"""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage"""
        return (tokens / 1000) * self.profile.cost_per_1k_tokens


class LocalModelProvider(BaseModelProvider):
    """Provider for local models using LlamaCpp"""
    
    def validate_config(self) -> bool:
        """Validate local model configuration"""
        model_path = self.profile.custom_params.get("model_path")
        if not model_path:
            raise ModelError(
                "Model path not specified for local model",
                error_code="MISSING_MODEL_PATH",
                details={"profile": self.profile.name}
            )
        
        if not os.path.exists(model_path):
            raise ModelError(
                f"Model file not found: {model_path}",
                error_code="MODEL_NOT_FOUND",
                details={"path": model_path}
            )
        
        return True
    
    @handle_errors(logger=None, raise_on_error=True)
    def get_model(self) -> BaseLLM:
        """Initialize and return local model"""
        self.validate_config()
        
        model_path = self.profile.custom_params.get("model_path")
        n_ctx = self.profile.custom_params.get("n_ctx", self.profile.context_window)
        n_threads = self.profile.custom_params.get("n_threads", 4)
        n_gpu_layers = self.profile.custom_params.get("n_gpu_layers", -1)
        
        self.logger.info(f"Loading local model: {model_path}")
        
        # Callback for streaming
        callback_manager = None
        if self.profile.streaming:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        try:
            # Try GPU first
            model = LlamaCpp(
                model_path=model_path,
                temperature=self.profile.temperature,
                max_tokens=self.profile.max_tokens,
                top_p=self.profile.top_p,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                callback_manager=callback_manager,
                verbose=False,
            )
            self.logger.info("Local model loaded successfully with GPU support")
            return model
            
        except Exception as e:
            if n_gpu_layers > 0:
                self.logger.warning(f"GPU loading failed: {e}. Falling back to CPU...")
                
                # Fallback to CPU
                model = LlamaCpp(
                    model_path=model_path,
                    temperature=self.profile.temperature,
                    max_tokens=self.profile.max_tokens,
                    top_p=self.profile.top_p,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=0,  # CPU only
                    callback_manager=callback_manager,
                    verbose=False,
                )
                self.logger.info("Local model loaded successfully with CPU")
                return model
            else:
                raise


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI models"""
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        api_key = os.environ.get("OPENAI_API_KEY") or self.profile.custom_params.get("api_key")
        if not api_key:
            raise ModelError(
                "OpenAI API key not found",
                error_code="MISSING_API_KEY",
                details={"env_var": "OPENAI_API_KEY"}
            )
        return True
    
    @retry_with_backoff(retries=3, exceptions=(Exception,))
    def get_model(self) -> BaseChatModel:
        """Initialize and return OpenAI model"""
        self.validate_config()
        
        api_key = os.environ.get("OPENAI_API_KEY") or self.profile.custom_params.get("api_key")
        
        self.logger.info(f"Initializing OpenAI model: {self.profile.model_id}")
        
        model = ChatOpenAI(
            model=self.profile.model_id,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            top_p=self.profile.top_p,
            openai_api_key=api_key,
            streaming=self.profile.streaming,
        )
        
        return model


class AnthropicProvider(BaseModelProvider):
    """Provider for Anthropic Claude models"""
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.profile.custom_params.get("api_key")
        if not api_key:
            raise ModelError(
                "Anthropic API key not found",
                error_code="MISSING_API_KEY",
                details={"env_var": "ANTHROPIC_API_KEY"}
            )
        return True
    
    def get_model(self) -> BaseChatModel:
        """Initialize and return Anthropic model"""
        self.validate_config()
        
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.profile.custom_params.get("api_key")
        
        self.logger.info(f"Initializing Anthropic model: {self.profile.model_id}")
        
        model = ChatAnthropic(
            model=self.profile.model_id,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            anthropic_api_key=api_key,
            streaming=self.profile.streaming,
        )
        
        return model


class GoogleProvider(BaseModelProvider):
    """Provider for Google models"""
    
    def validate_config(self) -> bool:
        """Validate Google configuration"""
        api_key = os.environ.get("GOOGLE_API_KEY") or self.profile.custom_params.get("api_key")
        if not api_key:
            raise ModelError(
                "Google API key not found",
                error_code="MISSING_API_KEY",
                details={"env_var": "GOOGLE_API_KEY"}
            )
        return True
    
    def get_model(self) -> BaseChatModel:
        """Initialize and return Google model"""
        self.validate_config()
        
        api_key = os.environ.get("GOOGLE_API_KEY") or self.profile.custom_params.get("api_key")
        
        self.logger.info(f"Initializing Google model: {self.profile.model_id}")
        
        model = ChatGoogleGenerativeAI(
            model=self.profile.model_id,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            google_api_key=api_key,
        )
        
        return model


class HuggingFaceProvider(BaseModelProvider):
    """Provider for HuggingFace models"""
    
    def validate_config(self) -> bool:
        """Validate HuggingFace configuration"""
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or self.profile.custom_params.get("api_key")
        if not api_key:
            raise ModelError(
                "HuggingFace API token not found",
                error_code="MISSING_API_KEY",
                details={"env_var": "HUGGINGFACEHUB_API_TOKEN"}
            )
        return True
    
    def get_model(self) -> BaseLLM:
        """Initialize and return HuggingFace model"""
        self.validate_config()
        
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or self.profile.custom_params.get("api_key")
        
        self.logger.info(f"Initializing HuggingFace model: {self.profile.model_id}")
        
        model = HuggingFaceHub(
            repo_id=self.profile.model_id,
            model_kwargs={
                "temperature": self.profile.temperature,
                "max_length": self.profile.max_tokens,
            },
            huggingfacehub_api_token=api_key,
        )
        
        return model


class ModelManager:
    """Manages multiple model providers and profiles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging("rag_chat.ModelManager")
        self.metrics = get_monitoring_instance(config)
        
        # Load model profiles
        self.profiles = self._load_profiles()
        
        # Provider mapping
        self.provider_classes = {
            ModelProvider.LOCAL: LocalModelProvider,
            ModelProvider.OPENAI: OpenAIProvider,
            ModelProvider.ANTHROPIC: AnthropicProvider,
            ModelProvider.GOOGLE: GoogleProvider,
            ModelProvider.HUGGINGFACE: HuggingFaceProvider,
        }
        
        # Model cache
        self._model_cache = {}
        
        # Fallback chain
        self.fallback_chain = config.get("models", {}).get("fallback_chain", [])
        
        self.logger.info(f"ModelManager initialized with {len(self.profiles)} profiles")
    
    def _load_profiles(self) -> Dict[str, ModelProfile]:
        """Load model profiles from configuration"""
        profiles = {}
        models_config = self.config.get("models", {})
        
        # Load predefined profiles
        for profile_name, profile_data in models_config.get("profiles", {}).items():
            try:
                provider = ModelProvider(profile_data["provider"])
                profile = ModelProfile(
                    name=profile_name,
                    provider=provider,
                    model_id=profile_data["model"],
                    max_tokens=profile_data.get("max_tokens", 512),
                    temperature=profile_data.get("temperature", 0.7),
                    top_p=profile_data.get("top_p", 0.95),
                    streaming=profile_data.get("streaming", False),
                    cost_per_1k_tokens=profile_data.get("cost_per_1k_tokens", 0.0),
                    supports_functions=profile_data.get("supports_functions", False),
                    context_window=profile_data.get("context_window", 4096),
                    custom_params=profile_data.get("custom_params", {})
                )
                profiles[profile_name] = profile
                self.logger.debug(f"Loaded profile: {profile_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load profile {profile_name}: {e}")
        
        # Ensure we have at least a default local profile
        if "default" not in profiles and self.config.get("llm", {}).get("local_model_path"):
            profiles["default"] = ModelProfile(
                name="default",
                provider=ModelProvider.LOCAL,
                model_id="local-model",
                max_tokens=self.config["llm"].get("max_tokens", 512),
                temperature=self.config["llm"].get("temperature", 0.7),
                custom_params={"model_path": self.config["llm"]["local_model_path"]}
            )
        
        return profiles
    
    @time_operation("model_loading")
    def get_model(
        self, 
        profile_name: str = None, 
        use_fallback: bool = True
    ) -> Union[BaseLLM, BaseChatModel]:
        """Get a model instance by profile name"""
        profile_name = profile_name or self.config.get("models", {}).get("default_profile", "default")
        
        # Check cache
        if profile_name in self._model_cache:
            self.logger.debug(f"Using cached model: {profile_name}")
            return self._model_cache[profile_name]
        
        # Try to load the requested model
        try:
            model = self._load_model(profile_name)
            self._model_cache[profile_name] = model
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {profile_name}: {e}")
            
            if use_fallback and self.fallback_chain:
                # Try fallback models
                for fallback_name in self.fallback_chain:
                    if fallback_name != profile_name:
                        try:
                            self.logger.info(f"Trying fallback model: {fallback_name}")
                            model = self._load_model(fallback_name)
                            self._model_cache[fallback_name] = model
                            return model
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback {fallback_name} also failed: {fallback_error}")
            
            raise ModelError(
                f"Failed to load model {profile_name} and all fallbacks",
                error_code="MODEL_LOAD_FAILED",
                details={"profile": profile_name, "fallback_chain": self.fallback_chain}
            )
    
    def _load_model(self, profile_name: str) -> Union[BaseLLM, BaseChatModel]:
        """Load a specific model by profile name"""
        if profile_name not in self.profiles:
            raise ModelError(
                f"Model profile not found: {profile_name}",
                error_code="PROFILE_NOT_FOUND",
                details={"available_profiles": list(self.profiles.keys())}
            )
        
        profile = self.profiles[profile_name]
        provider_class = self.provider_classes.get(profile.provider)
        
        if not provider_class:
            raise ModelError(
                f"Unsupported provider: {profile.provider}",
                error_code="UNSUPPORTED_PROVIDER",
                details={"provider": profile.provider}
            )
        
        # Create provider and get model
        provider = provider_class(profile, self.config)
        model = provider.get_model()
        
        self.logger.info(
            f"Successfully loaded model: {profile_name} "
            f"(provider={profile.provider}, model={profile.model_id})"
        )
        
        # Record metrics
        self.metrics.record_metric(
            "model_loaded",
            1.0,
            tags={"profile": profile_name, "provider": profile.provider.value}
        )
        
        return model
    
    def select_model_for_query(self, query: str, context_length: int = 0) -> str:
        """Select the best model profile for a given query"""
        # Simple heuristic-based selection
        # In production, this could use more sophisticated logic
        
        total_length = len(query) + context_length
        
        # For very long contexts, prefer models with larger context windows
        if total_length > 3000:
            for profile_name, profile in self.profiles.items():
                if profile.context_window >= total_length + 1000:  # Buffer
                    return profile_name
        
        # For complex analytical queries, prefer more capable models
        analytical_keywords = ["analyze", "compare", "evaluate", "explain why", "how does"]
        if any(keyword in query.lower() for keyword in analytical_keywords):
            # Prefer quality profiles
            for profile_name in ["quality", "balanced", "default"]:
                if profile_name in self.profiles:
                    return profile_name
        
        # For simple queries, use fast models
        simple_keywords = ["what is", "when", "where", "who", "list", "name"]
        if any(keyword in query.lower() for keyword in simple_keywords):
            for profile_name in ["fast", "default"]:
                if profile_name in self.profiles:
                    return profile_name
        
        # Default selection
        return self.config.get("models", {}).get("default_profile", "default")
    
    def get_model_info(self, profile_name: str = None) -> Dict[str, Any]:
        """Get information about a model profile"""
        if profile_name is None:
            # Return info about all profiles
            return {
                name: {
                    "provider": profile.provider.value,
                    "model": profile.model_id,
                    "max_tokens": profile.max_tokens,
                    "context_window": profile.context_window,
                    "cost_per_1k_tokens": profile.cost_per_1k_tokens,
                    "supports_functions": profile.supports_functions
                }
                for name, profile in self.profiles.items()
            }
        
        if profile_name not in self.profiles:
            return {"error": f"Profile {profile_name} not found"}
        
        profile = self.profiles[profile_name]
        return {
            "name": profile.name,
            "provider": profile.provider.value,
            "model": profile.model_id,
            "max_tokens": profile.max_tokens,
            "temperature": profile.temperature,
            "context_window": profile.context_window,
            "cost_per_1k_tokens": profile.cost_per_1k_tokens,
            "supports_functions": profile.supports_functions,
            "streaming": profile.streaming
        }


# Convenience functions for backward compatibility
def get_local_llm(config: Dict[str, Any], overrides: Dict[str, Any] = None) -> BaseLLM:
    """Get local LLM with overrides (backward compatibility)"""
    logger = setup_logging("rag_chat.get_local_llm")
    
    # Apply overrides to config
    if overrides:
        if "model_path" in overrides:
            config["llm"]["local_model_path"] = overrides["model_path"]
        if "temperature" in overrides:
            config["llm"]["temperature"] = overrides["temperature"]
    
    # Create a simple profile
    profile = ModelProfile(
        name="local",
        provider=ModelProvider.LOCAL,
        model_id="local-model",
        max_tokens=config["llm"].get("max_tokens", 512),
        temperature=config["llm"].get("temperature", 0.7),
        top_p=config["llm"].get("top_p", 0.95),
        custom_params={
            "model_path": config["llm"].get("local_model_path"),
            "n_ctx": config["llm"].get("n_ctx", 2048),
            "n_threads": config["llm"].get("n_threads", 4),
            "n_gpu_layers": config["llm"].get("n_gpu_layers", -1)
        }
    )
    
    provider = LocalModelProvider(profile, config)
    return provider.get_model()


# Example usage
if __name__ == "__main__":
    from utils import load_config
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Add some example profiles to config for testing
    config["models"] = {
        "profiles": {
            "fast": {
                "provider": "local",
                "model": "mistral-7b",
                "max_tokens": 256,
                "temperature": 0.3,
                "custom_params": {
                    "model_path": config["llm"]["local_model_path"]
                }
            },
            "quality": {
                "provider": "openai",
                "model": "gpt-4",
                "max_tokens": 1024,
                "temperature": 0.7,
                "cost_per_1k_tokens": 0.03,
                "supports_functions": True,
                "context_window": 8192
            }
        },
        "default_profile": "fast",
        "fallback_chain": ["fast", "quality"]
    }
    
    # Create model manager
    manager = ModelManager(config)
    
    # Get model info
    print("Available models:")
    for name, info in manager.get_model_info().items():
        print(f"  {name}: {info}")
    
    # Load a model
    try:
        model = manager.get_model("fast")
        print(f"\nLoaded model: {model}")
        
        # Test model selection
        test_queries = [
            "What is Python?",
            "Analyze the performance implications of using async/await in Python",
            "List the top 5 programming languages"
        ]
        
        print("\nModel selection for queries:")
        for query in test_queries:
            selected = manager.select_model_for_query(query)
            print(f"  '{query[:50]}...' -> {selected}")
            
    except Exception as e:
        print(f"Error: {e}")