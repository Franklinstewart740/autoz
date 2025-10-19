"""
OpenAI Backend
Implementation of OpenAI API backend for LLM routing.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
import openai

from .base_backend import BaseLLMBackend, LLMRequest, LLMResponse, BackendStatus


class OpenAIBackend(BaseLLMBackend):
    """
    OpenAI API backend implementation.
    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        
        self.client: Optional[OpenAI] = None
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.default_model = config.get("default_model", "gpt-3.5-turbo")
        
        # Model configurations
        self.model_configs = {
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "context_window": 16385,
                "cost_per_1k_tokens": {"input": 0.0015, "output": 0.002},
                "avg_latency": 2.0
            },
            "gpt-4": {
                "max_tokens": 8192,
                "context_window": 8192,
                "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
                "avg_latency": 5.0
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "context_window": 128000,
                "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
                "avg_latency": 3.0
            }
        }

    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            self.logger.info("Initializing OpenAI backend")
            
            # Initialize OpenAI client
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            self.client = OpenAI(**client_kwargs)
            
            # Test the connection
            if await self.check_availability():
                self.status = BackendStatus.AVAILABLE
                self.logger.info("OpenAI backend initialized successfully")
                return True
            else:
                self.status = BackendStatus.ERROR
                self.logger.error("OpenAI backend initialization failed - API not accessible")
                return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize OpenAI backend: {e}", exc_info=True)
            return False

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.client:
            return LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=self.default_model,
                error="Backend not initialized"
            )
        
        if self._check_rate_limit():
            return LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=self.default_model,
                error="Rate limited"
            )
        
        start_time = time.time()
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        
        try:
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Prepare API parameters
            api_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.model_configs.get(model_name, {}).get("max_tokens", 1000),
                "temperature": request.temperature or 0.7,
            }
            
            if request.top_p is not None:
                api_params["top_p"] = request.top_p
            if request.stop_sequences:
                api_params["stop"] = request.stop_sequences
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            response_time = time.time() - start_time
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                success=True,
                backend_name=self.backend_name,
                model_name=model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                confidence=0.8,  # Default confidence for OpenAI
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None
                }
            )
            
            # Update metrics
            self._update_metrics(llm_response)
            
            return llm_response
            
        except openai.RateLimitError as e:
            self._handle_rate_limit()
            response = LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=model_name,
                error=f"Rate limit exceeded: {str(e)}",
                response_time=time.time() - start_time
            )
            self._update_metrics(response)
            return response
            
        except Exception as e:
            self._handle_error(str(e))
            response = LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=model_name,
                error=str(e),
                response_time=time.time() - start_time
            )
            self._update_metrics(response)
            return response

    async def check_availability(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.client:
            return False
        
        try:
            # Make a minimal API call to test availability
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            self.status = BackendStatus.AVAILABLE
            return True
            
        except openai.RateLimitError:
            self.status = BackendStatus.RATE_LIMITED
            return False
            
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            return False

    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        return list(self.model_configs.keys())

    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for OpenAI API request."""
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        model_config = self.model_configs.get(model_name, {})
        
        if "cost_per_1k_tokens" not in model_config:
            return 0.0
        
        # Rough token estimation (4 characters per token)
        prompt_tokens = len(request.prompt) // 4
        if request.system_prompt:
            prompt_tokens += len(request.system_prompt) // 4
        
        max_tokens = request.max_tokens or model_config.get("max_tokens", 1000)
        
        # Calculate cost
        input_cost = (prompt_tokens / 1000) * model_config["cost_per_1k_tokens"]["input"]
        output_cost = (max_tokens / 1000) * model_config["cost_per_1k_tokens"]["output"]
        
        return input_cost + output_cost

    def estimate_latency(self, request: LLMRequest) -> float:
        """Estimate response latency for OpenAI API."""
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        model_config = self.model_configs.get(model_name, {})
        
        base_latency = model_config.get("avg_latency", 3.0)
        
        # Adjust based on max_tokens
        max_tokens = request.max_tokens or 1000
        token_factor = max_tokens / 1000  # Linear scaling
        
        return base_latency * token_factor

    def is_suitable_for_request(self, request: LLMRequest) -> bool:
        """Check if OpenAI backend is suitable for the request."""
        if not super().is_suitable_for_request(request):
            return False
        
        # Check model availability
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        if model_name not in self.model_configs:
            return False
        
        # Check token limits
        model_config = self.model_configs[model_name]
        if request.max_tokens and request.max_tokens > model_config["max_tokens"]:
            return False
        
        return True

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific OpenAI model."""
        if model_name not in self.model_configs:
            return super().get_model_capabilities(model_name)
        
        config = self.model_configs[model_name]
        return {
            "max_tokens": config["max_tokens"],
            "context_window": config["context_window"],
            "supports_system_prompt": True,
            "supports_streaming": True,
            "supports_function_calling": model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "cost_per_1k_tokens": config["cost_per_1k_tokens"],
            "avg_latency": config["avg_latency"]
        }

    async def cleanup(self) -> None:
        """Clean up OpenAI backend resources."""
        await super().cleanup()
        self.client = None

