"""
Ollama Backend
Implementation of Ollama backend for local LLM hosting.
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional
import requests

from .base_backend import BaseLLMBackend, LLMRequest, LLMResponse, BackendStatus


class OllamaBackend(BaseLLMBackend):
    """
    Ollama backend implementation.
    Supports local Ollama server for running various open-source models.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ollama", config)
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.default_model = config.get("default_model", "llama2")
        self.timeout = config.get("timeout", 60)
        
        # Model configurations
        self.model_configs = {
            "llama2": {
                "max_tokens": 4096,
                "context_window": 4096,
                "avg_latency": 5.0,
                "size": "7B"
            },
            "llama2:13b": {
                "max_tokens": 4096,
                "context_window": 4096,
                "avg_latency": 8.0,
                "size": "13B"
            },
            "llama2:70b": {
                "max_tokens": 4096,
                "context_window": 4096,
                "avg_latency": 15.0,
                "size": "70B"
            },
            "mistral": {
                "max_tokens": 8192,
                "context_window": 8192,
                "avg_latency": 4.0,
                "size": "7B"
            },
            "codellama": {
                "max_tokens": 4096,
                "context_window": 16384,
                "avg_latency": 6.0,
                "size": "7B"
            },
            "neural-chat": {
                "max_tokens": 4096,
                "context_window": 4096,
                "avg_latency": 4.5,
                "size": "7B"
            }
        }
        
        # Available models (will be populated during initialization)
        self.available_models = []

    async def initialize(self) -> bool:
        """Initialize the Ollama backend."""
        try:
            self.logger.info("Initializing Ollama backend")
            
            # Check if Ollama server is running
            if await self.check_availability():
                # Get list of available models
                await self._fetch_available_models()
                
                self.status = BackendStatus.AVAILABLE
                self.logger.info(f"Ollama backend initialized successfully with {len(self.available_models)} models")
                return True
            else:
                self.status = BackendStatus.ERROR
                self.logger.error("Ollama server not accessible")
                return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize Ollama backend: {e}", exc_info=True)
            return False

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama."""
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
            # Prepare the prompt
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {}
            }
            
            # Add generation parameters
            if request.max_tokens:
                payload["options"]["num_predict"] = request.max_tokens
            if request.temperature is not None:
                payload["options"]["temperature"] = request.temperature
            if request.top_p is not None:
                payload["options"]["top_p"] = request.top_p
            if request.stop_sequences:
                payload["options"]["stop"] = request.stop_sequences
            
            # Make API request
            url = f"{self.base_url}/api/generate"
            
            # Use asyncio to make the request non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=self.timeout)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                # Extract additional metadata
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 0)
                
                llm_response = LLMResponse(
                    content=content,
                    success=True,
                    backend_name=self.backend_name,
                    model_name=model_name,
                    tokens_used=eval_count,
                    response_time=response_time,
                    confidence=0.75,  # Default confidence for Ollama
                    metadata={
                        "eval_count": eval_count,
                        "eval_duration": eval_duration,
                        "total_duration": result.get("total_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                        "prompt_eval_count": result.get("prompt_eval_count", 0)
                    }
                )
                
                self._update_metrics(llm_response)
                return llm_response
                
            else:
                error_msg = f"Ollama API error {response.status_code}: {response.text}"
                self._handle_error(error_msg)
                llm_response = LLMResponse(
                    content="",
                    success=False,
                    backend_name=self.backend_name,
                    model_name=model_name,
                    error=error_msg,
                    response_time=response_time
                )
                self._update_metrics(llm_response)
                return llm_response
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            llm_response = LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=model_name,
                error=error_msg,
                response_time=time.time() - start_time
            )
            self._update_metrics(llm_response)
            return llm_response
            
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
        """Check if Ollama server is available."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                self.status = BackendStatus.AVAILABLE
                return True
            else:
                self.status = BackendStatus.ERROR
                return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            return False

    async def _fetch_available_models(self) -> None:
        """Fetch list of available models from Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                models = result.get("models", [])
                self.available_models = [model["name"] for model in models]
                self.logger.info(f"Found {len(self.available_models)} available models: {self.available_models}")
            else:
                self.logger.warning("Failed to fetch available models from Ollama")
                
        except Exception as e:
            self.logger.warning(f"Error fetching available models: {e}")

    def get_supported_models(self) -> List[str]:
        """Get list of supported Ollama models."""
        # Return available models if we have them, otherwise return configured models
        if self.available_models:
            return self.available_models
        else:
            return list(self.model_configs.keys())

    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Ollama request."""
        # Ollama is free for local usage
        return 0.0

    def estimate_latency(self, request: LLMRequest) -> float:
        """Estimate response latency for Ollama."""
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        model_config = self.model_configs.get(model_name, {})
        
        base_latency = model_config.get("avg_latency", 5.0)
        
        # Adjust based on max_tokens
        max_tokens = request.max_tokens or 1000
        token_factor = max_tokens / 1000
        
        return base_latency * token_factor

    def is_suitable_for_request(self, request: LLMRequest) -> bool:
        """Check if Ollama backend is suitable for the request."""
        if not super().is_suitable_for_request(request):
            return False
        
        # Check if model is available
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        supported_models = self.get_supported_models()
        
        if model_name not in supported_models:
            return False
        
        # Check token limits
        model_config = self.model_configs.get(model_name, {})
        if request.max_tokens and request.max_tokens > model_config.get("max_tokens", 4096):
            return False
        
        return True

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific Ollama model."""
        config = self.model_configs.get(model_name, {})
        
        return {
            "max_tokens": config.get("max_tokens", 4096),
            "context_window": config.get("context_window", 4096),
            "supports_system_prompt": True,  # Ollama supports system prompts via prompt formatting
            "supports_streaming": True,
            "supports_function_calling": False,
            "model_size": config.get("size", "Unknown"),
            "avg_latency": config.get("avg_latency", 5.0),
            "deployment": "local",
            "cost": 0.0
        }

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model to the Ollama server.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Pulling model: {model_name}")
            
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}
            
            # This can take a long time, so use a longer timeout
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                # Refresh available models list
                await self._fetch_available_models()
                self.logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to pull model {model_name}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}", exc_info=True)
            return False

    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the Ollama server.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Deleting model: {model_name}")
            
            url = f"{self.base_url}/api/delete"
            payload = {"name": model_name}
            
            response = requests.delete(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                # Refresh available models list
                await self._fetch_available_models()
                self.logger.info(f"Successfully deleted model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to delete model {model_name}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up Ollama backend resources."""
        await super().cleanup()
        self.available_models = []

