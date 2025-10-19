"""
Groq Backend
Implementation of Groq API backend for ultra-fast inference.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import requests

from .base_backend import BaseLLMBackend, LLMRequest, LLMResponse, BackendStatus


class GroqBackend(BaseLLMBackend):
    """
    Groq API backend implementation.
    Provides ultra-fast inference with specialized hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("groq", config)
        
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.groq.com/openai/v1")
        self.default_model = config.get("default_model", "mixtral-8x7b-32768")
        
        # Model configurations
        self.model_configs = {
            "mixtral-8x7b-32768": {
                "max_tokens": 32768,
                "context_window": 32768,
                "avg_latency": 0.5,  # Groq is very fast
                "cost_per_1k_tokens": {"input": 0.0002, "output": 0.0002}
            },
            "llama2-70b-4096": {
                "max_tokens": 4096,
                "context_window": 4096,
                "avg_latency": 0.3,
                "cost_per_1k_tokens": {"input": 0.0007, "output": 0.0008}
            },
            "gemma-7b-it": {
                "max_tokens": 8192,
                "context_window": 8192,
                "avg_latency": 0.4,
                "cost_per_1k_tokens": {"input": 0.0001, "output": 0.0001}
            }
        }

    async def initialize(self) -> bool:
        """Initialize the Groq backend."""
        try:
            self.logger.info("Initializing Groq backend")
            
            if not self.api_key:
                self.status = BackendStatus.ERROR
                self.last_error = "API key not provided"
                self.logger.error("Groq API key not provided")
                return False
            
            # Test the connection
            if await self.check_availability():
                self.status = BackendStatus.AVAILABLE
                self.logger.info("Groq backend initialized successfully")
                return True
            else:
                self.status = BackendStatus.ERROR
                self.logger.error("Groq backend initialization failed - API not accessible")
                return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize Groq backend: {e}", exc_info=True)
            return False

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Groq API."""
        if not self.api_key:
            return LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=self.default_model,
                error="API key not configured"
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
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.model_configs.get(model_name, {}).get("max_tokens", 1000),
                "temperature": request.temperature or 0.7,
            }
            
            if request.top_p is not None:
                payload["top_p"] = request.top_p
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            url = f"{self.base_url}/chat/completions"
            
            # Use asyncio to make the request non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, headers=headers, timeout=30)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response data
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                llm_response = LLMResponse(
                    content=content,
                    success=True,
                    backend_name=self.backend_name,
                    model_name=model_name,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    confidence=0.85,  # High confidence for Groq
                    metadata={
                        "finish_reason": result["choices"][0].get("finish_reason"),
                        "usage": usage,
                        "model": result.get("model"),
                        "x_groq": response.headers.get("x-groq-id")
                    }
                )
                
                # Update rate limit info
                self.rate_limit_remaining = response.headers.get("x-ratelimit-remaining-requests")
                
                self._update_metrics(llm_response)
                return llm_response
                
            elif response.status_code == 429:
                self._handle_rate_limit()
                llm_response = LLMResponse(
                    content="",
                    success=False,
                    backend_name=self.backend_name,
                    model_name=model_name,
                    error="Rate limit exceeded",
                    response_time=response_time
                )
                self._update_metrics(llm_response)
                return llm_response
                
            else:
                error_msg = f"Groq API error {response.status_code}: {response.text}"
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
        """Check if Groq API is available."""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make a minimal API call to test availability
            payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code in [200, 429]:  # 429 means rate limited but available
                self.status = BackendStatus.AVAILABLE
                return True
            else:
                self.status = BackendStatus.ERROR
                self.last_error = f"API returned status {response.status_code}"
                return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            return False

    def get_supported_models(self) -> List[str]:
        """Get list of supported Groq models."""
        return list(self.model_configs.keys())

    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Groq API request."""
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
        """Estimate response latency for Groq API."""
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        model_config = self.model_configs.get(model_name, {})
        
        base_latency = model_config.get("avg_latency", 0.5)
        
        # Groq is very fast, minimal scaling with tokens
        max_tokens = request.max_tokens or 1000
        token_factor = 1 + (max_tokens / 10000)  # Very small scaling factor
        
        return base_latency * token_factor

    def is_suitable_for_request(self, request: LLMRequest) -> bool:
        """Check if Groq backend is suitable for the request."""
        if not super().is_suitable_for_request(request):
            return False
        
        if not self.api_key:
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
        """Get capabilities for a specific Groq model."""
        if model_name not in self.model_configs:
            return super().get_model_capabilities(model_name)
        
        config = self.model_configs[model_name]
        return {
            "max_tokens": config["max_tokens"],
            "context_window": config["context_window"],
            "supports_system_prompt": True,
            "supports_streaming": True,
            "supports_function_calling": False,
            "cost_per_1k_tokens": config["cost_per_1k_tokens"],
            "avg_latency": config["avg_latency"],
            "deployment": "cloud",
            "hardware": "specialized_inference_chips"
        }

    async def cleanup(self) -> None:
        """Clean up Groq backend resources."""
        await super().cleanup()

