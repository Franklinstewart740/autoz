"""
HuggingFace Backend
Implementation of HuggingFace Transformers backend for local/hosted models.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import requests

from .base_backend import BaseLLMBackend, LLMRequest, LLMResponse, BackendStatus


class HuggingFaceBackend(BaseLLMBackend):
    """
    HuggingFace backend implementation.
    Supports both Inference API and local transformers.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("huggingface", config)
        
        self.api_key = config.get("api_key")
        self.use_inference_api = config.get("use_inference_api", True)
        self.default_model = config.get("default_model", "microsoft/DialoGPT-medium")
        self.api_base_url = "https://api-inference.huggingface.co/models"
        
        # Model configurations
        self.model_configs = {
            "microsoft/DialoGPT-medium": {
                "max_tokens": 1000,
                "context_window": 1000,
                "avg_latency": 3.0,
                "type": "conversational"
            },
            "google/flan-t5-large": {
                "max_tokens": 512,
                "context_window": 512,
                "avg_latency": 2.5,
                "type": "text2text"
            },
            "microsoft/DialoGPT-large": {
                "max_tokens": 1000,
                "context_window": 1000,
                "avg_latency": 4.0,
                "type": "conversational"
            },
            "facebook/blenderbot-400M-distill": {
                "max_tokens": 128,
                "context_window": 128,
                "avg_latency": 2.0,
                "type": "conversational"
            }
        }
        
        # Local transformers components (if not using API)
        self.tokenizer = None
        self.model = None

    async def initialize(self) -> bool:
        """Initialize the HuggingFace backend."""
        try:
            self.logger.info("Initializing HuggingFace backend")
            
            if self.use_inference_api:
                # Test API availability
                if await self.check_availability():
                    self.status = BackendStatus.AVAILABLE
                    self.logger.info("HuggingFace Inference API backend initialized successfully")
                    return True
                else:
                    self.status = BackendStatus.ERROR
                    self.logger.error("HuggingFace Inference API not accessible")
                    return False
            else:
                # Initialize local transformers
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(self.default_model)
                    self.model = AutoModelForCausalLM.from_pretrained(self.default_model)
                    
                    self.status = BackendStatus.AVAILABLE
                    self.logger.info("HuggingFace local backend initialized successfully")
                    return True
                    
                except ImportError:
                    self.status = BackendStatus.ERROR
                    self.last_error = "transformers library not installed"
                    self.logger.error("transformers library not installed")
                    return False
                except Exception as e:
                    self.status = BackendStatus.ERROR
                    self.last_error = str(e)
                    self.logger.error(f"Failed to load local model: {e}")
                    return False
                
        except Exception as e:
            self.status = BackendStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize HuggingFace backend: {e}", exc_info=True)
            return False

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using HuggingFace."""
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
            if self.use_inference_api:
                return await self._generate_via_api(request, model_name, start_time)
            else:
                return await self._generate_locally(request, model_name, start_time)
                
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

    async def _generate_via_api(self, request: LLMRequest, model_name: str, start_time: float) -> LLMResponse:
        """Generate response via HuggingFace Inference API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare input based on model type
        model_config = self.model_configs.get(model_name, {})
        model_type = model_config.get("type", "conversational")
        
        if model_type == "conversational":
            # For conversational models
            payload = {
                "inputs": {
                    "text": request.prompt
                },
                "parameters": {
                    "max_length": request.max_tokens or model_config.get("max_tokens", 100),
                    "temperature": request.temperature or 0.7,
                    "do_sample": True
                }
            }
        else:
            # For text2text models
            payload = {
                "inputs": request.prompt,
                "parameters": {
                    "max_length": request.max_tokens or model_config.get("max_tokens", 100),
                    "temperature": request.temperature or 0.7
                }
            }
        
        # Make API request
        url = f"{self.api_base_url}/{model_name}"
        
        # Use asyncio to make the request non-blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract content based on response format
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    content = result[0]["generated_text"]
                elif "translation_text" in result[0]:
                    content = result[0]["translation_text"]
                else:
                    content = str(result[0])
            else:
                content = str(result)
            
            llm_response = LLMResponse(
                content=content,
                success=True,
                backend_name=self.backend_name,
                model_name=model_name,
                response_time=response_time,
                confidence=0.7,  # Default confidence for HuggingFace
                metadata={
                    "model_type": model_type,
                    "api_response": result
                }
            )
            
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
            error_msg = f"API error {response.status_code}: {response.text}"
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

    async def _generate_locally(self, request: LLMRequest, model_name: str, start_time: float) -> LLMResponse:
        """Generate response using local transformers."""
        if not self.tokenizer or not self.model:
            return LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=model_name,
                error="Local model not loaded"
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(request.prompt, return_tensors="pt")
            
            # Generate response
            max_tokens = request.max_tokens or 100
            
            # Use asyncio to make generation non-blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=request.temperature or 0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            if response_text.startswith(request.prompt):
                content = response_text[len(request.prompt):].strip()
            else:
                content = response_text
            
            response_time = time.time() - start_time
            tokens_used = outputs.shape[1]
            
            llm_response = LLMResponse(
                content=content,
                success=True,
                backend_name=self.backend_name,
                model_name=model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                confidence=0.7,
                metadata={
                    "generation_method": "local",
                    "input_tokens": inputs.shape[1],
                    "output_tokens": outputs.shape[1] - inputs.shape[1]
                }
            )
            
            self._update_metrics(llm_response)
            return llm_response
            
        except Exception as e:
            response_time = time.time() - start_time
            llm_response = LLMResponse(
                content="",
                success=False,
                backend_name=self.backend_name,
                model_name=model_name,
                error=str(e),
                response_time=response_time
            )
            self._update_metrics(llm_response)
            return llm_response

    async def check_availability(self) -> bool:
        """Check if HuggingFace backend is available."""
        if self.use_inference_api:
            try:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                # Test with a simple request
                url = f"{self.api_base_url}/{self.default_model}"
                response = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": "test"},
                    timeout=10
                )
                
                if response.status_code in [200, 429]:  # 429 means rate limited but available
                    self.status = BackendStatus.AVAILABLE
                    return True
                else:
                    self.status = BackendStatus.ERROR
                    return False
                    
            except Exception as e:
                self.status = BackendStatus.ERROR
                self.last_error = str(e)
                return False
        else:
            # For local models, check if they're loaded
            return self.tokenizer is not None and self.model is not None

    def get_supported_models(self) -> List[str]:
        """Get list of supported HuggingFace models."""
        return list(self.model_configs.keys())

    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for HuggingFace request."""
        if self.use_inference_api:
            # HuggingFace Inference API is free for limited usage
            return 0.0
        else:
            # Local inference has no direct cost
            return 0.0

    def estimate_latency(self, request: LLMRequest) -> float:
        """Estimate response latency for HuggingFace."""
        model_name = request.metadata.get("model", self.default_model) if request.metadata else self.default_model
        model_config = self.model_configs.get(model_name, {})
        
        base_latency = model_config.get("avg_latency", 3.0)
        
        if not self.use_inference_api:
            # Local inference is typically faster
            base_latency *= 0.7
        
        # Adjust based on max_tokens
        max_tokens = request.max_tokens or 100
        token_factor = max_tokens / 100
        
        return base_latency * token_factor

    def is_suitable_for_request(self, request: LLMRequest) -> bool:
        """Check if HuggingFace backend is suitable for the request."""
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
        """Get capabilities for a specific HuggingFace model."""
        if model_name not in self.model_configs:
            return super().get_model_capabilities(model_name)
        
        config = self.model_configs[model_name]
        return {
            "max_tokens": config["max_tokens"],
            "context_window": config["context_window"],
            "supports_system_prompt": False,  # Most HF models don't support system prompts
            "supports_streaming": False,
            "supports_function_calling": False,
            "model_type": config["type"],
            "avg_latency": config["avg_latency"],
            "deployment": "inference_api" if self.use_inference_api else "local"
        }

    async def cleanup(self) -> None:
        """Clean up HuggingFace backend resources."""
        await super().cleanup()
        if not self.use_inference_api:
            self.tokenizer = None
            self.model = None

