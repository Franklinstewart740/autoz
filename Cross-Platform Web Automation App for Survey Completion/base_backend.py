"""
Base LLM Backend
Abstract base class for all LLM backend implementations.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class BackendStatus(Enum):
    """Backend status enumeration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class LLMResponse:
    """Standardized response structure from LLM backends."""
    content: str
    success: bool
    backend_name: str
    model_name: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Standardized request structure for LLM backends."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMBackend(ABC):
    """
    Abstract base class for all LLM backend implementations.
    Provides common functionality and defines the interface.
    """

    def __init__(self, backend_name: str, config: Dict[str, Any]):
        """
        Initialize the backend.
        
        Args:
            backend_name: Name of the backend (e.g., "openai", "huggingface")
            config: Configuration dictionary
        """
        self.backend_name = backend_name
        self.config = config
        self.logger = logging.getLogger(f"llm_backend.{backend_name}")
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.last_request_time = 0.0
        
        # Rate limiting
        self.rate_limit_remaining = None
        self.rate_limit_reset_time = None
        
        # Status
        self.status = BackendStatus.UNAVAILABLE
        self.last_error = None

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the backend (e.g., authenticate, load models).
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response using the LLM.
        
        Args:
            request: LLMRequest object with prompt and parameters
            
        Returns:
            LLMResponse object with the generated content
        """
        pass

    @abstractmethod
    async def check_availability(self) -> bool:
        """
        Check if the backend is currently available.
        
        Returns:
            True if available, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models for this backend.
        
        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """
        Estimate the cost for a request (in USD or credits).
        
        Args:
            request: LLMRequest object
            
        Returns:
            Estimated cost
        """
        pass

    @abstractmethod
    def estimate_latency(self, request: LLMRequest) -> float:
        """
        Estimate the response latency for a request (in seconds).
        
        Args:
            request: LLMRequest object
            
        Returns:
            Estimated latency in seconds
        """
        pass

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        self.logger.info(f"Cleaning up {self.backend_name} backend")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current backend status and metrics.
        
        Returns:
            Dictionary with status information
        """
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "backend_name": self.backend_name,
            "status": self.status.value,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "average_response_time": avg_response_time,
            "last_request_time": self.last_request_time,
            "last_error": self.last_error,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset_time": self.rate_limit_reset_time,
            "supported_models": self.get_supported_models()
        }

    def _update_metrics(self, response: LLMResponse) -> None:
        """
        Update performance metrics after a request.
        
        Args:
            response: LLMResponse object
        """
        self.request_count += 1
        self.last_request_time = time.time()
        
        if response.success:
            if response.tokens_used:
                self.total_tokens += response.tokens_used
            if response.response_time:
                self.total_response_time += response.response_time
        else:
            self.error_count += 1
            self.last_error = response.error

    def _check_rate_limit(self) -> bool:
        """
        Check if we're currently rate limited.
        
        Returns:
            True if rate limited, False otherwise
        """
        if self.rate_limit_reset_time and time.time() < self.rate_limit_reset_time:
            return True
        return False

    def _handle_rate_limit(self, reset_time: Optional[float] = None) -> None:
        """
        Handle rate limiting.
        
        Args:
            reset_time: Time when rate limit resets (Unix timestamp)
        """
        self.status = BackendStatus.RATE_LIMITED
        self.rate_limit_reset_time = reset_time or (time.time() + 60)  # Default 1 minute
        self.logger.warning(f"Rate limited until {self.rate_limit_reset_time}")

    def _handle_error(self, error: str) -> None:
        """
        Handle backend errors.
        
        Args:
            error: Error message
        """
        self.status = BackendStatus.ERROR
        self.last_error = error
        self.logger.error(f"Backend error: {error}")

    def get_performance_score(self) -> float:
        """
        Calculate a performance score for this backend.
        Higher score indicates better performance.
        
        Returns:
            Performance score (0.0 to 1.0)
        """
        if self.request_count == 0:
            return 0.5  # Neutral score for untested backends
        
        # Calculate success rate
        success_rate = (self.request_count - self.error_count) / self.request_count
        
        # Calculate speed score (lower response time = higher score)
        avg_response_time = self.total_response_time / self.request_count
        speed_score = max(0.0, 1.0 - (avg_response_time / 10.0))  # Normalize to 10 seconds max
        
        # Calculate availability score
        availability_score = 1.0 if self.status == BackendStatus.AVAILABLE else 0.0
        
        # Weighted combination
        performance_score = (
            success_rate * 0.4 +
            speed_score * 0.3 +
            availability_score * 0.3
        )
        
        return max(0.0, min(1.0, performance_score))

    def is_suitable_for_request(self, request: LLMRequest) -> bool:
        """
        Check if this backend is suitable for a specific request.
        
        Args:
            request: LLMRequest object
            
        Returns:
            True if suitable, False otherwise
        """
        # Basic checks
        if self.status != BackendStatus.AVAILABLE:
            return False
        
        if self._check_rate_limit():
            return False
        
        # Check if we support the required features
        # This can be overridden by specific backends
        return True

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get capabilities for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model capabilities
        """
        # Default capabilities - should be overridden by specific backends
        return {
            "max_tokens": 4096,
            "supports_system_prompt": True,
            "supports_streaming": False,
            "supports_function_calling": False,
            "context_window": 4096
        }

