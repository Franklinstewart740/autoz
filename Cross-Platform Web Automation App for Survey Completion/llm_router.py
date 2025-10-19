"""
LLM Router
Dynamic routing system for selecting optimal LLM backends based on requirements.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from .base_backend import BaseLLMBackend, LLMRequest, LLMResponse, BackendStatus
from .openai_backend import OpenAIBackend
from .huggingface_backend import HuggingFaceBackend
from .ollama_backend import OllamaBackend
from .groq_backend import GroqBackend


class RoutingStrategy(Enum):
    """Routing strategy enumeration."""
    PERFORMANCE = "performance"  # Best performance score
    COST = "cost"  # Lowest cost
    LATENCY = "latency"  # Fastest response
    AVAILABILITY = "availability"  # Most available
    ROUND_ROBIN = "round_robin"  # Round-robin selection
    WEIGHTED = "weighted"  # Weighted selection based on multiple factors


@dataclass
class RoutingCriteria:
    """Criteria for backend selection."""
    strategy: RoutingStrategy = RoutingStrategy.PERFORMANCE
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    min_confidence: Optional[float] = None
    preferred_backends: Optional[List[str]] = None
    excluded_backends: Optional[List[str]] = None
    require_features: Optional[List[str]] = None  # e.g., ["streaming", "function_calling"]


class LLMRouter:
    """
    Dynamic router for selecting optimal LLM backends.
    Supports multiple routing strategies and performance optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM Router.
        
        Args:
            config: Configuration dictionary with backend configurations
        """
        self.logger = logging.getLogger("llm_router")
        self.config = config
        
        # Initialize backends
        self.backends: Dict[str, BaseLLMBackend] = {}
        self.backend_configs = config.get("backends", {})
        
        # Routing state
        self.round_robin_index = 0
        self.routing_history = []
        self.performance_cache = {}
        
        # Default routing criteria
        self.default_criteria = RoutingCriteria()

    async def initialize(self) -> bool:
        """Initialize all configured backends."""
        try:
            self.logger.info("Initializing LLM Router")
            
            # Initialize each configured backend
            for backend_name, backend_config in self.backend_configs.items():
                if not backend_config.get("enabled", True):
                    continue
                
                try:
                    backend = self._create_backend(backend_name, backend_config)
                    if backend and await backend.initialize():
                        self.backends[backend_name] = backend
                        self.logger.info(f"Initialized {backend_name} backend")
                    else:
                        self.logger.warning(f"Failed to initialize {backend_name} backend")
                        
                except Exception as e:
                    self.logger.error(f"Error initializing {backend_name} backend: {e}")
            
            if not self.backends:
                self.logger.error("No backends successfully initialized")
                return False
            
            self.logger.info(f"LLM Router initialized with {len(self.backends)} backends: {list(self.backends.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Router: {e}", exc_info=True)
            return False

    def _create_backend(self, backend_name: str, config: Dict[str, Any]) -> Optional[BaseLLMBackend]:
        """Create a backend instance based on configuration."""
        backend_type = config.get("type", backend_name)
        
        if backend_type == "openai":
            return OpenAIBackend(config)
        elif backend_type == "huggingface":
            return HuggingFaceBackend(config)
        elif backend_type == "ollama":
            return OllamaBackend(config)
        elif backend_type == "groq":
            return GroqBackend(config)
        else:
            self.logger.error(f"Unknown backend type: {backend_type}")
            return None

    async def route_request(self, request: LLMRequest, 
                          criteria: Optional[RoutingCriteria] = None) -> LLMResponse:
        """
        Route a request to the optimal backend.
        
        Args:
            request: LLMRequest object
            criteria: Optional routing criteria (uses default if not provided)
            
        Returns:
            LLMResponse from the selected backend
        """
        if not self.backends:
            return LLMResponse(
                content="",
                success=False,
                backend_name="router",
                model_name="unknown",
                error="No backends available"
            )
        
        criteria = criteria or self.default_criteria
        
        try:
            # Select the best backend
            selected_backend = await self._select_backend(request, criteria)
            
            if not selected_backend:
                return LLMResponse(
                    content="",
                    success=False,
                    backend_name="router",
                    model_name="unknown",
                    error="No suitable backend found"
                )
            
            # Generate response
            response = await selected_backend.generate_response(request)
            
            # Update routing history
            self._update_routing_history(selected_backend.backend_name, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error routing request: {e}", exc_info=True)
            return LLMResponse(
                content="",
                success=False,
                backend_name="router",
                model_name="unknown",
                error=str(e)
            )

    async def _select_backend(self, request: LLMRequest, 
                            criteria: RoutingCriteria) -> Optional[BaseLLMBackend]:
        """
        Select the optimal backend based on routing criteria.
        
        Args:
            request: LLMRequest object
            criteria: Routing criteria
            
        Returns:
            Selected backend or None if no suitable backend found
        """
        # Filter backends based on criteria
        suitable_backends = []
        
        for backend_name, backend in self.backends.items():
            # Check if backend is excluded
            if criteria.excluded_backends and backend_name in criteria.excluded_backends:
                continue
            
            # Check if backend is in preferred list (if specified)
            if criteria.preferred_backends and backend_name not in criteria.preferred_backends:
                continue
            
            # Check if backend is suitable for the request
            if not backend.is_suitable_for_request(request):
                continue
            
            # Check cost constraint
            if criteria.max_cost is not None:
                estimated_cost = backend.estimate_cost(request)
                if estimated_cost > criteria.max_cost:
                    continue
            
            # Check latency constraint
            if criteria.max_latency is not None:
                estimated_latency = backend.estimate_latency(request)
                if estimated_latency > criteria.max_latency:
                    continue
            
            # Check required features
            if criteria.require_features:
                model_name = request.metadata.get("model") if request.metadata else None
                if model_name:
                    capabilities = backend.get_model_capabilities(model_name)
                    for feature in criteria.require_features:
                        feature_key = f"supports_{feature}"
                        if not capabilities.get(feature_key, False):
                            break
                    else:
                        suitable_backends.append(backend)
                else:
                    suitable_backends.append(backend)
            else:
                suitable_backends.append(backend)
        
        if not suitable_backends:
            self.logger.warning("No suitable backends found for request")
            return None
        
        # Select backend based on strategy
        return await self._apply_routing_strategy(suitable_backends, request, criteria)

    async def _apply_routing_strategy(self, backends: List[BaseLLMBackend], 
                                    request: LLMRequest,
                                    criteria: RoutingCriteria) -> Optional[BaseLLMBackend]:
        """
        Apply the routing strategy to select from suitable backends.
        
        Args:
            backends: List of suitable backends
            request: LLMRequest object
            criteria: Routing criteria
            
        Returns:
            Selected backend
        """
        if not backends:
            return None
        
        if len(backends) == 1:
            return backends[0]
        
        strategy = criteria.strategy
        
        if strategy == RoutingStrategy.PERFORMANCE:
            # Select backend with best performance score
            best_backend = max(backends, key=lambda b: b.get_performance_score())
            return best_backend
            
        elif strategy == RoutingStrategy.COST:
            # Select backend with lowest cost
            costs = [(backend, backend.estimate_cost(request)) for backend in backends]
            best_backend = min(costs, key=lambda x: x[1])[0]
            return best_backend
            
        elif strategy == RoutingStrategy.LATENCY:
            # Select backend with lowest latency
            latencies = [(backend, backend.estimate_latency(request)) for backend in backends]
            best_backend = min(latencies, key=lambda x: x[1])[0]
            return best_backend
            
        elif strategy == RoutingStrategy.AVAILABILITY:
            # Select backend with best availability status
            available_backends = [b for b in backends if b.status == BackendStatus.AVAILABLE]
            if available_backends:
                # Among available, select by performance
                return max(available_backends, key=lambda b: b.get_performance_score())
            else:
                return backends[0]  # Fallback to first backend
                
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Round-robin selection
            selected = backends[self.round_robin_index % len(backends)]
            self.round_robin_index += 1
            return selected
            
        elif strategy == RoutingStrategy.WEIGHTED:
            # Weighted selection based on multiple factors
            return self._weighted_selection(backends, request)
            
        else:
            # Default to performance
            return max(backends, key=lambda b: b.get_performance_score())

    def _weighted_selection(self, backends: List[BaseLLMBackend], 
                          request: LLMRequest) -> BaseLLMBackend:
        """
        Select backend using weighted scoring across multiple factors.
        
        Args:
            backends: List of backends to choose from
            request: LLMRequest object
            
        Returns:
            Selected backend
        """
        scores = []
        
        for backend in backends:
            # Calculate weighted score
            performance_score = backend.get_performance_score()
            
            # Normalize cost (lower is better)
            cost = backend.estimate_cost(request)
            cost_score = 1.0 / (1.0 + cost) if cost > 0 else 1.0
            
            # Normalize latency (lower is better)
            latency = backend.estimate_latency(request)
            latency_score = 1.0 / (1.0 + latency / 10.0)  # Normalize to ~10 seconds
            
            # Availability score
            availability_score = 1.0 if backend.status == BackendStatus.AVAILABLE else 0.5
            
            # Weighted combination
            total_score = (
                performance_score * 0.3 +
                cost_score * 0.2 +
                latency_score * 0.3 +
                availability_score * 0.2
            )
            
            scores.append((backend, total_score))
        
        # Select backend with highest score
        best_backend = max(scores, key=lambda x: x[1])[0]
        return best_backend

    def _update_routing_history(self, backend_name: str, response: LLMResponse) -> None:
        """
        Update routing history for analytics.
        
        Args:
            backend_name: Name of the backend used
            response: LLMResponse object
        """
        self.routing_history.append({
            "backend_name": backend_name,
            "success": response.success,
            "response_time": response.response_time,
            "tokens_used": response.tokens_used,
            "error": response.error
        })
        
        # Keep only last 1000 entries
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]

    async def get_backend_status(self) -> Dict[str, Any]:
        """
        Get status of all backends.
        
        Returns:
            Dictionary with backend status information
        """
        status = {}
        
        for backend_name, backend in self.backends.items():
            status[backend_name] = backend.get_status()
        
        return status

    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all backends.
        
        Returns:
            Dictionary with health status for each backend
        """
        health_status = {}
        
        # Run health checks concurrently
        tasks = []
        for backend_name, backend in self.backends.items():
            tasks.append(self._check_backend_health(backend_name, backend))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (backend_name, _) in enumerate(self.backends.items()):
            if isinstance(results[i], Exception):
                health_status[backend_name] = False
            else:
                health_status[backend_name] = results[i]
        
        return health_status

    async def _check_backend_health(self, backend_name: str, backend: BaseLLMBackend) -> bool:
        """Check health of a single backend."""
        try:
            return await backend.check_availability()
        except Exception as e:
            self.logger.error(f"Health check failed for {backend_name}: {e}")
            return False

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics and analytics.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {"total_requests": 0}
        
        # Calculate statistics
        total_requests = len(self.routing_history)
        successful_requests = sum(1 for entry in self.routing_history if entry["success"])
        
        # Backend usage statistics
        backend_usage = {}
        for entry in self.routing_history:
            backend_name = entry["backend_name"]
            backend_usage[backend_name] = backend_usage.get(backend_name, 0) + 1
        
        # Average response times
        response_times = [entry["response_time"] for entry in self.routing_history if entry["response_time"]]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Token usage
        total_tokens = sum(entry["tokens_used"] for entry in self.routing_history if entry["tokens_used"])
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "backend_usage": backend_usage,
            "average_response_time": avg_response_time,
            "total_tokens": total_tokens,
            "available_backends": len([b for b in self.backends.values() if b.status == BackendStatus.AVAILABLE])
        }

    def set_default_criteria(self, criteria: RoutingCriteria) -> None:
        """
        Set default routing criteria.
        
        Args:
            criteria: New default routing criteria
        """
        self.default_criteria = criteria
        self.logger.info(f"Updated default routing criteria: {criteria.strategy.value}")

    async def cleanup(self) -> None:
        """Clean up all backends."""
        self.logger.info("Cleaning up LLM Router")
        
        cleanup_tasks = []
        for backend in self.backends.values():
            cleanup_tasks.append(backend.cleanup())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.backends.clear()
        self.logger.info("LLM Router cleanup completed")

