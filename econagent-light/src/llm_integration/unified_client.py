"""
Unified LLM client that manages Nemotron primary + Ollama fallback.
Provides seamless switching between local LLM services.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .nemotron_client import NemotronClient, NemotronError, validate_economic_decision
from .ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

class UnifiedLLMClient:
    """
    Unified client managing Nemotron (primary) and Ollama (fallback) services.
    Automatically switches to fallback when primary service fails.
    """
    
    def __init__(
        self,
        nemotron_url: str = "http://localhost:8000/v1",
        ollama_url: str = "http://localhost:11434/v1",
        nemotron_model: str = "nvidia-nemotron-nano-9b-v2",
        ollama_model: str = "llama2:7b-chat",
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        self.nemotron = NemotronClient(
            base_url=nemotron_url,
            model=nemotron_model
        )
        
        self.ollama = OllamaClient(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Service health tracking
        self.nemotron_healthy = self.nemotron.health_check()
        self.ollama_healthy = self.ollama.health_check()
        
        # Response caching
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.cache_size = cache_size
        
        # Statistics
        self.stats = {
            "nemotron_calls": 0,
            "ollama_calls": 0,
            "cache_hits": 0,
            "fallback_switches": 0,
            "total_calls": 0
        }
        
        logger.info(f"UnifiedLLMClient initialized - Nemotron: {self.nemotron_healthy}, Ollama: {self.ollama_healthy}")
    
    def _get_cache_key(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate cache key for request."""
        import hashlib
        content = json.dumps(messages, sort_keys=True) + f"_{temperature}_{max_tokens}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if response is cached."""
        if not self.enable_caching or not self.cache:
            return None
        
        cached_response = self.cache.get(cache_key)
        if cached_response:
            self.stats["cache_hits"] += 1
            logger.debug("Cache hit for LLM request")
            return cached_response
        
        return None
    
    def _store_cache(self, cache_key: str, response: Dict[str, Any]):
        """Store response in cache."""
        if not self.enable_caching or not self.cache:
            return
        
        # Simple LRU: remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
    
    def call_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        prefer_nemotron: bool = True
    ) -> Dict[str, Any]:
        """
        Call LLM with automatic fallback between services.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prefer_nemotron: Whether to prefer Nemotron over Ollama
            
        Returns:
            Dict with response content and metadata
        """
        self.stats["total_calls"] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(messages, temperature, max_tokens)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Determine service order
        if prefer_nemotron and self.nemotron_healthy:
            primary, fallback = self.nemotron, self.ollama
            primary_name, fallback_name = "nemotron", "ollama"
        elif self.ollama_healthy:
            primary, fallback = self.ollama, self.nemotron
            primary_name, fallback_name = "ollama", "nemotron"
        else:
            # Both services unhealthy, return fallback decision
            logger.error("Both LLM services are unhealthy, using fallback decision")
            return self._get_fallback_response()
        
        # Try primary service
        try:
            response = primary.call_model(messages, temperature, max_tokens)
            response["service_used"] = primary_name
            
            # Update stats
            if primary_name == "nemotron":
                self.stats["nemotron_calls"] += 1
            else:
                self.stats["ollama_calls"] += 1
            
            # Cache successful response
            self._store_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.warning(f"Primary service ({primary_name}) failed: {e}")
            self.stats["fallback_switches"] += 1
            
            # Try fallback service
            try:
                response = fallback.call_model(messages, temperature, max_tokens)
                response["service_used"] = fallback_name
                response["fallback_used"] = True
                
                # Update stats
                if fallback_name == "nemotron":
                    self.stats["nemotron_calls"] += 1
                else:
                    self.stats["ollama_calls"] += 1
                
                # Cache successful response
                self._store_cache(cache_key, response)
                
                return response
                
            except Exception as e2:
                logger.error(f"Fallback service ({fallback_name}) also failed: {e2}")
                return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get hardcoded fallback response when all services fail."""
        return {
            "content": '{"work": 0.2, "consumption": 0.1}',
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "response_time": 0.0,
            "model": "fallback",
            "service_used": "fallback",
            "fallback_used": True
        }
    
    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Make multiple LLM calls concurrently with fallback support.
        
        Args:
            message_batches: List of message lists for each call
            temperature: Sampling temperature
            max_tokens: Maximum tokens per call
            max_workers: Maximum concurrent workers
            
        Returns:
            List of response dictionaries
        """
        results = [None] * len(message_batches)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.call_model,
                    messages,
                    temperature,
                    max_tokens
                ): i
                for i, messages in enumerate(message_batches)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Batch call {index} failed: {e}")
                    results[index] = self._get_fallback_response()
        
        return results
    
    def make_economic_decision(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Dict[str, float]:
        """
        Make economic decision with validation and fallback.
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Dict with 'work' and 'consumption' keys (normalized values)
        """
        try:
            response = self.call_model(messages, temperature, max_tokens)
            decision = validate_economic_decision(response["content"])
            
            # Add metadata
            decision["_metadata"] = {
                "service_used": response.get("service_used", "unknown"),
                "response_time": response.get("response_time", 0.0),
                "fallback_used": response.get("fallback_used", False)
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Economic decision failed: {e}")
            return {
                "work": 0.2,
                "consumption": 0.1,
                "_metadata": {
                    "service_used": "fallback",
                    "response_time": 0.0,
                    "fallback_used": True,
                    "error": str(e)
                }
            }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        self.nemotron_healthy = self.nemotron.health_check()
        self.ollama_healthy = self.ollama.health_check()
        
        return {
            "nemotron": self.nemotron_healthy,
            "ollama": self.ollama_healthy,
            "any_healthy": self.nemotron_healthy or self.ollama_healthy
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.stats,
            "cache_size": len(self.cache) if self.cache else 0,
            "services_healthy": self.health_check()
        }