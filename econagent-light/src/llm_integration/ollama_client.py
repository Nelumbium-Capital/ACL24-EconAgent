"""
Ollama client for fallback LLM processing.
Provides backup when Nemotron is unavailable.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .nemotron_client import normalize_decision_value, validate_economic_decision

logger = logging.getLogger(__name__)

class OllamaClient:
    """HTTP client for local Ollama service as LLM fallback."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama2:7b-chat",
        timeout: int = 45,  # Ollama can be slower
        max_retries: int = 2
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Connection validation
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """Validate connection to Ollama service."""
        try:
            # Try Ollama's native API first
            response = self.session.get(
                f"{self.base_url.replace('/v1', '')}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama at {self.base_url}")
                return True
            else:
                logger.warning(f"Ollama service returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
            return False
    
    def call_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,  # Slightly higher for Ollama
        max_tokens: int = 512,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Call Ollama model with OpenAI-compatible format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Dict with 'content' and 'usage' information
        """
        # Try OpenAI-compatible endpoint first
        try:
            return self._call_openai_compatible(messages, temperature, max_tokens, stream)
        except Exception as e:
            logger.warning(f"OpenAI-compatible endpoint failed: {e}")
            # Fallback to native Ollama API
            return self._call_native_api(messages, temperature, max_tokens)
    
    def _call_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """Call using OpenAI-compatible endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Ollama call completed in {elapsed_time:.2f}s")
        
        return {
            "content": content,
            "usage": usage,
            "response_time": elapsed_time,
            "model": self.model
        }
    
    def _call_native_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Call using native Ollama API format."""
        # Convert messages to single prompt for native API
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url.replace('/v1', '')}/api/generate",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        result = response.json()
        
        content = result.get("response", "")
        elapsed_time = time.time() - start_time
        
        logger.debug(f"Ollama native API call completed in {elapsed_time:.2f}s")
        
        return {
            "content": content,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "response_time": elapsed_time,
            "model": self.model
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to single prompt."""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Make multiple calls to Ollama (sequential).
        
        Args:
            message_batches: List of message lists for each call
            temperature: Sampling temperature
            max_tokens: Maximum tokens per call
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, messages in enumerate(message_batches):
            try:
                result = self.call_model(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Ollama batch call {i} failed: {e}")
                # Return fallback response for failed calls
                results.append({
                    "content": '{"work": 0.2, "consumption": 0.1}',
                    "usage": {},
                    "response_time": 0.0,
                    "model": "fallback",
                    "error": str(e)
                })
        
        return results
    
    def health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        return self._validate_connection()

class OllamaError(Exception):
    """Custom exception for Ollama client errors."""
    pass