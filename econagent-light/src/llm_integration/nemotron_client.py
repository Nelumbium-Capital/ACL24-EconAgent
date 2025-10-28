"""
NVIDIA Nemotron client for local Docker container communication.
Provides OpenAI-compatible interface for economic agent decision-making.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class NemotronClient:
    """HTTP client for NVIDIA Nemotron - supports both local Docker and NVIDIA API."""
    
    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-4-340b-instruct",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Connection validation
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """Validate connection to Nemotron service."""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Successfully connected to Nemotron at {self.base_url}")
                return True
            else:
                logger.warning(f"Nemotron service returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to Nemotron: {e}")
            return False
    
    def call_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Call Nemotron model with OpenAI-compatible format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Dict with 'content' and 'usage' information
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content and usage info
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Nemotron call completed in {elapsed_time:.2f}s")
            
            return {
                "content": content,
                "usage": usage,
                "response_time": elapsed_time,
                "model": self.model
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Nemotron API request failed: {e}")
            raise NemotronError(f"API request failed: {e}")
        
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Invalid response format from Nemotron: {e}")
            raise NemotronError(f"Invalid response format: {e}")
    
    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Make multiple calls to Nemotron (sequential for now).
        
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
                
            except NemotronError as e:
                logger.error(f"Batch call {i} failed: {e}")
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
        """Check if Nemotron service is healthy."""
        return self._validate_connection()

class NemotronError(Exception):
    """Custom exception for Nemotron client errors."""
    pass

def normalize_decision_value(value: Union[str, float]) -> float:
    """
    Normalize decision values to [0,1] range with 0.02 step precision.
    
    Args:
        value: Raw value from LLM response
        
    Returns:
        Normalized float value in [0,1] with 0.02 steps
    """
    try:
        # Convert to float if string
        if isinstance(value, str):
            value = float(value)
        
        # Clamp to [0,1] range
        value = max(0.0, min(1.0, float(value)))
        
        # Round to 0.02 step increments
        value = round(value * 50) / 50.0
        
        return value
        
    except (ValueError, TypeError):
        logger.warning(f"Failed to normalize value: {value}, using fallback 0.5")
        return 0.5

def validate_economic_decision(response_text: str) -> Dict[str, float]:
    """
    Validate and parse economic decision from LLM response.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        Dict with 'work' and 'consumption' keys, normalized values
    """
    try:
        # Try to parse as JSON
        if response_text.strip().startswith('{'):
            data = json.loads(response_text)
        else:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        
        # Extract and normalize work and consumption values
        work = normalize_decision_value(data.get('work', 0.5))
        consumption = normalize_decision_value(data.get('consumption', 0.5))
        
        return {
            "work": work,
            "consumption": consumption
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse economic decision: {e}")
        logger.warning(f"Response text: {response_text[:200]}...")
        
        # Return conservative fallback values
        return {
            "work": 0.2,
            "consumption": 0.1
        }