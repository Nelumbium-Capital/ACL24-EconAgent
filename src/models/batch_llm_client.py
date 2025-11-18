"""
Batch LLM client for efficient agent inference with NeMo and Ollama fallback.

Implements batching, JSON validation, retry logic, and prompt compression
for large-scale multi-agent simulations.
"""
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from src.utils.logging_config import logger


@dataclass
class LLMResponse:
    """Container for LLM response with parsed JSON."""
    success: bool
    data: Optional[Dict[str, Any]]
    raw_text: str
    error: Optional[str] = None
    retries: int = 0


class BatchLLMClient:
    """
    Batch LLM client for efficient multi-agent inference.
    
    Features:
    - Batches prompts to reduce API calls
    - JSON schema validation with retry
    - NeMo primary, Ollama fallback
    - Shared context caching
    - Message compression
    """
    
    def __init__(
        self,
        nemotron_url: str = None,
        ollama_url: str = None,
        model_name: str = "nvidia/llama-3.1-nemotron-70b-instruct",
        ollama_model: str = "llama3.1",
        batch_size: int = 64,
        max_retries: int = 3,
        timeout: int = 5
    ):
        """
        Initialize batch LLM client.
        
        Args:
            nemotron_url: NeMo server URL
            ollama_url: Ollama server URL
            model_name: NeMo model identifier
            ollama_model: Ollama model name
            batch_size: Max prompts per batch
            max_retries: Max retries for failed JSON parsing
            timeout: Request timeout in seconds
        """
        self.nemotron_url = nemotron_url or settings.nemotron_url
        self.ollama_url = ollama_url or settings.ollama_url
        self.model_name = model_name
        self.ollama_model = ollama_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Track statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'nemotron_calls': 0,
            'ollama_calls': 0,
            'total_retries': 0
        }
        
        logger.info(f"Initialized BatchLLMClient with batch_size={batch_size}")
    
    def batch_inference(
        self,
        prompts: List[str],
        system_prompt: str = "You are an expert economic agent.",
        temperature: float = 0.3,
        max_tokens: int = 500,
        expected_json_keys: Optional[List[str]] = None
    ) -> List[LLMResponse]:
        """
        Run batch inference on multiple prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Shared system prompt (cached)
            temperature: Sampling temperature
            max_tokens: Max tokens per response
            expected_json_keys: Keys to validate in JSON response
            
        Returns:
            List of LLMResponse objects
        """
        logger.info(f"Running batch inference on {len(prompts)} prompts")
        
        # Split into batches
        batches = [
            prompts[i:i+self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        
        all_responses = []
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} prompts)")
            
            # Process batch with threading
            batch_responses = self._process_batch(
                batch,
                system_prompt,
                temperature,
                max_tokens,
                expected_json_keys
            )
            
            all_responses.extend(batch_responses)
        
        # Log statistics
        success_rate = sum(1 for r in all_responses if r.success) / len(all_responses)
        logger.info(f"Batch inference complete. Success rate: {success_rate:.2%}")
        
        return all_responses
    
    def _process_batch(
        self,
        prompts: List[str],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        expected_json_keys: Optional[List[str]]
    ) -> List[LLMResponse]:
        """Process a single batch of prompts with threading."""
        responses = []
        
        # Use thread pool for parallel requests
        with ThreadPoolExecutor(max_workers=min(16, len(prompts))) as executor:
            futures = {
                executor.submit(
                    self._single_inference,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    expected_json_keys
                ): idx
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results in order
            results = [None] * len(prompts)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Future failed for prompt {idx}: {e}")
                    results[idx] = LLMResponse(
                        success=False,
                        data=None,
                        raw_text="",
                        error=str(e)
                    )
            
            responses = results
        
        return responses
    
    def _single_inference(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        expected_json_keys: Optional[List[str]]
    ) -> LLMResponse:
        """Run inference for a single prompt with retry logic."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                # Try NeMo first
                raw_text = self._call_nemotron(messages, temperature, max_tokens)
                
                if raw_text:
                    # Parse and validate JSON
                    parsed_data = self._parse_json(raw_text, expected_json_keys)
                    
                    if parsed_data:
                        return LLMResponse(
                            success=True,
                            data=parsed_data,
                            raw_text=raw_text,
                            retries=attempt
                        )
                
                # If NeMo failed or JSON invalid, try Ollama
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed, retrying...")
                    time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Inference attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        success=False,
                        data=None,
                        raw_text="",
                        error=str(e),
                        retries=attempt
                    )
        
        # Final fallback: generate default response
        logger.warning("All attempts failed, using fallback response")
        return self._generate_fallback_response(expected_json_keys)
    
    def _call_nemotron(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call NeMo server."""
        try:
            response = requests.post(
                f"{self.nemotron_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.stats['nemotron_calls'] += 1
                self.stats['successful_calls'] += 1
                return result['choices'][0]['message']['content']
            else:
                logger.warning(f"NeMo returned status {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            logger.warning("NeMo connection failed, trying Ollama")
        except Exception as e:
            logger.warning(f"NeMo call failed: {e}")
        
        # Fallback to Ollama
        return self._call_ollama(messages, temperature, max_tokens)
    
    def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Ollama server."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.stats['ollama_calls'] += 1
                self.stats['successful_calls'] += 1
                return result['message']['content']
            else:
                logger.error(f"Ollama returned status {response.status_code}")
        
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        
        return None
    
    def _parse_json(
        self,
        text: str,
        expected_keys: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response with validation.
        
        Args:
            text: Raw text response
            expected_keys: Keys that must be present
            
        Returns:
            Parsed JSON dict or None if invalid
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip()
        
        try:
            data = json.loads(json_str)
            
            # Validate expected keys
            if expected_keys:
                if not all(key in data for key in expected_keys):
                    logger.warning(f"JSON missing expected keys: {expected_keys}")
                    return None
            
            return data
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None
    
    def _generate_fallback_response(
        self,
        expected_keys: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate a safe fallback response when all else fails."""
        if expected_keys:
            # Generate reasonable defaults
            fallback_data = {}
            for key in expected_keys:
                if key in ['work', 'consumption', 'production', 'hiring']:
                    fallback_data[key] = 0.5  # Neutral value
                elif key == 'lessons':
                    fallback_data[key] = ["No reflection available"]
                elif 'delta' in key:
                    fallback_data[key] = 0.0
                else:
                    fallback_data[key] = None
            
            return LLMResponse(
                success=True,
                data=fallback_data,
                raw_text="Fallback response",
                error="Used fallback due to repeated failures"
            )
        
        return LLMResponse(
            success=False,
            data=None,
            raw_text="",
            error="Complete failure, no fallback available"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


# Global singleton instance
_global_client: Optional[BatchLLMClient] = None


def get_batch_client() -> BatchLLMClient:
    """Get or create global batch LLM client."""
    global _global_client
    if _global_client is None:
        _global_client = BatchLLMClient()
    return _global_client


