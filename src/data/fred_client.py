"""
FRED API client with local caching and retry logic.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings
from src.utils.logging_config import logger
from src.utils.error_handler import error_handler, retry_on_failure, DataFetchError


class FREDClient:
    """
    Client for fetching US macroeconomic data from Federal Reserve Economic Data API.
    Implements local file-based caching and retry logic for resilience.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_hours: int = 24
    ):
        """
        Initialize FRED client.
        
        Args:
            api_key: FRED API key (defaults to settings)
            cache_dir: Directory for caching data (defaults to settings)
            cache_hours: Hours before cache is considered stale
        """
        self.api_key = api_key or settings.fred_api_key
        self.cache_dir = cache_dir or settings.data_cache_dir
        self.cache_hours = cache_hours
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with retry logic
        self.session = self._create_session()
        
        logger.info(f"FRED Client initialized with cache dir: {self.cache_dir}")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        
        # Retry strategy: 3 retries with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_cache_path(self, series_id: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path for a series."""
        cache_key = f"{series_id}_{start_date}_{end_date}.json"
        return self.cache_dir / cache_key
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        is_valid = age < timedelta(hours=self.cache_hours)
        
        if not is_valid:
            logger.debug(f"Cache expired for {cache_path.name} (age: {age})")
        
        return is_valid
    
    def _load_from_cache(
        self, 
        series_id: str, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        cache_path = self._get_cache_path(series_id, start_date, end_date)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')
            
            logger.info(f"Loaded {series_id} from cache ({len(df)} observations)")
            return df
        
        except Exception as e:
            logger.warning(f"Failed to load cache for {series_id}: {e}")
            return None
    
    def _save_to_cache(
        self, 
        series_id: str, 
        start_date: str, 
        end_date: str, 
        data: Dict
    ):
        """Save data to cache."""
        cache_path = self._get_cache_path(series_id, start_date, end_date)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {series_id} to cache")
        
        except Exception as e:
            logger.warning(f"Failed to save cache for {series_id}: {e}")
    
    def _parse_response(self, response_data: Dict) -> pd.DataFrame:
        """Parse FRED API response into DataFrame."""
        observations = response_data.get('observations', [])
        
        if not observations:
            logger.warning("No observations in response")
            return pd.DataFrame()
        
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop rows with non-numeric values
        df = df.dropna(subset=['value'])
        
        df = df.set_index('date')
        
        return df[['value']]
    
    @retry_on_failure(max_retries=3, exceptions=(requests.RequestException,))
    def fetch_series(
        self, 
        series_id: str, 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch time series data for a given FRED series ID.
        
        Args:
            series_id: FRED series identifier (e.g., 'UNRATE', 'FEDFUNDS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with date index and series values
            
        Raises:
            requests.HTTPError: If API request fails
            ValueError: If API key is invalid
            DataFetchError: If fetch fails and no fallback available
        """
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(series_id, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Fetch from API
        logger.info(f"Fetching {series_id} from FRED API ({start_date} to {end_date})")
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/series/observations",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check for API errors
            if 'error_code' in response_data:
                error_msg = response_data.get('error_message', 'Unknown error')
                if response_data['error_code'] == 400:
                    raise ValueError(f"Invalid FRED API key or parameters: {error_msg}")
                raise requests.HTTPError(f"FRED API error: {error_msg}")
            
            # Parse response
            df = self._parse_response(response_data)
            
            # Cache the result
            self._save_to_cache(series_id, start_date, end_date, response_data)
            
            logger.info(f"Successfully fetched {series_id} ({len(df)} observations)")
            return df
        
        except (requests.HTTPError, requests.RequestException) as e:
            # Use error handler for fallback to stale cache
            cache_path = self._get_cache_path(series_id, start_date, end_date)
            fallback_data = error_handler.handle_data_error(
                error=e,
                series_id=series_id,
                cache_path=cache_path,
                use_stale_cache=True
            )
            
            if fallback_data is not None:
                return fallback_data
            
            # No fallback available, raise error
            raise DataFetchError(
                f"Failed to fetch {series_id} and no cache available",
                context={
                    'series_id': series_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'error': str(e)
                }
            )
        
        except Exception as e:
            logger.error(f"Unexpected error fetching {series_id}: {e}")
            raise
    
    def fetch_multiple_series(
        self, 
        series_ids: List[str], 
        start_date: str, 
        end_date: str,
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple series concurrently.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_workers: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping series IDs to DataFrames
        """
        logger.info(f"Fetching {len(series_ids)} series concurrently")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            futures = {
                executor.submit(
                    self.fetch_series, 
                    sid, 
                    start_date, 
                    end_date
                ): sid
                for sid in series_ids
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                series_id = futures[future]
                try:
                    results[series_id] = future.result()
                except Exception as e:
                    logger.error(f"Failed to fetch {series_id}: {e}")
                    results[series_id] = pd.DataFrame()  # Empty DataFrame on error
        
        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(f"Successfully fetched {successful}/{len(series_ids)} series")
        
        return results
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata for a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with series metadata
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/series",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if 'seriess' in data and len(data['seriess']) > 0:
                return data['seriess'][0]
            
            return {}
        
        except Exception as e:
            logger.error(f"Failed to get info for {series_id}: {e}")
            return {}
