"""
FRED (Federal Reserve Economic Data) API client for EconAgent-Light.
Fetches real economic data to calibrate and validate simulations.
Enhanced for MVP with better error handling, data validation, and economic indicators.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path
import json
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EconomicSnapshot:
    """Current economic conditions snapshot from FRED data."""
    timestamp: datetime
    unemployment_rate: float
    inflation_rate: float
    fed_funds_rate: float
    gdp_growth: float
    wage_growth: float
    labor_participation: float
    consumer_sentiment: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'unemployment_rate': self.unemployment_rate,
            'inflation_rate': self.inflation_rate,
            'fed_funds_rate': self.fed_funds_rate,
            'gdp_growth': self.gdp_growth,
            'wage_growth': self.wage_growth,
            'labor_participation': self.labor_participation,
            'consumer_sentiment': self.consumer_sentiment
        }

@dataclass
class DataQualityReport:
    """Report on FRED data quality and completeness."""
    series_id: str
    total_observations: int
    missing_values: int
    data_quality_score: float
    last_updated: datetime
    anomalies_detected: List[str]
    
    def is_reliable(self) -> bool:
        """Check if data is reliable for simulation use."""
        return (
            self.data_quality_score >= 0.8 and
            self.missing_values / max(self.total_observations, 1) < 0.1 and
            len(self.anomalies_detected) < 3
        )

class FREDClient:
    """
    Client for fetching economic data from FRED API.
    """
    
    # Essential FRED series for economic simulation
    CORE_SERIES = {
        # GDP & Production
        'gdp': 'GDP',                    # Gross Domestic Product (Quarterly)
        'real_gdp': 'GDPC1',            # Real GDP (Quarterly)
        'potential_gdp': 'GDPPOT',      # Real Potential GDP (Quarterly)
        
        # Employment & Labor
        'unemployment': 'UNRATE',        # Unemployment Rate (Monthly)
        'labor_participation': 'CIVPART', # Labor Force Participation (Monthly)
        'employment': 'PAYEMS',          # Total Nonfarm Employment (Monthly)
        'wages': 'AHETPI',              # Average Hourly Earnings (Monthly)
        
        # Inflation & Prices
        'cpi': 'CPIAUCSL',              # Consumer Price Index (Monthly)
        'core_cpi': 'CPILFESL',         # Core CPI (Monthly)
        'pce': 'PCEPI',                 # PCE Price Index (Monthly)
        
        # Interest Rates & Monetary Policy
        'fed_funds': 'FEDFUNDS',        # Federal Funds Rate (Monthly)
        'treasury_10y': 'DGS10',        # 10-Year Treasury Rate (Daily)
        'treasury_3m': 'DGS3MO',        # 3-Month Treasury Rate (Daily)
        
        # Income & Wealth
        'median_income': 'MEHOINUSA672N', # Real Median Household Income (Annual)
        'disposable_income': 'DSPIC96',   # Real Disposable Personal Income (Monthly)
        'saving_rate': 'PSAVERT',         # Personal Saving Rate (Monthly)
        
        # Government & Fiscal
        'federal_deficit': 'FYFSGDA188S', # Federal Surplus/Deficit (Annual)
        'debt_to_gdp': 'GFDEGDQ188S',     # Federal Debt to GDP (Quarterly)
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.stlouisfed.org/fred",
        cache_dir: str = "./data_cache",
        cache_hours: int = 24,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3
    ):
        """
        Initialize FRED client with enhanced error handling and validation.
        
        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
            base_url: FRED API base URL
            cache_dir: Directory for caching data
            cache_hours: Hours to cache data before refresh
            rate_limit_delay: Delay between API requests (seconds)
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_hours = cache_hours
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for requests with timeout and retries
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Request statistics
        self.request_count = 0
        self.cache_hits = 0
        self.api_errors = 0
        
        if not api_key:
            logger.warning("No FRED API key provided. Using public access (limited requests).")
            logger.info("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        
        # Validate connection
        self._validate_connection()
    
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch a single economic time series from FRED.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with date index and series values
        """
        # Check cache first
        if use_cache:
            cached_data = self._get_cached_data(series_id, start_date, end_date)
            if cached_data is not None:
                self.cache_hits += 1
                logger.debug(f"Using cached data for {series_id}")
                return cached_data
        
        # Build API request
        params = {
            'series_id': series_id,
            'file_type': 'json',
            'sort_order': 'asc'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        if start_date:
            params['observation_start'] = start_date
        
        if end_date:
            params['observation_end'] = end_date
            
        if frequency:
            params['frequency'] = frequency
        
        # Try API request with retries
        for attempt in range(self.max_retries):
            try:
                # Make API request
                url = f"{self.base_url}/series/observations"
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                self.request_count += 1
                
                data = response.json()
                
                if 'observations' not in data:
                    raise ValueError(f"No data found for series {series_id}")
                
                # Convert to DataFrame
                observations = data['observations']
                df = pd.DataFrame(observations)
                
                # Clean and process data
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].set_index('date')
                df.columns = [series_id]
                
                # Remove missing values but warn if too many
                initial_count = len(df)
                df = df.dropna()
                final_count = len(df)
                
                if initial_count > 0 and (initial_count - final_count) / initial_count > 0.1:
                    logger.warning(f"Series {series_id} has {initial_count - final_count} missing values "
                                 f"({((initial_count - final_count) / initial_count) * 100:.1f}%)")
                
                # Validate data quality
                quality_report = self.validate_data_quality(series_id, df)
                if not quality_report.is_reliable():
                    logger.warning(f"Data quality issues detected for {series_id}: "
                                 f"score={quality_report.data_quality_score:.2f}, "
                                 f"anomalies={quality_report.anomalies_detected}")
                
                # Cache the data
                if use_cache:
                    self._cache_data(series_id, df, start_date, end_date)
                
                logger.info(f"Fetched {len(df)} observations for {series_id} "
                           f"(quality score: {quality_report.data_quality_score:.2f})")
                return df
                
            except requests.exceptions.RequestException as e:
                self.api_errors += 1
                logger.warning(f"API request failed for {series_id} (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for series {series_id}")
                    raise
                    
            except Exception as e:
                self.api_errors += 1
                logger.error(f"Failed to fetch series {series_id}: {e}")
                raise
    
    def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch multiple economic time series from FRED.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with date index and multiple series columns
        """
        dataframes = []
        
        for series_id in series_ids:
            try:
                df = self.get_series(series_id, start_date, end_date, frequency, use_cache)
                dataframes.append(df)
                
                # Rate limiting for API
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No series data could be fetched")
        
        # Combine all series
        combined_df = pd.concat(dataframes, axis=1, sort=True)
        
        logger.info(f"Fetched {len(series_ids)} series with {len(combined_df)} total observations")
        return combined_df
    
    def get_core_economic_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all core economic data series needed for simulation.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping series names to DataFrames
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching core economic data from {start_date} to {end_date}")
        
        data = {}
        
        for name, series_id in self.CORE_SERIES.items():
            try:
                df = self.get_series(series_id, start_date, end_date, use_cache=use_cache)
                data[name] = df
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {e}")
                continue
        
        logger.info(f"Successfully fetched {len(data)} core economic series")
        return data
    
    def _get_cached_data(
        self,
        series_id: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Check if cached data exists and is fresh."""
        cache_file = self.cache_dir / f"{series_id}_{start_date}_{end_date}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is fresh
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > timedelta(hours=self.cache_hours):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            logger.debug(f"Using cached data for {series_id}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load cached data for {series_id}: {e}")
            return None
    
    def _cache_data(
        self,
        series_id: str,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ):
        """Cache data to disk."""
        cache_file = self.cache_dir / f"{series_id}_{start_date}_{end_date}.json"
        
        try:
            # Convert DataFrame to JSON-serializable format
            data = {
                'series_id': series_id,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now().isoformat(),
                'data': df.reset_index().to_dict('records')
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug(f"Cached data for {series_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cache data for {series_id}: {e}")
    
    def get_series_info(self, series_id: str) -> Dict:
        """Get metadata about a FRED series."""
        params = {
            'series_id': series_id,
            'file_type': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            url = f"{self.base_url}/series"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data['seriess'][0] if 'seriess' in data else {}
            
        except Exception as e:
            logger.error(f"Failed to get info for series {series_id}: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict]:
        """Search for FRED series by text."""
        params = {
            'search_text': search_text,
            'file_type': 'json',
            'limit': limit
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            url = f"{self.base_url}/series/search"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('seriess', [])
            
        except Exception as e:
            logger.error(f"Failed to search series: {e}")
            return []
    
    def _validate_connection(self) -> bool:
        """Validate connection to FRED API."""
        try:
            # Test with a simple series info request
            params = {'series_id': 'GDP', 'file_type': 'json'}
            if self.api_key:
                params['api_key'] = self.api_key
                
            response = self.session.get(f"{self.base_url}/series", params=params, timeout=10)
            response.raise_for_status()
            
            logger.info("FRED API connection validated successfully")
            return True
            
        except Exception as e:
            logger.warning(f"FRED API connection validation failed: {e}")
            return False
    
    def get_current_economic_snapshot(self) -> EconomicSnapshot:
        """
        Get current economic snapshot with key indicators.
        
        Returns:
            EconomicSnapshot with current economic conditions
        """
        logger.info("Fetching current economic snapshot from FRED")
        
        # Get latest data for key indicators
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            # Fetch key indicators
            unemployment_df = self.get_series('UNRATE', start_date=start_date, end_date=end_date)
            fed_funds_df = self.get_series('FEDFUNDS', start_date=start_date, end_date=end_date)
            cpi_df = self.get_series('CPIAUCSL', start_date=start_date, end_date=end_date)
            wages_df = self.get_series('AHETPI', start_date=start_date, end_date=end_date)
            participation_df = self.get_series('CIVPART', start_date=start_date, end_date=end_date)
            
            # Calculate rates and growth
            unemployment_rate = unemployment_df['UNRATE'].iloc[-1] if not unemployment_df.empty else 5.0
            fed_funds_rate = fed_funds_df['FEDFUNDS'].iloc[-1] if not fed_funds_df.empty else 2.0
            labor_participation = participation_df['CIVPART'].iloc[-1] if not participation_df.empty else 63.0
            
            # Calculate inflation rate (year-over-year CPI change)
            if len(cpi_df) >= 12:
                current_cpi = cpi_df['CPIAUCSL'].iloc[-1]
                year_ago_cpi = cpi_df['CPIAUCSL'].iloc[-12]
                inflation_rate = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100
            else:
                inflation_rate = 2.0  # Default assumption
            
            # Calculate wage growth (year-over-year change)
            if len(wages_df) >= 12:
                current_wage = wages_df['AHETPI'].iloc[-1]
                year_ago_wage = wages_df['AHETPI'].iloc[-12]
                wage_growth = ((current_wage - year_ago_wage) / year_ago_wage) * 100
            else:
                wage_growth = 3.0  # Default assumption
            
            # GDP growth (quarterly, so approximate)
            gdp_growth = 2.5  # Default - would need quarterly data for accurate calculation
            
            snapshot = EconomicSnapshot(
                timestamp=datetime.now(),
                unemployment_rate=unemployment_rate,
                inflation_rate=inflation_rate,
                fed_funds_rate=fed_funds_rate,
                gdp_growth=gdp_growth,
                wage_growth=wage_growth,
                labor_participation=labor_participation
            )
            
            logger.info(f"Economic snapshot created: unemployment={unemployment_rate:.1f}%, "
                       f"inflation={inflation_rate:.1f}%, fed_funds={fed_funds_rate:.1f}%")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create economic snapshot: {e}")
            # Return default snapshot
            return EconomicSnapshot(
                timestamp=datetime.now(),
                unemployment_rate=5.0,
                inflation_rate=2.0,
                fed_funds_rate=2.0,
                gdp_growth=2.5,
                wage_growth=3.0,
                labor_participation=63.0
            )
    
    def validate_data_quality(self, series_id: str, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate data quality for a FRED series.
        
        Args:
            series_id: FRED series ID
            df: DataFrame with series data
            
        Returns:
            DataQualityReport with quality metrics
        """
        if df.empty:
            return DataQualityReport(
                series_id=series_id,
                total_observations=0,
                missing_values=0,
                data_quality_score=0.0,
                last_updated=datetime.now(),
                anomalies_detected=['No data available']
            )
        
        total_obs = len(df)
        missing_vals = df.isnull().sum().sum()
        
        # Detect anomalies
        anomalies = []
        values = df.iloc[:, 0].dropna()
        
        if len(values) > 0:
            # Check for extreme outliers (beyond 3 standard deviations)
            mean_val = values.mean()
            std_val = values.std()
            outliers = values[abs(values - mean_val) > 3 * std_val]
            
            if len(outliers) > 0:
                anomalies.append(f"{len(outliers)} extreme outliers detected")
            
            # Check for sudden jumps (>50% change)
            pct_changes = values.pct_change().abs()
            large_changes = pct_changes[pct_changes > 0.5]
            
            if len(large_changes) > 0:
                anomalies.append(f"{len(large_changes)} sudden value jumps detected")
        
        # Calculate quality score
        completeness_score = 1 - (missing_vals / total_obs) if total_obs > 0 else 0
        anomaly_penalty = min(len(anomalies) * 0.1, 0.3)
        quality_score = max(0, completeness_score - anomaly_penalty)
        
        return DataQualityReport(
            series_id=series_id,
            total_observations=total_obs,
            missing_values=missing_vals,
            data_quality_score=quality_score,
            last_updated=datetime.now(),
            anomalies_detected=anomalies
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        cache_hit_rate = (self.cache_hits / max(self.request_count, 1)) * 100
        
        return {
            'total_requests': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'api_errors': self.api_errors,
            'cache_directory': str(self.cache_dir),
            'cached_files': len(list(self.cache_dir.glob("*.json")))
        }
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cached data files.
        
        Args:
            older_than_hours: Only clear files older than this many hours
            
        Returns:
            Number of files cleared
        """
        cleared_count = 0
        cutoff_time = None
        
        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if cutoff_time:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time > cutoff_time:
                        continue
                
                cache_file.unlink()
                cleared_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count


# Example usage and testing
if __name__ == "__main__":
    # Initialize client (no API key needed for testing)
    fred = FREDClient()
    
    # Test fetching unemployment rate
    try:
        unemployment = fred.get_series('UNRATE', start_date='2020-01-01')
        print(f"Unemployment data: {len(unemployment)} observations")
        print(unemployment.head())
        
        # Test data quality validation
        quality_report = fred.validate_data_quality('UNRATE', unemployment)
        print(f"Data quality score: {quality_report.data_quality_score:.2f}")
        print(f"Is reliable: {quality_report.is_reliable()}")
        
        # Test economic snapshot
        snapshot = fred.get_current_economic_snapshot()
        print(f"Current economic snapshot:")
        print(f"  Unemployment: {snapshot.unemployment_rate:.1f}%")
        print(f"  Inflation: {snapshot.inflation_rate:.1f}%")
        print(f"  Fed Funds Rate: {snapshot.fed_funds_rate:.1f}%")
        
        # Test fetching multiple series
        core_data = fred.get_core_economic_data(start_date='2020-01-01')
        print(f"Core data: {len(core_data)} series fetched")
        
        # Show statistics
        stats = fred.get_statistics()
        print(f"Client statistics: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You may need a FRED API key for full functionality")