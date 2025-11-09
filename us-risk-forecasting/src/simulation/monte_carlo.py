"""
Monte Carlo simulation engine for stress testing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from .model import RiskSimulationModel
from .scenarios import EconomicScenario

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for running multiple simulation instances
    and computing probability distributions of outcomes.
    """
    
    def __init__(
        self,
        n_banks: int = 10,
        n_firms: int = 50,
        n_steps: int = 100
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_banks: Number of bank agents per simulation
            n_firms: Number of firm agents per simulation
            n_steps: Number of time steps per simulation
        """
        self.n_banks = n_banks
        self.n_firms = n_firms
        self.n_steps = n_steps
        
        logger.info(f"Initialized Monte Carlo engine: {n_banks} banks, {n_firms} firms, {n_steps} steps")
    
    def run_monte_carlo(
        self,
        scenario: EconomicScenario,
        n_simulations: int = 100,
        random_seed: Optional[int] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run Monte Carlo simulations with the given scenario.
        
        Args:
            scenario: Economic scenario to simulate
            n_simulations: Number of simulation runs
            random_seed: Base random seed for reproducibility
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers (None = auto)
            
        Returns:
            Dictionary containing:
                - 'all_runs': DataFrame with all simulation runs
                - 'summary_stats': DataFrame with summary statistics
                - 'distributions': DataFrame with probability distributions
        """
        logger.info(f"Starting Monte Carlo with {n_simulations} simulations for scenario: {scenario.name}")
        
        # Generate seeds for each simulation
        if random_seed is not None:
            np.random.seed(random_seed)
        
        seeds = [np.random.randint(0, 1000000) for _ in range(n_simulations)]
        
        # Run simulations
        if parallel and n_simulations > 1:
            results = self._run_parallel(scenario, seeds, max_workers)
        else:
            results = self._run_sequential(scenario, seeds)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        logger.info(f"Monte Carlo complete. Mean default rate: {aggregated['summary_stats']['default_rate']['mean']:.4f}")
        
        return aggregated
    
    def _run_sequential(
        self,
        scenario: EconomicScenario,
        seeds: List[int]
    ) -> List[pd.DataFrame]:
        """Run simulations sequentially."""
        results = []
        
        for i, seed in enumerate(seeds):
            logger.debug(f"Running simulation {i+1}/{len(seeds)}")
            result = self._run_single_simulation(scenario, seed, i)
            results.append(result)
        
        return results
    
    def _run_parallel(
        self,
        scenario: EconomicScenario,
        seeds: List[int],
        max_workers: Optional[int]
    ) -> List[pd.DataFrame]:
        """Run simulations in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            futures = {
                executor.submit(self._run_single_simulation, scenario, seed, i): i
                for i, seed in enumerate(seeds)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                sim_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed simulation {sim_id+1}/{len(seeds)}")
                except Exception as e:
                    logger.error(f"Simulation {sim_id} failed: {e}")
        
        return results
    
    def _run_single_simulation(
        self,
        scenario: EconomicScenario,
        seed: int,
        sim_id: int
    ) -> pd.DataFrame:
        """
        Run a single simulation instance.
        
        Args:
            scenario: Economic scenario
            seed: Random seed
            sim_id: Simulation identifier
            
        Returns:
            DataFrame with simulation results
        """
        # Create model
        model = RiskSimulationModel(
            n_banks=self.n_banks,
            n_firms=self.n_firms,
            scenario=scenario,
            random_seed=seed
        )
        
        # Run simulation
        results = model.run_simulation(n_steps=self.n_steps)
        
        # Add simulation ID
        results['simulation_id'] = sim_id
        
        return results
    
    def _aggregate_results(
        self,
        results: List[pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate results from multiple simulations.
        
        Args:
            results: List of DataFrames from individual simulations
            
        Returns:
            Dictionary with aggregated results
        """
        # Combine all runs
        all_runs = pd.concat(results, ignore_index=True)
        
        # Compute summary statistics for each metric at each time step
        metrics = [
            'system_liquidity',
            'default_rate',
            'network_stress',
            'avg_capital_ratio',
            'avg_liquidity_ratio'
        ]
        
        summary_stats = {}
        distributions = {}
        
        for metric in metrics:
            # Group by step and compute statistics
            grouped = all_runs.groupby(all_runs.index % self.n_steps)[metric]
            
            summary_stats[metric] = {
                'mean': grouped.mean().values,
                'median': grouped.median().values,
                'std': grouped.std().values,
                'min': grouped.min().values,
                'max': grouped.max().values,
                'p5': grouped.quantile(0.05).values,
                'p25': grouped.quantile(0.25).values,
                'p75': grouped.quantile(0.75).values,
                'p95': grouped.quantile(0.95).values
            }
            
            # Compute final distribution (last step)
            final_values = all_runs.groupby('simulation_id')[metric].last()
            distributions[metric] = final_values.values
        
        # Convert to DataFrames
        summary_df = self._create_summary_dataframe(summary_stats)
        dist_df = pd.DataFrame(distributions)
        
        return {
            'all_runs': all_runs,
            'summary_stats': summary_df,
            'distributions': dist_df
        }
    
    def _create_summary_dataframe(
        self,
        summary_stats: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """Create a structured DataFrame from summary statistics."""
        records = []
        
        for metric, stats in summary_stats.items():
            for stat_name, values in stats.items():
                for step, value in enumerate(values):
                    records.append({
                        'metric': metric,
                        'statistic': stat_name,
                        'step': step,
                        'value': value
                    })
        
        return pd.DataFrame(records)
    
    def compute_risk_metrics(
        self,
        distributions: pd.DataFrame,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute risk metrics from probability distributions.
        
        Args:
            distributions: DataFrame with final distributions from Monte Carlo
            confidence_levels: Confidence levels for VaR/CVaR calculations
            
        Returns:
            Dictionary of risk metrics per KRI
        """
        risk_metrics = {}
        
        for metric in distributions.columns:
            values = distributions[metric].dropna().values
            
            if len(values) == 0:
                continue
            
            risk_metrics[metric] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'skewness': float(self._compute_skewness(values)),
                'kurtosis': float(self._compute_kurtosis(values))
            }
            
            # Value at Risk (VaR) and Conditional VaR (CVaR)
            for cl in confidence_levels:
                var = np.percentile(values, (1 - cl) * 100)
                cvar_values = values[values <= var]
                cvar = np.mean(cvar_values) if len(cvar_values) > 0 else var
                
                risk_metrics[metric][f'var_{int(cl*100)}'] = float(var)
                risk_metrics[metric][f'cvar_{int(cl*100)}'] = float(cvar)
        
        return risk_metrics
    
    @staticmethod
    def _compute_skewness(values: np.ndarray) -> float:
        """Compute skewness of distribution."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)
    
    @staticmethod
    def _compute_kurtosis(values: np.ndarray) -> float:
        """Compute excess kurtosis of distribution."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3
    
    def compare_scenarios(
        self,
        scenarios: List[EconomicScenario],
        n_simulations: int = 100,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios using Monte Carlo simulation.
        
        Args:
            scenarios: List of scenarios to compare
            n_simulations: Number of simulations per scenario
            random_seed: Base random seed
            
        Returns:
            DataFrame comparing risk metrics across scenarios
        """
        logger.info(f"Comparing {len(scenarios)} scenarios")
        
        comparison_results = []
        
        for scenario in scenarios:
            # Run Monte Carlo for this scenario
            results = self.run_monte_carlo(
                scenario=scenario,
                n_simulations=n_simulations,
                random_seed=random_seed,
                parallel=True
            )
            
            # Compute risk metrics
            risk_metrics = self.compute_risk_metrics(results['distributions'])
            
            # Add scenario name to each metric
            for metric, values in risk_metrics.items():
                record = {'scenario': scenario.name, 'metric': metric}
                record.update(values)
                comparison_results.append(record)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        logger.info("Scenario comparison complete")
        
        return comparison_df


def run_stress_test(
    scenario: EconomicScenario,
    n_simulations: int = 100,
    n_banks: int = 10,
    n_firms: int = 50,
    n_steps: int = 100,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Convenience function to run a complete stress test.
    
    Args:
        scenario: Economic scenario to test
        n_simulations: Number of Monte Carlo simulations
        n_banks: Number of bank agents
        n_firms: Number of firm agents
        n_steps: Number of time steps
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (summary statistics DataFrame, risk metrics dictionary)
    """
    engine = MonteCarloEngine(
        n_banks=n_banks,
        n_firms=n_firms,
        n_steps=n_steps
    )
    
    results = engine.run_monte_carlo(
        scenario=scenario,
        n_simulations=n_simulations,
        random_seed=random_seed
    )
    
    risk_metrics = engine.compute_risk_metrics(results['distributions'])
    
    return results['summary_stats'], risk_metrics
