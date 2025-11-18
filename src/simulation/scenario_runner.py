"""
Scenario runner for orchestrating stress tests and KRI comparison.

Runs all scenarios (Baseline, Recession, Rate Shock, Credit Crisis),
collects KRI outputs, and generates comparison reports.
"""
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np
import json

from src.simulation.model import RiskSimulationModel
from src.simulation.scenarios import (
    BaselineScenario,
    RecessionScenario,
    InterestRateShockScenario,
    CreditCrisisScenario
)
from src.kri.calculator import KRICalculator
from src.utils.logging_config import logger


class ScenarioRunner:
    """
    Orchestrates stress test scenarios and KRI calculation.
    
    Runs multiple economic scenarios, computes KRIs for each,
    and generates comparison reports.
    """
    
    def __init__(
        self,
        n_banks: int = 10,
        n_firms: int = 50,
        n_workers: int = 20,
        n_steps: int = 100,
        use_llm_agents: bool = False,
        output_dir: str = "data/processed/scenarios"
    ):
        """
        Initialize scenario runner.
        
        Args:
            n_banks: Number of bank agents
            n_firms: Number of firm agents
            n_workers: Number of worker agents
            n_steps: Simulation steps per scenario
            use_llm_agents: Whether to use LLM-based agents
            output_dir: Directory for output files
        """
        self.n_banks = n_banks
        self.n_firms = n_firms
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.use_llm_agents = use_llm_agents
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize KRI calculator
        self.kri_calculator = KRICalculator()
        
        # Results storage
        self.scenario_results = {}
        self.scenario_kris = {}
        
        logger.info(f"Initialized ScenarioRunner with {n_banks} banks, {n_firms} firms, {n_workers} workers")
    
    def run_all_scenarios(
        self,
        scenarios: List[str] = None,
        random_seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all stress test scenarios.
        
        Args:
            scenarios: List of scenario names (None = all)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping scenario name to results DataFrame
        """
        if scenarios is None:
            scenarios = ['baseline', 'recession', 'rate_shock', 'credit_crisis']
        
        logger.info(f"Running {len(scenarios)} scenarios: {scenarios}")
        
        for scenario_name in scenarios:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running scenario: {scenario_name.upper()}")
            logger.info(f"{'='*70}")
            
            try:
                results = self.run_scenario(scenario_name, random_seed)
                self.scenario_results[scenario_name] = results
                
                # Compute KRIs for this scenario
                kris = self._compute_scenario_kris(scenario_name, results)
                self.scenario_kris[scenario_name] = kris
                
                logger.info(f"✓ Scenario '{scenario_name}' completed successfully")
                
            except Exception as e:
                logger.error(f"✗ Scenario '{scenario_name}' failed: {e}")
                self.scenario_results[scenario_name] = None
                self.scenario_kris[scenario_name] = {}
        
        # Generate comparison report
        self._generate_comparison_report()
        
        return self.scenario_results
    
    def run_scenario(
        self,
        scenario_name: str,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Run a single scenario.
        
        Args:
            scenario_name: Name of scenario ('baseline', 'recession', etc.)
            random_seed: Random seed
            
        Returns:
            DataFrame with simulation results
        """
        # Create scenario object
        scenario_map = {
            'baseline': BaselineScenario(),
            'recession': RecessionScenario(),
            'rate_shock': InterestRateShockScenario(),
            'credit_crisis': CreditCrisisScenario()
        }
        
        if scenario_name not in scenario_map:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = scenario_map[scenario_name]
        
        # Create and run model
        model = RiskSimulationModel(
            n_banks=self.n_banks,
            n_firms=self.n_firms,
            n_workers=self.n_workers,
            scenario=scenario,
            random_seed=random_seed,
            use_llm_agents=self.use_llm_agents
        )
        
        results = model.run_simulation(n_steps=self.n_steps)
        
        # Save results
        output_file = self.output_dir / f"{scenario_name}_results.csv"
        results.to_csv(output_file)
        logger.info(f"Saved results to {output_file}")
        
        return results
    
    def _compute_scenario_kris(
        self,
        scenario_name: str,
        results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute KRIs for a scenario's results.
        
        Args:
            scenario_name: Name of scenario
            results: Simulation results DataFrame
            
        Returns:
            Dictionary of KRI values
        """
        logger.info(f"Computing KRIs for scenario: {scenario_name}")
        
        # Extract final period values
        final_default_rate = results['default_rate'].iloc[-1]
        final_liquidity = results['system_liquidity'].iloc[-1]
        final_capital_ratio = results['avg_capital_ratio'].iloc[-1]
        avg_network_stress = results['network_stress'].mean()
        
        # Compute KRIs
        kris = {
            'default_rate': final_default_rate,
            'system_liquidity': final_liquidity,
            'avg_capital_ratio': final_capital_ratio,
            'network_stress': avg_network_stress,
            'max_default_rate': results['default_rate'].max(),
            'min_liquidity': results['system_liquidity'].min(),
            'scenario': scenario_name
        }
        
        logger.info(f"KRIs for {scenario_name}:")
        for kri, value in kris.items():
            if kri != 'scenario':
                logger.info(f"  {kri}: {value:.4f}")
        
        return kris
    
    def _generate_comparison_report(self):
        """Generate comparison report across all scenarios."""
        logger.info("\n" + "="*70)
        logger.info("SCENARIO COMPARISON REPORT")
        logger.info("="*70)
        
        if not self.scenario_kris:
            logger.warning("No scenario results to compare")
            return
        
        # Create comparison DataFrame
        kri_names = ['default_rate', 'system_liquidity', 'avg_capital_ratio', 'network_stress']
        comparison_data = []
        
        for scenario_name, kris in self.scenario_kris.items():
            row = {'Scenario': scenario_name}
            for kri_name in kri_names:
                row[kri_name] = kris.get(kri_name, np.nan)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison
        logger.info("\nKRI Comparison Table:")
        logger.info(comparison_df.to_string(index=False))
        
        # Calculate changes vs baseline
        if 'baseline' in self.scenario_kris:
            baseline_kris = self.scenario_kris['baseline']
            
            logger.info("\n" + "-"*70)
            logger.info("Changes vs Baseline:")
            logger.info("-"*70)
            
            for scenario_name, kris in self.scenario_kris.items():
                if scenario_name == 'baseline':
                    continue
                
                logger.info(f"\n{scenario_name.upper()}:")
                for kri_name in kri_names:
                    baseline_val = baseline_kris.get(kri_name, 0)
                    scenario_val = kris.get(kri_name, 0)
                    
                    if baseline_val != 0:
                        pct_change = ((scenario_val - baseline_val) / baseline_val) * 100
                        logger.info(f"  {kri_name}: {scenario_val:.4f} ({pct_change:+.1f}% vs baseline)")
                    else:
                        logger.info(f"  {kri_name}: {scenario_val:.4f} (baseline=0)")
        
        # Save comparison to file
        output_file = self.output_dir / "scenario_comparison.csv"
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Comparison saved to {output_file}")
        
        # Save detailed JSON
        json_file = self.output_dir / "scenario_kris.json"
        with open(json_file, 'w') as f:
            json.dump(self.scenario_kris, f, indent=2, default=str)
        logger.info(f"✓ Detailed KRIs saved to {json_file}")
    
    def get_scenario_summary(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a scenario.
        
        Args:
            scenario_name: Name of scenario
            
        Returns:
            Dictionary with summary statistics
        """
        if scenario_name not in self.scenario_results:
            raise ValueError(f"Scenario '{scenario_name}' not found. Run scenario first.")
        
        results = self.scenario_results[scenario_name]
        kris = self.scenario_kris.get(scenario_name, {})
        
        summary = {
            'scenario': scenario_name,
            'n_steps': len(results),
            'kris': kris,
            'statistics': {
                'default_rate': {
                    'mean': results['default_rate'].mean(),
                    'max': results['default_rate'].max(),
                    'final': results['default_rate'].iloc[-1]
                },
                'system_liquidity': {
                    'mean': results['system_liquidity'].mean(),
                    'min': results['system_liquidity'].min(),
                    'final': results['system_liquidity'].iloc[-1]
                },
                'avg_capital_ratio': {
                    'mean': results['avg_capital_ratio'].mean(),
                    'min': results['avg_capital_ratio'].min(),
                    'final': results['avg_capital_ratio'].iloc[-1]
                }
            }
        }
        
        return summary
    
    def plot_scenario_comparison(self, metric: str = 'default_rate'):
        """
        Plot comparison of scenarios for a specific metric.
        
        Args:
            metric: Metric to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for scenario_name, results in self.scenario_results.items():
                if results is not None and metric in results.columns:
                    ax.plot(results.index, results[metric], label=scenario_name, linewidth=2)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Scenario Comparison: {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = self.output_dir / f"comparison_{metric}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Plot saved to {plot_file}")
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def export_results(self, format: str = 'csv'):
        """
        Export all scenario results.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for scenario_name, results in self.scenario_results.items():
            if results is None:
                continue
            
            filename = f"{scenario_name}_{timestamp}"
            
            if format == 'csv':
                output_file = self.output_dir / f"{filename}.csv"
                results.to_csv(output_file)
            elif format == 'json':
                output_file = self.output_dir / f"{filename}.json"
                results.to_json(output_file, orient='records', indent=2)
            elif format == 'excel':
                output_file = self.output_dir / f"{filename}.xlsx"
                results.to_excel(output_file)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"✓ Exported {scenario_name} to {output_file}")


def main():
    """Run scenario comparison demo."""
    logger.info("Starting Scenario Runner Demo")
    
    # Create runner
    runner = ScenarioRunner(
        n_banks=10,
        n_firms=50,
        n_workers=20,
        n_steps=100,
        use_llm_agents=False  # Set to True to enable LLM agents
    )
    
    # Run all scenarios
    results = runner.run_all_scenarios()
    
    # Generate plots
    for metric in ['default_rate', 'system_liquidity', 'avg_capital_ratio']:
        runner.plot_scenario_comparison(metric)
    
    # Export results
    runner.export_results(format='csv')
    
    logger.info("\n" + "="*70)
    logger.info("✓ Scenario Runner Demo Complete")
    logger.info("="*70)


if __name__ == "__main__":
    main()


