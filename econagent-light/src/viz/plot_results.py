"""
Visualization and analysis tools for EconAgent-Light results.
Reproduces key figures from the original ACL24-EconAgent paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EconResultsAnalyzer:
    """Analyzer for EconAgent-Light simulation results."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to Excel file with simulation results
        """
        self.results_file = Path(results_file)
        self.model_data = None
        self.agent_data = None
        
        self._load_data()
    
    def _load_data(self):
        """Load simulation data from Excel file."""
        try:
            with pd.ExcelFile(self.results_file) as xls:
                self.model_data = pd.read_excel(xls, sheet_name='Model_Data', index_col=0)
                self.agent_data = pd.read_excel(xls, sheet_name='Agent_Data', index_col=0)
            
            logger.info(f"Loaded data: {len(self.model_data)} model steps, {len(self.agent_data)} agent records")
            
        except Exception as e:
            logger.error(f"Failed to load results file {self.results_file}: {e}")
            raise
    
    def plot_economic_indicators(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot key economic indicators over time.
        Reproduces Figure 2 from the original paper.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Economic Indicators Over Time', fontsize=16, fontweight='bold')
        
        # GDP over time
        axes[0, 0].plot(self.model_data['Year'], self.model_data['GDP'], 'b-', linewidth=2)
        axes[0, 0].set_title('GDP Growth')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('GDP ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Unemployment rate
        axes[0, 1].plot(self.model_data['Year'], self.model_data['Unemployment'] * 100, 'r-', linewidth=2)
        axes[0, 1].set_title('Unemployment Rate')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Unemployment (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Inflation rate
        axes[1, 0].plot(self.model_data['Year'], self.model_data['Inflation'] * 100, 'g-', linewidth=2)
        axes[1, 0].set_title('Inflation Rate')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Inflation (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Interest rate
        axes[1, 1].plot(self.model_data['Year'], self.model_data['Interest_Rate'] * 100, 'm-', linewidth=2)
        axes[1, 1].set_title('Interest Rate')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Interest Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved economic indicators plot to {save_path}")
        
        return fig
    
    def plot_phillips_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Phillips curve (inflation vs unemployment).
        Reproduces Figure 3 from the original paper.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create scatter plot with color gradient over time
        scatter = ax.scatter(
            self.model_data['Unemployment'] * 100,
            self.model_data['Inflation'] * 100,
            c=self.model_data['Year'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # Add trend line
        z = np.polyfit(self.model_data['Unemployment'] * 100, self.model_data['Inflation'] * 100, 1)
        p = np.poly1d(z)
        ax.plot(self.model_data['Unemployment'] * 100, p(self.model_data['Unemployment'] * 100), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        ax.set_xlabel('Unemployment Rate (%)', fontsize=12)
        ax.set_ylabel('Inflation Rate (%)', fontsize=12)
        ax.set_title('Phillips Curve: Inflation vs Unemployment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Phillips curve plot to {save_path}")
        
        return fig
    
    def plot_okun_law(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Okun's law (GDP growth vs unemployment change).
        Reproduces Figure 4 from the original paper.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate GDP growth rate and unemployment change
        gdp_growth = self.model_data['GDP'].pct_change() * 100
        unemployment_change = self.model_data['Unemployment'].diff() * 100
        
        # Remove NaN values
        valid_mask = ~(gdp_growth.isna() | unemployment_change.isna())
        gdp_growth_clean = gdp_growth[valid_mask]
        unemployment_change_clean = unemployment_change[valid_mask]
        
        # Create scatter plot
        scatter = ax.scatter(
            gdp_growth_clean,
            unemployment_change_clean,
            c=self.model_data['Year'][valid_mask],
            cmap='plasma',
            alpha=0.7,
            s=50
        )
        
        # Add trend line
        if len(gdp_growth_clean) > 1:
            z = np.polyfit(gdp_growth_clean, unemployment_change_clean, 1)
            p = np.poly1d(z)
            ax.plot(gdp_growth_clean, p(gdp_growth_clean), 
                    "r--", alpha=0.8, linewidth=2, label=f'Okun Coefficient: {z[0]:.3f}')
        
        ax.set_xlabel('GDP Growth Rate (%)', fontsize=12)
        ax.set_ylabel('Change in Unemployment Rate (pp)', fontsize=12)
        ax.set_title("Okun's Law: GDP Growth vs Unemployment Change", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Okun's law plot to {save_path}")
        
        return fig
    
    def plot_wealth_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot wealth distribution evolution.
        Reproduces Figure 5 from the original paper.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gini coefficient over time
        axes[0].plot(self.model_data['Year'], self.model_data['Gini_Coefficient'], 'purple', linewidth=2)
        axes[0].set_title('Wealth Inequality (Gini Coefficient)')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Gini Coefficient')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Wealth distribution histogram (final year)
        if self.agent_data is not None:
            final_step = self.agent_data['Step'].max()
            final_wealth = self.agent_data[self.agent_data['Step'] == final_step]['Wealth']
            
            axes[1].hist(final_wealth, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].set_title('Final Wealth Distribution')
            axes[1].set_xlabel('Wealth ($)')
            axes[1].set_ylabel('Number of Agents')
            axes[1].grid(True, alpha=0.3)
            
            # Add statistics
            mean_wealth = final_wealth.mean()
            median_wealth = final_wealth.median()
            axes[1].axvline(mean_wealth, color='red', linestyle='--', label=f'Mean: ${mean_wealth:.2f}')
            axes[1].axvline(median_wealth, color='orange', linestyle='--', label=f'Median: ${median_wealth:.2f}')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved wealth distribution plot to {save_path}")
        
        return fig
    
    def plot_agent_behavior(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot agent behavior patterns.
        Reproduces Figure 6 from the original paper.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Behavior Patterns', fontsize=16, fontweight='bold')
        
        if self.agent_data is not None:
            # Average work decisions over time
            work_by_step = self.agent_data.groupby('Step')['Last_Work'].mean()
            axes[0, 0].plot(work_by_step.index, work_by_step.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Average Work Propensity')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Work Propensity')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Average consumption decisions over time
            consumption_by_step = self.agent_data.groupby('Step')['Last_Consumption'].mean()
            axes[0, 1].plot(consumption_by_step.index, consumption_by_step.values, 'g-', linewidth=2)
            axes[0, 1].set_title('Average Consumption Propensity')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Consumption Propensity')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Work vs Consumption scatter (final step)
            final_step = self.agent_data['Step'].max()
            final_data = self.agent_data[self.agent_data['Step'] == final_step]
            
            scatter = axes[1, 0].scatter(
                final_data['Last_Work'],
                final_data['Last_Consumption'],
                c=final_data['Wealth'],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            axes[1, 0].set_title('Work vs Consumption Decisions (Final)')
            axes[1, 0].set_xlabel('Work Propensity')
            axes[1, 0].set_ylabel('Consumption Propensity')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Wealth ($)')
            
            # Employment rate over time
            employment_by_step = self.model_data['Employment_Rate']
            axes[1, 1].plot(self.model_data['Step'], employment_by_step, 'r-', linewidth=2)
            axes[1, 1].set_title('Employment Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Employment Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved agent behavior plot to {save_path}")
        
        return fig
    
    def generate_all_plots(self, output_dir: str = "./plots"):
        """
        Generate all standard plots and save to directory.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Generating all plots in {output_path}")
        
        # Generate each plot
        self.plot_economic_indicators(output_path / "economic_indicators.png")
        self.plot_phillips_curve(output_path / "phillips_curve.png")
        self.plot_okun_law(output_path / "okun_law.png")
        self.plot_wealth_distribution(output_path / "wealth_distribution.png")
        self.plot_agent_behavior(output_path / "agent_behavior.png")
        
        logger.info("All plots generated successfully")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the simulation.
        
        Returns:
            Dictionary of summary statistics
        """
        stats = {}
        
        # Model-level statistics
        stats['simulation_length'] = len(self.model_data)
        stats['final_gdp'] = self.model_data['GDP'].iloc[-1]
        stats['avg_gdp_growth'] = self.model_data['GDP'].pct_change().mean() * 100
        stats['avg_unemployment'] = self.model_data['Unemployment'].mean() * 100
        stats['avg_inflation'] = self.model_data['Inflation'].mean() * 100
        stats['final_gini'] = self.model_data['Gini_Coefficient'].iloc[-1]
        stats['avg_interest_rate'] = self.model_data['Interest_Rate'].mean() * 100
        
        # Phillips curve correlation
        phillips_corr = np.corrcoef(
            self.model_data['Unemployment'], 
            self.model_data['Inflation']
        )[0, 1]
        stats['phillips_correlation'] = phillips_corr
        
        # Okun's law coefficient
        if len(self.model_data) > 1:
            gdp_growth = self.model_data['GDP'].pct_change()
            unemployment_change = self.model_data['Unemployment'].diff()
            valid_mask = ~(gdp_growth.isna() | unemployment_change.isna())
            
            if valid_mask.sum() > 1:
                okun_coeff = np.polyfit(
                    gdp_growth[valid_mask], 
                    unemployment_change[valid_mask], 
                    1
                )[0]
                stats['okun_coefficient'] = okun_coeff
        
        # Agent-level statistics (if available)
        if self.agent_data is not None:
            final_step = self.agent_data['Step'].max()
            final_agents = self.agent_data[self.agent_data['Step'] == final_step]
            
            stats['final_avg_wealth'] = final_agents['Wealth'].mean()
            stats['final_median_wealth'] = final_agents['Wealth'].median()
            stats['final_wealth_std'] = final_agents['Wealth'].std()
            stats['avg_work_propensity'] = final_agents['Last_Work'].mean()
            stats['avg_consumption_propensity'] = final_agents['Last_Consumption'].mean()
        
        return stats
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """
        Compare results with baseline (original paper results).
        
        Args:
            baseline_file: Path to baseline results file
            
        Returns:
            Dictionary of comparison metrics
        """
        try:
            baseline_analyzer = EconResultsAnalyzer(baseline_file)
            
            current_stats = self.get_summary_statistics()
            baseline_stats = baseline_analyzer.get_summary_statistics()
            
            comparison = {}
            
            for key in current_stats:
                if key in baseline_stats:
                    current_val = current_stats[key]
                    baseline_val = baseline_stats[key]
                    
                    if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                        diff = current_val - baseline_val
                        pct_diff = (diff / baseline_val * 100) if baseline_val != 0 else 0
                        
                        comparison[key] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'difference': diff,
                            'percent_difference': pct_diff
                        }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare with baseline: {e}")
            return {}

def create_analysis_report(results_file: str, output_dir: str = "./analysis"):
    """
    Create comprehensive analysis report with plots and statistics.
    
    Args:
        results_file: Path to simulation results Excel file
        output_dir: Directory to save analysis outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Creating analysis report from {results_file}")
    
    # Initialize analyzer
    analyzer = EconResultsAnalyzer(results_file)
    
    # Generate plots
    plots_dir = output_path / "plots"
    analyzer.generate_all_plots(plots_dir)
    
    # Generate statistics
    stats = analyzer.get_summary_statistics()
    
    # Create report
    report_file = output_path / "analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("EconAgent-Light Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write(f"\nPlots saved to: {plots_dir}\n")
        f.write(f"Report generated: {pd.Timestamp.now()}\n")
    
    logger.info(f"Analysis report saved to {report_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results_file.xlsx> [output_dir]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./analysis"
    
    create_analysis_report(results_file, output_dir)