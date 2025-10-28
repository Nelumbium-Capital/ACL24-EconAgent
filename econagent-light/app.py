#!/usr/bin/env python3
"""
EconAgent-Light Web UI
Beautiful Streamlit interface for running economic simulations and viewing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure page
st.set_page_config(
    page_title="EconAgent-Light",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #ff6b6b;
        font-weight: bold;
    }
    .status-completed {
        color: #51cf66;
        font-weight: bold;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_simulation_results(results_file):
    """Load simulation results with caching."""
    try:
        with pd.ExcelFile(results_file) as xls:
            model_data = pd.read_excel(xls, sheet_name='Model_Data', index_col=0)
            agent_data = pd.read_excel(xls, sheet_name='Agent_Data', index_col=0)
        return model_data, agent_data
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None, None

def create_economic_indicators_plot(model_data):
    """Create economic indicators dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDP Over Time', 'Unemployment Rate', 'Inflation Rate', 'Interest Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GDP
    fig.add_trace(
        go.Scatter(x=model_data['Year'], y=model_data['GDP'], 
                  name='GDP', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Unemployment
    fig.add_trace(
        go.Scatter(x=model_data['Year'], y=model_data['Unemployment']*100, 
                  name='Unemployment', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Inflation
    fig.add_trace(
        go.Scatter(x=model_data['Year'], y=model_data['Inflation']*100, 
                  name='Inflation', line=dict(color='green', width=3)),
        row=2, col=1
    )
    
    # Interest Rate
    fig.add_trace(
        go.Scatter(x=model_data['Year'], y=model_data['Interest_Rate']*100, 
                  name='Interest Rate', line=dict(color='purple', width=3)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="GDP ($)", row=1, col=1)
    fig.update_yaxes(title_text="Unemployment (%)", row=1, col=2)
    fig.update_yaxes(title_text="Inflation (%)", row=2, col=1)
    fig.update_yaxes(title_text="Interest Rate (%)", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Economic Indicators Dashboard",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig

def create_phillips_curve(model_data):
    """Create Phillips curve plot."""
    fig = px.scatter(
        x=model_data['Unemployment']*100,
        y=model_data['Inflation']*100,
        color=model_data['Year'],
        title="Phillips Curve: Inflation vs Unemployment",
        labels={'x': 'Unemployment Rate (%)', 'y': 'Inflation Rate (%)', 'color': 'Year'},
        color_continuous_scale='viridis'
    )
    
    # Add trend line with error handling
    try:
        z = np.polyfit(model_data['Unemployment']*100, model_data['Inflation']*100, 1)
        x_trend = np.linspace(model_data['Unemployment'].min()*100, model_data['Unemployment'].max()*100, 100)
        y_trend = z[0] * x_trend + z[1]
        
        fig.add_trace(
            go.Scatter(x=x_trend, y=y_trend, mode='lines', 
                      name=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}',
                      line=dict(color='red', dash='dash', width=2))
        )
    except np.linalg.LinAlgError:
        # SVD didn't converge - skip trend line
        fig.add_annotation(text="Trend line unavailable (insufficient variation)", 
                         xref="paper", yref="paper", x=0.5, y=0.1)
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def create_okun_law_plot(model_data):
    """Create Okun's law plot."""
    gdp_growth = model_data['GDP'].pct_change() * 100
    unemployment_change = model_data['Unemployment'].diff() * 100
    
    # Remove NaN values
    valid_mask = ~(gdp_growth.isna() | unemployment_change.isna())
    
    if valid_mask.sum() > 1:
        fig = px.scatter(
            x=gdp_growth[valid_mask],
            y=unemployment_change[valid_mask],
            color=model_data['Year'][valid_mask],
            title="Okun's Law: GDP Growth vs Unemployment Change",
            labels={'x': 'GDP Growth Rate (%)', 'y': 'Change in Unemployment Rate (pp)', 'color': 'Year'},
            color_continuous_scale='plasma'
        )
        
        # Add trend line with error handling
        try:
            z = np.polyfit(gdp_growth[valid_mask], unemployment_change[valid_mask], 1)
            x_trend = np.linspace(gdp_growth[valid_mask].min(), gdp_growth[valid_mask].max(), 100)
            y_trend = z[0] * x_trend + z[1]
            
            fig.add_trace(
                go.Scatter(x=x_trend, y=y_trend, mode='lines',
                          name=f'Okun Coefficient: {z[0]:.3f}',
                          line=dict(color='red', dash='dash', width=2))
            )
        except np.linalg.LinAlgError:
            # SVD didn't converge - skip trend line
            fig.add_annotation(text="Trend line unavailable (insufficient variation)", 
                             xref="paper", yref="paper", x=0.5, y=0.1)
        
        fig.update_layout(height=500, showlegend=True)
        return fig
    else:
        return go.Figure().add_annotation(text="Insufficient data for Okun's Law", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)

def create_wealth_distribution_plot(model_data, agent_data):
    """Create wealth distribution plots."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Gini Coefficient Over Time', 'Final Wealth Distribution')
    )
    
    # Gini coefficient over time
    fig.add_trace(
        go.Scatter(x=model_data['Year'], y=model_data['Gini_Coefficient'],
                  name='Gini Coefficient', line=dict(color='purple', width=3)),
        row=1, col=1
    )
    
    # Wealth distribution histogram
    if agent_data is not None and len(agent_data) > 0:
        # Handle Mesa's MultiIndex DataFrame structure
        if hasattr(agent_data.index, 'levels'):
            # MultiIndex case - get the final step
            final_step = agent_data.index.get_level_values(0).max()
            final_wealth = agent_data.loc[final_step]['Wealth']
        else:
            # Regular index case
            final_wealth = agent_data['Wealth']
        
        fig.add_trace(
            go.Histogram(x=final_wealth, nbinsx=30, name='Wealth Distribution',
                        marker_color='skyblue', opacity=0.7),
            row=1, col=2
        )
        
        # Add mean and median lines
        mean_wealth = final_wealth.mean()
        median_wealth = final_wealth.median()
        
        fig.add_vline(x=mean_wealth, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: ${mean_wealth:.0f}", row=1, col=2)
        fig.add_vline(x=median_wealth, line_dash="dash", line_color="orange",
                     annotation_text=f"Median: ${median_wealth:.0f}", row=1, col=2)
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Wealth ($)", row=1, col=2)
    fig.update_yaxes(title_text="Gini Coefficient", row=1, col=1)
    fig.update_yaxes(title_text="Number of Agents", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=False, title_text="Wealth Distribution Analysis", title_x=0.5)
    return fig

def create_agent_behavior_plot(model_data, agent_data):
    """Create agent behavior analysis plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Work Propensity', 'Average Consumption Propensity', 
                       'Work vs Consumption (Final)', 'Employment Rate')
    )
    
    if agent_data is not None and len(agent_data) > 0:
        # Handle Mesa's MultiIndex DataFrame structure
        if hasattr(agent_data.index, 'levels'):
            # MultiIndex case - group by step level
            work_by_step = agent_data.groupby(level=0)['Last_Work'].mean()
            consumption_by_step = agent_data.groupby(level=0)['Last_Consumption'].mean()
            
            fig.add_trace(
                go.Scatter(x=work_by_step.index, y=work_by_step.values,
                          name='Work Propensity', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=consumption_by_step.index, y=consumption_by_step.values,
                          name='Consumption Propensity', line=dict(color='green', width=2)),
                row=1, col=2
            )
            
            # Work vs Consumption scatter - final step
            final_step = agent_data.index.get_level_values(0).max()
            final_data = agent_data.loc[final_step]
        else:
            # Regular DataFrame case
            final_data = agent_data
        
        fig.add_trace(
            go.Scatter(x=final_data['Last_Work'], y=final_data['Last_Consumption'],
                      mode='markers', marker=dict(color=final_data['Wealth'], 
                      colorscale='viridis', size=8, opacity=0.7),
                      name='Agents', showlegend=False),
            row=2, col=1
        )
    
    # Employment rate
    if 'Employment_Rate' in model_data.columns:
        fig.add_trace(
            go.Scatter(x=model_data['Step'], y=model_data['Employment_Rate'],
                      name='Employment Rate', line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Agent Behavior Analysis", title_x=0.5)
    return fig

def run_simulation(params):
    """Run simulation with given parameters."""
    try:
        # Use the main Mesa model
        from src.mesa_model.model import EconModel
        
        # Create model with original parameters
        model = EconModel(
            n_agents=params['n_agents'],
            episode_length=params['years'] * 12,
            random_seed=params['seed'],
            # Original economic parameters
            productivity=params.get('productivity', 1.0),
            max_price_inflation=params.get('max_inflation', 0.10),
            max_wage_inflation=0.05,
            pareto_param=8.0,  # Original value
            payment_max_skill_multiplier=950.0,  # Original value
            labor_hours=168,  # Original value
            # Disable LLM features for web UI
            llm_client=None,
            enable_lightagent=False,
            log_frequency=max(1, params['years'] * 12 // 10)
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulation
        total_steps = params['years'] * 12
        for step in range(total_steps):
            model.step()
            
            # Update progress
            progress = (step + 1) / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Running simulation... Step {step + 1}/{total_steps} "
                           f"(Year {(step // 12) + 1}, Month {(step % 12) + 1})")
            
            if not model.running:
                break
        
        # Save results
        results_dir = Path("./web_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"simulation_{timestamp}.xlsx"
        
        model.save_results(str(results_file))
        
        progress_bar.progress(1.0)
        status_text.text("✅ Simulation completed successfully!")
        
        return str(results_file), model.get_summary_stats()
        
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None, None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 EconAgent-Light Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Modern Economic Simulation with AI Agents**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Simulation Controls")
    
    # Simulation parameters
    with st.sidebar.expander("📋 Simulation Parameters", expanded=True):
        n_agents = st.slider("Number of Agents", min_value=10, max_value=200, value=50, step=10)
        years = st.slider("Simulation Years", min_value=1, max_value=10, value=3, step=1)
        seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
    
    # Advanced parameters
    with st.sidebar.expander("⚙️ Advanced Settings"):
        st.info("Advanced economic parameters coming soon!")
        productivity = st.slider("Productivity", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        max_inflation = st.slider("Max Inflation", min_value=0.05, max_value=0.20, value=0.10, step=0.01)
    
    # Run simulation button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        params = {
            'n_agents': n_agents,
            'years': years,
            'seed': seed,
            'productivity': productivity,
            'max_inflation': max_inflation
        }
        
        with st.spinner("Running economic simulation..."):
            results_file, summary = run_simulation(params)
            
            if results_file:
                st.session_state['results_file'] = results_file
                st.session_state['summary'] = summary
                st.success("🎉 Simulation completed! Check the results below.")
                st.rerun()
    
    # Load existing results
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📁 Load Results")
    
    # List available results
    results_dir = Path("./web_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.xlsx"))
        if result_files:
            selected_file = st.sidebar.selectbox(
                "Select Results File",
                options=[f.name for f in result_files],
                index=0
            )
            
            if st.sidebar.button("📊 Load Results", use_container_width=True):
                st.session_state['results_file'] = str(results_dir / selected_file)
                st.rerun()
    
    # Main content area
    if 'results_file' in st.session_state:
        results_file = st.session_state['results_file']
        
        # Load data
        with st.spinner("Loading simulation results..."):
            model_data, agent_data = load_simulation_results(results_file)
        
        if model_data is not None:
            # Summary metrics
            st.markdown("## 📈 Simulation Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                final_gdp = model_data['GDP'].iloc[-1]
                st.metric("Final GDP", f"${final_gdp:,.0f}", 
                         delta=f"{model_data['GDP'].pct_change().iloc[-1]*100:.1f}%")
            
            with col2:
                avg_unemployment = model_data['Unemployment'].mean() * 100
                st.metric("Avg Unemployment", f"{avg_unemployment:.1f}%")
            
            with col3:
                avg_inflation = model_data['Inflation'].mean() * 100
                st.metric("Avg Inflation", f"{avg_inflation:.1f}%")
            
            with col4:
                final_gini = model_data['Gini_Coefficient'].iloc[-1]
                st.metric("Final Gini", f"{final_gini:.3f}")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Economic Indicators", "📈 Phillips Curve", "📉 Okun's Law", 
                "💰 Wealth Distribution", "👥 Agent Behavior", "📋 Raw Data"
            ])
            
            with tab1:
                st.markdown("### Economic Indicators Over Time")
                fig = create_economic_indicators_plot(model_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Key Statistics")
                    st.write(f"**Simulation Length:** {len(model_data)} months")
                    st.write(f"**GDP Growth Rate:** {model_data['GDP'].pct_change().mean()*100:.2f}% per month")
                    st.write(f"**Unemployment Range:** {model_data['Unemployment'].min()*100:.1f}% - {model_data['Unemployment'].max()*100:.1f}%")
                    st.write(f"**Inflation Range:** {model_data['Inflation'].min()*100:.1f}% - {model_data['Inflation'].max()*100:.1f}%")
                
                with col2:
                    st.markdown("#### Economic Correlations")
                    corr_matrix = model_data[['GDP', 'Unemployment', 'Inflation', 'Interest_Rate']].corr()
                    st.dataframe(corr_matrix.round(3))
            
            with tab2:
                st.markdown("### Phillips Curve Analysis")
                fig = create_phillips_curve(model_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Phillips curve statistics
                phillips_corr = np.corrcoef(model_data['Unemployment'], model_data['Inflation'])[0, 1]
                st.info(f"**Phillips Curve Correlation:** {phillips_corr:.3f}")
                
                if phillips_corr < -0.3:
                    st.success("✅ Strong negative correlation observed (consistent with Phillips Curve theory)")
                elif phillips_corr < 0:
                    st.warning("⚠️ Weak negative correlation observed")
                else:
                    st.error("❌ Positive correlation observed (inconsistent with theory)")
            
            with tab3:
                st.markdown("### Okun's Law Analysis")
                fig = create_okun_law_plot(model_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Okun's law statistics
                if len(model_data) > 1:
                    gdp_growth = model_data['GDP'].pct_change()
                    unemployment_change = model_data['Unemployment'].diff()
                    valid_mask = ~(gdp_growth.isna() | unemployment_change.isna())
                    
                    if valid_mask.sum() > 1:
                        try:
                            okun_coeff = np.polyfit(gdp_growth[valid_mask], unemployment_change[valid_mask], 1)[0]
                            st.info(f"**Okun's Coefficient:** {okun_coeff:.3f}")
                            
                            if -0.6 < okun_coeff < -0.2:
                                st.success("✅ Coefficient within expected range (-0.2 to -0.6)")
                            else:
                                st.warning("⚠️ Coefficient outside typical range")
                        except np.linalg.LinAlgError:
                            st.warning("⚠️ Okun's coefficient could not be calculated (insufficient data variation)")
            
            with tab4:
                st.markdown("### Wealth Distribution Analysis")
                fig = create_wealth_distribution_plot(model_data, agent_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Wealth statistics
                if agent_data is not None:
                    # Handle Mesa's MultiIndex DataFrame structure
                    if hasattr(agent_data.index, 'levels'):
                        final_step = agent_data.index.get_level_values(0).max()
                        final_wealth = agent_data.loc[final_step]['Wealth']
                    else:
                        final_wealth = agent_data['Wealth']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Wealth Statistics")
                        st.write(f"**Mean Wealth:** ${final_wealth.mean():.2f}")
                        st.write(f"**Median Wealth:** ${final_wealth.median():.2f}")
                        st.write(f"**Wealth Std Dev:** ${final_wealth.std():.2f}")
                        st.write(f"**Min Wealth:** ${final_wealth.min():.2f}")
                        st.write(f"**Max Wealth:** ${final_wealth.max():.2f}")
                    
                    with col2:
                        st.markdown("#### Inequality Metrics")
                        gini_final = model_data['Gini_Coefficient'].iloc[-1]
                        gini_initial = model_data['Gini_Coefficient'].iloc[0]
                        st.write(f"**Final Gini:** {gini_final:.3f}")
                        st.write(f"**Initial Gini:** {gini_initial:.3f}")
                        st.write(f"**Gini Change:** {gini_final - gini_initial:+.3f}")
                        
                        if gini_final < 0.3:
                            st.success("✅ Low inequality")
                        elif gini_final < 0.5:
                            st.warning("⚠️ Moderate inequality")
                        else:
                            st.error("❌ High inequality")
            
            with tab5:
                st.markdown("### Agent Behavior Analysis")
                fig = create_agent_behavior_plot(model_data, agent_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Agent behavior statistics
                if agent_data is not None:
                    # Handle Mesa's MultiIndex DataFrame structure
                    if hasattr(agent_data.index, 'levels'):
                        final_step = agent_data.index.get_level_values(0).max()
                        final_agents = agent_data.loc[final_step]
                    else:
                        final_agents = agent_data
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Decision Patterns")
                        st.write(f"**Avg Work Propensity:** {final_agents['Last_Work'].mean():.3f}")
                        st.write(f"**Avg Consumption Propensity:** {final_agents['Last_Consumption'].mean():.3f}")
                        st.write(f"**Work Propensity Std:** {final_agents['Last_Work'].std():.3f}")
                        st.write(f"**Consumption Propensity Std:** {final_agents['Last_Consumption'].std():.3f}")
                    
                    with col2:
                        st.markdown("#### Employment Stats")
                        employment_rate = model_data['Employment_Rate'].iloc[-1] if 'Employment_Rate' in model_data.columns else 0
                        st.write(f"**Final Employment Rate:** {employment_rate:.1%}")
                        st.write(f"**Avg Employment Rate:** {model_data.get('Employment_Rate', pd.Series([0])).mean():.1%}")
            
            with tab6:
                st.markdown("### Raw Data")
                
                # Model data
                st.markdown("#### Model-Level Data")
                st.dataframe(model_data.round(4))
                
                # Download button for model data
                csv = model_data.to_csv(index=True)
                st.download_button(
                    label="📥 Download Model Data (CSV)",
                    data=csv,
                    file_name=f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Agent data (if available)
                if agent_data is not None:
                    st.markdown("#### Agent-Level Data")
                    st.dataframe(agent_data.head(100))  # Show first 100 rows
                    
                    if len(agent_data) > 100:
                        st.info(f"Showing first 100 rows of {len(agent_data)} total agent records")
                    
                    # Download button for agent data
                    csv_agents = agent_data.to_csv(index=True)
                    st.download_button(
                        label="📥 Download Agent Data (CSV)",
                        data=csv_agents,
                        file_name=f"agent_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        else:
            st.error("Failed to load simulation results. Please check the file format.")
    
    else:
        # Welcome screen
        st.markdown("## 🎯 Welcome to EconAgent-Light!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Modern Economic Simulation Platform
            
            EconAgent-Light is a cutting-edge economic simulation system that combines:
            
            - **🤖 AI-Powered Agents**: Intelligent economic agents with memory and reasoning
            - **📊 Mesa Framework**: Standard agent-based modeling with robust economics
            - **🎨 Beautiful Visualizations**: Interactive charts and comprehensive analysis
            - **⚡ Real-time Results**: Run simulations and see results instantly
            
            ### 📈 Key Features
            
            - **Economic Indicators**: GDP, unemployment, inflation, interest rates
            - **Phillips Curve**: Inflation vs unemployment relationship analysis
            - **Okun's Law**: GDP growth vs unemployment change correlation
            - **Wealth Distribution**: Inequality metrics and distribution analysis
            - **Agent Behavior**: Individual and aggregate decision patterns
            
            ### 🎛️ Getting Started
            
            1. **Set Parameters**: Use the sidebar to configure your simulation
            2. **Run Simulation**: Click "🚀 Run Simulation" to start
            3. **Analyze Results**: Explore the interactive dashboards and charts
            4. **Download Data**: Export results for further analysis
            
            ### 🚀 Quick Start
            
            - **Demo Mode**: Try 50 agents for 3 years (recommended for first run)
            - **Full Simulation**: 100+ agents for 5-10 years for comprehensive results
            - **Research Mode**: 200 agents for 10+ years for publication-quality data
            """)
        
        with col2:
            st.markdown("### 📊 Sample Results")
            
            # Create sample visualization
            sample_data = pd.DataFrame({
                'Year': range(1, 11),
                'GDP': np.cumsum(np.random.normal(100, 20, 10)) + 1000,
                'Unemployment': np.random.uniform(0.04, 0.12, 10),
                'Inflation': np.random.uniform(-0.02, 0.08, 10)
            })
            
            fig = px.line(sample_data, x='Year', y='GDP', 
                         title='Sample GDP Growth',
                         labels={'GDP': 'GDP ($)'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("👆 This is what your results will look like!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>EconAgent-Light Dashboard | Built with ❤️ using Streamlit</p>
        <p>Based on ACL24-EconAgent research | Enhanced with LightAgent framework</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()