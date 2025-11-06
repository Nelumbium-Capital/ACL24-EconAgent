#!/usr/bin/env python3
"""
Simple web interface for EconAgent-Light with real FRED data integration.
No numpy dependencies - uses the working real economic model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import json
import time
from datetime import datetime
from src.mesa_model.simple_real_model import RealEconModel

def create_simple_chart_data(results):
    """Create simple chart data for display."""
    if not results:
        return {}
    
    return {
        'steps': [r['step'] for r in results],
        'gdp': [r['gdp'] for r in results],
        'unemployment': [r['unemployment'] * 100 for r in results],
        'inflation': [r['inflation'] * 100 for r in results],
        'interest_rate': [r['interest_rate'] * 100 for r in results],
        'real_unemployment': [r['real_unemployment'] * 100 for r in results],
        'real_fed_funds': [r['real_fed_funds'] * 100 for r in results]
    }

def run_simulation_simple(n_agents, years, seed):
    """Run simulation with simple progress tracking."""
    print(f"🏦 Starting Economic Simulation with Real FRED Data")
    print(f"📊 Agents: {n_agents}, Years: {years}, Seed: {seed}")
    print("=" * 60)
    
    # Create model with real FRED data
    model = RealEconModel(
        n_agents=n_agents,
        episode_length=years * 12,
        random_seed=seed,
        enable_real_data=True,
        real_data_update_frequency=6  # Update every 6 months
    )
    
    print(f"✅ Model initialized with real FRED data")
    print(f"✅ Real unemployment rate: {model.real_unemployment_rate:.1%}")
    print(f"✅ Real Fed funds rate: {model.real_fed_funds_rate:.1%}")
    print(f"✅ Real CPI level: {model.real_cpi_level:.1f}")
    print()
    
    # Run simulation
    total_steps = years * 12
    for step in range(total_steps):
        model.step()
        
        # Progress updates
        if (step + 1) % 6 == 0 or step == 0:
            data = model.model_data[-1]
            print(f"📈 Month {step + 1:2d}: GDP=${data['gdp']:6.0f}, "
                  f"Unemployment={data['unemployment']:5.1%}, "
                  f"Inflation={data['inflation']:6.1%}, "
                  f"Interest Rate={data['interest_rate']:5.1%}")
        
        if not model.running:
            break
    
    print()
    print("🎉 Simulation completed!")
    
    # Get results
    results = model.get_results()
    summary = model.get_summary_stats()
    
    return results, summary

def main_cli():
    """Command line interface."""
    print("=" * 80)
    print("🏦 ECONAGENT-LIGHT WITH REAL FRED DATA")
    print("=" * 80)
    print()
    
    # Get user input
    try:
        n_agents = int(input("Number of agents (default 50): ") or "50")
        years = int(input("Number of years (default 2): ") or "2")
        seed = int(input("Random seed (default 42): ") or "42")
    except ValueError:
        print("Using default values...")
        n_agents, years, seed = 50, 2, 42
    
    # Run simulation
    results, summary = run_simulation_simple(n_agents, years, seed)
    
    # Display results
    print()
    print("=" * 80)
    print("📊 SIMULATION RESULTS")
    print("=" * 80)
    print(f"Simulation Length: {summary['simulation_length']} months")
    print(f"Final GDP: ${summary['final_gdp']:,.2f}")
    print(f"Average Unemployment: {summary['avg_unemployment']:.1%}")
    print(f"Average Inflation: {summary['avg_inflation']:.1%}")
    print(f"Final Gini Coefficient: {summary['final_gini']:.3f}")
    print(f"Total Agents: {summary['total_agents']}")
    print()
    print("🔍 REAL DATA COMPARISON:")
    print(f"Real Unemployment Rate: {summary['real_unemployment']:.1%}")
    print(f"Real Fed Funds Rate: {summary['real_fed_funds']:.1%}")
    print(f"Real Data Integration: {'✅ Enabled' if summary['real_data_enabled'] else '❌ Disabled'}")
    print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': summary,
            'parameters': {
                'n_agents': n_agents,
                'years': years,
                'seed': seed
            }
        }, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    print("=" * 80)

def main_streamlit():
    """Streamlit web interface."""
    st.set_page_config(
        page_title="EconAgent-Light with Real FRED Data",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 EconAgent-Light with Real FRED Data")
    st.markdown("**Economic Simulation with Live Federal Reserve Data**")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Parameters")
    
    n_agents = st.sidebar.slider("Number of Agents", 10, 200, 50, 10)
    years = st.sidebar.slider("Simulation Years", 1, 5, 2, 1)
    seed = st.sidebar.number_input("Random Seed", 1, 9999, 42, 1)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running economic simulation with real FRED data..."):
            # Create model
            model = RealEconModel(
                n_agents=n_agents,
                episode_length=years * 12,
                random_seed=seed,
                enable_real_data=True
            )
            
            # Display real data info
            st.success(f"✅ Real FRED data loaded!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Real Unemployment", f"{model.real_unemployment_rate:.1%}")
            with col2:
                st.metric("Real Fed Funds Rate", f"{model.real_fed_funds_rate:.1%}")
            with col3:
                st.metric("Real CPI Level", f"{model.real_cpi_level:.1f}")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation
            total_steps = years * 12
            for step in range(total_steps):
                model.step()
                
                # Update progress
                progress = (step + 1) / total_steps
                progress_bar.progress(progress)
                status_text.text(f"Running... Month {step + 1}/{total_steps}")
                
                if not model.running:
                    break
            
            progress_bar.progress(1.0)
            status_text.text("✅ Simulation completed!")
            
            # Store results in session state
            st.session_state['results'] = model.get_results()
            st.session_state['summary'] = model.get_summary_stats()
    
    # Display results
    if 'results' in st.session_state and 'summary' in st.session_state:
        results = st.session_state['results']
        summary = st.session_state['summary']
        
        st.markdown("## 📊 Simulation Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final GDP", f"${summary['final_gdp']:,.0f}")
        with col2:
            st.metric("Avg Unemployment", f"{summary['avg_unemployment']:.1%}")
        with col3:
            st.metric("Avg Inflation", f"{summary['avg_inflation']:.1%}")
        with col4:
            st.metric("Final Gini", f"{summary['final_gini']:.3f}")
        
        # Charts
        chart_data = create_simple_chart_data(results)
        
        if chart_data:
            # GDP Chart
            st.markdown("### GDP Over Time")
            st.line_chart({
                'GDP': chart_data['gdp']
            })
            
            # Economic Indicators
            st.markdown("### Economic Indicators")
            st.line_chart({
                'Unemployment (%)': chart_data['unemployment'],
                'Inflation (%)': chart_data['inflation'],
                'Interest Rate (%)': chart_data['interest_rate']
            })
            
            # Real vs Simulated Comparison
            st.markdown("### Real vs Simulated Data")
            st.line_chart({
                'Simulated Unemployment (%)': chart_data['unemployment'],
                'Real Unemployment (%)': chart_data['real_unemployment'],
                'Simulated Interest Rate (%)': chart_data['interest_rate'],
                'Real Fed Funds Rate (%)': chart_data['real_fed_funds']
            })
        
        # Real data comparison
        st.markdown("### 🔍 Real Data Integration")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Real Economic Data (from FRED):**")
            st.write(f"• Unemployment Rate: {summary['real_unemployment']:.1%}")
            st.write(f"• Fed Funds Rate: {summary['real_fed_funds']:.1%}")
            st.write(f"• Data Integration: {'✅ Enabled' if summary['real_data_enabled'] else '❌ Disabled'}")
        
        with col2:
            st.write("**Simulation Results:**")
            st.write(f"• Avg Unemployment: {summary['avg_unemployment']:.1%}")
            st.write(f"• Final GDP: ${summary['final_gdp']:,.2f}")
            st.write(f"• Simulation Length: {summary['simulation_length']} months")
        
        # Raw data
        with st.expander("📋 Raw Simulation Data"):
            st.json(results[-5:] if len(results) > 5 else results)  # Show last 5 records
    
    else:
        # Welcome message
        st.markdown("## 🎯 Welcome!")
        st.markdown("""
        This is **EconAgent-Light** with real Federal Reserve Economic Data (FRED) integration.
        
        ### 🏦 Features:
        - **Real Economic Data**: Live unemployment, inflation, and interest rates from FRED
        - **ACL24-EconAgent Model**: Complete economic simulation based on the research paper
        - **Agent-Based Modeling**: Individual economic agents making decisions
        - **Economic Relationships**: Phillips Curve, Okun's Law, Taylor Rule
        
        ### 🚀 Getting Started:
        1. Set your simulation parameters in the sidebar
        2. Click "🚀 Run Simulation" to start
        3. View real-time results and comparisons with actual economic data
        
        **Your simulation will use live data from the Federal Reserve!**
        """)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        if STREAMLIT_AVAILABLE:
            main_streamlit()
        else:
            print("❌ Streamlit not available. Install with: pip install streamlit")
            print("🔄 Running CLI version instead...")
            main_cli()
    else:
        main_cli()