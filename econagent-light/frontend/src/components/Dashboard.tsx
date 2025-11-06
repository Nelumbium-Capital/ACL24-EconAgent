import React, { useState, useEffect } from 'react';
import { Plus, RefreshCw, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import toast from 'react-hot-toast';
import SimulationControls from './SimulationControls.tsx';
import EconomicCharts from './EconomicCharts.tsx';
import FREDDataPanel from './FREDDataPanel.tsx';
import SimulationsList from './SimulationsList.tsx';
import EconomicInsights from './EconomicInsights.tsx';
import { apiClient } from '../services/api.ts';
import { EconomicSnapshot, SimulationConfig, SimulationStatus } from '../types/index.ts';

const Dashboard: React.FC = () => {
  const [fredData, setFredData] = useState<EconomicSnapshot | null>(null);
  const [simulations, setSimulations] = useState<SimulationStatus[]>([]);
  const [activeSimulation, setActiveSimulation] = useState<SimulationStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreatingSimulation, setIsCreatingSimulation] = useState(false);

  // Fetch initial data
  useEffect(() => {
    loadDashboardData();
    
    // Set up polling for active simulations
    const interval = setInterval(() => {
      if (activeSimulation && activeSimulation.status === 'running') {
        refreshSimulationStatus(activeSimulation.simulation_id);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeSimulation]);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      
      // Load FRED data and simulations in parallel
      const [fredResponse, simulationsResponse] = await Promise.all([
        apiClient.getCurrentEconomicData(),
        apiClient.getSimulations()
      ]);
      
      setFredData(fredResponse);
      setSimulations(simulationsResponse.simulations);
      
      // Set active simulation to the most recent running one
      const runningSimulation = simulationsResponse.simulations.find(
        (sim: SimulationStatus) => sim.status === 'running'
      );
      if (runningSimulation) {
        setActiveSimulation(runningSimulation);
      }
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshSimulationStatus = async (simulationId: string) => {
    try {
      const status = await apiClient.getSimulationStatus(simulationId);
      
      // Update simulations list
      setSimulations(prev => 
        prev.map(sim => 
          sim.simulation_id === simulationId ? status : sim
        )
      );
      
      // Update active simulation
      if (activeSimulation?.simulation_id === simulationId) {
        setActiveSimulation(status);
        
        // Show completion notification
        if (status.status === 'completed' && activeSimulation.status === 'running') {
          toast.success(`Simulation "${status.name}" completed successfully!`);
        } else if (status.status === 'failed' && activeSimulation.status === 'running') {
          toast.error(`Simulation "${status.name}" failed: ${status.error_message}`);
        }
      }
      
    } catch (error) {
      console.error('Failed to refresh simulation status:', error);
    }
  };

  const handleCreateSimulation = async (config: SimulationConfig) => {
    try {
      setIsCreatingSimulation(true);
      
      // Create simulation
      const createResponse = await apiClient.createSimulation(config);
      toast.success(`Simulation "${config.name}" created successfully`);
      
      // Start simulation
      await apiClient.startSimulation(createResponse.simulation_id);
      toast.success('Simulation started');
      
      // Refresh data
      await loadDashboardData();
      
      // Set as active simulation
      const newSimulation = await apiClient.getSimulationStatus(createResponse.simulation_id);
      setActiveSimulation(newSimulation);
      
    } catch (error) {
      console.error('Failed to create simulation:', error);
      toast.error('Failed to create simulation');
    } finally {
      setIsCreatingSimulation(false);
    }
  };

  const handleRefreshFredData = async () => {
    try {
      const fredResponse = await apiClient.getCurrentEconomicData();
      setFredData(fredResponse);
      toast.success('FRED data refreshed');
    } catch (error) {
      console.error('Failed to refresh FRED data:', error);
      toast.error('Failed to refresh FRED data');
    }
  };

  const handleSimulationSelect = (simulation: SimulationStatus) => {
    setActiveSimulation(simulation);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-6 h-6 animate-spin text-primary-600" />
          <span className="text-lg text-gray-600">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Economic Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Real-time FRED data integration with agent-based economic modeling
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={loadDashboardData}
            disabled={isLoading}
            className="btn-secondary flex items-center"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh All
          </button>
          
          <button
            onClick={handleRefreshFredData}
            className="btn-primary flex items-center"
          >
            <TrendingUp className="w-4 h-4 mr-2" />
            Update FRED Data
          </button>
        </div>
      </div>

      {/* Enhanced Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="metric-card hover:shadow-md transition-shadow cursor-pointer">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Unemployment Rate</p>
              <p className="text-2xl font-bold text-gray-900">
                {fredData?.unemployment_rate.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Target: ≤ 4.0% | {fredData && fredData.unemployment_rate <= 4.0 ? '✅ On Target' : '⚠️ Above Target'}
              </p>
            </div>
            <div className={`p-2 rounded-lg ${
              fredData && fredData.unemployment_rate <= 4.0 ? 'bg-green-100' : 
              fredData && fredData.unemployment_rate <= 6.0 ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              {fredData && fredData.unemployment_rate <= 4.0 ? (
                <TrendingDown className="w-6 h-6 text-green-600" />
              ) : fredData && fredData.unemployment_rate <= 6.0 ? (
                <Minus className="w-6 h-6 text-yellow-600" />
              ) : (
                <TrendingUp className="w-6 h-6 text-red-600" />
              )}
            </div>
          </div>
        </div>

        <div className="metric-card hover:shadow-md transition-shadow cursor-pointer">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Inflation Rate</p>
              <p className="text-2xl font-bold text-gray-900">
                {fredData?.inflation_rate.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Target: 2.0% | {fredData && Math.abs(fredData.inflation_rate - 2.0) <= 0.5 ? '✅ On Target' : '⚠️ Off Target'}
              </p>
            </div>
            <div className={`p-2 rounded-lg ${
              fredData && Math.abs(fredData.inflation_rate - 2.0) <= 0.5 ? 'bg-green-100' : 
              fredData && Math.abs(fredData.inflation_rate - 2.0) <= 1.0 ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              {fredData && Math.abs(fredData.inflation_rate - 2.0) <= 0.5 ? (
                <TrendingDown className="w-6 h-6 text-green-600" />
              ) : (
                <TrendingUp className="w-6 h-6 text-yellow-600" />
              )}
            </div>
          </div>
        </div>

        <div className="metric-card hover:shadow-md transition-shadow cursor-pointer">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">GDP Growth</p>
              <p className="text-2xl font-bold text-gray-900">
                {fredData?.gdp_growth.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Target: ≥ 2.5% | {fredData && fredData.gdp_growth >= 2.5 ? '✅ Strong' : fredData && fredData.gdp_growth >= 1.0 ? '⚠️ Moderate' : '❌ Weak'}
              </p>
            </div>
            <div className={`p-2 rounded-lg ${
              fredData && fredData.gdp_growth >= 2.5 ? 'bg-green-100' : 
              fredData && fredData.gdp_growth >= 1.0 ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              {fredData && fredData.gdp_growth >= 2.5 ? (
                <TrendingUp className="w-6 h-6 text-green-600" />
              ) : fredData && fredData.gdp_growth >= 1.0 ? (
                <Minus className="w-6 h-6 text-yellow-600" />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-600" />
              )}
            </div>
          </div>
        </div>

        <div className="metric-card hover:shadow-md transition-shadow cursor-pointer">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Simulations</p>
              <p className="text-2xl font-bold text-gray-900">
                {simulations.filter(sim => sim.status === 'running').length}/{simulations.length}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Running/Total | {simulations.filter(sim => sim.status === 'completed').length} Completed
              </p>
            </div>
            <div className="p-2 bg-primary-100 rounded-lg">
              <RefreshCw className={`w-6 h-6 text-primary-600 ${
                simulations.some(sim => sim.status === 'running') ? 'animate-spin' : ''
              }`} />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="space-y-8">
        
        {/* Top Row - Charts (Full Width) */}
        <div className="w-full">
          <EconomicCharts 
            simulation={activeSimulation}
            fredData={fredData}
          />
        </div>

        {/* Bottom Row - Controls and Data */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left - Controls */}
          <div className="lg:col-span-1">
            <SimulationControls 
              onCreateSimulation={handleCreateSimulation}
              isCreating={isCreatingSimulation}
              fredData={fredData}
            />
          </div>

          {/* Center - FRED Data */}
          <div className="lg:col-span-1">
            <FREDDataPanel 
              data={fredData}
              onRefresh={handleRefreshFredData}
            />
          </div>

          {/* Right - Insights */}
          <div className="lg:col-span-1">
            <EconomicInsights 
              fredData={fredData}
              activeSimulation={activeSimulation}
            />
          </div>
        </div>
      </div>

      {/* Simulations List */}
      <div className="mt-8">
        <SimulationsList 
          simulations={simulations}
          activeSimulation={activeSimulation}
          onSimulationSelect={handleSimulationSelect}
          onRefresh={loadDashboardData}
        />
      </div>
    </div>
  );
};

export default Dashboard;