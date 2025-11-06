import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download, RefreshCw, Calendar, Users, Clock, TrendingUp } from 'lucide-react';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import { SimulationStatus, SimulationResult } from '../types/index.ts';
import { apiClient } from '../services/api.ts';
import EconomicCharts from './EconomicCharts.tsx';

const SimulationDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [simulation, setSimulation] = useState<SimulationStatus | null>(null);
  const [results, setResults] = useState<SimulationResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (id) {
      loadSimulationData(id);
      
      // Set up polling for running simulations
      const interval = setInterval(() => {
        if (simulation && simulation.status === 'running') {
          refreshSimulationStatus(id);
        }
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [id, simulation?.status]);

  const loadSimulationData = async (simulationId: string) => {
    try {
      setIsLoading(true);
      
      const status = await apiClient.getSimulationStatus(simulationId);
      setSimulation(status);
      
      if (status.status === 'completed') {
        const simulationResults = await apiClient.getSimulationResults(simulationId);
        setResults(simulationResults);
      }
      
    } catch (error) {
      console.error('Failed to load simulation data:', error);
      toast.error('Failed to load simulation data');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshSimulationStatus = async (simulationId: string) => {
    try {
      setIsRefreshing(true);
      const status = await apiClient.getSimulationStatus(simulationId);
      setSimulation(status);
      
      if (status.status === 'completed' && !results) {
        const simulationResults = await apiClient.getSimulationResults(simulationId);
        setResults(simulationResults);
        toast.success('Simulation completed!');
      }
      
    } catch (error) {
      console.error('Failed to refresh simulation status:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleExport = async () => {
    if (!id || !simulation) return;

    try {
      await apiClient.downloadFile(
        `/api/simulations/${id}/export?format=json`,
        `${simulation.name}_results.json`
      );
      toast.success('Results exported successfully');
    } catch (error) {
      console.error('Failed to export results:', error);
      toast.error('Failed to export results');
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy HH:mm:ss');
    } catch {
      return 'Unknown';
    }
  };

  const getDuration = () => {
    if (!simulation?.started_at) return null;
    
    const startTime = new Date(simulation.started_at);
    const endTime = simulation.completed_at ? new Date(simulation.completed_at) : new Date();
    const durationMs = endTime.getTime() - startTime.getTime();
    const durationMinutes = Math.floor(durationMs / 60000);
    
    if (durationMinutes < 1) return '< 1 minute';
    if (durationMinutes < 60) return `${durationMinutes} minutes`;
    
    const hours = Math.floor(durationMinutes / 60);
    const minutes = durationMinutes % 60;
    return `${hours} hours ${minutes} minutes`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-6 h-6 animate-spin text-primary-600" />
          <span className="text-lg text-gray-600">Loading simulation...</span>
        </div>
      </div>
    );
  }

  if (!simulation) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Simulation Not Found</h2>
        <p className="text-gray-600 mb-6">The requested simulation could not be found.</p>
        <Link to="/" className="btn-primary">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to="/"
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{simulation.name}</h1>
            <p className="text-gray-600">Simulation Details</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => id && refreshSimulationStatus(id)}
            disabled={isRefreshing}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
          
          {simulation.status === 'completed' && (
            <button onClick={handleExport} className="btn-primary">
              <Download className="w-4 h-4 mr-2" />
              Export Results
            </button>
          )}
        </div>
      </div>

      {/* Status and Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        
        {/* Status */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Status</p>
              <p className={`text-lg font-bold capitalize ${
                simulation.status === 'completed' ? 'text-success-600' :
                simulation.status === 'running' ? 'text-primary-600' :
                simulation.status === 'failed' ? 'text-danger-600' :
                'text-gray-600'
              }`}>
                {simulation.status}
              </p>
            </div>
            <div className={`status-badge status-${simulation.status}`}>
              {simulation.status}
            </div>
          </div>
        </div>

        {/* Progress */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Progress</p>
              <p className="text-2xl font-bold text-gray-900">
                {simulation.progress_percent.toFixed(1)}%
              </p>
            </div>
            <div className="p-2 bg-primary-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-primary-600" />
            </div>
          </div>
          {simulation.status === 'running' && (
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${simulation.progress_percent}%` }}
              ></div>
            </div>
          )}
        </div>

        {/* Duration */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Duration</p>
              <p className="text-2xl font-bold text-gray-900">
                {getDuration() || 'Not started'}
              </p>
            </div>
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Clock className="w-6 h-6 text-indigo-600" />
            </div>
          </div>
        </div>

        {/* Steps */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Steps</p>
              <p className="text-2xl font-bold text-gray-900">
                {simulation.current_step}/{simulation.total_steps}
              </p>
            </div>
            <div className="p-2 bg-success-100 rounded-lg">
              <Users className="w-6 h-6 text-success-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Current Metrics (if running) */}
      {simulation.status === 'running' && simulation.current_metrics && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Current Economic Indicators</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-sm font-medium text-blue-700">Unemployment Rate</div>
              <div className="text-2xl font-bold text-blue-900">
                {simulation.current_metrics.unemployment_rate.toFixed(1)}%
              </div>
            </div>
            <div className="bg-amber-50 rounded-lg p-4">
              <div className="text-sm font-medium text-amber-700">Inflation Rate</div>
              <div className="text-2xl font-bold text-amber-900">
                {simulation.current_metrics.inflation_rate.toFixed(1)}%
              </div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-sm font-medium text-green-700">GDP Growth</div>
              <div className="text-2xl font-bold text-green-900">
                {simulation.current_metrics.gdp_growth.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <EconomicCharts simulation={simulation} fredData={null} />

      {/* Simulation Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Configuration */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Created:</span>
              <span className="font-medium">{formatDate(simulation.created_at)}</span>
            </div>
            {simulation.started_at && (
              <div className="flex justify-between">
                <span className="text-gray-600">Started:</span>
                <span className="font-medium">{formatDate(simulation.started_at)}</span>
              </div>
            )}
            {simulation.completed_at && (
              <div className="flex justify-between">
                <span className="text-gray-600">Completed:</span>
                <span className="font-medium">{formatDate(simulation.completed_at)}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600">Total Steps:</span>
              <span className="font-medium">{simulation.total_steps}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Simulation Years:</span>
              <span className="font-medium">{simulation.total_steps / 12}</span>
            </div>
          </div>
        </div>

        {/* Results Summary */}
        {results && (
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Results Summary</h2>
            <div className="space-y-3">
              {results.final_metrics && (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Unemployment:</span>
                    <span className="font-medium">{results.final_metrics.avg_unemployment.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Inflation:</span>
                    <span className="font-medium">{results.final_metrics.avg_inflation.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg GDP Growth:</span>
                    <span className="font-medium">{results.final_metrics.avg_gdp_growth.toFixed(1)}%</span>
                  </div>
                </>
              )}
              {results.duration_seconds && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Execution Time:</span>
                  <span className="font-medium">{Math.round(results.duration_seconds / 60)} minutes</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Error Details */}
      {simulation.status === 'failed' && simulation.error_message && (
        <div className="card border-danger-200 bg-danger-50">
          <h2 className="text-lg font-semibold text-danger-900 mb-4">Error Details</h2>
          <div className="bg-white border border-danger-200 rounded-lg p-4">
            <pre className="text-sm text-danger-700 whitespace-pre-wrap">
              {simulation.error_message}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationDetail;