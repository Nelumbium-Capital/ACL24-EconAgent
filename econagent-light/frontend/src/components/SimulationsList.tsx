import React, { useState } from 'react';
import { Play, Pause, Trash2, Download, Eye, RefreshCw, Clock, Users, Calendar } from 'lucide-react';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import { SimulationStatus } from '../types/index.ts';
import { apiClient } from '../services/api.ts';

interface SimulationsListProps {
  simulations: SimulationStatus[];
  activeSimulation: SimulationStatus | null;
  onSimulationSelect: (simulation: SimulationStatus) => void;
  onRefresh: () => Promise<void>;
}

const SimulationsList: React.FC<SimulationsListProps> = ({
  simulations,
  activeSimulation,
  onSimulationSelect,
  onRefresh
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await onRefresh();
      toast.success('Simulations refreshed');
    } catch (error) {
      toast.error('Failed to refresh simulations');
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleDelete = async (simulationId: string, simulationName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${simulationName}"?`)) {
      return;
    }

    setDeletingId(simulationId);
    try {
      await apiClient.deleteSimulation(simulationId);
      toast.success('Simulation deleted successfully');
      await onRefresh();
    } catch (error) {
      console.error('Failed to delete simulation:', error);
      toast.error('Failed to delete simulation');
    } finally {
      setDeletingId(null);
    }
  };

  const handleStop = async (simulationId: string, simulationName: string) => {
    try {
      await apiClient.stopSimulation(simulationId);
      toast.success(`Simulation "${simulationName}" stopped`);
      await onRefresh();
    } catch (error) {
      console.error('Failed to stop simulation:', error);
      toast.error('Failed to stop simulation');
    }
  };

  const handleExport = async (simulationId: string, simulationName: string) => {
    try {
      await apiClient.downloadFile(
        `/api/simulations/${simulationId}/export?format=json`,
        `${simulationName}_results.json`
      );
      toast.success('Results exported successfully');
    } catch (error) {
      console.error('Failed to export results:', error);
      toast.error('Failed to export results');
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return format(new Date(dateString), 'MMM dd, HH:mm');
    } catch {
      return 'Unknown';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-primary-100 text-primary-800';
      case 'completed':
        return 'bg-success-100 text-success-800';
      case 'failed':
        return 'bg-danger-100 text-danger-800';
      case 'stopped':
        return 'bg-warning-100 text-warning-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getDuration = (simulation: SimulationStatus) => {
    if (!simulation.started_at) return null;
    
    const startTime = new Date(simulation.started_at);
    const endTime = simulation.completed_at ? new Date(simulation.completed_at) : new Date();
    const durationMs = endTime.getTime() - startTime.getTime();
    const durationMinutes = Math.floor(durationMs / 60000);
    
    if (durationMinutes < 1) return '< 1 min';
    if (durationMinutes < 60) return `${durationMinutes} min`;
    
    const hours = Math.floor(durationMinutes / 60);
    const minutes = durationMinutes % 60;
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">
          Recent Simulations
        </h2>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          title="Refresh simulations"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {simulations.length === 0 ? (
        <div className="text-center py-12">
          <Play className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Simulations Yet</h3>
          <p className="text-gray-500">Create your first simulation to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {simulations.map((simulation) => (
            <div
              key={simulation.simulation_id}
              className={`border rounded-lg p-4 transition-all cursor-pointer hover:shadow-md ${
                activeSimulation?.simulation_id === simulation.simulation_id
                  ? 'border-primary-300 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => onSimulationSelect(simulation)}
            >
              <div className="flex items-center justify-between">
                
                {/* Simulation Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-sm font-medium text-gray-900 truncate">
                      {simulation.name}
                    </h3>
                    <span className={`status-badge ${getStatusColor(simulation.status)}`}>
                      {simulation.status}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <div className="flex items-center">
                      <Calendar className="w-3 h-3 mr-1" />
                      {formatDate(simulation.created_at)}
                    </div>
                    
                    {simulation.started_at && (
                      <div className="flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {getDuration(simulation)}
                      </div>
                    )}
                    
                    <div className="flex items-center">
                      <Users className="w-3 h-3 mr-1" />
                      {simulation.total_steps / 12} years
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {simulation.status === 'running' && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>{simulation.progress_percent.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div 
                          className="bg-primary-600 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${simulation.progress_percent}%` }}
                        ></div>
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        Step {simulation.current_step} of {simulation.total_steps}
                      </div>
                    </div>
                  )}

                  {/* Current Metrics */}
                  {simulation.status === 'running' && simulation.current_metrics && (
                    <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                      <div className="bg-white rounded p-2">
                        <div className="text-gray-500">Unemployment</div>
                        <div className="font-medium text-blue-600">
                          {simulation.current_metrics.unemployment_rate.toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-gray-500">Inflation</div>
                        <div className="font-medium text-amber-600">
                          {simulation.current_metrics.inflation_rate.toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-gray-500">GDP Growth</div>
                        <div className="font-medium text-green-600">
                          {simulation.current_metrics.gdp_growth.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {simulation.status === 'failed' && simulation.error_message && (
                    <div className="mt-2 p-2 bg-danger-50 border border-danger-200 rounded text-xs text-danger-700">
                      {simulation.error_message}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-2 ml-4">
                  
                  {/* View Details */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSimulationSelect(simulation);
                    }}
                    className="p-2 text-gray-600 hover:text-primary-600 hover:bg-primary-100 rounded-lg transition-colors"
                    title="View details"
                  >
                    <Eye className="w-4 h-4" />
                  </button>

                  {/* Stop (if running) */}
                  {simulation.status === 'running' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleStop(simulation.simulation_id, simulation.name);
                      }}
                      className="p-2 text-gray-600 hover:text-warning-600 hover:bg-warning-100 rounded-lg transition-colors"
                      title="Stop simulation"
                    >
                      <Pause className="w-4 h-4" />
                    </button>
                  )}

                  {/* Export (if completed) */}
                  {simulation.status === 'completed' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleExport(simulation.simulation_id, simulation.name);
                      }}
                      className="p-2 text-gray-600 hover:text-success-600 hover:bg-success-100 rounded-lg transition-colors"
                      title="Export results"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  )}

                  {/* Delete (if not running) */}
                  {simulation.status !== 'running' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(simulation.simulation_id, simulation.name);
                      }}
                      disabled={deletingId === simulation.simulation_id}
                      className="p-2 text-gray-600 hover:text-danger-600 hover:bg-danger-100 rounded-lg transition-colors disabled:opacity-50"
                      title="Delete simulation"
                    >
                      {deletingId === simulation.simulation_id ? (
                        <div className="w-4 h-4 border-2 border-danger-600 border-t-transparent rounded-full animate-spin"></div>
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Summary Stats */}
      {simulations.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-4 gap-4 text-center text-sm">
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {simulations.length}
              </div>
              <div className="text-gray-500">Total</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-primary-600">
                {simulations.filter(s => s.status === 'running').length}
              </div>
              <div className="text-gray-500">Running</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-success-600">
                {simulations.filter(s => s.status === 'completed').length}
              </div>
              <div className="text-gray-500">Completed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-danger-600">
                {simulations.filter(s => s.status === 'failed').length}
              </div>
              <div className="text-gray-500">Failed</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationsList;