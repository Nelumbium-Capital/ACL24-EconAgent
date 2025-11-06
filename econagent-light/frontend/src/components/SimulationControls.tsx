import React, { useState } from 'react';
import { Play, Settings, Zap, Calendar, Users, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';
import { SimulationConfig, EconomicSnapshot, FormErrors } from '../types/index.ts';

interface SimulationControlsProps {
  onCreateSimulation: (config: SimulationConfig) => Promise<void>;
  isCreating: boolean;
  fredData: EconomicSnapshot | null;
}

const SimulationControls: React.FC<SimulationControlsProps> = ({
  onCreateSimulation,
  isCreating,
  fredData
}) => {
  const [config, setConfig] = useState<SimulationConfig>({
    name: `Simulation ${new Date().toLocaleDateString()}`,
    num_agents: 100,
    num_years: 20,
    use_fred_calibration: true,
    economic_scenario: 'baseline',
    productivity: 1.0,
    skill_change: 0.02,
    price_change: 0.02
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [errors, setErrors] = useState<FormErrors>({});

  const validateConfig = (): boolean => {
    const newErrors: FormErrors = {};

    if (!config.name.trim()) {
      newErrors.name = 'Simulation name is required';
    }

    if (config.num_agents < 10 || config.num_agents > 1000) {
      newErrors.num_agents = 'Number of agents must be between 10 and 1000';
    }

    if (config.num_years < 1 || config.num_years > 50) {
      newErrors.num_years = 'Simulation years must be between 1 and 50';
    }

    if (config.productivity && (config.productivity < 0.1 || config.productivity > 5.0)) {
      newErrors.productivity = 'Productivity must be between 0.1 and 5.0';
    }

    if (config.skill_change && (config.skill_change < 0 || config.skill_change > 0.1)) {
      newErrors.skill_change = 'Skill change rate must be between 0 and 0.1';
    }

    if (config.price_change && (config.price_change < 0 || config.price_change > 0.1)) {
      newErrors.price_change = 'Price change rate must be between 0 and 0.1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateConfig()) {
      toast.error('Please fix the validation errors');
      return;
    }

    try {
      await onCreateSimulation(config);
    } catch (error) {
      console.error('Failed to create simulation:', error);
    }
  };

  const handleInputChange = (field: keyof SimulationConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
    
    // Clear error for this field
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const getEstimatedDuration = () => {
    const steps = config.num_years * 12;
    const estimatedMinutes = Math.ceil((steps * config.num_agents) / 1000);
    return estimatedMinutes;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-primary-600" />
          Simulation Configuration
        </h2>
        
        {fredData && (
          <div className="flex items-center text-xs text-success-600">
            <div className="w-2 h-2 bg-success-500 rounded-full mr-1 animate-pulse"></div>
            FRED Connected
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        
        {/* Basic Configuration */}
        <div>
          <label className="label">
            Simulation Name
          </label>
          <input
            type="text"
            value={config.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
            className={`input-field ${errors.name ? 'border-danger-300 focus:ring-danger-500' : ''}`}
            placeholder="Enter simulation name"
          />
          {errors.name && (
            <p className="text-xs text-danger-600 mt-1">{errors.name}</p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label flex items-center">
              <Users className="w-4 h-4 mr-1" />
              Agents
            </label>
            <input
              type="number"
              value={config.num_agents}
              onChange={(e) => handleInputChange('num_agents', parseInt(e.target.value))}
              className={`input-field ${errors.num_agents ? 'border-danger-300 focus:ring-danger-500' : ''}`}
              min="10"
              max="1000"
              step="10"
            />
            {errors.num_agents && (
              <p className="text-xs text-danger-600 mt-1">{errors.num_agents}</p>
            )}
          </div>

          <div>
            <label className="label flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              Years
            </label>
            <input
              type="number"
              value={config.num_years}
              onChange={(e) => handleInputChange('num_years', parseInt(e.target.value))}
              className={`input-field ${errors.num_years ? 'border-danger-300 focus:ring-danger-500' : ''}`}
              min="1"
              max="50"
            />
            {errors.num_years && (
              <p className="text-xs text-danger-600 mt-1">{errors.num_years}</p>
            )}
          </div>
        </div>

        {/* Economic Scenario */}
        <div>
          <label className="label">
            Economic Scenario
          </label>
          <select
            value={config.economic_scenario}
            onChange={(e) => handleInputChange('economic_scenario', e.target.value)}
            className="input-field"
          >
            <option value="baseline">Baseline Economy</option>
            <option value="recession">Recession Scenario</option>
            <option value="boom">Economic Boom</option>
            <option value="stagflation">Stagflation</option>
          </select>
        </div>

        {/* FRED Calibration */}
        <div className="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="fred-calibration"
              checked={config.use_fred_calibration}
              onChange={(e) => handleInputChange('use_fred_calibration', e.target.checked)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="fred-calibration" className="ml-2 text-sm font-medium text-gray-700">
              Use FRED Data Calibration
            </label>
          </div>
          
          {config.use_fred_calibration && fredData && (
            <div className="text-xs text-primary-600">
              <TrendingUp className="w-4 h-4 inline mr-1" />
              Real-time calibration
            </div>
          )}
        </div>

        {/* Advanced Settings */}
        <div className="border-t pt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            <Settings className="w-4 h-4 mr-1" />
            Advanced Settings
            <span className="ml-1">{showAdvanced ? '▼' : '▶'}</span>
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="label">
                  Productivity Factor
                </label>
                <input
                  type="number"
                  value={config.productivity}
                  onChange={(e) => handleInputChange('productivity', parseFloat(e.target.value))}
                  className={`input-field ${errors.productivity ? 'border-danger-300 focus:ring-danger-500' : ''}`}
                  min="0.1"
                  max="5.0"
                  step="0.1"
                />
                {errors.productivity && (
                  <p className="text-xs text-danger-600 mt-1">{errors.productivity}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">
                    Skill Change Rate
                  </label>
                  <input
                    type="number"
                    value={config.skill_change}
                    onChange={(e) => handleInputChange('skill_change', parseFloat(e.target.value))}
                    className={`input-field ${errors.skill_change ? 'border-danger-300 focus:ring-danger-500' : ''}`}
                    min="0"
                    max="0.1"
                    step="0.01"
                  />
                  {errors.skill_change && (
                    <p className="text-xs text-danger-600 mt-1">{errors.skill_change}</p>
                  )}
                </div>

                <div>
                  <label className="label">
                    Price Change Rate
                  </label>
                  <input
                    type="number"
                    value={config.price_change}
                    onChange={(e) => handleInputChange('price_change', parseFloat(e.target.value))}
                    className={`input-field ${errors.price_change ? 'border-danger-300 focus:ring-danger-500' : ''}`}
                    min="0"
                    max="0.1"
                    step="0.01"
                  />
                  {errors.price_change && (
                    <p className="text-xs text-danger-600 mt-1">{errors.price_change}</p>
                  )}
                </div>
              </div>

              <div>
                <label className="label">
                  Random Seed (Optional)
                </label>
                <input
                  type="number"
                  value={config.random_seed || ''}
                  onChange={(e) => handleInputChange('random_seed', e.target.value ? parseInt(e.target.value) : undefined)}
                  className="input-field"
                  placeholder="Leave empty for random"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Set a seed for reproducible results
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Simulation Info */}
        <div className="bg-gray-50 rounded-lg p-4 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Total Steps:</span>
            <span className="font-medium">{config.num_years * 12} months</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Estimated Duration:</span>
            <span className="font-medium">~{getEstimatedDuration()} minutes</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Data Points:</span>
            <span className="font-medium">{(config.num_years * 12 * config.num_agents).toLocaleString()}</span>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isCreating}
          className="w-full btn-primary flex items-center justify-center"
        >
          {isCreating ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
              Creating Simulation...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Start Simulation
            </>
          )}
        </button>

        {config.use_fred_calibration && !fredData && (
          <div className="text-center">
            <p className="text-xs text-warning-600">
              <Zap className="w-4 h-4 inline mr-1" />
              FRED data not available - using default parameters
            </p>
          </div>
        )}
      </form>
    </div>
  );
};

export default SimulationControls;