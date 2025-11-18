import React, { useState, useEffect } from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, Activity, RefreshCw } from 'lucide-react';
import apiService from '../services/api';
import { 
  BarChart, Bar, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area 
} from 'recharts';

// Card component for scenarios page
const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      {children}
    </div>
  );
};

interface ScenarioKRIs {
  default_rate: number;
  system_liquidity: number;
  avg_capital_ratio: number;
  network_stress: number;
  max_default_rate: number;
  min_liquidity: number;
  scenario: string;
}

interface ScenarioResults {
  status: string;
  timestamp: string;
  scenarios: {
    baseline: ScenarioKRIs;
    recession: ScenarioKRIs;
    rate_shock: ScenarioKRIs;
    credit_crisis: ScenarioKRIs;
  };
  metadata: {
    n_banks: number;
    n_firms: number;
    n_steps: number;
    method: string;
  };
}

const Scenarios: React.FC = () => {
  const [scenarioData, setScenarioData] = useState<ScenarioResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadScenarios();
  }, []);

  const loadScenarios = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.runScenarioSimulations();
      setScenarioData(data);
    } catch (err) {
      setError('Failed to load scenario simulations');
      console.error('Error loading scenarios:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-900 text-lg font-semibold">Running scenario simulations...</p>
          <p className="text-gray-600 mt-2">This may take 30-60 seconds</p>
        </div>
      </div>
    );
  }

  if (error || !scenarioData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <p className="text-gray-900 text-lg font-semibold mb-2">{error || 'No data available'}</p>
          <p className="text-gray-600 mb-6">Failed to load scenario simulations</p>
          <button
            onClick={loadScenarios}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-lg"
          >
            Retry Simulations
          </button>
        </div>
      </div>
    );
  }

  // Prepare data for comparison chart
  const comparisonData = Object.entries(scenarioData.scenarios).map(([name, kris]) => ({
    scenario: name.replace('_', ' ').toUpperCase(),
    'Default Rate': parseFloat((kris.default_rate * 100).toFixed(2)),
    'System Liquidity': parseFloat((kris.system_liquidity * 100).toFixed(2)),
    'Network Stress': parseFloat((kris.network_stress * 100).toFixed(2)),
  }));

  // Prepare radar chart data
  const radarData = [
    {
      metric: 'Default Rate',
      baseline: scenarioData.scenarios.baseline.default_rate * 100,
      recession: scenarioData.scenarios.recession.default_rate * 100,
      rate_shock: scenarioData.scenarios.rate_shock.default_rate * 100,
      credit_crisis: scenarioData.scenarios.credit_crisis.default_rate * 100,
    },
    {
      metric: 'Liquidity',
      baseline: scenarioData.scenarios.baseline.system_liquidity * 100,
      recession: scenarioData.scenarios.recession.system_liquidity * 100,
      rate_shock: scenarioData.scenarios.rate_shock.system_liquidity * 100,
      credit_crisis: scenarioData.scenarios.credit_crisis.system_liquidity * 100,
    },
    {
      metric: 'Network Stress',
      baseline: scenarioData.scenarios.baseline.network_stress * 100,
      recession: scenarioData.scenarios.recession.network_stress * 100,
      rate_shock: scenarioData.scenarios.rate_shock.network_stress * 100,
      credit_crisis: scenarioData.scenarios.credit_crisis.network_stress * 100,
    },
  ];

  // Timeline comparison data
  const timelineData = Object.entries(scenarioData.scenarios).map(([name, kris]) => ({
    scenario: name.replace('_', ' '),
    min: kris.min_liquidity * 100,
    current: kris.system_liquidity * 100,
    max: kris.max_default_rate * 100,
  }));

  const getScenarioColor = (scenario: string) => {
    const colors: Record<string, string> = {
      baseline: 'bg-green-500',
      recession: 'bg-yellow-500',
      rate_shock: 'bg-orange-500',
      credit_crisis: 'bg-red-500',
    };
    return colors[scenario] || 'bg-gray-500';
  };

  const getScenarioDescription = (scenario: string) => {
    const descriptions: Record<string, string> = {
      baseline: 'Normal economic conditions with historical trends',
      recession: 'Economic downturn with increased defaults and reduced liquidity',
      rate_shock: 'Sudden increase in interest rates affecting credit markets',
      credit_crisis: 'Severe credit market disruption with systemic stress',
    };
    return descriptions[scenario] || '';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Scenario Analysis</h1>
          <p className="text-gray-600 mt-2">Agent-Based Model (ABM) Stress Testing Results</p>
        </div>
        <button
          onClick={loadScenarios}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
        >
          {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Activity className="w-5 h-5" />}
          {loading ? 'Running...' : 'Re-run Simulations'}
        </button>
      </div>

      {/* Metadata */}
      <Card>
        <div className="grid grid-cols-4 gap-6">
          <div className="text-center">
            <p className="text-gray-600 text-sm uppercase tracking-wider mb-2">Banks</p>
            <p className="text-3xl font-bold text-blue-600">{scenarioData.metadata.n_banks}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-600 text-sm uppercase tracking-wider mb-2">Firms</p>
            <p className="text-3xl font-bold text-green-600">{scenarioData.metadata.n_firms}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-600 text-sm uppercase tracking-wider mb-2">Steps</p>
            <p className="text-3xl font-bold text-purple-600">{scenarioData.metadata.n_steps}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-600 text-sm uppercase tracking-wider mb-2">Method</p>
            <p className="text-lg font-semibold text-orange-600">{scenarioData.metadata.method}</p>
          </div>
        </div>
      </Card>

      {/* Bar Comparison Chart */}
      <Card>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Key Metrics Comparison</h2>
        <ResponsiveContainer width="100%" height={450}>
          <BarChart data={comparisonData} barGap={8} barCategoryGap="20%">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="scenario" 
              stroke="#6B7280" 
              tick={{ fill: '#374151', fontSize: 14 }}
              tickLine={{ stroke: '#9CA3AF' }}
            />
            <YAxis 
              stroke="#6B7280" 
              label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft', fill: '#374151', fontSize: 14 }}
              tick={{ fill: '#374151', fontSize: 12 }}
              tickLine={{ stroke: '#9CA3AF' }}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: '#ffffff', 
                border: '1px solid #e5e7eb', 
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
              labelStyle={{ color: '#111827', fontWeight: 'bold', fontSize: 15 }}
              itemStyle={{ color: '#374151', fontSize: 13 }}
              cursor={{ fill: 'rgba(229, 231, 235, 0.3)' }}
            />
            <Legend 
              wrapperStyle={{ color: '#374151', paddingTop: 20 }}
              iconType="circle"
            />
            <Bar dataKey="Default Rate" fill="#EF4444" radius={[8, 8, 0, 0]} />
            <Bar dataKey="System Liquidity" fill="#10B981" radius={[8, 8, 0, 0]} />
            <Bar dataKey="Network Stress" fill="#F59E0B" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Radar Chart for Multi-dimensional View */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Risk Profile Comparison</h2>
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#d1d5db" />
              <PolarAngleAxis 
                dataKey="metric" 
                stroke="#6B7280" 
                tick={{ fill: '#374151', fontSize: 13, fontWeight: 500 }} 
              />
              <PolarRadiusAxis 
                stroke="#9CA3AF" 
                tick={{ fill: '#6B7280', fontSize: 11 }}
              />
              <Radar 
                name="Baseline" 
                dataKey="baseline" 
                stroke="#10B981" 
                fill="#10B981" 
                fillOpacity={0.25} 
                strokeWidth={2}
              />
              <Radar 
                name="Recession" 
                dataKey="recession" 
                stroke="#F59E0B" 
                fill="#F59E0B" 
                fillOpacity={0.25} 
                strokeWidth={2}
              />
              <Radar 
                name="Rate Shock" 
                dataKey="rate_shock" 
                stroke="#F97316" 
                fill="#F97316" 
                fillOpacity={0.25} 
                strokeWidth={2}
              />
              <Radar 
                name="Credit Crisis" 
                dataKey="credit_crisis" 
                stroke="#EF4444" 
                fill="#EF4444" 
                fillOpacity={0.25} 
                strokeWidth={2}
              />
              <Legend 
                wrapperStyle={{ color: '#374151', paddingTop: 15 }} 
                iconType="circle"
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '1px solid #e5e7eb', 
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                }}
                labelStyle={{ color: '#111827', fontWeight: 'bold' }}
                itemStyle={{ color: '#374151' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </Card>

        {/* Liquidity Range Chart */}
        <Card>
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Liquidity & Risk Analysis</h2>
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={timelineData} margin={{ bottom: 20, left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis 
                dataKey="scenario" 
                stroke="#6B7280" 
                tick={{ fill: '#374151', fontSize: 12 }}
                tickLine={{ stroke: '#9CA3AF' }}
                label={{ value: 'Scenario', position: 'insideBottom', offset: -15, fill: '#374151', fontSize: 13, fontWeight: 500 }}
              />
              <YAxis 
                stroke="#6B7280" 
                tick={{ fill: '#374151', fontSize: 12 }} 
                tickLine={{ stroke: '#9CA3AF' }}
                label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft', fill: '#374151', fontSize: 13, fontWeight: 500 }}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '1px solid #e5e7eb', 
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                }}
                labelStyle={{ color: '#111827', fontWeight: 'bold' }}
                itemStyle={{ color: '#374151' }}
              />
              <Legend wrapperStyle={{ color: '#374151', paddingTop: 15 }} iconType="circle" />
              <Area 
                dataKey="max" 
                fill="#EF4444" 
                fillOpacity={0.2} 
                stroke="#EF4444" 
                name="Max Default Rate"
                strokeWidth={2}
              />
              <Bar 
                dataKey="current" 
                fill="#3B82F6" 
                name="Current Liquidity" 
                radius={[4, 4, 0, 0]}
              />
              <Line 
                dataKey="min" 
                stroke="#10B981" 
                strokeWidth={3} 
                name="Min Liquidity" 
                dot={{ r: 5, fill: '#10B981', strokeWidth: 2, stroke: '#ffffff' }} 
              />
            </ComposedChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Individual Scenario Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(scenarioData.scenarios).map(([scenarioName, kris]) => (
          <Card key={scenarioName} className="hover:shadow-lg transition-shadow duration-300">
            <div className="flex items-center justify-between mb-6">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-4 h-4 rounded-full ${getScenarioColor(scenarioName)} shadow-lg`}></div>
                  <h3 className="text-2xl font-bold text-gray-900 capitalize">
                    {scenarioName.replace('_', ' ')}
                  </h3>
                </div>
                <p className="text-sm text-gray-600 ml-7">
                  {getScenarioDescription(scenarioName)}
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                <span className="text-gray-700 font-medium">Default Rate</span>
                <div className="flex items-center gap-3">
                  <span className="text-gray-900 font-bold text-lg">
                    {(kris.default_rate * 100).toFixed(1)}%
                  </span>
                  {kris.default_rate > 0.15 ? (
                    <TrendingUp className="w-5 h-5 text-red-600" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                <span className="text-gray-700 font-medium">System Liquidity</span>
                <div className="flex items-center gap-3">
                  <span className="text-gray-900 font-bold text-lg">
                    {(kris.system_liquidity * 100).toFixed(1)}%
                  </span>
                  {kris.system_liquidity < 0.1 ? (
                    <TrendingDown className="w-5 h-5 text-red-600" />
                  ) : (
                    <TrendingUp className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                <span className="text-gray-700 font-medium">Network Stress</span>
                <div className="flex items-center gap-3">
                  <span className="text-gray-900 font-bold text-lg">
                    {(kris.network_stress * 100).toFixed(1)}%
                  </span>
                  {kris.network_stress > 0.3 ? (
                    <TrendingUp className="w-5 h-5 text-red-600" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                <span className="text-gray-700 font-medium">Avg Capital Ratio</span>
                <span className="text-gray-900 font-bold text-lg">
                  {kris.avg_capital_ratio.toFixed(1)}%
                </span>
              </div>

              <div className="border-t border-gray-200 pt-4 mt-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 bg-red-50 rounded text-center border border-red-200">
                    <span className="text-gray-600 text-xs uppercase tracking-wider block mb-1">Max Default</span>
                    <span className="text-red-600 font-bold text-base">{(kris.max_default_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2 bg-green-50 rounded text-center border border-green-200">
                    <span className="text-gray-600 text-xs uppercase tracking-wider block mb-1">Min Liquidity</span>
                    <span className="text-green-600 font-bold text-base">{(kris.min_liquidity * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default Scenarios;
