import React, { useState, useEffect } from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, Activity, RefreshCw } from 'lucide-react';
import apiService from '../services/api';
import { 
  BarChart, Bar, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area 
} from 'recharts';

// Dark Card component for scenarios page
const DarkCard: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => {
  return (
    <div className={`bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6 ${className}`}>
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
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-200 text-lg font-semibold">Running scenario simulations...</p>
          <p className="text-gray-400 mt-2">This may take 30-60 seconds</p>
        </div>
      </div>
    );
  }

  if (error || !scenarioData) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <p className="text-gray-200 text-lg font-semibold mb-2">{error || 'No data available'}</p>
          <p className="text-gray-400 mb-6">Failed to load scenario simulations</p>
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
    <div className="space-y-6 p-6 bg-gray-900 min-h-screen">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-white mb-2">Scenario Analysis</h1>
          <p className="text-gray-300 text-lg">Agent-Based Model (ABM) Stress Testing Results</p>
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
      <DarkCard>
        <div className="grid grid-cols-4 gap-6">
          <div className="text-center">
            <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Banks</p>
            <p className="text-3xl font-bold text-blue-400">{scenarioData.metadata.n_banks}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Firms</p>
            <p className="text-3xl font-bold text-green-400">{scenarioData.metadata.n_firms}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Steps</p>
            <p className="text-3xl font-bold text-purple-400">{scenarioData.metadata.n_steps}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Method</p>
            <p className="text-lg font-semibold text-orange-400">{scenarioData.metadata.method}</p>
          </div>
        </div>
      </DarkCard>

      {/* Bar Comparison Chart */}
      <DarkCard>
        <h2 className="text-2xl font-bold text-white mb-6">Key Metrics Comparison</h2>
        <ResponsiveContainer width="100%" height={450}>
          <BarChart data={comparisonData} barGap={8} barCategoryGap="20%">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="scenario" 
              stroke="#9CA3AF" 
              tick={{ fill: '#D1D5DB', fontSize: 14 }}
              tickLine={{ stroke: '#6B7280' }}
            />
            <YAxis 
              stroke="#9CA3AF" 
              label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft', fill: '#D1D5DB', fontSize: 14 }}
              tick={{ fill: '#D1D5DB', fontSize: 12 }}
              tickLine={{ stroke: '#6B7280' }}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151', 
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
              }}
              labelStyle={{ color: '#F3F4F6', fontWeight: 'bold', fontSize: 15 }}
              itemStyle={{ color: '#E5E7EB', fontSize: 13 }}
              cursor={{ fill: 'rgba(55, 65, 81, 0.3)' }}
            />
            <Legend 
              wrapperStyle={{ color: '#D1D5DB', paddingTop: 20 }}
              iconType="circle"
            />
            <Bar dataKey="Default Rate" fill="#EF4444" radius={[8, 8, 0, 0]} />
            <Bar dataKey="System Liquidity" fill="#10B981" radius={[8, 8, 0, 0]} />
            <Bar dataKey="Network Stress" fill="#F59E0B" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </DarkCard>

      {/* Radar Chart for Multi-dimensional View */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DarkCard>
          <h2 className="text-2xl font-bold text-white mb-6">Risk Profile Comparison</h2>
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#4B5563" />
              <PolarAngleAxis 
                dataKey="metric" 
                stroke="#9CA3AF" 
                tick={{ fill: '#D1D5DB', fontSize: 13, fontWeight: 500 }} 
              />
              <PolarRadiusAxis 
                stroke="#6B7280" 
                tick={{ fill: '#9CA3AF', fontSize: 11 }}
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
                wrapperStyle={{ color: '#D1D5DB', paddingTop: 15 }} 
                iconType="circle"
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151', 
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
                }}
                labelStyle={{ color: '#F3F4F6', fontWeight: 'bold' }}
                itemStyle={{ color: '#E5E7EB' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </DarkCard>

        {/* Liquidity Range Chart */}
        <DarkCard>
          <h2 className="text-2xl font-bold text-white mb-6">Liquidity & Risk Analysis</h2>
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={timelineData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                type="number" 
                stroke="#9CA3AF" 
                tick={{ fill: '#D1D5DB', fontSize: 12 }}
                tickLine={{ stroke: '#6B7280' }}
              />
              <YAxis 
                dataKey="scenario" 
                type="category" 
                stroke="#9CA3AF" 
                width={100} 
                tick={{ fill: '#D1D5DB', fontSize: 13, fontWeight: 500 }} 
                tickLine={{ stroke: '#6B7280' }}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151', 
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
                }}
                labelStyle={{ color: '#F3F4F6', fontWeight: 'bold' }}
                itemStyle={{ color: '#E5E7EB' }}
              />
              <Legend wrapperStyle={{ color: '#D1D5DB', paddingTop: 15 }} iconType="circle" />
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
                radius={[0, 4, 4, 0]}
              />
              <Line 
                dataKey="min" 
                stroke="#10B981" 
                strokeWidth={3} 
                name="Min Liquidity" 
                dot={{ r: 5, fill: '#10B981', strokeWidth: 2, stroke: '#1F2937' }} 
              />
            </ComposedChart>
          </ResponsiveContainer>
        </DarkCard>
      </div>

      {/* Individual Scenario Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(scenarioData.scenarios).map(([scenarioName, kris]) => (
          <DarkCard key={scenarioName} className="hover:shadow-2xl transition-shadow duration-300">
            <div className="flex items-center justify-between mb-6">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-4 h-4 rounded-full ${getScenarioColor(scenarioName)} shadow-lg`}></div>
                  <h3 className="text-2xl font-bold text-white capitalize">
                    {scenarioName.replace('_', ' ')}
                  </h3>
                </div>
                <p className="text-sm text-gray-400 ml-7">
                  {getScenarioDescription(scenarioName)}
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-900 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
                <span className="text-gray-300 font-medium">Default Rate</span>
                <div className="flex items-center gap-3">
                  <span className="text-white font-bold text-lg">
                    {(kris.default_rate * 100).toFixed(1)}%
                  </span>
                  {kris.default_rate > 0.15 ? (
                    <TrendingUp className="w-5 h-5 text-red-400" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-green-400" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-900 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
                <span className="text-gray-300 font-medium">System Liquidity</span>
                <div className="flex items-center gap-3">
                  <span className="text-white font-bold text-lg">
                    {(kris.system_liquidity * 100).toFixed(1)}%
                  </span>
                  {kris.system_liquidity < 0.1 ? (
                    <TrendingDown className="w-5 h-5 text-red-400" />
                  ) : (
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-900 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
                <span className="text-gray-300 font-medium">Network Stress</span>
                <div className="flex items-center gap-3">
                  <span className="text-white font-bold text-lg">
                    {(kris.network_stress * 100).toFixed(1)}%
                  </span>
                  {kris.network_stress > 0.3 ? (
                    <TrendingUp className="w-5 h-5 text-red-400" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-green-400" />
                  )}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-900 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
                <span className="text-gray-300 font-medium">Avg Capital Ratio</span>
                <span className="text-white font-bold text-lg">
                  {kris.avg_capital_ratio.toFixed(1)}%
                </span>
              </div>

              <div className="border-t border-gray-700 pt-4 mt-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 bg-gray-900 rounded text-center border border-gray-700">
                    <span className="text-gray-400 text-xs uppercase tracking-wider block mb-1">Max Default</span>
                    <span className="text-red-300 font-bold text-base">{(kris.max_default_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-2 bg-gray-900 rounded text-center border border-gray-700">
                    <span className="text-gray-400 text-xs uppercase tracking-wider block mb-1">Min Liquidity</span>
                    <span className="text-green-300 font-bold text-base">{(kris.min_liquidity * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </DarkCard>
        ))}
      </div>
    </div>
  );
};

export default Scenarios;
