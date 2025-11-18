import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface DashboardData {
  kpis: {
    unemployment: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    inflation: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    interest_rate: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    credit_spread: { value: number; unit: string; change: number; trend: 'up' | 'down' };
  };
  risk_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

const DashboardSimple: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      setRefreshing(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch('http://localhost:8000/api/dashboard/summary', {
        signal: controller.signal,
        headers: { 'Accept': 'application/json' }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err: any) {
      console.error('Error:', err);
      setError(err.name === 'AbortError' ? 'Timeout' : err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mx-auto mb-4"></div>
          <div className="text-xl text-gray-900 font-semibold">Loading Dashboard...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <div className="text-xl text-gray-900 mb-4">Error: {error}</div>
          <button
            onClick={fetchData}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors shadow-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return <div className="text-gray-900">No data</div>;

  // Prepare data for charts
  const kpiData = [
    { name: 'Unemployment', value: data.kpis.unemployment.value, change: data.kpis.unemployment.change },
    { name: 'Inflation', value: data.kpis.inflation.value, change: data.kpis.inflation.change },
    { name: 'Interest Rate', value: data.kpis.interest_rate.value, change: data.kpis.interest_rate.change },
    { name: 'Credit Spread', value: data.kpis.credit_spread.value, change: data.kpis.credit_spread.change },
  ];

  const riskData = [
    { name: 'Low', value: data.risk_summary.low, color: '#10b981' },
    { name: 'Medium', value: data.risk_summary.medium, color: '#3b82f6' },
    { name: 'High', value: data.risk_summary.high, color: '#f59e0b' },
    { name: 'Critical', value: data.risk_summary.critical, color: '#ef4444' },
  ];

  const totalRisks = data.risk_summary.critical + data.risk_summary.high + data.risk_summary.medium + data.risk_summary.low;
  const overallHealth = totalRisks > 0 ? ((data.risk_summary.low / totalRisks) * 100).toFixed(0) : '100';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Economic Risk Dashboard</h1>
          <p className="text-gray-600 mt-2">Real-time monitoring powered by Kumo Graph Transformer + ABM</p>
        </div>
        <button
          onClick={fetchData}
          disabled={refreshing}
          className="flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all shadow-lg disabled:opacity-50"
        >
          <RefreshCw className={`w-5 h-5 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Unemployment Card */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg mr-3">
                <Activity className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Unemployment</h3>
            </div>
            <div className={`flex items-center ${data.kpis.unemployment.trend === 'up' ? 'text-red-600' : 'text-green-600'}`}>
              {data.kpis.unemployment.trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span className="text-sm font-bold">
                {data.kpis.unemployment.change > 0 ? '+' : ''}{data.kpis.unemployment.change.toFixed(2)}%
              </span>
            </div>
          </div>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {data.kpis.unemployment.value.toFixed(2)}%
          </div>
          <div className="text-sm text-gray-600">
            {data.kpis.unemployment.trend === 'up' ? 'Increasing' : 'Decreasing'}
          </div>
        </div>

        {/* Inflation Card */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 bg-red-100 rounded-lg mr-3">
                <TrendingUp className="w-5 h-5 text-red-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Inflation</h3>
            </div>
            <div className={`flex items-center ${data.kpis.inflation.trend === 'up' ? 'text-red-600' : 'text-green-600'}`}>
              {data.kpis.inflation.trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span className="text-sm font-bold">
                {data.kpis.inflation.change > 0 ? '+' : ''}{data.kpis.inflation.change.toFixed(2)}%
              </span>
            </div>
          </div>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {data.kpis.inflation.value.toFixed(2)}%
          </div>
          <div className="text-sm text-gray-600">
            {data.kpis.inflation.trend === 'up' ? 'Increasing' : 'Decreasing'}
          </div>
        </div>

        {/* Interest Rate Card */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg mr-3">
                <TrendingUp className="w-5 h-5 text-green-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Interest Rate</h3>
            </div>
            <div className={`flex items-center ${data.kpis.interest_rate.trend === 'up' ? 'text-red-600' : 'text-green-600'}`}>
              {data.kpis.interest_rate.trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span className="text-sm font-bold">
                {data.kpis.interest_rate.change > 0 ? '+' : ''}{data.kpis.interest_rate.change.toFixed(2)}%
              </span>
            </div>
          </div>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {data.kpis.interest_rate.value.toFixed(2)}%
          </div>
          <div className="text-sm text-gray-600">
            {data.kpis.interest_rate.trend === 'up' ? 'Increasing' : 'Decreasing'}
          </div>
        </div>

        {/* Credit Spread Card */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg mr-3">
                <AlertTriangle className="w-5 h-5 text-yellow-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Credit Spread</h3>
            </div>
            <div className={`flex items-center ${data.kpis.credit_spread.trend === 'up' ? 'text-red-600' : 'text-green-600'}`}>
              {data.kpis.credit_spread.trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span className="text-sm font-bold">
                {data.kpis.credit_spread.change > 0 ? '+' : ''}{data.kpis.credit_spread.change.toFixed(2)}%
              </span>
            </div>
          </div>
          <div className="text-4xl font-bold text-gray-900 mb-2">
            {data.kpis.credit_spread.value.toFixed(2)}%
          </div>
          <div className="text-sm text-gray-600">
            {data.kpis.credit_spread.trend === 'up' ? 'Increasing' : 'Decreasing'}
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* KPI Comparison Chart */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Economic Indicators Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={kpiData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
              <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#60a5fa' }}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => percent > 0 ? `${name}: ${(percent * 100).toFixed(0)}%` : ''}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend 
                verticalAlign="bottom" 
                height={36}
                formatter={(value) => {
                  const dataEntry = riskData.find(d => d.name === value);
                  return dataEntry ? `${value} (${dataEntry.value})` : value;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-8 shadow-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">System Health Score</h3>
            <p className="text-gray-600">Based on {totalRisks} risk indicators</p>
          </div>
          <div className="text-center">
            <div className="text-6xl font-bold text-green-600 mb-2">{overallHealth}%</div>
            <div className="flex items-center justify-center text-green-600">
              <CheckCircle className="w-6 h-6 mr-2" />
              <span className="font-semibold">Healthy</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-6">
          <div className="text-center p-6 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-5xl font-bold text-red-600 mb-2">{data.risk_summary.critical}</div>
            <div className="text-sm font-semibold text-red-700 uppercase tracking-wide">Critical</div>
            <div className="mt-2 h-1 bg-red-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500"
                style={{ width: totalRisks > 0 ? `${(data.risk_summary.critical / totalRisks) * 100}%` : '0%' }}
              />
            </div>
          </div>

          <div className="text-center p-6 bg-orange-50 border border-orange-200 rounded-lg">
            <div className="text-5xl font-bold text-orange-600 mb-2">{data.risk_summary.high}</div>
            <div className="text-sm font-semibold text-orange-700 uppercase tracking-wide">High</div>
            <div className="mt-2 h-1 bg-orange-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-orange-500"
                style={{ width: totalRisks > 0 ? `${(data.risk_summary.high / totalRisks) * 100}%` : '0%' }}
              />
            </div>
          </div>

          <div className="text-center p-6 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-5xl font-bold text-blue-600 mb-2">{data.risk_summary.medium}</div>
            <div className="text-sm font-semibold text-blue-700 uppercase tracking-wide">Medium</div>
            <div className="mt-2 h-1 bg-blue-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: totalRisks > 0 ? `${(data.risk_summary.medium / totalRisks) * 100}%` : '0%' }}
              />
            </div>
          </div>

          <div className="text-center p-6 bg-green-50 border border-green-200 rounded-lg">
            <div className="text-5xl font-bold text-green-600 mb-2">{data.risk_summary.low}</div>
            <div className="text-sm font-semibold text-green-700 uppercase tracking-wide">Low</div>
            <div className="mt-2 h-1 bg-green-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: totalRisks > 0 ? `${(data.risk_summary.low / totalRisks) * 100}%` : '0%' }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Footer Note */}
      <div className="mt-6 text-center text-gray-600 text-sm">
        <p>Powered by Kumo Graph Transformer (40%) + ARIMA (30%) + ETS (30%)</p>
        <p className="mt-1">Mesa ABM with {totalRisks} KRI indicators • Auto-refresh every 30s</p>
      </div>
    </div>
  );
};

export default DashboardSimple;

