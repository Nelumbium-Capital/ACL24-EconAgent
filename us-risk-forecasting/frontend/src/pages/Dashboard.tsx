import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import apiService, { DashboardSummary, EconomicDataPoint } from '../services/api';

const Dashboard: React.FC = () => {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [economicData, setEconomicData] = useState<EconomicDataPoint[]>([]);
  const [forecasts, setForecasts] = useState<EconomicDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [summaryData, econData, forecastData] = await Promise.all([
        apiService.getDashboardSummary(),
        apiService.getEconomicData(),
        apiService.getForecasts(),
      ]);
      
      setSummary(summaryData);
      setEconomicData(econData);
      setForecasts(forecastData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading || !summary) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  const KPICard = ({ title, value, unit, change, trend }: any) => (
    <div className="bg-card rounded-xl shadow-sm p-6 border border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        {trend === 'down' ? (
          <TrendingDown className="w-5 h-5 text-success" />
        ) : (
          <TrendingUp className="w-5 h-5 text-danger" />
        )}
      </div>
      <div className="flex items-baseline">
        <span className="text-3xl font-bold text-gray-900">{value.toFixed(2)}</span>
        <span className="ml-2 text-lg text-gray-600">{unit}</span>
      </div>
      <div className={`mt-2 text-sm ${change < 0 ? 'text-success' : 'text-danger'}`}>
        {change > 0 ? '+' : ''}{change.toFixed(2)}{unit} from last month
      </div>
    </div>
  );

  const riskData = [
    { name: 'Low', value: summary.risk_summary.low, color: '#24a148' },
    { name: 'Medium', value: summary.risk_summary.medium, color: '#f1c21b' },
    { name: 'High', value: summary.risk_summary.high, color: '#fd7e14' },
    { name: 'Critical', value: summary.risk_summary.critical, color: '#da1e28' },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Risk Dashboard</h1>
          <p className="text-gray-600 mt-1">Real-time financial risk monitoring and forecasting</p>
        </div>
        <button
          onClick={fetchData}
          className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh Data
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Unemployment Rate"
          value={summary.kpis.unemployment.value}
          unit={summary.kpis.unemployment.unit}
          change={summary.kpis.unemployment.change}
          trend={summary.kpis.unemployment.trend}
        />
        <KPICard
          title="Inflation Rate"
          value={summary.kpis.inflation.value}
          unit={summary.kpis.inflation.unit}
          change={summary.kpis.inflation.change}
          trend={summary.kpis.inflation.trend}
        />
        <KPICard
          title="Interest Rate"
          value={summary.kpis.interest_rate.value}
          unit={summary.kpis.interest_rate.unit}
          change={summary.kpis.interest_rate.change}
          trend={summary.kpis.interest_rate.trend}
        />
        <KPICard
          title="Credit Spread"
          value={summary.kpis.credit_spread.value}
          unit={summary.kpis.credit_spread.unit}
          change={summary.kpis.credit_spread.change}
          trend={summary.kpis.credit_spread.trend}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Economic Indicators Chart */}
        <div className="lg:col-span-2 bg-card rounded-xl shadow-sm p-6 border border-gray-200">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Economic Indicators</h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={economicData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="unemployment" stroke="#0f62fe" strokeWidth={2} name="Unemployment %" />
              <Line type="monotone" dataKey="inflation" stroke="#da1e28" strokeWidth={2} name="Inflation %" />
              <Line type="monotone" dataKey="interest_rate" stroke="#24a148" strokeWidth={2} name="Interest Rate %" />
              <Line type="monotone" dataKey="credit_spread" stroke="#f1c21b" strokeWidth={2} name="Credit Spread %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-card rounded-xl shadow-sm p-6 border border-gray-200">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Risk Distribution</h2>
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecasts Chart */}
      <div className="bg-card rounded-xl shadow-sm p-6 border border-gray-200">
        <h2 className="text-xl font-bold text-gray-900 mb-4">12-Month Forecasts</h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={[...economicData.slice(-24), ...forecasts]}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="unemployment" stroke="#0f62fe" strokeWidth={2} name="Unemployment %" />
            <Line type="monotone" dataKey="inflation" stroke="#da1e28" strokeWidth={2} name="Inflation %" />
            <Line type="monotone" dataKey="interest_rate" stroke="#24a148" strokeWidth={2} name="Interest Rate %" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Dashboard;
