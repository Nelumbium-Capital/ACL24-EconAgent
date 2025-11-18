import React, { useState, useEffect } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Area,
  ComposedChart
} from 'recharts';
import { 
  TrendingUp, 
  Target, 
  Brain, 
  Info, 
  RefreshCw, 
  Activity,
  BarChart3,
  Lightbulb,
  Gauge,
  Clock
} from 'lucide-react';
import Card from '../components/ui/Card';
import { 
  apiService, 
  ModelPerformance, 
  ModelInsight, 
  ForecastAnalysis, 
  EconomicContext 
} from '../services/api';
import { EconomicDataPoint } from '../types';

const Forecasting: React.FC = () => {
  const [economicData, setEconomicData] = useState<EconomicDataPoint[]>([]);
  const [forecasts, setForecasts] = useState<EconomicDataPoint[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [modelInsights, setModelInsights] = useState<ModelInsight[]>([]);
  const [forecastAnalysis, setForecastAnalysis] = useState<ForecastAnalysis | null>(null);
  const [economicContext, setEconomicContext] = useState<EconomicContext | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string>('unemployment');
  const [performanceLoading, setPerformanceLoading] = useState(false);

  useEffect(() => {
    fetchAllData();
  }, []);

  useEffect(() => {
    if (selectedMetric) {
      fetchForecastAnalysis(selectedMetric);
    }
  }, [selectedMetric]);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      const [economicDataRes, forecastsRes, contextRes, insightsRes] = await Promise.all([
        apiService.getEconomicData(),
        apiService.getForecasts(),
        apiService.getEconomicContext(),
        apiService.getModelInsights()
      ]);
      
      setEconomicData(economicDataRes);
      setForecasts(forecastsRes);
      setEconomicContext(contextRes);
      setModelInsights(insightsRes.models);
      
      // Fetch model performance separately as it may take longer
      fetchModelPerformance();
    } catch (error) {
      console.error('Failed to fetch forecasting data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchModelPerformance = async () => {
    setPerformanceLoading(true);
    try {
      const performanceRes = await apiService.getModelPerformance();
      setModelPerformance(performanceRes.model_performance);
    } catch (error) {
      console.error('Failed to fetch model performance:', error);
      setModelPerformance([]);
    } finally {
      setPerformanceLoading(false);
    }
  };

  const fetchForecastAnalysis = async (seriesName: string) => {
    try {
      const analysis = await apiService.getForecastAnalysis(seriesName);
      setForecastAnalysis(analysis);
    } catch (error) {
      console.error('Failed to fetch forecast analysis:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg text-gray-600">Loading forecasting data...</p>
        </div>
      </div>
    );
  }

  const metrics = [
    { key: 'unemployment', label: 'Unemployment Rate', color: '#0f62fe', unit: '%', icon: Activity },
    { key: 'inflation', label: 'Inflation Rate', color: '#da1e28', unit: '%', icon: TrendingUp },
    { key: 'interest_rate', label: 'Interest Rate', color: '#24a148', unit: '%', icon: BarChart3 },
    { key: 'credit_spread', label: 'Credit Spread', color: '#f1c21b', unit: '%', icon: Gauge }
  ];

  const selectedMetricInfo = metrics.find(m => m.key === selectedMetric)!;

  // Combine historical and forecast data - SPLIT into separate series for visualization
  const historicalData = economicData.slice(-24).map(d => ({ 
    date: d.date, 
    historical: d[selectedMetric as keyof EconomicDataPoint] as number,
    forecast: null as number | null
  }));
  
  const forecastData = forecasts.map(d => ({ 
    date: d.date, 
    historical: null as number | null,
    forecast: d[selectedMetric as keyof EconomicDataPoint] as number,
    forecast_lower: d[`${selectedMetric}_lower` as keyof EconomicDataPoint] as number,
    forecast_upper: d[`${selectedMetric}_upper` as keyof EconomicDataPoint] as number
  }));
  
  // Bridge the gap between historical and forecast
  if (historicalData.length > 0 && forecastData.length > 0) {
    const lastHistorical = historicalData[historicalData.length - 1];
    forecastData[0] = {
      ...forecastData[0],
      historical: lastHistorical.historical // Connect the lines
    };
  }
  
  const combinedData = [...historicalData, ...forecastData];

  // Calculate trends and insights
  const lastHistorical = economicData[economicData.length - 1];
  const firstForecast = forecasts[0];
  const forecastChange = firstForecast && lastHistorical 
    ? ((firstForecast[selectedMetric as keyof EconomicDataPoint] as number) - 
       (lastHistorical[selectedMetric as keyof EconomicDataPoint] as number))
    : 0;

  const bestModel = modelPerformance.length > 0 
    ? modelPerformance.reduce((prev, current) => (prev.accuracy > current.accuracy) ? prev : current)
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Advanced Forecasting</h1>
          <p className="text-gray-600 mt-2">
            Real-time 12-month economic forecasts with AI-powered model ensemble
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
          >
            {metrics.map(metric => (
              <option key={metric.key} value={metric.key}>
                {metric.label}
              </option>
            ))}
          </select>
          
          <button
            onClick={fetchAllData}
            className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Economic Context Banner */}
      {economicContext && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200">
          <div className="flex items-start space-x-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Info className="w-6 h-6 text-blue-600" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Current Economic Context</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Market Conditions</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {economicContext.current_conditions.map((condition, idx) => (
                      <li key={idx}>• {condition}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Market Sentiment: {economicContext.market_sentiment}</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Unemployment: {economicContext.key_metrics.unemployment_rate.value.toFixed(1)}%</div>
                    <div>Inflation: {economicContext.key_metrics.inflation_rate.value.toFixed(1)}%</div>
                    <div>Fed Rate: {economicContext.key_metrics.fed_funds_rate.value.toFixed(2)}%</div>
                    <div>Credit Spread: {economicContext.key_metrics.credit_spread.value.toFixed(2)}%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 rounded-lg mr-4">
              <TrendingUp className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {forecastChange > 0 ? '+' : ''}{forecastChange.toFixed(2)}{selectedMetricInfo.unit}
              </div>
              <div className="text-sm text-gray-600">
                12-Month Change
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center">
            <div className="p-3 bg-green-100 rounded-lg mr-4">
              <Target className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {forecasts.length}
              </div>
              <div className="text-sm text-gray-600">
                Months Forecasted
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center">
            <div className="p-3 bg-yellow-100 rounded-lg mr-4">
              <Brain className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {bestModel ? bestModel.accuracy.toFixed(1) : 'N/A'}%
              </div>
              <div className="text-sm text-gray-600">
                Best Model Accuracy
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center">
            <div className="p-3 bg-purple-100 rounded-lg mr-4">
              <Clock className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {economicData.length}
              </div>
              <div className="text-sm text-gray-600">
                Historical Data Points
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Main Forecast Chart */}
      <Card title={`${selectedMetricInfo.label} - Historical & Forecast`}>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                month: 'short', 
                year: '2-digit' 
              })}
            />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              labelFormatter={(label) => new Date(label).toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long' 
              })}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}${selectedMetricInfo.unit}`,
                name === 'value' ? selectedMetricInfo.label : name
              ]}
            />
            <Legend />
            
            {/* Confidence band */}
            <Area
              type="monotoneX"
              dataKey="forecast_upper"
              stroke="none"
              fill={selectedMetricInfo.color}
              fillOpacity={0.15}
              connectNulls
              name="95% Confidence"
            />
            <Area
              type="monotoneX"
              dataKey="forecast_lower"
              stroke="none"
              fill="white"
              fillOpacity={1}
              connectNulls
            />

            {/* Historical line (solid) */}
            <Line
              type="monotoneX"
              dataKey="historical"
              stroke={selectedMetricInfo.color}
              strokeWidth={2.5}
              dot={{ fill: selectedMetricInfo.color, r: 3.5 }}
              connectNulls={true}
              name="Historical"
            />

            {/* Forecast line (dashed) */}
            <Line
              type="monotoneX"
              dataKey="forecast"
              stroke={selectedMetricInfo.color}
              strokeWidth={2.5}
              strokeDasharray="5 5"
              dot={{ fill: selectedMetricInfo.color, r: 3.5, strokeDasharray: 'none' }}
              connectNulls={true}
              name="Forecast"
            />
          </ComposedChart>
        </ResponsiveContainer>
        
        <div className="mt-4 flex items-center justify-center space-x-8 text-sm text-gray-600">
          <div className="flex items-center">
            <div className="w-8 h-0.5 bg-gray-800 mr-2"></div>
            <span>Historical</span>
          </div>
          <div className="flex items-center">
            <div className="w-8 h-0.5 border-t-2 border-dashed border-gray-800 mr-2"></div>
            <span>Forecast</span>
          </div>
          <div className="flex items-center">
            <div className="w-8 h-3 bg-gray-300 bg-opacity-30 mr-2"></div>
            <span>95% Confidence Interval</span>
          </div>
        </div>
      </Card>

      {/* Model Performance and Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance */}
        <Card title="Real Model Performance (Backtesting)">
          {performanceLoading ? (
            <div className="flex items-center justify-center h-48">
              <RefreshCw className="w-8 h-8 animate-spin text-gray-400" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="model" 
                  tick={{ fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value: number, name: string) => [
                    name === 'accuracy' ? `${value}%` : value.toFixed(3),
                    name === 'accuracy' ? 'Accuracy' : 'MAE'
                  ]}
                />
                <Legend />
                <Bar dataKey="accuracy" fill="#0f62fe" name="Accuracy (%)" />
              </BarChart>
            </ResponsiveContainer>
          )}
          <div className="mt-4 text-sm text-gray-600">
            <p>Real backtest results from time-series cross-validation on unemployment data.</p>
          </div>
        </Card>

        {/* Forecast Analysis */}
        <Card title={`${selectedMetricInfo.label} Analysis`}>
          {forecastAnalysis ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Current Value:</span>
                  <div className="text-lg font-bold text-gray-900">
                    {forecastAnalysis.current_value.toFixed(2)}{selectedMetricInfo.unit}
                  </div>
                </div>
                <div>
                  <span className="font-medium">Recent Trend:</span>
                  <div className={`text-lg font-bold ${forecastAnalysis.recent_trend > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {forecastAnalysis.recent_trend > 0 ? '+' : ''}{forecastAnalysis.recent_trend.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <span className="font-medium">Volatility:</span>
                  <div className="text-lg font-bold text-gray-900">
                    {forecastAnalysis.volatility.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <span className="font-medium">Data Points:</span>
                  <div className="text-lg font-bold text-gray-900">
                    {forecastAnalysis.statistics.data_points}
                  </div>
                </div>
              </div>
              
              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-700 mb-2">Economic Context</h4>
                <p className="text-sm text-gray-600">
                  {forecastAnalysis.analysis.economic_context}
                </p>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-700 mb-2">Risk Factors</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  {forecastAnalysis.analysis.risk_factors.map((factor, idx) => (
                    <li key={idx}>• {factor}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 text-gray-500">
              <Lightbulb className="w-8 h-8 mr-2" />
              Loading analysis...
            </div>
          )}
        </Card>
      </div>

      {/* Model Insights */}
      <Card title="AI Model Explanations">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {modelInsights.map((model, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <div className="p-2 bg-blue-100 rounded-lg mr-3">
                  <Brain className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{model.name}</h3>
                  <p className="text-sm text-gray-500">{model.type}</p>
                </div>
              </div>
              
              <p className="text-sm text-gray-600 mb-3">{model.description}</p>
              
              <div className="space-y-2">
                <div>
                  <span className="text-xs font-medium text-gray-500">STRENGTHS:</span>
                  <div className="text-sm text-gray-600">
                    {model.strengths.join(', ')}
                  </div>
                </div>
                
                <div>
                  <span className="text-xs font-medium text-gray-500">BEST FOR:</span>
                  <div className="text-sm text-gray-600">
                    {model.use_cases.join(', ')}
                  </div>
                </div>
                
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">
                    Complexity: <span className="font-medium">{model.computational_complexity}</span>
                  </span>
                  <span className="text-gray-500">
                    Interpretability: <span className="font-medium">{model.interpretability}</span>
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Forecast Table */}
      <Card title="12-Month Detailed Forecast">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Month
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  {selectedMetricInfo.label}
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Change from Previous
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Trend
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {forecasts.slice(0, 12).map((forecast, index) => {
                const prevValue = index === 0 
                  ? lastHistorical[selectedMetric as keyof EconomicDataPoint] as number
                  : forecasts[index - 1][selectedMetric as keyof EconomicDataPoint] as number;
                const currentValue = forecast[selectedMetric as keyof EconomicDataPoint] as number;
                const change = currentValue - prevValue;
                
                return (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {new Date(forecast.date).toLocaleDateString('en-US', { 
                        year: 'numeric', 
                        month: 'short' 
                      })}
                    </td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">
                      {currentValue.toFixed(2)}{selectedMetricInfo.unit}
                    </td>
                    <td className={`px-4 py-3 text-sm font-medium ${change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {change > 0 ? '+' : ''}{change.toFixed(2)}{selectedMetricInfo.unit}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      {change > 0.1 ? (
                        <TrendingUp className="w-4 h-4 text-red-500" />
                      ) : change < -0.1 ? (
                        <TrendingUp className="w-4 h-4 text-green-500 rotate-180" />
                      ) : (
                        <div className="w-4 h-4 bg-gray-300 rounded-full" />
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default Forecasting;
