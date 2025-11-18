import React, { useState, useEffect } from 'react';
import {
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart
} from 'recharts';
import { TrendingUp, TrendingDown, Activity, Info, AlertCircle } from 'lucide-react';
import Card from '../components/ui/Card';
import { apiService } from '../services/api';
import { EconomicDataPoint } from '../types';

const MarketDynamics: React.FC = () => {
  const [economicData, setEconomicData] = useState<EconomicDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const economicDataRes = await apiService.getEconomicData();
        setEconomicData(economicDataRes);
      } catch (error) {
        console.error('Failed to fetch market dynamics data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading market dynamics...</div>
      </div>
    );
  }

  // Calculate volatility (rolling std dev)
  const calculateVolatility = (data: number[], window: number = 6) => {
    const result = [];
    for (let i = window; i < data.length; i++) {
      const slice = data.slice(i - window, i);
      const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / slice.length;
      result.push(Math.sqrt(variance));
    }
    return result;
  };

  const recentData = economicData.slice(-36); // Last 3 years
  const unemploymentVol = calculateVolatility(recentData.map(d => d.unemployment));
  const inflationVol = calculateVolatility(recentData.map(d => d.inflation * 100));

  // Prepare volatility chart data
  const volatilityData = recentData.slice(6).map((d, i) => ({
    date: d.date,
    unemployment_vol: unemploymentVol[i],
    inflation_vol: inflationVol[i]
  }));

  // Calculate momentum (percentage change month-over-month)
  const momentumData = recentData.slice(1).map((d, i) => {
    const prev = recentData[i];
    return {
      date: d.date,
      unemployment_momentum: prev.unemployment !== 0
        ? ((d.unemployment - prev.unemployment) / prev.unemployment) * 100
        : 0,
      inflation_momentum: prev.inflation !== 0
        ? ((d.inflation - prev.inflation) / prev.inflation) * 100
        : 0
    };
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-800 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-3xl font-bold mb-2">Market Dynamics Analysis</h1>
        <p className="text-purple-100 text-lg">
          Real-time analysis of economic trends, volatility, and momentum indicators
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Current Unemployment</div>
              <div className="text-2xl font-bold text-gray-900">
                {recentData[recentData.length - 1]?.unemployment.toFixed(2)}%
              </div>
            </div>
            <TrendingDown className="w-8 h-8 text-green-500" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            {recentData[recentData.length - 1]?.unemployment < recentData[recentData.length - 2]?.unemployment ? 
              '↓ Decreasing' : '↑ Increasing'}
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Current Inflation</div>
              <div className="text-2xl font-bold text-gray-900">
                {(recentData[recentData.length - 1]?.inflation * 100).toFixed(2)}%
              </div>
            </div>
            <Activity className="w-8 h-8 text-blue-500" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            {recentData[recentData.length - 1]?.inflation < recentData[recentData.length - 2]?.inflation ? 
              '↓ Cooling' : '↑ Heating'}
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Interest Rate</div>
              <div className="text-2xl font-bold text-gray-900">
                {recentData[recentData.length - 1]?.interest_rate.toFixed(2)}%
              </div>
            </div>
            <TrendingUp className="w-8 h-8 text-orange-500" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Federal Reserve policy rate
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Credit Spread</div>
              <div className="text-2xl font-bold text-gray-900">
                {recentData[recentData.length - 1]?.credit_spread.toFixed(2)}%
              </div>
            </div>
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Corporate risk premium
          </div>
        </Card>
      </div>

      {/* Economic Indicators with Explanation */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold text-gray-900">Economic Indicators Over Time</h3>
            <p className="text-sm text-gray-600 mt-1">36-month historical view of key metrics</p>
          </div>
          <div className="group relative">
            <Info className="w-5 h-5 text-gray-400 cursor-help" />
            <div className="absolute right-0 top-6 w-96 p-4 bg-gray-900 text-white text-sm rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
              <p className="font-semibold mb-2">Understanding Economic Indicators</p>
              <p className="mb-2"><span className="font-semibold">Unemployment Rate:</span> Percentage of labor force without jobs. Higher = weaker economy.</p>
              <p className="mb-2"><span className="font-semibold">Inflation Rate:</span> Rate of price increases. Moderate inflation (2-3%) is healthy.</p>
              <p className="mb-2"><span className="font-semibold">Interest Rate:</span> Federal Reserve's benchmark rate. Higher rates slow economy.</p>
              <p><span className="font-semibold">Credit Spread:</span> Risk premium investors demand. Higher = more credit risk.</p>
            </div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={recentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                month: 'short', 
                year: '2-digit' 
              })}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              yAxisId="left"
              label={{ value: 'Unemployment & Interest Rate (%)', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              yAxisId="right" 
              orientation="right"
              label={{ value: 'Inflation & Spread (%)', angle: 90, position: 'insideRight' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.98)', 
                border: '2px solid #333',
                borderRadius: '8px'
              }}
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <Legend />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="unemployment" 
              stroke="#3B82F6" 
              strokeWidth={3}
              dot={false}
              name="Unemployment (%)"
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="inflation" 
              stroke="#EF4444" 
              strokeWidth={3}
              dot={false}
              name="Inflation (%)"
            />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="interest_rate" 
              stroke="#10B981" 
              strokeWidth={3}
              dot={false}
              name="Interest Rate (%)"
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="credit_spread" 
              stroke="#F59E0B" 
              strokeWidth={3}
              dot={false}
              name="Credit Spread (%)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      {/* Volatility Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900">Market Volatility</h3>
              <p className="text-sm text-gray-600 mt-1">6-month rolling standard deviation</p>
            </div>
            <div className="group relative">
              <Info className="w-5 h-5 text-gray-400 cursor-help" />
              <div className="absolute right-0 top-6 w-80 p-4 bg-gray-900 text-white text-sm rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                <p className="font-semibold mb-2">What is Volatility?</p>
                <p className="mb-2">Volatility measures how much an indicator fluctuates over time.</p>
                <p className="mb-2"><span className="font-semibold">High volatility:</span> Large swings, more uncertainty</p>
                <p><span className="font-semibold">Low volatility:</span> Stable, predictable conditions</p>
              </div>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={volatilityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                  month: 'short', 
                  year: '2-digit' 
                })}
                tick={{ fontSize: 11 }}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number) => value.toFixed(3)}
              />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="unemployment_vol" 
                stroke="#3B82F6" 
                fill="#3B82F6"
                fillOpacity={0.3}
                name="Unemployment Volatility"
              />
              <Area 
                type="monotone" 
                dataKey="inflation_vol" 
                stroke="#EF4444" 
                fill="#EF4444"
                fillOpacity={0.3}
                name="Inflation Volatility"
              />
            </AreaChart>
          </ResponsiveContainer>

          <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-700">
            <p><span className="font-semibold">Current Status:</span> {
              volatilityData[volatilityData.length - 1]?.unemployment_vol > 0.2 ? 
              'Elevated volatility detected' : 
              'Normal market conditions'
            }</p>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900">Momentum Indicators</h3>
              <p className="text-sm text-gray-600 mt-1">Month-over-month rate of change</p>
            </div>
            <div className="group relative">
              <Info className="w-5 h-5 text-gray-400 cursor-help" />
              <div className="absolute right-0 top-6 w-80 p-4 bg-gray-900 text-white text-sm rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                <p className="font-semibold mb-2">What is Momentum?</p>
                <p className="mb-2">Momentum shows the direction and speed of change.</p>
                <p className="mb-2"><span className="font-semibold">Positive:</span> Indicator is accelerating upward</p>
                <p><span className="font-semibold">Negative:</span> Indicator is decelerating or falling</p>
              </div>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={momentumData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                  month: 'short', 
                  year: '2-digit' 
                })}
                tick={{ fontSize: 11 }}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number) => `${value > 0 ? '+' : ''}${value.toFixed(3)}%`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="unemployment_momentum" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={{ r: 3 }}
                name="Unemployment Momentum"
              />
              <Line 
                type="monotone" 
                dataKey="inflation_momentum" 
                stroke="#EF4444" 
                strokeWidth={2}
                dot={{ r: 3 }}
                name="Inflation Momentum"
              />
            </ComposedChart>
          </ResponsiveContainer>

          <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-700">
            <p><span className="font-semibold">Trend:</span> {
              momentumData[momentumData.length - 1]?.unemployment_momentum > 0 ? 
              'Unemployment rising' : 
              'Unemployment falling'
            }</p>
          </div>
        </Card>
      </div>

      {/* Explanation Panel */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-300 rounded-xl p-6">
        <div className="flex items-start">
          <div className="p-3 bg-blue-600 rounded-lg mr-4">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-900 mb-3">Understanding Market Dynamics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
              <div>
                <p className="font-semibold mb-2">Why Track These Indicators?</p>
                <p>
                  These four indicators (unemployment, inflation, interest rates, credit spreads) 
                  form the foundation of macroeconomic risk assessment. Together, they provide a 
                  comprehensive view of economic health and stability.
                </p>
              </div>
              <div>
                <p className="font-semibold mb-2">How They Interact</p>
                <p>
                  High unemployment typically correlates with low inflation. The Federal Reserve 
                  adjusts interest rates to balance these forces. Credit spreads widen when 
                  investors perceive higher default risk.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketDynamics;
