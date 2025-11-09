import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area
} from 'recharts';
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

  // Phillips Curve data (Unemployment vs Inflation)
  const phillipsData = economicData.map(d => ({
    unemployment: d.unemployment,
    inflation: d.inflation,
    date: d.date
  }));

  // Interest Rate vs Inflation (Taylor Rule)
  const taylorData = economicData.map(d => ({
    inflation: d.inflation,
    interest_rate: d.interest_rate,
    date: d.date
  }));

  // Volatility analysis
  const volatilityData = economicData.map((d, i) => {
    if (i === 0) return { date: d.date, volatility: 0 };
    
    const prev = economicData[i - 1];
    const volatility = Math.abs(d.unemployment - prev.unemployment) + 
                      Math.abs(d.inflation - prev.inflation) + 
                      Math.abs(d.interest_rate - prev.interest_rate);
    
    return {
      date: d.date,
      volatility,
      unemployment: d.unemployment,
      inflation: d.inflation,
      interest_rate: d.interest_rate
    };
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Market Dynamics</h1>
        <p className="text-gray-600 mt-2">
          Analysis of macroeconomic relationships and market interactions
        </p>
      </div>

      {/* Economic Relationships */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Phillips Curve */}
        <Card title="Phillips Curve (Unemployment vs Inflation)">
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart data={phillipsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="unemployment" 
                name="Unemployment"
                unit="%"
                type="number"
              />
              <YAxis 
                dataKey="inflation" 
                name="Inflation"
                unit="%"
                type="number"
              />
              <Tooltip 
                formatter={(value: number, name: string) => [
                  `${value.toFixed(2)}%`, 
                  name === 'unemployment' ? 'Unemployment' : 'Inflation'
                ]}
              />
              <Scatter 
                dataKey="inflation" 
                fill="#0f62fe"
                name="Data Points"
              />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="mt-4 text-sm text-gray-600">
            The Phillips Curve shows the historical relationship between unemployment and inflation rates.
          </div>
        </Card>

        {/* Taylor Rule */}
        <Card title="Taylor Rule (Interest Rate vs Inflation)">
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart data={taylorData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="inflation" 
                name="Inflation"
                unit="%"
                type="number"
              />
              <YAxis 
                dataKey="interest_rate" 
                name="Interest Rate"
                unit="%"
                type="number"
              />
              <Tooltip 
                formatter={(value: number, name: string) => [
                  `${value.toFixed(2)}%`, 
                  name === 'inflation' ? 'Inflation' : 'Interest Rate'
                ]}
              />
              <Scatter 
                dataKey="interest_rate" 
                fill="#24a148"
                name="Data Points"
              />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="mt-4 text-sm text-gray-600">
            The Taylor Rule relationship between inflation and central bank interest rate policy.
          </div>
        </Card>
      </div>

      {/* Market Volatility */}
      <Card title="Market Volatility Analysis">
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={volatilityData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
              formatter={(value: number) => [`${value.toFixed(2)}`, 'Volatility Index']}
            />
            <Area 
              type="monotone" 
              dataKey="volatility" 
              stroke="#da1e28" 
              fill="#da1e28"
              fillOpacity={0.3}
              name="Volatility Index"
            />
          </AreaChart>
        </ResponsiveContainer>
        <div className="mt-4 text-sm text-gray-600">
          Composite volatility index based on changes in unemployment, inflation, and interest rates.
        </div>
      </Card>

      {/* Correlation Matrix */}
      <Card title="Economic Indicator Correlations">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Interest Rate vs Credit Spread */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Interest Rate vs Credit Spread</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={economicData.slice(-24)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="interest_rate" 
                  stroke="#24a148" 
                  strokeWidth={2}
                  name="Interest Rate"
                />
                <Line 
                  type="monotone" 
                  dataKey="credit_spread" 
                  stroke="#f1c21b" 
                  strokeWidth={2}
                  name="Credit Spread"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Unemployment vs Inflation Trend */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Unemployment vs Inflation Trend</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={economicData.slice(-24)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="unemployment" 
                  stroke="#0f62fe" 
                  strokeWidth={2}
                  name="Unemployment"
                />
                <Line 
                  type="monotone" 
                  dataKey="inflation" 
                  stroke="#da1e28" 
                  strokeWidth={2}
                  name="Inflation"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Card>

      {/* Market Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 mb-2">
              {(economicData[economicData.length - 1]?.unemployment - 
                economicData[economicData.length - 12]?.unemployment || 0).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Unemployment Change (YoY)</div>
          </div>
        </Card>
        
        <Card>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600 mb-2">
              {((economicData[economicData.length - 1]?.inflation || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Current Inflation Rate</div>
          </div>
        </Card>
        
        <Card>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600 mb-2">
              {(economicData[economicData.length - 1]?.credit_spread || 0).toFixed(2)}%
            </div>
            <div className="text-sm text-gray-600">Current Credit Spread</div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default MarketDynamics;
