import React, { useState } from 'react';
import { RefreshCw, TrendingUp, TrendingDown, Minus, ExternalLink, Calendar } from 'lucide-react';
import { format } from 'date-fns';
import { EconomicSnapshot } from '../types/index.ts';

interface FREDDataPanelProps {
  data: EconomicSnapshot | null;
  onRefresh: () => Promise<void>;
}

const FREDDataPanel: React.FC<FREDDataPanelProps> = ({ data, onRefresh }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  };

  const getTrendIcon = (value: number, threshold: { good: number; warning: number }) => {
    if (value <= threshold.good) {
      return <TrendingDown className="w-4 h-4 text-success-600" />;
    } else if (value <= threshold.warning) {
      return <Minus className="w-4 h-4 text-warning-600" />;
    } else {
      return <TrendingUp className="w-4 h-4 text-danger-600" />;
    }
  };

  const getValueColor = (value: number, threshold: { good: number; warning: number }) => {
    if (value <= threshold.good) {
      return 'text-success-700';
    } else if (value <= threshold.warning) {
      return 'text-warning-700';
    } else {
      return 'text-danger-700';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return format(new Date(timestamp), 'MMM dd, yyyy HH:mm');
    } catch {
      return 'Unknown';
    }
  };

  if (!data) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">
            FRED Economic Data
          </h2>
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>
        
        <div className="flex items-center justify-center h-48 text-gray-500">
          <div className="text-center">
            <Calendar className="w-8 h-8 mx-auto mb-2" />
            <p>Loading FRED data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">
          FRED Economic Data
        </h2>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          title="Refresh FRED data"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="space-y-4">
        
        {/* Unemployment Rate */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getTrendIcon(data.unemployment_rate, { good: 4.0, warning: 6.0 })}
            <div>
              <p className="text-sm font-medium text-gray-700">Unemployment Rate</p>
              <p className="text-xs text-gray-500">UNRATE</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${getValueColor(data.unemployment_rate, { good: 4.0, warning: 6.0 })}`}>
              {data.unemployment_rate.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Inflation Rate */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getTrendIcon(Math.abs(data.inflation_rate - 2.0), { good: 0.5, warning: 1.0 })}
            <div>
              <p className="text-sm font-medium text-gray-700">Inflation Rate</p>
              <p className="text-xs text-gray-500">CPI-U (YoY)</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${getValueColor(Math.abs(data.inflation_rate - 2.0), { good: 0.5, warning: 1.0 })}`}>
              {data.inflation_rate.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Fed Funds Rate */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            <Minus className="w-4 h-4 text-primary-600" />
            <div>
              <p className="text-sm font-medium text-gray-700">Fed Funds Rate</p>
              <p className="text-xs text-gray-500">FEDFUNDS</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-primary-700">
              {data.fed_funds_rate.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* GDP Growth */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getTrendIcon(-data.gdp_growth + 3.0, { good: 1.0, warning: 2.0 })}
            <div>
              <p className="text-sm font-medium text-gray-700">GDP Growth</p>
              <p className="text-xs text-gray-500">Real GDP (Annualized)</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${data.gdp_growth >= 2.0 ? 'text-success-700' : data.gdp_growth >= 0 ? 'text-warning-700' : 'text-danger-700'}`}>
              {data.gdp_growth.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Wage Growth */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getTrendIcon(-data.wage_growth + 4.0, { good: 1.0, warning: 2.0 })}
            <div>
              <p className="text-sm font-medium text-gray-700">Wage Growth</p>
              <p className="text-xs text-gray-500">Avg Hourly Earnings (YoY)</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${data.wage_growth >= 3.0 ? 'text-success-700' : data.wage_growth >= 2.0 ? 'text-warning-700' : 'text-danger-700'}`}>
              {data.wage_growth.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Labor Participation */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            {getTrendIcon(-data.labor_participation + 65.0, { good: 2.0, warning: 4.0 })}
            <div>
              <p className="text-sm font-medium text-gray-700">Labor Participation</p>
              <p className="text-xs text-gray-500">CIVPART</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${data.labor_participation >= 63.0 ? 'text-success-700' : data.labor_participation >= 61.0 ? 'text-warning-700' : 'text-danger-700'}`}>
              {data.labor_participation.toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Data Timestamp */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>Last Updated:</span>
          <span>{formatTimestamp(data.timestamp)}</span>
        </div>
      </div>

      {/* Economic Summary */}
      <div className="mt-4 p-3 bg-primary-50 rounded-lg">
        <h3 className="text-sm font-medium text-primary-900 mb-2">Economic Summary</h3>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-primary-700">Economic Health:</span>
            <span className={`font-medium ${
              data.unemployment_rate <= 5.0 && data.gdp_growth >= 2.0 
                ? 'text-success-700' 
                : 'text-warning-700'
            }`}>
              {data.unemployment_rate <= 5.0 && data.gdp_growth >= 2.0 ? 'Good' : 'Moderate'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-primary-700">Inflation Target:</span>
            <span className={`font-medium ${
              Math.abs(data.inflation_rate - 2.0) <= 0.5 
                ? 'text-success-700' 
                : 'text-warning-700'
            }`}>
              {Math.abs(data.inflation_rate - 2.0) <= 0.5 ? 'On Target' : 'Off Target'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-primary-700">Policy Stance:</span>
            <span className="font-medium text-primary-800">
              {data.fed_funds_rate < 2.0 ? 'Accommodative' : 
               data.fed_funds_rate > 4.0 ? 'Restrictive' : 'Neutral'}
            </span>
          </div>
        </div>
      </div>

      {/* FRED Link */}
      <div className="mt-4 text-center">
        <a
          href="https://fred.stlouisfed.org/"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center text-xs text-primary-600 hover:text-primary-700 transition-colors"
        >
          <ExternalLink className="w-3 h-3 mr-1" />
          View on FRED
        </a>
      </div>
    </div>
  );
};

export default FREDDataPanel;