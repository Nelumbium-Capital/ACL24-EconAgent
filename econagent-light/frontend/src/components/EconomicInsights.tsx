import React from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Info, Target } from 'lucide-react';
import { EconomicSnapshot, SimulationStatus } from '../types/index.ts';

interface EconomicInsightsProps {
  fredData: EconomicSnapshot | null;
  activeSimulation: SimulationStatus | null;
}

const EconomicInsights: React.FC<EconomicInsightsProps> = ({ fredData, activeSimulation }) => {
  
  const getEconomicHealth = () => {
    if (!fredData) return null;
    
    const { unemployment_rate, inflation_rate, gdp_growth, fed_funds_rate } = fredData;
    
    // Economic health scoring
    let score = 0;
    const factors = [];
    
    // Unemployment assessment
    if (unemployment_rate <= 4.0) {
      score += 25;
      factors.push({ name: 'Low Unemployment', status: 'good', value: `${unemployment_rate.toFixed(1)}%` });
    } else if (unemployment_rate <= 6.0) {
      score += 15;
      factors.push({ name: 'Moderate Unemployment', status: 'warning', value: `${unemployment_rate.toFixed(1)}%` });
    } else {
      score += 5;
      factors.push({ name: 'High Unemployment', status: 'poor', value: `${unemployment_rate.toFixed(1)}%` });
    }
    
    // Inflation assessment
    if (Math.abs(inflation_rate - 2.0) <= 0.5) {
      score += 25;
      factors.push({ name: 'Target Inflation', status: 'good', value: `${inflation_rate.toFixed(1)}%` });
    } else if (Math.abs(inflation_rate - 2.0) <= 1.0) {
      score += 15;
      factors.push({ name: 'Near-Target Inflation', status: 'warning', value: `${inflation_rate.toFixed(1)}%` });
    } else {
      score += 5;
      factors.push({ name: 'Off-Target Inflation', status: 'poor', value: `${inflation_rate.toFixed(1)}%` });
    }
    
    // GDP growth assessment
    if (gdp_growth >= 2.5) {
      score += 25;
      factors.push({ name: 'Strong GDP Growth', status: 'good', value: `${gdp_growth.toFixed(1)}%` });
    } else if (gdp_growth >= 1.0) {
      score += 15;
      factors.push({ name: 'Moderate GDP Growth', status: 'warning', value: `${gdp_growth.toFixed(1)}%` });
    } else {
      score += 5;
      factors.push({ name: 'Weak GDP Growth', status: 'poor', value: `${gdp_growth.toFixed(1)}%` });
    }
    
    // Interest rate assessment
    if (fed_funds_rate >= 2.0 && fed_funds_rate <= 4.0) {
      score += 25;
      factors.push({ name: 'Neutral Policy Rate', status: 'good', value: `${fed_funds_rate.toFixed(1)}%` });
    } else if (fed_funds_rate < 2.0) {
      score += 15;
      factors.push({ name: 'Accommodative Policy', status: 'warning', value: `${fed_funds_rate.toFixed(1)}%` });
    } else {
      score += 10;
      factors.push({ name: 'Restrictive Policy', status: 'warning', value: `${fed_funds_rate.toFixed(1)}%` });
    }
    
    return { score, factors };
  };

  const getSimulationInsights = () => {
    if (!activeSimulation?.current_metrics) return null;
    
    const { unemployment_rate, inflation_rate, gdp_growth } = activeSimulation.current_metrics;
    
    const insights = [];
    
    // Compare with FRED data
    if (fredData) {
      const unemploymentDiff = unemployment_rate - fredData.unemployment_rate;
      const inflationDiff = inflation_rate - fredData.inflation_rate;
      const gdpDiff = gdp_growth - fredData.gdp_growth;
      
      insights.push({
        title: 'vs Real Economy',
        items: [
          {
            label: 'Unemployment',
            diff: unemploymentDiff,
            better: unemploymentDiff < 0,
            value: `${unemploymentDiff > 0 ? '+' : ''}${unemploymentDiff.toFixed(1)}pp`
          },
          {
            label: 'Inflation',
            diff: Math.abs(inflationDiff),
            better: Math.abs(inflationDiff) < Math.abs(fredData.inflation_rate - 2.0),
            value: `${inflationDiff > 0 ? '+' : ''}${inflationDiff.toFixed(1)}pp`
          },
          {
            label: 'GDP Growth',
            diff: gdpDiff,
            better: gdpDiff > 0,
            value: `${gdpDiff > 0 ? '+' : ''}${gdpDiff.toFixed(1)}pp`
          }
        ]
      });
    }
    
    // Economic relationships
    const phillipsSlope = unemployment_rate * inflation_rate;
    insights.push({
      title: 'Economic Relationships',
      items: [
        {
          label: 'Phillips Curve Position',
          value: phillipsSlope < 10 ? 'Favorable' : phillipsSlope < 20 ? 'Moderate' : 'Challenging',
          better: phillipsSlope < 15
        },
        {
          label: 'Policy Stance Needed',
          value: inflation_rate > 3 ? 'Restrictive' : inflation_rate < 1 ? 'Accommodative' : 'Neutral',
          better: inflation_rate >= 1.5 && inflation_rate <= 2.5
        }
      ]
    });
    
    return insights;
  };

  const economicHealth = getEconomicHealth();
  const simulationInsights = getSimulationInsights();

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-600" />;
      case 'poor':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <Info className="w-4 h-4 text-blue-600" />;
    }
  };

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };

  return (
    <div className="space-y-6">
      
      {/* Economic Health Score */}
      {economicHealth && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Economic Health Score</h3>
            <div className={`text-2xl font-bold ${getHealthColor(economicHealth.score)}`}>
              {economicHealth.score}/100
            </div>
          </div>
          
          <div className="mb-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span>Overall Health</span>
              <span className={`font-medium ${getHealthColor(economicHealth.score)}`}>
                {getHealthLabel(economicHealth.score)}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-500 ${
                  economicHealth.score >= 80 ? 'bg-green-500' :
                  economicHealth.score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${economicHealth.score}%` }}
              ></div>
            </div>
          </div>

          <div className="space-y-2">
            {economicHealth.factors.map((factor, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(factor.status)}
                  <span className="text-sm font-medium">{factor.name}</span>
                </div>
                <span className="text-sm text-gray-600">{factor.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Simulation Insights */}
      {simulationInsights && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Insights</h3>
          
          <div className="space-y-4">
            {simulationInsights.map((section, sectionIndex) => (
              <div key={sectionIndex}>
                <h4 className="text-sm font-medium text-gray-700 mb-2">{section.title}</h4>
                <div className="space-y-2">
                  {section.items.map((item, itemIndex) => (
                    <div key={itemIndex} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center space-x-2">
                        {item.better ? (
                          <TrendingUp className="w-4 h-4 text-green-600" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-600" />
                        )}
                        <span className="text-sm">{item.label}</span>
                      </div>
                      <span className={`text-sm font-medium ${
                        item.better ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {item.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Economic Targets */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Economic Targets</h3>
        
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium">Inflation Target</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-blue-900">2.0%</div>
              <div className="text-xs text-blue-600">Fed Target</div>
            </div>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium">Full Employment</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-green-900">≤ 4.0%</div>
              <div className="text-xs text-green-600">Natural Rate</div>
            </div>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-medium">GDP Growth</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-purple-900">≥ 2.5%</div>
              <div className="text-xs text-purple-600">Trend Growth</div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        
        <div className="grid grid-cols-2 gap-3">
          <button className="p-3 text-left bg-primary-50 hover:bg-primary-100 rounded-lg transition-colors">
            <div className="text-sm font-medium text-primary-900">Export Data</div>
            <div className="text-xs text-primary-600">Download FRED data</div>
          </button>
          
          <button className="p-3 text-left bg-green-50 hover:bg-green-100 rounded-lg transition-colors">
            <div className="text-sm font-medium text-green-900">Compare Scenarios</div>
            <div className="text-xs text-green-600">Run multiple simulations</div>
          </button>
          
          <button className="p-3 text-left bg-amber-50 hover:bg-amber-100 rounded-lg transition-colors">
            <div className="text-sm font-medium text-amber-900">Policy Analysis</div>
            <div className="text-xs text-amber-600">Analyze interventions</div>
          </button>
          
          <button className="p-3 text-left bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors">
            <div className="text-sm font-medium text-purple-900">Forecast</div>
            <div className="text-xs text-purple-600">Predict trends</div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default EconomicInsights;