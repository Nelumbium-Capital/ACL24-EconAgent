import React, { useState, useEffect } from 'react';
import { AlertTriangle, Info, TrendingUp, TrendingDown, Shield } from 'lucide-react';
import Card from '../components/ui/Card';
import { apiService } from '../services/api';

interface KRI {
  name: string;
  display_name: string;
  value: number;
  unit: string;
  category: string;
  risk_level: string;
  is_leading: boolean;
  description: string;
  thresholds: Record<string, number>;
}

// KRI Explanations
const KRI_EXPLANATIONS: Record<string, string> = {
  'loan_default_rate': 'Percentage of loans where borrowers have failed to make payments. Critical indicator of credit portfolio health.',
  'delinquency_rate': 'Leading indicator: Percentage of loans 30+ days past due. Predicts future defaults.',
  'credit_quality_score': 'Weighted average credit score of loan portfolio. Higher scores indicate lower risk borrowers.',
  'loan_concentration': 'Concentration of loans in top sectors/borrowers. High concentration increases portfolio risk.',
  'portfolio_volatility': 'Leading indicator: Standard deviation of portfolio returns. Measures market risk exposure.',
  'var_95': 'Value at Risk (95%): Maximum expected loss at 95% confidence level over a given time period.',
  'interest_rate_risk': 'Duration measure: Sensitivity to interest rate changes. Higher = more interest rate risk.',
  'liquidity_coverage_ratio': 'Leading indicator: Ratio of liquid assets to net cash outflows. Must be >100% for safety.',
  'deposit_flow_ratio': 'Leading indicator: Net deposit inflows/outflows as % of total deposits. Negative = deposit flight.',
  'capital_adequacy_ratio': 'Ratio of capital to risk-weighted assets. Regulatory minimum is typically 8-10%.',
  'leverage_ratio': 'Total assets divided by equity. Higher leverage = higher risk.',
  'net_interest_margin': 'Difference between interest income and interest paid, as % of assets.',
  'return_on_assets': 'Net income divided by total assets. Measures profitability efficiency.',
  'return_on_equity': 'Net income divided by equity. Measures shareholder returns.',
  'efficiency_ratio': 'Operating expenses divided by revenue. Lower is better (more efficient).'
};

const RiskAnalysis: React.FC = () => {
  const [kris, setKris] = useState<KRI[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');

  useEffect(() => {
    const fetchKRIs = async () => {
      try {
        const data = await apiService.getKRIs();
        setKris(data);
      } catch (error) {
        console.error('Failed to fetch KRIs:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchKRIs();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading risk indicators...</div>
      </div>
    );
  }

  const categories = ['all', ...Array.from(new Set(kris.map(k => k.category)))];
  const riskLevels = ['all', 'critical', 'high', 'medium', 'low'];

  const filteredKRIs = kris.filter(kri => {
    const categoryMatch = selectedCategory === 'all' || kri.category === selectedCategory;
    const riskMatch = selectedRiskLevel === 'all' || kri.risk_level === selectedRiskLevel;
    return categoryMatch && riskMatch;
  });

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-300';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'medium': return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'low': return 'bg-green-100 text-green-800 border-green-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'critical':
      case 'high':
        return <AlertTriangle className="w-5 h-5" />;
      case 'medium':
        return <TrendingUp className="w-5 h-5" />;
      case 'low':
        return <Shield className="w-5 h-5" />;
      default:
        return <Info className="w-5 h-5" />;
    }
  };

  const riskSummary = {
    critical: kris.filter(k => k.risk_level === 'critical').length,
    high: kris.filter(k => k.risk_level === 'high').length,
    medium: kris.filter(k => k.risk_level === 'medium').length,
    low: kris.filter(k => k.risk_level === 'low').length
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-600 to-red-800 rounded-xl p-6 text-white shadow-lg">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Risk Analysis</h1>
            <p className="text-red-100 text-lg">
              Comprehensive Key Risk Indicators (KRIs) across all risk categories
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-red-200">Total Indicators</div>
            <div className="text-4xl font-bold">{kris.length}</div>
          </div>
        </div>
      </div>

      {/* Risk Level Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Critical Risk</div>
              <div className="text-3xl font-bold text-red-600">{riskSummary.critical}</div>
            </div>
            <AlertTriangle className="w-10 h-10 text-red-500" />
          </div>
          <div className="mt-2 text-xs text-gray-600">
            Immediate action required
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">High Risk</div>
              <div className="text-3xl font-bold text-orange-600">{riskSummary.high}</div>
            </div>
            <AlertTriangle className="w-10 h-10 text-orange-500" />
          </div>
          <div className="mt-2 text-xs text-gray-600">
            Close monitoring needed
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Medium Risk</div>
              <div className="text-3xl font-bold text-blue-600">{riskSummary.medium}</div>
            </div>
            <TrendingUp className="w-10 h-10 text-blue-500" />
          </div>
          <div className="mt-2 text-xs text-gray-600">
            Normal monitoring
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 mb-1">Low Risk</div>
              <div className="text-3xl font-bold text-green-600">{riskSummary.low}</div>
            </div>
            <Shield className="w-10 h-10 text-green-500" />
          </div>
          <div className="mt-2 text-xs text-gray-600">
            Within acceptable limits
          </div>
        </Card>
      </div>

      {/* Explanation Panel */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-300 rounded-xl p-6">
        <div className="flex items-start">
          <div className="p-3 bg-blue-600 rounded-lg mr-4">
            <Info className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-900 mb-3">Understanding Key Risk Indicators (KRIs)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-700">
              <div>
                <p className="font-semibold mb-2">What are KRIs?</p>
                <p className="mb-2">
                  Key Risk Indicators are metrics that provide early warning signals about 
                  potential risk exposures. They help organizations monitor and manage risk 
                  before issues become critical.
                </p>
                <p className="font-semibold mb-1">Leading vs. Lagging:</p>
                <p className="mb-1">• <span className="font-semibold">Leading indicators</span> predict future problems (e.g., delinquency rate)</p>
                <p>• <span className="font-semibold">Lagging indicators</span> confirm past performance (e.g., default rate)</p>
              </div>
              <div>
                <p className="font-semibold mb-2">Risk Categories</p>
                <p className="mb-1">• <span className="font-semibold text-blue-700">Credit Risk:</span> Loan defaults and credit quality</p>
                <p className="mb-1">• <span className="font-semibold text-purple-700">Market Risk:</span> Volatility and interest rate exposure</p>
                <p className="mb-1">• <span className="font-semibold text-green-700">Liquidity Risk:</span> Cash availability and deposit stability</p>
                <p>• <span className="font-semibold text-orange-700">Operational Risk:</span> Capital adequacy and profitability</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Risk Category
            </label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {categories.map(cat => (
                <option key={cat} value={cat}>
                  {cat === 'all' ? 'All Categories' : cat.charAt(0).toUpperCase() + cat.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Risk Level
            </label>
            <select
              value={selectedRiskLevel}
              onChange={(e) => setSelectedRiskLevel(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {riskLevels.map(level => (
                <option key={level} value={level}>
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="ml-auto text-sm text-gray-600">
            Showing {filteredKRIs.length} of {kris.length} indicators
          </div>
        </div>
      </Card>

      {/* KRI Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredKRIs.map((kri) => (
          <Card key={kri.name}>
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className="text-lg font-bold text-gray-900">{kri.display_name}</h3>
                    {kri.is_leading && (
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-semibold rounded">
                        LEADING
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 uppercase font-medium">{kri.category} Risk</p>
                </div>
                <div className={`flex items-center space-x-2 px-3 py-1 border rounded-lg ${getRiskColor(kri.risk_level)}`}>
                  {getRiskIcon(kri.risk_level)}
                  <span className="text-sm font-semibold uppercase">{kri.risk_level}</span>
                </div>
              </div>

              {/* Current Value */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Current Value</div>
                <div className="text-3xl font-bold text-gray-900">
                  {kri.value.toFixed(2)}{kri.unit}
                </div>
              </div>

              {/* Explanation */}
              <div className="p-3 bg-blue-50 border-l-4 border-blue-500 rounded">
                <p className="text-sm text-gray-700">
                  <span className="font-semibold">What this means:</span> {KRI_EXPLANATIONS[kri.name] || kri.description}
                </p>
              </div>

              {/* Thresholds */}
              <div>
                <div className="text-sm font-semibold text-gray-700 mb-2">Risk Thresholds</div>
                <div className="grid grid-cols-4 gap-2">
                  <div className="text-center">
                    <div className="text-xs text-gray-600 mb-1">Low</div>
                    <div className="text-sm font-semibold text-green-700">
                      {kri.thresholds.low?.toFixed(1) || 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-600 mb-1">Medium</div>
                    <div className="text-sm font-semibold text-blue-700">
                      {kri.thresholds.medium?.toFixed(1) || 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-600 mb-1">High</div>
                    <div className="text-sm font-semibold text-orange-700">
                      {kri.thresholds.high?.toFixed(1) || 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-600 mb-1">Critical</div>
                    <div className="text-sm font-semibold text-red-700">
                      {kri.thresholds.critical?.toFixed(1) || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Visual Threshold Bar */}
              <div className="relative h-4 bg-gradient-to-r from-green-200 via-blue-200 via-orange-200 to-red-200 rounded-full overflow-hidden">
                <div 
                  className="absolute top-0 bottom-0 w-1 bg-gray-900"
                  style={{ 
                    left: `${Math.min(100, Math.max(0, (kri.value / (kri.thresholds.critical * 1.2)) * 100))}%` 
                  }}
                >
                  <div className="absolute -top-1 -left-2 w-5 h-6 bg-gray-900 rounded-sm"></div>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {filteredKRIs.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <Info className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-lg text-gray-600">No KRIs match the selected filters</p>
            <button
              onClick={() => {
                setSelectedCategory('all');
                setSelectedRiskLevel('all');
              }}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Clear Filters
            </button>
          </div>
        </Card>
      )}
    </div>
  );
};

export default RiskAnalysis;
