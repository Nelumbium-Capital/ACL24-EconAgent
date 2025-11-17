import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Info, 
  TrendingUp, 
  TrendingDown, 
  Shield, 
  RefreshCw,
  Target,
  Activity,
  BarChart3,
  Brain,
  Eye,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  ScatterChart,
  Scatter
} from 'recharts';
import Card from '../components/ui/Card';
import { apiService, KRI, RiskInsights, ForecastAnalysis } from '../services/api';

// KRI Explanations with detailed methodology
const KRI_EXPLANATIONS: Record<string, { 
  description: string; 
  methodology: string; 
  thresholds: string;
  regulatory_context: string;
}> = {
  'loan_default_rate': {
    description: 'Percentage of loans where borrowers have failed to make payments for 90+ days',
    methodology: 'Calculated as (Total defaulted loans / Total loans) × 100',
    thresholds: 'Low: <2%, Medium: 2-4%, High: 4-6%, Critical: >6%',
    regulatory_context: 'Basel III expects banks to maintain provisions for expected credit losses'
  },
  'delinquency_rate': {
    description: 'Leading indicator: Percentage of loans 30+ days past due but not yet defaulted',
    methodology: 'Rolling 12-month average of (Delinquent loans / Total loans) × 100',
    thresholds: 'Low: <3%, Medium: 3-6%, High: 6-9%, Critical: >9%',
    regulatory_context: 'Early warning indicator used by FDIC for prompt corrective action'
  },
  'credit_quality_score': {
    description: 'Weighted average credit score of entire loan portfolio',
    methodology: 'Σ(Loan Amount × FICO Score) / Total Portfolio Value',
    thresholds: 'Low risk: >740, Medium: 670-740, High: 600-670, Critical: <600',
    regulatory_context: 'Used in stress testing scenarios and capital adequacy calculations'
  },
  'portfolio_volatility': {
    description: 'Leading indicator: 30-day rolling volatility of portfolio returns',
    methodology: 'Standard deviation of daily portfolio returns over 30-day window',
    thresholds: 'Low: <15%, Medium: 15-25%, High: 25-35%, Critical: >35%',
    regulatory_context: 'Core component of market risk capital requirements under Basel III'
  },
  'liquidity_coverage_ratio': {
    description: 'Ratio of high-quality liquid assets to net cash outflows over 30 days',
    methodology: 'HQLA / Net Cash Outflows (30-day stress scenario)',
    thresholds: 'Regulatory minimum: 100%, Target: >110%, Strong: >120%',
    regulatory_context: 'Basel III mandates minimum 100% LCR for internationally active banks'
  }
};

const RiskAnalysis: React.FC = () => {
  const [kris, setKris] = useState<KRI[]>([]);
  const [riskInsights, setRiskInsights] = useState<RiskInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');
  const [selectedKRI, setSelectedKRI] = useState<KRI | null>(null);
  const [forecastAnalyses, setForecastAnalyses] = useState<Record<string, ForecastAnalysis>>({});

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [kriData, insightsData] = await Promise.all([
        apiService.getKRIs(),
        apiService.getRiskInsights()
      ]);
      
      setKris(kriData);
      setRiskInsights(insightsData);
      
      // Fetch forecast analyses for key indicators
      const economicIndicators = ['unemployment', 'inflation', 'interest_rate', 'credit_spread'];
      const analyses: Record<string, ForecastAnalysis> = {};
      
      for (const indicator of economicIndicators) {
        try {
          const analysis = await apiService.getForecastAnalysis(indicator);
          analyses[indicator] = analysis;
        } catch (error) {
          console.error(`Failed to fetch analysis for ${indicator}:`, error);
        }
      }
      
      setForecastAnalyses(analyses);
    } catch (error) {
      console.error('Failed to fetch risk data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg text-gray-600">Loading risk analysis...</p>
        </div>
      </div>
    );
  }

  // Filter KRIs
  const filteredKRIs = kris.filter(kri => {
    const categoryMatch = selectedCategory === 'all' || kri.category === selectedCategory;
    const riskMatch = selectedRiskLevel === 'all' || kri.risk_level === selectedRiskLevel;
    return categoryMatch && riskMatch;
  });

  // Group KRIs by category
  const krisByCategory = filteredKRIs.reduce((acc, kri) => {
    if (!acc[kri.category]) acc[kri.category] = [];
    acc[kri.category].push(kri);
    return acc;
  }, {} as Record<string, KRI[]>);

  // Risk level distribution
  const riskDistribution = kris.reduce((acc, kri) => {
    acc[kri.risk_level] = (acc[kri.risk_level] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const riskDistributionData = Object.entries(riskDistribution).map(([level, count]) => ({
    level: level.charAt(0).toUpperCase() + level.slice(1),
    count,
    color: level === 'critical' ? '#da1e28' : 
           level === 'high' ? '#ff832b' : 
           level === 'medium' ? '#f1c21b' : '#24a148'
  }));

  // Create radar chart data for risk categories
  const categoryRiskData = Object.entries(krisByCategory).map(([category, categoryKRIs]) => {
    const avgRisk = categoryKRIs.reduce((sum, kri) => {
      const riskScore = kri.risk_level === 'critical' ? 4 : 
                       kri.risk_level === 'high' ? 3 : 
                       kri.risk_level === 'medium' ? 2 : 1;
      return sum + riskScore;
    }, 0) / categoryKRIs.length;
    
    return {
      category: category.replace('_', ' ').toUpperCase(),
      risk: avgRisk,
      count: categoryKRIs.length
    };
  });

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'critical': return <XCircle className="w-5 h-5" />;
      case 'high': return <AlertCircle className="w-5 h-5" />;
      case 'medium': return <AlertTriangle className="w-5 h-5" />;
      default: return <CheckCircle className="w-5 h-5" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Risk Analysis</h1>
          <p className="text-gray-600 mt-2">Comprehensive Key Risk Indicator (KRI) monitoring and analysis</p>
        </div>
        
        <button
          onClick={fetchData}
          className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh Analysis
        </button>
      </div>

      {/* Risk Overview */}
      {riskInsights && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className={`p-6 rounded-xl border-2 ${
            riskInsights.overall_risk_level === 'Critical' ? 'bg-red-50 border-red-200' :
            riskInsights.overall_risk_level === 'High' ? 'bg-orange-50 border-orange-200' :
            riskInsights.overall_risk_level === 'Medium' ? 'bg-yellow-50 border-yellow-200' :
            'bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center mb-3">
              <Shield className={`w-6 h-6 mr-3 ${
                riskInsights.overall_risk_level === 'Critical' ? 'text-red-600' :
                riskInsights.overall_risk_level === 'High' ? 'text-orange-600' :
                riskInsights.overall_risk_level === 'Medium' ? 'text-yellow-600' :
                'text-green-600'
              }`} />
              <h3 className="font-semibold text-gray-900">Overall Risk</h3>
            </div>
            <div className="text-2xl font-bold mb-2">{riskInsights.overall_risk_level}</div>
            <p className="text-sm text-gray-600">{riskInsights.recommendation}</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-3">
              <AlertTriangle className="w-6 h-6 text-red-600 mr-3" />
              <h3 className="font-semibold text-gray-900">Critical</h3>
            </div>
            <div className="text-2xl font-bold mb-2">{riskInsights.risk_summary.critical}</div>
            <p className="text-sm text-gray-600">Immediate attention required</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-3">
              <Activity className="w-6 h-6 text-orange-600 mr-3" />
              <h3 className="font-semibold text-gray-900">High Risk</h3>
            </div>
            <div className="text-2xl font-bold mb-2">{riskInsights.risk_summary.high}</div>
            <p className="text-sm text-gray-600">Enhanced monitoring needed</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-3">
              <Target className="w-6 h-6 text-blue-600 mr-3" />
              <h3 className="font-semibold text-gray-900">Total KRIs</h3>
            </div>
            <div className="text-2xl font-bold mb-2">{kris.length}</div>
            <p className="text-sm text-gray-600">Actively monitored</p>
          </div>
        </div>
      )}

      {/* AI Insights */}
      {riskInsights && riskInsights.insights.length > 0 && (
        <Card title="🤖 AI Risk Insights">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {riskInsights.insights.map((insight, idx) => (
              <div key={idx} className={`p-4 rounded-lg border ${
                insight.type === 'alert' ? 'bg-red-50 border-red-200' :
                insight.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                'bg-blue-50 border-blue-200'
              }`}>
                <div className="flex items-start">
                  <Brain className={`w-5 h-5 mt-0.5 mr-3 ${
                    insight.type === 'alert' ? 'text-red-600' :
                    insight.type === 'warning' ? 'text-yellow-600' :
                    'text-blue-600'
                  }`} />
                  <div>
                    <h4 className="font-medium mb-1">{insight.category}</h4>
                    <p className="text-sm text-gray-700 mb-2">{insight.message}</p>
                    <p className="text-xs text-gray-600">Impact: {insight.impact}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Risk Distribution and Category Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Risk Level Distribution">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={riskDistributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#0f62fe" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Risk by Category">
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={categoryRiskData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis domain={[0, 4]} tick={false} />
              <Radar
                name="Risk Level"
                dataKey="risk"
                stroke="#0f62fe"
                fill="#0f62fe"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Tooltip 
                formatter={(value: number) => [
                  value === 4 ? 'Critical' : 
                  value === 3 ? 'High' : 
                  value === 2 ? 'Medium' : 'Low',
                  'Average Risk Level'
                ]}
              />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 items-center">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Category</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Categories</option>
            <option value="credit">Credit Risk</option>
            <option value="market">Market Risk</option>
            <option value="liquidity">Liquidity Risk</option>
            <option value="operational">Operational Risk</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Risk Level</label>
          <select
            value={selectedRiskLevel}
            onChange={(e) => setSelectedRiskLevel(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Risk Levels</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
      </div>

      {/* KRI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredKRIs.map((kri) => (
          <div 
            key={kri.name}
            className={`p-6 rounded-xl border-2 cursor-pointer transition-all hover:shadow-lg ${getRiskLevelColor(kri.risk_level)}`}
            onClick={() => setSelectedKRI(kri)}
          >
            <div className="flex justify-between items-start mb-4">
              <div className="flex items-center">
                {getRiskIcon(kri.risk_level)}
                <h3 className="font-semibold text-gray-900 ml-2">{kri.display_name}</h3>
              </div>
              {kri.is_leading && (
                <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                  Leading
                </span>
              )}
            </div>
            
            <div className="mb-4">
              <div className="text-2xl font-bold">
                {kri.value.toFixed(2)}{kri.unit}
              </div>
              <div className="text-sm text-gray-600 capitalize">
                {kri.risk_level} Risk
              </div>
            </div>
            
            <div className="text-sm text-gray-600 mb-3">
              {kri.description}
            </div>
            
            <div className="text-xs text-gray-500">
              Category: {kri.category.charAt(0).toUpperCase() + kri.category.slice(1)}
            </div>
          </div>
        ))}
      </div>

      {/* Economic Context Integration */}
      <Card title="Economic Context & Risk Correlation">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(forecastAnalyses).map(([indicator, analysis]) => (
            <div key={indicator} className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">
                {indicator.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </h4>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Current Value:</span>
                  <span className="font-medium">
                    {analysis.current_value.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Recent Trend:</span>
                  <span className={`font-medium ${analysis.recent_trend > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {analysis.recent_trend > 0 ? '+' : ''}{analysis.recent_trend.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Volatility:</span>
                  <span className="font-medium">{analysis.volatility.toFixed(2)}%</span>
                </div>
              </div>
              
              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-xs text-gray-600">
                  {analysis.analysis.current_interpretation}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* KRI Detail Modal */}
      {selectedKRI && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-screen overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{selectedKRI.display_name}</h2>
                  <p className="text-gray-600">Detailed Risk Analysis</p>
                </div>
                <button
                  onClick={() => setSelectedKRI(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>

              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Current Value</label>
                    <div className="text-2xl font-bold">{selectedKRI.value.toFixed(2)}{selectedKRI.unit}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700">Risk Level</label>
                    <div className={`text-2xl font-bold capitalize ${
                      selectedKRI.risk_level === 'critical' ? 'text-red-600' :
                      selectedKRI.risk_level === 'high' ? 'text-orange-600' :
                      selectedKRI.risk_level === 'medium' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {selectedKRI.risk_level}
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
                  <p className="text-gray-600">{selectedKRI.description}</p>
                </div>

                {KRI_EXPLANATIONS[selectedKRI.name] && (
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Methodology</h3>
                      <p className="text-gray-600">{KRI_EXPLANATIONS[selectedKRI.name].methodology}</p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Risk Thresholds</h3>
                      <p className="text-gray-600">{KRI_EXPLANATIONS[selectedKRI.name].thresholds}</p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Regulatory Context</h3>
                      <p className="text-gray-600">{KRI_EXPLANATIONS[selectedKRI.name].regulatory_context}</p>
                    </div>
                  </div>
                )}

                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Risk Thresholds</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                    {Object.entries(selectedKRI.thresholds).map(([level, value]) => (
                      <div key={level} className={`p-2 rounded ${
                        level === 'critical' ? 'bg-red-100 text-red-800' :
                        level === 'high' ? 'bg-orange-100 text-orange-800' :
                        level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        <div className="font-medium capitalize">{level}</div>
                        <div>{value}{selectedKRI.unit}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <div>Category: <span className="font-medium capitalize">{selectedKRI.category}</span></div>
                  {selectedKRI.is_leading && (
                    <div className="flex items-center">
                      <TrendingUp className="w-4 h-4 mr-1" />
                      Leading Indicator
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskAnalysis;
