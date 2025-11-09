import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import Card from '../components/ui/Card';
import { apiService } from '../services/api';
import { KRI } from '../types';

const RISK_COLORS = {
  low: '#24a148',
  medium: '#f1c21b', 
  high: '#fd7e14',
  critical: '#da1e28'
};

const RISK_ICONS = {
  low: CheckCircle,
  medium: AlertTriangle,
  high: AlertTriangle,
  critical: XCircle
};

const RiskAnalysis: React.FC = () => {
  const [kris, setKris] = useState<KRI[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

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
        <div className="text-lg text-gray-600">Loading risk analysis...</div>
      </div>
    );
  }

  const categories = ['all', 'credit', 'market', 'liquidity'];
  const filteredKRIs = selectedCategory === 'all' 
    ? kris 
    : kris.filter(kri => kri.category === selectedCategory);

  const riskCounts = {
    critical: kris.filter(k => k.risk_level === 'critical').length,
    high: kris.filter(k => k.risk_level === 'high').length,
    medium: kris.filter(k => k.risk_level === 'medium').length,
    low: kris.filter(k => k.risk_level === 'low').length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Risk Analysis</h1>
          <p className="text-gray-600 mt-2">
            Comprehensive analysis of key risk indicators across all categories
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category.charAt(0).toUpperCase() + category.slice(1)} Risk
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Risk Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {Object.entries(riskCounts).map(([level, count]) => {
          const color = RISK_COLORS[level as keyof typeof RISK_COLORS];
          const Icon = RISK_ICONS[level as keyof typeof RISK_ICONS];
          
          return (
            <Card key={level}>
              <div className="flex items-center">
                <div 
                  className="p-3 rounded-lg mr-4"
                  style={{ backgroundColor: `${color}20`, color }}
                >
                  <Icon className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-gray-900">{count}</div>
                  <div className="text-sm text-gray-600 capitalize">
                    {level} Risk KRIs
                  </div>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Risk Heatmap */}
      <Card title="Risk Heatmap">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {filteredKRIs.map((kri) => {
            const color = RISK_COLORS[kri.risk_level];
            const Icon = RISK_ICONS[kri.risk_level];
            
            return (
              <div
                key={kri.name}
                className="p-4 rounded-lg border-2 transition-all hover:shadow-md"
                style={{ borderColor: color }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <Icon 
                      className="w-5 h-5 mr-2" 
                      style={{ color }}
                    />
                    <span className="font-medium text-gray-900">
                      {kri.display_name}
                    </span>
                  </div>
                  <span 
                    className="px-2 py-1 text-xs font-medium rounded-full text-white"
                    style={{ backgroundColor: color }}
                  >
                    {kri.risk_level.toUpperCase()}
                  </span>
                </div>
                
                <div className="text-2xl font-bold text-gray-900 mb-1">
                  {kri.value.toFixed(2)} {kri.unit}
                </div>
                
                <div className="text-sm text-gray-600 mb-2">
                  {kri.description}
                </div>
                
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span className="capitalize">{kri.category} Risk</span>
                  <span>{kri.is_leading ? 'Leading' : 'Lagging'}</span>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Detailed KRI Table */}
      <Card title="Detailed Risk Indicators">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  KRI Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredKRIs.map((kri) => {
                const color = RISK_COLORS[kri.risk_level];
                const Icon = RISK_ICONS[kri.risk_level];
                
                return (
                  <tr key={kri.name} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Icon 
                          className="w-4 h-4 mr-2" 
                          style={{ color }}
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {kri.display_name}
                          </div>
                          <div className="text-sm text-gray-500">
                            {kri.description}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded-full capitalize">
                        {kri.category}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {kri.value.toFixed(2)} {kri.unit}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span 
                        className="px-2 py-1 text-xs font-medium rounded-full text-white"
                        style={{ backgroundColor: color }}
                      >
                        {kri.risk_level.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        kri.is_leading 
                          ? 'bg-blue-100 text-blue-800' 
                          : 'bg-purple-100 text-purple-800'
                      }`}>
                        {kri.is_leading ? 'Leading' : 'Lagging'}
                      </span>
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

export default RiskAnalysis;
