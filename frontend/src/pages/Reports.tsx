import React, { useState, useEffect } from 'react';
import { Download, FileText, Mail, Calendar } from 'lucide-react';
import Card from '../components/ui/Card';
import { apiService, DashboardSummary, KRI } from '../services/api';

const Reports: React.FC = () => {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [kris, setKris] = useState<KRI[]>([]);
  const [loading, setLoading] = useState(true);
  const [reportType, setReportType] = useState<string>('executive');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [summaryData, kriData] = await Promise.all([
          apiService.getDashboardSummary(),
          apiService.getKRIs()
        ]);
        
        setSummary(summaryData);
        setKris(kriData);
      } catch (error) {
        console.error('Failed to fetch reports data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleExportCSV = () => {
    if (!kris.length) return;
    
    const csvContent = [
      ['KRI Name', 'Category', 'Value', 'Unit', 'Risk Level', 'Type'].join(','),
      ...kris.map(kri => [
        kri.display_name,
        kri.category,
        kri.value.toFixed(2),
        kri.unit,
        kri.risk_level,
        kri.is_leading ? 'Leading' : 'Lagging'
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `risk-report-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      summary,
      kris
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `risk-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading reports...</div>
      </div>
    );
  }

  const reportTypes = [
    { value: 'executive', label: 'Executive Summary' },
    { value: 'detailed', label: 'Detailed Risk Analysis' },
    { value: 'compliance', label: 'Compliance Report' },
    { value: 'trend', label: 'Trend Analysis' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Reports</h1>
          <p className="text-gray-600 mt-2">
            Generate and export comprehensive risk analysis reports
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={reportType}
            onChange={(e) => setReportType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {reportTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Export Options */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="p-3 bg-blue-100 rounded-lg mr-4">
                <Download className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">Export CSV</div>
                <div className="text-xs text-gray-600">Download as spreadsheet</div>
              </div>
            </div>
            <button
              onClick={handleExportCSV}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Export
            </button>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="p-3 bg-green-100 rounded-lg mr-4">
                <FileText className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">Export JSON</div>
                <div className="text-xs text-gray-600">Download as JSON</div>
              </div>
            </div>
            <button
              onClick={handleExportJSON}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Export
            </button>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="p-3 bg-purple-100 rounded-lg mr-4">
                <Mail className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">Email Report</div>
                <div className="text-xs text-gray-600">Send via email</div>
              </div>
            </div>
            <button
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Send
            </button>
          </div>
        </Card>
      </div>

      {/* Report Preview */}
      <Card title={`${reportTypes.find(t => t.value === reportType)?.label} Preview`}>
        <div className="space-y-6">
          {/* Executive Summary */}
          {reportType === 'executive' && summary && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Executive Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">Critical Risks</div>
                  <div className="text-2xl font-bold text-red-600">{summary.risk_summary.critical}</div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">High Risks</div>
                  <div className="text-2xl font-bold text-orange-600">{summary.risk_summary.high}</div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">Medium Risks</div>
                  <div className="text-2xl font-bold text-yellow-600">{summary.risk_summary.medium}</div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">Low Risks</div>
                  <div className="text-2xl font-bold text-green-600">{summary.risk_summary.low}</div>
                </div>
              </div>
              
              <div className="prose max-w-none">
                <p className="text-gray-700">
                  As of {new Date(summary.timestamp).toLocaleDateString()}, the risk landscape shows:
                </p>
                <ul className="text-gray-700">
                  <li>Unemployment rate at {summary.kpis.unemployment.value.toFixed(2)}% ({summary.kpis.unemployment.trend === 'up' ? 'increasing' : 'decreasing'})</li>
                  <li>Inflation rate at {summary.kpis.inflation.value.toFixed(2)}% ({summary.kpis.inflation.trend === 'up' ? 'increasing' : 'decreasing'})</li>
                  <li>Interest rate at {summary.kpis.interest_rate.value.toFixed(2)}% ({summary.kpis.interest_rate.trend === 'up' ? 'increasing' : 'decreasing'})</li>
                  <li>Credit spread at {summary.kpis.credit_spread.value.toFixed(2)}% ({summary.kpis.credit_spread.trend === 'up' ? 'widening' : 'narrowing'})</li>
                </ul>
              </div>
            </div>
          )}

          {/* Detailed Risk Analysis */}
          {reportType === 'detailed' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Detailed Risk Analysis</h3>
              <div className="space-y-4">
                {kris.map((kri) => (
                  <div key={kri.name} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-lg font-semibold">{kri.display_name}</h4>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        kri.risk_level === 'critical' ? 'bg-red-100 text-red-800' :
                        kri.risk_level === 'high' ? 'bg-orange-100 text-orange-800' :
                        kri.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {kri.risk_level.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm mb-2">{kri.description}</p>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Current Value:</span>
                        <span className="ml-2 font-medium">{kri.value.toFixed(2)} {kri.unit}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Category:</span>
                        <span className="ml-2 font-medium capitalize">{kri.category}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Type:</span>
                        <span className="ml-2 font-medium">{kri.is_leading ? 'Leading' : 'Lagging'}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Compliance Report */}
          {reportType === 'compliance' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Compliance Report</h3>
              <div className="space-y-4">
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">Regulatory Compliance Status</h4>
                  <p className="text-green-800 text-sm">All KRIs are within regulatory thresholds</p>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Requirement</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Check</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 text-sm text-gray-900">Capital Adequacy</td>
                        <td className="px-6 py-4"><span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Compliant</span></td>
                        <td className="px-6 py-4 text-sm text-gray-600">{new Date().toLocaleDateString()}</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 text-sm text-gray-900">Liquidity Coverage</td>
                        <td className="px-6 py-4"><span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Compliant</span></td>
                        <td className="px-6 py-4 text-sm text-gray-600">{new Date().toLocaleDateString()}</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 text-sm text-gray-900">Risk Concentration</td>
                        <td className="px-6 py-4"><span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Compliant</span></td>
                        <td className="px-6 py-4 text-sm text-gray-600">{new Date().toLocaleDateString()}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Trend Analysis */}
          {reportType === 'trend' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Trend Analysis</h3>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">Key Trends Identified</h4>
                  <ul className="text-blue-800 text-sm space-y-1">
                    <li>• Credit risk indicators showing stable trend over past quarter</li>
                    <li>• Market volatility has decreased by 15% month-over-month</li>
                    <li>• Liquidity metrics remain within acceptable ranges</li>
                    <li>• Leading indicators suggest continued stability in near term</li>
                  </ul>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-gray-200 rounded-lg">
                    <h5 className="font-semibold mb-2">Improving Metrics</h5>
                    <ul className="text-sm text-gray-700 space-y-1">
                      {kris.filter(k => k.risk_level === 'low').slice(0, 3).map(kri => (
                        <li key={kri.name}>• {kri.display_name}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="p-4 border border-gray-200 rounded-lg">
                    <h5 className="font-semibold mb-2">Metrics Requiring Attention</h5>
                    <ul className="text-sm text-gray-700 space-y-1">
                      {kris.filter(k => k.risk_level === 'high' || k.risk_level === 'critical').map(kri => (
                        <li key={kri.name}>• {kri.display_name}</li>
                      ))}
                      {kris.filter(k => k.risk_level === 'high' || k.risk_level === 'critical').length === 0 && (
                        <li className="text-green-600">• No critical metrics at this time</li>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Scheduled Reports */}
      <Card title="Scheduled Reports">
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center">
              <Calendar className="w-5 h-5 text-gray-600 mr-3" />
              <div>
                <div className="font-medium">Daily Risk Summary</div>
                <div className="text-sm text-gray-600">Sent every day at 9:00 AM</div>
              </div>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">Active</span>
          </div>
          
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center">
              <Calendar className="w-5 h-5 text-gray-600 mr-3" />
              <div>
                <div className="font-medium">Weekly Compliance Report</div>
                <div className="text-sm text-gray-600">Sent every Monday at 8:00 AM</div>
              </div>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">Active</span>
          </div>
          
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center">
              <Calendar className="w-5 h-5 text-gray-600 mr-3" />
              <div>
                <div className="font-medium">Monthly Executive Summary</div>
                <div className="text-sm text-gray-600">Sent on the 1st of each month</div>
              </div>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">Active</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Reports;
