import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement,
  TimeScale
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { Line, Bar, Scatter } from 'react-chartjs-2';
import { TrendingUp, BarChart3, Activity, Download, Zap as ScatterIcon, BarChart2, LineChart } from 'lucide-react';
import { SimulationStatus, EconomicSnapshot, SimulationResult } from '../types/index.ts';
import { apiClient } from '../services/api.ts';
import toast from 'react-hot-toast';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement,
  TimeScale,
  annotationPlugin
);

interface EconomicChartsProps {
  simulation: SimulationStatus | null;
  fredData: EconomicSnapshot | null;
}

const EconomicCharts: React.FC<EconomicChartsProps> = ({ simulation, fredData }) => {
  const [simulationResults, setSimulationResults] = useState<SimulationResult | null>(null);
  const [activeChart, setActiveChart] = useState<'overview' | 'phillips' | 'okun' | 'correlation' | 'distribution'>('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [useRealisticData, setUseRealisticData] = useState(false);

  // Load simulation results when simulation completes
  useEffect(() => {
    if (simulation && simulation.status === 'completed') {
      loadSimulationResults(simulation.simulation_id);
    }
  }, [simulation]);

  const loadSimulationResults = async (simulationId: string) => {
    try {
      setIsLoading(true);
      const results = await apiClient.getSimulationResults(simulationId);
      setSimulationResults(results);
    } catch (error) {
      console.error('Failed to load simulation results:', error);
      toast.error('Failed to load simulation results');
    } finally {
      setIsLoading(false);
    }
  };

  // Generate comprehensive economic overview chart
  const generateOverviewChart = () => {
    if (!simulationResults?.economic_indicators) return null;

    const indicators = simulationResults.economic_indicators;
    const totalSteps = indicators.unemployment_rates.length;
    const labels = Array.from({ length: totalSteps }, (_, i) => {
      const quarter = Math.floor(i / 3) + 1;
      const month = (i % 3) + 1;
      return `Q${quarter}.${month}`;
    });

    // Calculate moving averages for smoother trends
    const movingAverage = (data: number[], window: number = 3) => {
      return data.map((_, i) => {
        const start = Math.max(0, i - window + 1);
        const slice = data.slice(start, i + 1);
        return slice.reduce((sum, val) => sum + val, 0) / slice.length;
      });
    };

    const smoothedUnemployment = movingAverage(indicators.unemployment_rates);
    const smoothedInflation = movingAverage(indicators.inflation_rates);
    const smoothedGDP = movingAverage(indicators.gdp_growth);

    return {
      labels: labels,
      datasets: [
        {
          label: 'Unemployment Rate (%)',
          data: smoothedUnemployment,
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          yAxisID: 'y',
          tension: 0.4,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 8,
          pointBackgroundColor: '#3B82F6',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          fill: 'origin'
        },
        {
          label: 'Inflation Rate (%)',
          data: smoothedInflation,
          borderColor: '#F59E0B',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          yAxisID: 'y',
          tension: 0.4,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 8,
          pointBackgroundColor: '#F59E0B',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          fill: 'origin'
        },
        {
          label: 'GDP Growth (%)',
          data: smoothedGDP,
          borderColor: '#10B981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          yAxisID: 'y1',
          tension: 0.4,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 8,
          pointBackgroundColor: '#10B981',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          fill: 'origin'
        }
      ]
    };
  };

  // Generate Phillips Curve (Unemployment vs Inflation) - Fixed to show proper relationship
  const generatePhillipsCurve = () => {
    if (!simulationResults?.economic_indicators) return null;

    const indicators = simulationResults.economic_indicators;
    
    // Create meaningful Phillips Curve data by sorting and grouping
    const combinedData = indicators.unemployment_rates.map((unemployment, i) => ({
      unemployment,
      inflation: indicators.inflation_rates[i],
      period: i
    }));

    // Sort by unemployment to create a proper curve
    combinedData.sort((a, b) => a.unemployment - b.unemployment);

    // Create time-based color gradient (early periods = blue, later = red)
    const maxPeriod = Math.max(...combinedData.map(d => d.period));
    
    const datasets = [
      {
        label: 'Phillips Curve Relationship',
        data: combinedData.map(d => ({ x: d.unemployment, y: d.inflation })),
        backgroundColor: combinedData.map(d => {
          const intensity = d.period / maxPeriod;
          return `rgba(${59 + intensity * 180}, ${130 - intensity * 62}, ${246 - intensity * 178}, 0.7)`;
        }),
        borderColor: combinedData.map(d => {
          const intensity = d.period / maxPeriod;
          return `rgb(${59 + intensity * 180}, ${130 - intensity * 62}, ${246 - intensity * 178})`;
        }),
        pointRadius: 6,
        pointHoverRadius: 10,
        pointBorderWidth: 2,
        pointBorderColor: '#ffffff',
        showLine: true,
        tension: 0.3,
        borderWidth: 2
      }
    ];

    // Add FRED current point if available
    if (fredData) {
      datasets.push({
        label: 'Current Real Economy (FRED)',
        data: [{ x: fredData.unemployment_rate, y: fredData.inflation_rate }],
        backgroundColor: '#EF4444',
        borderColor: '#DC2626',
        pointRadius: 15,
        pointHoverRadius: 18,
        pointBorderWidth: 4,
        pointBorderColor: '#ffffff',
        showLine: false
      });
    }

    // Add theoretical Phillips Curve line
    const minUnemployment = Math.min(...indicators.unemployment_rates);
    const maxUnemployment = Math.max(...indicators.unemployment_rates);
    const theoreticalCurve = [];
    for (let u = minUnemployment; u <= maxUnemployment; u += 0.1) {
      // Theoretical Phillips Curve: π = a - b*u (simplified)
      const inflation = 4.5 - 0.5 * u; // Example relationship
      theoreticalCurve.push({ x: u, y: Math.max(0, inflation) });
    }

    datasets.push({
      label: 'Theoretical Phillips Curve',
      data: theoreticalCurve,
      backgroundColor: 'transparent',
      borderColor: '#6B7280',
      borderDash: [5, 5],
      pointRadius: 0,
      showLine: true,
      tension: 0.4,
      borderWidth: 2
    });

    return { datasets };
  };

  // Generate Okun's Law (GDP Growth vs Unemployment Change) - Fixed with proper economic relationship
  const generateOkunsLaw = () => {
    if (!simulationResults?.economic_indicators) return null;

    const indicators = simulationResults.economic_indicators;
    
    // Calculate unemployment changes properly
    const unemploymentChanges = indicators.unemployment_rates.slice(1).map((current, i) => 
      current - indicators.unemployment_rates[i]
    );
    
    // Pair with GDP growth (skip first point since we need previous unemployment)
    const gdpGrowthData = indicators.gdp_growth.slice(1);
    
    // Create Okun's Law data points with proper relationship
    const okunData = gdpGrowthData.map((gdp, i) => ({
      x: gdp,
      y: unemploymentChanges[i],
      period: i + 1
    }));

    // Filter out extreme outliers for better visualization
    const filteredData = okunData.filter(point => 
      Math.abs(point.y) < 2.0 && point.x > -5 && point.x < 8
    );

    // Create theoretical Okun's Law line (negative relationship)
    const minGDP = Math.min(...filteredData.map(d => d.x));
    const maxGDP = Math.max(...filteredData.map(d => d.x));
    const theoreticalLine = [];
    
    for (let gdp = minGDP; gdp <= maxGDP; gdp += 0.2) {
      // Okun's Law: Change in unemployment ≈ -0.5 * (GDP growth - trend growth)
      const trendGrowth = 2.5; // Assume 2.5% trend growth
      const unemploymentChange = -0.5 * (gdp - trendGrowth);
      theoreticalLine.push({ x: gdp, y: unemploymentChange });
    }

    return {
      datasets: [
        {
          label: "Okun's Law Data Points",
          data: filteredData.map(d => ({ x: d.x, y: d.y })),
          backgroundColor: '#10B981',
          borderColor: '#059669',
          pointRadius: 8,
          pointHoverRadius: 12,
          pointBorderWidth: 2,
          pointBorderColor: '#ffffff',
          showLine: false
        },
        {
          label: "Theoretical Okun's Law",
          data: theoreticalLine,
          backgroundColor: 'transparent',
          borderColor: '#6B7280',
          borderDash: [8, 4],
          pointRadius: 0,
          showLine: true,
          tension: 0.1,
          borderWidth: 3
        }
      ]
    };
  };

  // Generate correlation matrix with better visualization
  const generateCorrelationChart = () => {
    if (!simulationResults?.economic_indicators) return null;

    const indicators = simulationResults.economic_indicators;
    
    // Calculate correlations with proper error handling
    const correlations = {
      'Unemployment vs Inflation': calculateCorrelation(indicators.unemployment_rates, indicators.inflation_rates),
      'Unemployment vs GDP Growth': calculateCorrelation(indicators.unemployment_rates, indicators.gdp_growth),
      'Inflation vs GDP Growth': calculateCorrelation(indicators.inflation_rates, indicators.gdp_growth),
    };

    // Add expected theoretical relationships for comparison
    const theoreticalCorrelations = {
      'Expected: Unemployment vs Inflation': -0.6, // Phillips Curve
      'Expected: Unemployment vs GDP': -0.7, // Okun's Law
      'Expected: Inflation vs GDP': 0.3, // Moderate positive
    };

    const allLabels = [...Object.keys(correlations), ...Object.keys(theoreticalCorrelations)];
    const actualValues = Object.values(correlations);
    const theoreticalValues = Object.values(theoreticalCorrelations);

    return {
      labels: allLabels,
      datasets: [
        {
          label: 'Observed Correlations',
          data: [...actualValues, ...Array(theoreticalValues.length).fill(null)],
          backgroundColor: actualValues.map(val => 
            val > 0.3 ? '#10B981' :
            val < -0.3 ? '#EF4444' :
            '#F59E0B'
          ).concat(Array(theoreticalValues.length).fill('transparent')),
          borderColor: actualValues.map(val => 
            val > 0.3 ? '#059669' :
            val < -0.3 ? '#DC2626' :
            '#D97706'
          ).concat(Array(theoreticalValues.length).fill('transparent')),
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: 'Expected Correlations',
          data: [...Array(actualValues.length).fill(null), ...theoreticalValues],
          backgroundColor: theoreticalValues.map(val => 
            val > 0.3 ? 'rgba(16, 185, 129, 0.3)' :
            val < -0.3 ? 'rgba(239, 68, 68, 0.3)' :
            'rgba(245, 158, 11, 0.3)'
          ),
          borderColor: theoreticalValues.map(val => 
            val > 0.3 ? '#10B981' :
            val < -0.3 ? '#EF4444' :
            '#F59E0B'
          ),
          borderWidth: 2,
          borderRadius: 6,
          borderDash: [5, 5],
        }
      ]
    };
  };

  // Generate economic indicators distribution with better visualization
  const generateDistributionChart = () => {
    if (!simulationResults?.economic_indicators) return null;

    const indicators = simulationResults.economic_indicators;
    
    // Helper function to calculate statistics safely
    const calculateStats = (data: number[]) => {
      const validData = data.filter(val => !isNaN(val) && isFinite(val));
      if (validData.length === 0) return { mean: 0, std: 0, min: 0, max: 0 };
      
      const mean = validData.reduce((a, b) => a + b, 0) / validData.length;
      const variance = validData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validData.length;
      const std = Math.sqrt(variance);
      
      return {
        mean: Number(mean.toFixed(2)),
        std: Number(std.toFixed(2)),
        min: Number(Math.min(...validData).toFixed(2)),
        max: Number(Math.max(...validData).toFixed(2))
      };
    };

    const unemploymentStats = calculateStats(indicators.unemployment_rates);
    const inflationStats = calculateStats(indicators.inflation_rates);
    const gdpStats = calculateStats(indicators.gdp_growth);

    // Create comparison with FRED data if available
    const datasets = [
      {
        label: 'Unemployment Rate (%)',
        data: [unemploymentStats.mean, unemploymentStats.std, unemploymentStats.min, unemploymentStats.max],
        backgroundColor: '#3B82F6',
        borderColor: '#2563EB',
        borderWidth: 2,
        borderRadius: 6,
      },
      {
        label: 'Inflation Rate (%)',
        data: [inflationStats.mean, inflationStats.std, inflationStats.min, inflationStats.max],
        backgroundColor: '#F59E0B',
        borderColor: '#D97706',
        borderWidth: 2,
        borderRadius: 6,
      },
      {
        label: 'GDP Growth (%)',
        data: [gdpStats.mean, gdpStats.std, gdpStats.min, gdpStats.max],
        backgroundColor: '#10B981',
        borderColor: '#059669',
        borderWidth: 2,
        borderRadius: 6,
      }
    ];

    // Add FRED comparison if available
    if (fredData) {
      datasets.push({
        label: 'Current FRED Values',
        data: [fredData.unemployment_rate, fredData.inflation_rate, fredData.gdp_growth, 0],
        backgroundColor: '#EF4444',
        borderColor: '#DC2626',
        borderWidth: 2,
        borderRadius: 6,
      });
    }

    return {
      labels: ['Average', 'Volatility (Std Dev)', 'Minimum', 'Maximum'],
      datasets
    };
  };

  // Helper function to calculate correlation
  const calculateCorrelation = (x: number[], y: number[]) => {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  };

  // Chart options for different chart types
  const getChartOptions = (chartType: string) => {
    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
          labels: {
            padding: 20,
            font: {
              size: 14,
              weight: 'bold' as const
            },
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          mode: chartType === 'overview' ? 'index' as const : 'nearest' as const,
          intersect: false,
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#F9FAFB',
          bodyColor: '#F3F4F6',
          borderColor: '#6B7280',
          borderWidth: 1,
          cornerRadius: 12,
          padding: 16,
          titleFont: {
            size: 15,
            weight: 'bold' as const
          },
          bodyFont: {
            size: 14
          },
          callbacks: {
            title: function(context: any) {
              if (chartType === 'overview') {
                return `Period: ${context[0].label}`;
              } else if (chartType === 'phillips') {
                return 'Phillips Curve Analysis';
              } else if (chartType === 'okun') {
                return "Okun's Law Analysis";
              } else if (chartType === 'correlation') {
                return 'Economic Correlation';
              } else if (chartType === 'distribution') {
                return 'Statistical Summary';
              }
              return context[0].label;
            },
            label: function(context: any) {
              const value = context.parsed.y || context.parsed;
              const dataset = context.dataset.label;
              
              if (chartType === 'phillips') {
                return `${dataset}: Unemployment ${context.parsed.x.toFixed(1)}%, Inflation ${context.parsed.y.toFixed(1)}%`;
              } else if (chartType === 'okun') {
                return `${dataset}: GDP Growth ${context.parsed.x.toFixed(1)}%, Unemployment Change ${context.parsed.y.toFixed(2)}pp`;
              } else if (chartType === 'correlation') {
                return `${dataset}: ${value.toFixed(3)} ${value > 0.5 ? '(Strong Positive)' : value < -0.5 ? '(Strong Negative)' : '(Weak)'}`;
              } else if (chartType === 'distribution') {
                return `${dataset}: ${value.toFixed(2)}${chartType === 'distribution' ? '%' : ''}`;
              } else {
                return `${dataset}: ${value.toFixed(2)}%`;
              }
            },
            afterBody: function(context: any) {
              if (chartType === 'phillips' && context.length > 0) {
                return ['', 'Phillips Curve shows the trade-off between unemployment and inflation.', 'Points closer to origin indicate better economic conditions.'];
              } else if (chartType === 'okun' && context.length > 0) {
                return ['', "Okun's Law shows the relationship between GDP growth and unemployment.", 'Negative correlation indicates economic efficiency.'];
              }
              return [];
            }
          }
        },
      },
      layout: {
        padding: {
          top: 20,
          right: 20,
          bottom: 20,
          left: 20
        }
      },
      elements: {
        point: {
          radius: 6,
          hoverRadius: 8,
          borderWidth: 2
        },
        line: {
          borderWidth: 3,
          tension: 0.4
        }
      }
    };

    switch (chartType) {
      case 'overview':
        return {
          ...baseOptions,
          scales: {
            x: {
              display: true,
              title: { 
                display: true, 
                text: 'Time Period',
                font: { size: 14, weight: 'bold' as const },
                padding: { top: 10 }
              },
              ticks: {
                font: { size: 12 },
                maxTicksLimit: 10,
                padding: 8
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            },
            y: {
              type: 'linear' as const,
              display: true,
              position: 'left' as const,
              title: { 
                display: true, 
                text: 'Unemployment & Inflation (%)',
                font: { size: 14, weight: 'bold' as const },
                padding: { bottom: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(1) + '%';
                }
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            },
            y1: {
              type: 'linear' as const,
              display: true,
              position: 'right' as const,
              title: { 
                display: true, 
                text: 'GDP Growth (%)',
                font: { size: 14, weight: 'bold' as const },
                padding: { bottom: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(1) + '%';
                }
              },
              grid: { 
                drawOnChartArea: false,
                color: 'rgba(156, 163, 175, 0.3)'
              },
            },
          },
        };
      
      case 'phillips':
      case 'okun':
        return {
          ...baseOptions,
          scales: {
            x: {
              display: true,
              title: { 
                display: true, 
                text: chartType === 'phillips' ? 'Unemployment Rate (%)' : 'GDP Growth (%)',
                font: { size: 14, weight: 'bold' as const },
                padding: { top: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(1) + '%';
                }
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            },
            y: {
              display: true,
              title: { 
                display: true, 
                text: chartType === 'phillips' ? 'Inflation Rate (%)' : 'Change in Unemployment (pp)',
                font: { size: 14, weight: 'bold' as const },
                padding: { bottom: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(1) + (chartType === 'phillips' ? '%' : 'pp');
                }
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            }
          },
        };
      
      case 'correlation':
        return {
          ...baseOptions,
          scales: {
            x: {
              ticks: {
                font: { size: 12 },
                padding: 8,
                maxRotation: 45
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            },
            y: {
              beginAtZero: true,
              min: -1,
              max: 1,
              title: { 
                display: true, 
                text: 'Correlation Coefficient',
                font: { size: 14, weight: 'bold' as const },
                padding: { bottom: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(2);
                }
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            }
          },
        };
      
      case 'distribution':
        return {
          ...baseOptions,
          scales: {
            x: {
              ticks: {
                font: { size: 12 },
                padding: 8,
                maxRotation: 45
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            },
            y: {
              beginAtZero: true,
              title: { 
                display: true, 
                text: 'Value',
                font: { size: 14, weight: 'bold' as const },
                padding: { bottom: 10 }
              },
              ticks: {
                font: { size: 12 },
                padding: 8,
                callback: function(value: any) {
                  return value.toFixed(2);
                }
              },
              grid: {
                color: 'rgba(156, 163, 175, 0.3)',
                lineWidth: 1
              }
            }
          },
        };
      
      default:
        return baseOptions;
    }
  };

  const handleExportChart = async () => {
    if (!simulation) return;

    try {
      await apiClient.downloadFile(
        `/api/simulations/${simulation.simulation_id}/export?format=csv`,
        `${simulation.name}_results.csv`
      );
      toast.success('Chart data exported successfully');
    } catch (error) {
      console.error('Failed to export chart data:', error);
      toast.error('Failed to export chart data');
    }
  };

  const getChartTitle = () => {
    switch (activeChart) {
      case 'overview':
        return 'Economic Indicators Overview';
      case 'phillips':
        return 'Phillips Curve Analysis';
      case 'okun':
        return "Okun's Law Analysis";
      case 'correlation':
        return 'Economic Correlations';
      case 'distribution':
        return 'Statistical Distribution';
      default:
        return 'Economic Analysis';
    }
  };

  const getChartData = () => {
    switch (activeChart) {
      case 'overview':
        return generateOverviewChart();
      case 'phillips':
        return generatePhillipsCurve();
      case 'okun':
        return generateOkunsLaw();
      case 'correlation':
        return generateCorrelationChart();
      case 'distribution':
        return generateDistributionChart();
      default:
        return null;
    }
  };

  const renderChart = () => {
    const chartData = getChartData();
    if (!chartData) return null;

    const options = getChartOptions(activeChart);

    switch (activeChart) {
      case 'overview':
        return <Line data={chartData} options={options} />;
      case 'phillips':
      case 'okun':
        return <Scatter data={chartData} options={options} />;
      case 'correlation':
      case 'distribution':
        return <Bar data={chartData} options={options} />;
      default:
        return <Line data={chartData} options={options} />;
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-primary-600" />
          {getChartTitle()}
        </h2>
        
        <div className="flex items-center space-x-2">
          {simulation && simulation.status === 'completed' && (
            <button
              onClick={handleExportChart}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              title="Export Data"
            >
              <Download className="w-4 h-4" />
            </button>
          )}
          
          {simulation && (
            <div className={`status-badge status-${simulation.status}`}>
              {simulation.status}
            </div>
          )}
        </div>
      </div>

      {/* Chart Type Selector */}
      <div className="flex flex-wrap gap-2 mb-8 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
        <button
          onClick={() => setActiveChart('overview')}
          className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
            activeChart === 'overview'
              ? 'bg-primary-600 text-white shadow-lg transform scale-105'
              : 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-700 shadow-sm hover:shadow-md'
          }`}
        >
          <LineChart className="w-5 h-5 mr-2" />
          Economic Overview
        </button>
        <button
          onClick={() => setActiveChart('phillips')}
          className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
            activeChart === 'phillips'
              ? 'bg-primary-600 text-white shadow-lg transform scale-105'
              : 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-700 shadow-sm hover:shadow-md'
          }`}
        >
          <ScatterIcon className="w-5 h-5 mr-2" />
          Phillips Curve
        </button>
        <button
          onClick={() => setActiveChart('okun')}
          className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
            activeChart === 'okun'
              ? 'bg-primary-600 text-white shadow-lg transform scale-105'
              : 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-700 shadow-sm hover:shadow-md'
          }`}
        >
          <ScatterIcon className="w-5 h-5 mr-2" />
          Okun's Law
        </button>
        <button
          onClick={() => setActiveChart('correlation')}
          className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
            activeChart === 'correlation'
              ? 'bg-primary-600 text-white shadow-lg transform scale-105'
              : 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-700 shadow-sm hover:shadow-md'
          }`}
        >
          <BarChart2 className="w-5 h-5 mr-2" />
          Correlations
        </button>
        <button
          onClick={() => setActiveChart('distribution')}
          className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
            activeChart === 'distribution'
              ? 'bg-primary-600 text-white shadow-lg transform scale-105'
              : 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-700 shadow-sm hover:shadow-md'
          }`}
        >
          <BarChart3 className="w-5 h-5 mr-2" />
          Statistics
        </button>
      </div>

      {/* Current Metrics Display */}
      {simulation && simulation.status === 'running' && simulation.current_metrics && (
        <div className="mb-4 grid grid-cols-3 gap-3">
          <div className="p-3 bg-blue-50 rounded-lg">
            <div className="text-xs font-medium text-blue-700">Unemployment</div>
            <div className="text-lg font-bold text-blue-900">
              {simulation.current_metrics.unemployment_rate.toFixed(1)}%
            </div>
          </div>
          <div className="p-3 bg-amber-50 rounded-lg">
            <div className="text-xs font-medium text-amber-700">Inflation</div>
            <div className="text-lg font-bold text-amber-900">
              {simulation.current_metrics.inflation_rate.toFixed(1)}%
            </div>
          </div>
          <div className="p-3 bg-green-50 rounded-lg">
            <div className="text-xs font-medium text-green-700">GDP Growth</div>
            <div className="text-lg font-bold text-green-900">
              {simulation.current_metrics.gdp_growth.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Chart Description */}
      {simulationResults && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-700">
            {activeChart === 'overview' && 'Comprehensive view of all economic indicators over time with dual y-axes.'}
            {activeChart === 'phillips' && 'Relationship between unemployment and inflation rates. Points closer to origin indicate better economic conditions.'}
            {activeChart === 'okun' && 'Relationship between GDP growth and unemployment changes. Negative correlation expected.'}
            {activeChart === 'correlation' && 'Correlation coefficients between economic indicators. Values closer to ±1 indicate stronger relationships.'}
            {activeChart === 'distribution' && 'Statistical summary showing mean, standard deviation, minimum and maximum values for each indicator.'}
          </p>
        </div>
      )}

      {/* Chart Container */}
      <div className="relative h-[500px]">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-gray-600">Loading chart data...</span>
            </div>
          </div>
        ) : simulationResults ? (
          renderChart()
        ) : simulation ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Activity className="w-12 h-12 mb-2" />
            <p className="text-lg font-medium">
              {simulation.status === 'running' ? 'Simulation in Progress' : 'Waiting for Results'}
            </p>
            <p className="text-sm">
              {simulation.status === 'running' 
                ? `${simulation.progress_percent.toFixed(1)}% complete`
                : 'Charts will appear when simulation completes'
              }
            </p>
            
            {simulation.status === 'running' && (
              <div className="mt-4 w-64 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${simulation.progress_percent}%` }}
                ></div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <TrendingUp className="w-12 h-12 mb-2" />
            <p className="text-lg font-medium">No Active Simulation</p>
            <p className="text-sm">Start a simulation to see comprehensive economic analysis</p>
          </div>
        )}
      </div>

      {/* Economic Insights */}
      {simulationResults && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Key Insights:</h4>
              {activeChart === 'overview' && (
                <ul className="space-y-1 text-gray-600">
                  <li>• Track multiple indicators simultaneously</li>
                  <li>• Identify economic cycles and trends</li>
                  <li>• Compare with FRED baseline data</li>
                </ul>
              )}
              {activeChart === 'phillips' && (
                <ul className="space-y-1 text-gray-600">
                  <li>• Negative slope indicates Phillips Curve</li>
                  <li>• Scatter shows economic trade-offs</li>
                  <li>• Red dot shows current FRED position</li>
                </ul>
              )}
              {activeChart === 'okun' && (
                <ul className="space-y-1 text-gray-600">
                  <li>• Negative correlation expected</li>
                  <li>• Higher GDP growth → Lower unemployment</li>
                  <li>• Coefficient shows economic efficiency</li>
                </ul>
              )}
              {activeChart === 'correlation' && (
                <ul className="space-y-1 text-gray-600">
                  <li>• Green: Strong positive correlation (&gt;0.5)</li>
                  <li>• Red: Strong negative correlation (&lt;-0.5)</li>
                  <li>• Yellow: Weak correlation (-0.5 to 0.5)</li>
                </ul>
              )}
              {activeChart === 'distribution' && (
                <ul className="space-y-1 text-gray-600">
                  <li>• Mean: Average value over simulation</li>
                  <li>• Std Dev: Measure of volatility</li>
                  <li>• Min/Max: Range of values observed</li>
                </ul>
              )}
            </div>
            
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Legend:</h4>
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                  <span>Unemployment</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-amber-500 rounded-full mr-2"></div>
                  <span>Inflation</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                  <span>GDP Growth</span>
                </div>
                {fredData && (
                  <div className="flex items-center">
                    <div className="w-3 h-0.5 bg-red-500 mr-2" style={{ borderStyle: 'dashed' }}></div>
                    <span>FRED Data</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EconomicCharts;