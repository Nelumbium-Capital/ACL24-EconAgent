import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export interface KPI {
  value: number;
  unit: string;
  change: number;
  trend: 'up' | 'down';
}

export interface DashboardSummary {
  timestamp: string;
  kpis: {
    unemployment: KPI;
    inflation: KPI;
    interest_rate: KPI;
    credit_spread: KPI;
  };
  risk_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

export interface EconomicDataPoint {
  date: string;
  unemployment: number;
  inflation: number;
  interest_rate: number;
  credit_spread: number;
}

export interface KRI {
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

export interface ModelInsight {
  name: string;
  type: string;
  description: string;
  methodology: string;
  strengths: string[];
  use_cases: string[];
  computational_complexity: string;
  interpretability: string;
}

export interface ModelPerformance {
  model: string;
  accuracy: number;
  mae: number;
  rmse?: number;
  n_folds?: number;
}

export interface ForecastAnalysis {
  series_name: string;
  current_value: number;
  recent_trend: number;
  volatility: number;
  forecast_horizon: number;
  analysis: {
    current_interpretation: string;
    forecast_implications: string;
    risk_factors: string[];
    economic_context: string;
  };
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
    recent_mean: number;
    data_points: number;
  };
}

export interface RiskInsight {
  type: string;
  category: string;
  message: string;
  impact: string;
}

export interface RiskInsights {
  risk_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  critical_indicators: string[];
  high_risk_indicators: string[];
  insights: RiskInsight[];
  overall_risk_level: string;
  recommendation: string;
}

export interface EconomicContext {
  current_conditions: string[];
  key_metrics: {
    unemployment_rate: { value: number; change: number };
    inflation_rate: { value: number; change: number };
    fed_funds_rate: { value: number; change: number };
    credit_spread: { value: number; change: number };
  };
  market_sentiment: string;
  data_freshness: string | null;
}

export const apiService = {
  getDashboardSummary: async (): Promise<DashboardSummary> => {
    const response = await api.get('/dashboard/summary');
    return response.data;
  },

  getEconomicData: async (): Promise<EconomicDataPoint[]> => {
    const response = await api.get('/economic-data');
    return response.data.data;
  },

  getForecasts: async (): Promise<EconomicDataPoint[]> => {
    const response = await api.get('/forecasts');
    return response.data.data;
  },

  getKRIs: async (): Promise<KRI[]> => {
    const response = await api.get('/kris');
    return response.data.kris;
  },

  getModelPerformance: async (): Promise<{ model_performance: ModelPerformance[]; status: string; metadata: any }> => {
    const response = await api.get('/model-performance');
    return response.data;
  },

  getModelInsights: async (): Promise<{ models: ModelInsight[]; ensemble_strategy: any }> => {
    const response = await api.get('/model-insights');
    return response.data;
  },

  getForecastAnalysis: async (seriesName: string): Promise<ForecastAnalysis> => {
    const response = await api.get(`/forecast-analysis/${seriesName}`);
    return response.data;
  },

  getRiskInsights: async (): Promise<RiskInsights> => {
    const response = await api.get('/risk-insights');
    return response.data;
  },

  getEconomicContext: async (): Promise<EconomicContext> => {
    const response = await api.get('/economic-context');
    return response.data;
  },

  refreshData: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await api.post('/refresh');
    return response.data;
  },

  // NEW ECONAGENT ENDPOINTS
  
  runEnsembleForecast: async (params: {
    series_id?: string;
    horizon?: number;
    use_rolling_cv?: boolean;
  }): Promise<any> => {
    const response = await api.post('/forecast/ensemble', params);
    return response.data;
  },

  runLLMSimulation: async (params: {
    n_banks?: number;
    n_firms?: number;
    n_workers?: number;
    n_steps?: number;
    use_llm_agents?: boolean;
    scenarios?: string[];
  }): Promise<any> => {
    const response = await api.post('/simulation/llm', params);
    return response.data;
  },

  getScenarioKRIs: async (scenarioName: string): Promise<any> => {
    const response = await api.get(`/scenarios/${scenarioName}/kris`);
    return response.data;
  },

  runScenarioSimulations: async (): Promise<any> => {
    const response = await api.get('/scenarios/run');
    return response.data;
  },

  getModelWeights: async (): Promise<any> => {
    const response = await api.get('/models/weights');
    return response.data;
  },

  runBacktest: async (params: {
    series_id?: string;
    start_date?: string;
    end_date?: string;
    models?: string[];
  }): Promise<any> => {
    const response = await api.post('/backtest/run', params);
    return response.data;
  },
};

export default apiService;
