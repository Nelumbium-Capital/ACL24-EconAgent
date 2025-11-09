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

  refreshData: async (): Promise<void> => {
    await api.post('/refresh');
  },
};

export default apiService;
