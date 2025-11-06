import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  EconomicSnapshot,
  SimulationConfig,
  SimulationStatus,
  SimulationResult,
  SimulationsListResponse,
  CreateSimulationResponse,
  CalibrationResult,
  FREDSeries,
  DashboardIndicators
} from '../types';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_URL || '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        
        // Handle common error cases
        if (error.response?.status === 404) {
          throw new Error('Resource not found');
        } else if (error.response?.status === 500) {
          throw new Error('Server error occurred');
        } else if (error.response?.status === 503) {
          throw new Error('Service temporarily unavailable');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Request timeout');
        }
        
        throw new Error(error.response?.data?.detail || error.message || 'Unknown error occurred');
      }
    );
  }

  // Health check
  async checkHealth(): Promise<{ status: string; service: string; version: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // FRED Data endpoints
  async getCurrentEconomicData(): Promise<EconomicSnapshot> {
    const response = await this.client.get('/fred/current');
    return response.data;
  }

  async getFredSeries(
    seriesId: string,
    startDate?: string,
    endDate?: string,
    frequency?: string,
    useCache: boolean = true
  ): Promise<FREDSeries> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (frequency) params.append('frequency', frequency);
    params.append('use_cache', useCache.toString());

    const response = await this.client.get(`/fred/series/${seriesId}?${params}`);
    return response.data;
  }

  async getCoreEconomicData(
    startDate?: string,
    endDate?: string,
    useCache: boolean = true
  ): Promise<{ period: string; series_count: number; data: Record<string, any> }> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    params.append('use_cache', useCache.toString());

    const response = await this.client.get(`/fred/core-data?${params}`);
    return response.data;
  }

  async getDashboardIndicators(): Promise<DashboardIndicators> {
    const response = await this.client.get('/fred/indicators/dashboard');
    return response.data;
  }

  async calibrateParameters(
    forceRecalibrate: boolean = false,
    historicalYears?: number,
    baseConfig?: Record<string, number>
  ): Promise<CalibrationResult> {
    const response = await this.client.post('/fred/calibrate', {
      force_recalibrate: forceRecalibrate,
      historical_years: historicalYears,
      base_config: baseConfig,
    });
    return response.data;
  }

  async getCalibrationSummary(): Promise<any> {
    const response = await this.client.get('/fred/calibration/summary');
    return response.data;
  }

  async getFredStatistics(): Promise<any> {
    const response = await this.client.get('/fred/statistics');
    return response.data;
  }

  async clearFredCache(olderThanHours?: number): Promise<{ status: string; files_cleared: number; message: string }> {
    const params = olderThanHours ? `?older_than_hours=${olderThanHours}` : '';
    const response = await this.client.post(`/fred/cache/clear${params}`);
    return response.data;
  }

  async searchFredSeries(query: string, limit: number = 10): Promise<{ query: string; results_count: number; results: any[] }> {
    const response = await this.client.get(`/fred/search?query=${encodeURIComponent(query)}&limit=${limit}`);
    return response.data;
  }

  // Simulation endpoints
  async createSimulation(config: SimulationConfig): Promise<CreateSimulationResponse> {
    const response = await this.client.post('/simulations/', config);
    return response.data;
  }

  async startSimulation(simulationId: string): Promise<{ simulation_id: string; status: string; message: string }> {
    const response = await this.client.post(`/simulations/${simulationId}/start`);
    return response.data;
  }

  async getSimulationStatus(simulationId: string): Promise<SimulationStatus> {
    const response = await this.client.get(`/simulations/${simulationId}/status`);
    return response.data;
  }

  async getSimulationResults(simulationId: string): Promise<SimulationResult> {
    const response = await this.client.get(`/simulations/${simulationId}/results`);
    return response.data;
  }

  async getSimulations(status?: string, limit: number = 50): Promise<SimulationsListResponse> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());

    const response = await this.client.get(`/simulations/?${params}`);
    return response.data;
  }

  async deleteSimulation(simulationId: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/simulations/${simulationId}`);
    return response.data;
  }

  async stopSimulation(simulationId: string): Promise<{ message: string }> {
    const response = await this.client.post(`/simulations/${simulationId}/stop`);
    return response.data;
  }

  async exportSimulationResults(
    simulationId: string,
    format: 'json' | 'csv' = 'json'
  ): Promise<any> {
    const response = await this.client.get(`/simulations/${simulationId}/export?format=${format}`);
    return response.data;
  }

  // Utility methods
  async ping(): Promise<boolean> {
    try {
      await this.checkHealth();
      return true;
    } catch {
      return false;
    }
  }

  // File download helper
  async downloadFile(url: string, filename: string): Promise<void> {
    const response = await this.client.get(url, {
      responseType: 'blob',
    });

    const blob = new Blob([response.data]);
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }
}

// Create and export singleton instance
export const apiClient = new ApiClient();
export default apiClient;