// API Response Types
export interface EconomicSnapshot {
  timestamp: string;
  unemployment_rate: number;
  inflation_rate: number;
  fed_funds_rate: number;
  gdp_growth: number;
  wage_growth: number;
  labor_participation: number;
  consumer_sentiment?: number;
}

export interface SimulationConfig {
  name: string;
  num_agents: number;
  num_years: number;
  use_fred_calibration: boolean;
  fred_start_date?: string;
  economic_scenario: string;
  random_seed?: number;
  productivity?: number;
  skill_change?: number;
  price_change?: number;
}

export interface SimulationStatus {
  simulation_id: string;
  name: string;
  status: 'created' | 'running' | 'completed' | 'failed' | 'stopped';
  current_step: number;
  total_steps: number;
  progress_percent: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  current_metrics?: {
    unemployment_rate: number;
    inflation_rate: number;
    gdp_growth: number;
    step: number;
  };
  error_message?: string;
}

export interface SimulationResult {
  simulation_id: string;
  name: string;
  config: SimulationConfig;
  status: string;
  duration_seconds?: number;
  final_metrics?: {
    avg_unemployment: number;
    avg_inflation: number;
    avg_gdp_growth: number;
    total_steps: number;
  };
  economic_indicators?: {
    unemployment_rates: number[];
    inflation_rates: number[];
    gdp_growth: number[];
  };
  agent_statistics?: {
    num_agents: number;
    simulation_years: number;
  };
}

export interface CalibrationResult {
  unemployment_target: number;
  inflation_target: number;
  natural_interest_rate: number;
  productivity_growth: number;
  wage_adjustment_rate: number;
  price_adjustment_rate: number;
  calibration_date: string;
  confidence_score: number;
  data_period: string;
}

export interface FREDSeries {
  series_id: string;
  title: string;
  observations: number;
  start_date: string;
  end_date: string;
  data: Array<{
    date: string;
    value: number | null;
  }>;
}

// Chart Data Types
export interface ChartDataset {
  label: string;
  data: number[];
  borderColor: string;
  backgroundColor: string;
  fill?: boolean;
  tension?: number;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

// UI State Types
export interface DashboardState {
  fredData: EconomicSnapshot | null;
  simulations: SimulationStatus[];
  activeSimulation: SimulationStatus | null;
  isLoading: boolean;
  error?: string;
}

export interface FormErrors {
  [key: string]: string;
}

// API Response Wrappers
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface SimulationsListResponse {
  simulations: SimulationStatus[];
  total_count: number;
}

export interface CreateSimulationResponse {
  simulation_id: string;
  status: string;
  message: string;
}

// Economic Indicator Types
export interface EconomicIndicator {
  name: string;
  value: number;
  trend: 'rising' | 'falling' | 'stable';
  unit: string;
  description?: string;
}

export interface EconomicSummary {
  economic_health: 'good' | 'moderate' | 'concerning';
  inflation_status: 'target' | 'off_target';
  policy_stance: 'accommodative' | 'neutral' | 'restrictive';
}

export interface DashboardIndicators {
  timestamp: string;
  indicators: {
    unemployment: EconomicIndicator;
    inflation: EconomicIndicator;
    fed_funds_rate: EconomicIndicator;
    wage_growth: EconomicIndicator;
    labor_participation: EconomicIndicator;
  };
  summary: EconomicSummary;
}