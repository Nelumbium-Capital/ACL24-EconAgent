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
  category: 'credit' | 'market' | 'liquidity';
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  is_leading: boolean;
  description: string;
  thresholds: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
}
