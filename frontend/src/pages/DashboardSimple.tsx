import React, { useState, useEffect } from 'react';
import Card from '../components/ui/Card';
import KPICard from '../components/ui/KPICard';

interface DashboardData {
  kpis: {
    unemployment: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    inflation: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    interest_rate: { value: number; unit: string; change: number; trend: 'up' | 'down' };
    credit_spread: { value: number; unit: string; change: number; trend: 'up' | 'down' };
  };
  risk_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

const DashboardSimple: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching from: http://localhost:8000/api/dashboard/summary');

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch('http://localhost:8000/api/dashboard/summary', {
          signal: controller.signal,
          headers: { 'Accept': 'application/json' }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Data received:', result);
        setData(result);
        setError(null);
      } catch (err: any) {
        console.error('Error:', err);
        setError(err.name === 'AbortError' ? 'Timeout' : err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-2xl text-gray-600">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-2xl text-red-600 mb-4">Error: {error}</div>
          <button onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return <div>No data</div>;

  return (
    <div className="space-y-8 p-6">
      <div className="bg-blue-600 rounded-xl p-6 text-white">
        <h1 className="text-3xl font-bold">Risk Dashboard</h1>
      </div>

      <div className="grid grid-cols-4 gap-6">
        <KPICard
          title="Unemployment"
          kpi={data.kpis.unemployment}
        />
        <KPICard
          title="Inflation"
          kpi={data.kpis.inflation}
        />
        <KPICard
          title="Interest Rate"
          kpi={data.kpis.interest_rate}
        />
        <KPICard
          title="Credit Spread"
          kpi={data.kpis.credit_spread}
        />
      </div>

      <Card>
        <h3 className="text-xl font-bold mb-4">Risk Summary</h3>
        <div className="grid grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600">{data.risk_summary.critical}</div>
            <div className="text-sm text-gray-600">Critical</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-orange-600">{data.risk_summary.high}</div>
            <div className="text-sm text-gray-600">High</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">{data.risk_summary.medium}</div>
            <div className="text-sm text-gray-600">Medium</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">{data.risk_summary.low}</div>
            <div className="text-sm text-gray-600">Low</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DashboardSimple;
