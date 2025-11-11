import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { KPI } from '../../types';

interface KPICardProps {
  title: string;
  kpi: KPI;
  icon?: React.ReactNode;
}

const KPICard: React.FC<KPICardProps> = ({ title, kpi, icon }) => {
  const isPositive = kpi.trend === 'up';
  const changeColor = isPositive ? 'text-green-600' : 'text-red-600';
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          {icon && <div className="mr-3 text-blue-600">{icon}</div>}
          <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        </div>
        <div className={`flex items-center ${changeColor}`}>
          <TrendIcon className="w-4 h-4 mr-1" />
          <span className="text-sm font-medium">
            {kpi.change > 0 ? '+' : ''}{kpi.change.toFixed(2)}{kpi.unit}
          </span>
        </div>
      </div>
      
      <div className="mt-4">
        <div className="text-3xl font-bold text-gray-900">
          {kpi.value.toFixed(2)}{kpi.unit}
        </div>
      </div>
    </div>
  );
};

export default KPICard;
