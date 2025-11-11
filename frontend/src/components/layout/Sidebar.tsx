import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Shield, 
  TrendingUp, 
  BarChart3, 
  FileText,
  Activity
} from 'lucide-react';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Risk Analysis', href: '/risk-analysis', icon: Shield },
  { name: 'Market Dynamics', href: '/market-dynamics', icon: TrendingUp },
  { name: 'Forecasting', href: '/forecasting', icon: BarChart3 },
  { name: 'Reports', href: '/reports', icon: FileText },
];

const Sidebar: React.FC = () => {
  return (
    <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
      <div className="p-6">
        <div className="flex items-center mb-8">
          <Activity className="w-8 h-8 text-primary mr-3" />
          <span className="text-xl font-bold text-gray-900">EconAgent</span>
        </div>
        
        <nav className="space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  `flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary text-white'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`
                }
              >
                <Icon className="w-5 h-5 mr-3" />
                {item.name}
              </NavLink>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;
