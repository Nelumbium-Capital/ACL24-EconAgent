import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, TrendingUp, Activity, FileText, AlertTriangle, Layers } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: BarChart3 },
    { path: '/risk-analysis', label: 'Risk Analysis', icon: AlertTriangle },
    { path: '/market-dynamics', label: 'Market Dynamics', icon: Activity },
    { path: '/forecasting', label: 'Forecasting', icon: TrendingUp },
    { path: '/scenarios', label: 'Scenarios', icon: Layers },
    { path: '/reports', label: 'Reports', icon: FileText },
  ];

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className="w-64 bg-secondary text-white flex flex-col">
        <div className="p-6 border-b border-gray-700">
          <h1 className="text-xl font-bold">Risk Forecasting</h1>
          <p className="text-sm text-gray-400 mt-1">Enterprise Dashboard</p>
        </div>
        
        <nav className="flex-1 p-4">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center px-4 py-3 mb-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary text-white'
                    : 'text-gray-300 hover:bg-gray-700'
                }`}
              >
                <Icon className="w-5 h-5 mr-3" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        
        <div className="p-4 border-t border-gray-700">
          <div className="text-xs text-gray-400">
            <div>Version 1.0.0</div>
            <div className="mt-1">© 2025 Risk Analytics</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto p-8">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
