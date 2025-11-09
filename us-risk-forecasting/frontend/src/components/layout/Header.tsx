import React from 'react';
import { RefreshCw, Bell, Settings } from 'lucide-react';
import { apiService } from '../../services/api';

const Header: React.FC = () => {
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [lastUpdate, setLastUpdate] = React.useState<string>('');

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await apiService.refreshData();
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  React.useEffect(() => {
    setLastUpdate(new Date().toLocaleTimeString());
  }, []);

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Risk Forecasting Dashboard
            </h1>
            <p className="text-sm text-gray-600">
              Real-time financial risk monitoring and analysis
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              Last updated: {lastUpdate}
            </div>
            
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <RefreshCw 
                className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} 
              />
              Refresh
            </button>
            
            <button className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100">
              <Bell className="w-5 h-5" />
            </button>
            
            <button className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
