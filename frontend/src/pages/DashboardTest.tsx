import React from 'react';

const DashboardTest: React.FC = () => {
  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold text-blue-600">Frontend is Working!</h1>
      <p className="mt-4 text-xl">If you see this, React is rendering successfully.</p>

      <div className="mt-8 p-6 bg-green-100 border-2 border-green-500 rounded-lg">
        <h2 className="text-2xl font-semibold text-green-800">System Status</h2>
        <ul className="mt-4 space-y-2 text-green-700">
          <li>✓ React app is loading</li>
          <li>✓ Tailwind CSS is working</li>
          <li>✓ Router is functional</li>
        </ul>
      </div>

      <div className="mt-8">
        <button
          onClick={async () => {
            try {
              const response = await fetch('http://localhost:8000/api/health');
              const data = await response.json();
              alert(`API Status: ${data.status}\nTime: ${data.timestamp}`);
            } catch (error) {
              alert(`API Error: ${error}`);
            }
          }}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Test API Connection
        </button>
      </div>
    </div>
  );
};

export default DashboardTest;
