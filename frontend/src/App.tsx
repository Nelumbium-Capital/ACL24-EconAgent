import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import DashboardSimple from './pages/DashboardSimple';
import Forecasting from './pages/Forecasting';
import MarketDynamics from './pages/MarketDynamics';
import RiskAnalysis from './pages/RiskAnalysis';
import Reports from './pages/Reports';
import Scenarios from './pages/Scenarios';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Layout><DashboardSimple /></Layout>} />
        <Route path="/forecasting" element={<Layout><Forecasting /></Layout>} />
        <Route path="/market-dynamics" element={<Layout><MarketDynamics /></Layout>} />
        <Route path="/risk-analysis" element={<Layout><RiskAnalysis /></Layout>} />
        <Route path="/scenarios" element={<Layout><Scenarios /></Layout>} />
        <Route path="/reports" element={<Layout><Reports /></Layout>} />
      </Routes>
    </Router>
  );
}

export default App;
