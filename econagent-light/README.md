# EconAgent-Light 🚀

A modern economic simulation platform that integrates real-time Federal Reserve Economic Data (FRED) with intelligent agent-based modeling. Built with Mesa for ABM simulation, LightAgent framework for intelligent agents, FastAPI for the backend, and a beautiful React/Tailwind CSS frontend.

## ✨ Features

- **Real-time FRED Integration**: Live economic data fetching and calibration from Federal Reserve
- **Professional React/Tailwind UI**: Clean, modern dashboard with interactive visualizations
- **Local Nemotron LLM**: Intelligent agents powered by local NVIDIA Nemotron
- **LightAgent Framework**: Enhanced agent capabilities with memory and reasoning
- **Mesa Simulation**: Robust agent-based economic modeling
- **Enterprise Architecture**: FastAPI backend with proper API design and error handling

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm
- Git

### 2. Clone and Setup
```bash
git clone <your-repo-url>
cd econagent-light
```

### 3. Install Backend Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 5. Start Development Servers
```bash
# Start both backend and frontend
python start_dev.py
```

This will start:
- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/api/docs

### 6. Access the Application
Open your browser and navigate to http://localhost:3000 to access the beautiful React dashboard!

## 🏗️ Architecture

### Frontend (React/Tailwind)
- **Dashboard**: Real-time economic indicators and simulation controls
- **Charts**: Interactive visualizations using Chart.js
- **FRED Panel**: Live Federal Reserve economic data display
- **Simulation Management**: Create, monitor, and analyze simulations

### Backend (FastAPI)
- **FRED Integration**: Real-time economic data fetching and caching
- **Simulation API**: RESTful endpoints for simulation management
- **Calibration Engine**: Automatic parameter adjustment using FRED data
- **Data Validation**: Quality checks and economic indicator validation

### Simulation Engine (Mesa + LightAgent)
- **Mesa Model**: Agent-based economic modeling framework
- **Economic Agents**: LightAgent-powered agents with memory and reasoning
- **FRED Calibration**: Real-world economic data integration
- **Local Nemotron**: Intelligent decision-making using local LLM

## 📊 Key Features

### Real-time FRED Data Integration
- Live economic indicators from Federal Reserve Economic Data
- Automatic calibration of simulation parameters
- Historical trend analysis and validation
- Economic snapshot functionality

### Professional Web Interface
- Modern React dashboard with Tailwind CSS styling
- Interactive charts and real-time updates
- Simulation configuration and monitoring
- Export functionality for results and data

### Intelligent Economic Agents
- LightAgent framework with memory and reasoning
- Local Nemotron LLM integration
- FRED data-aware decision making
- Quarterly reflection and learning mechanisms

## 🛠️ Usage Guide

### Creating a Simulation
1. Open the dashboard at http://localhost:3000
2. Configure simulation parameters in the left panel
3. Toggle FRED calibration for real-world data integration
4. Click "Start Simulation" to begin
5. Monitor progress in real-time through the dashboard

### FRED Data Integration
- Automatic fetching of current economic indicators
- Parameter calibration based on historical trends
- Real-time comparison with simulation results
- Data quality validation and caching

### API Usage
```python
# Example API usage
import requests

# Get current FRED data
response = requests.get('http://localhost:8000/api/fred/current')
economic_data = response.json()

# Create a simulation
config = {
    "name": "My Simulation",
    "num_agents": 100,
    "num_years": 20,
    "use_fred_calibration": True
}
response = requests.post('http://localhost:8000/api/simulations/', json=config)
simulation = response.json()
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# FRED API Configuration
FRED_API_KEY=your_fred_api_key_here

# LLM Configuration (optional)
NEMOTRON_URL=http://localhost:8000/v1
OLLAMA_URL=http://localhost:11434/v1

# Development settings
REACT_APP_API_URL=http://localhost:8000/api
```

### FRED API Setup
1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Add it to your `.env` file or set as environment variable
3. The system works without an API key but with limited requests

### Local LLM Setup (Optional)
For enhanced agent intelligence, you can set up local LLM services:

**NVIDIA Nemotron:**
```bash
# Docker setup (requires NVIDIA GPU)
docker run -it --rm --gpus all \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest
```

**Ollama (CPU/GPU):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama2:7b-chat
```

## 📈 Performance

### Simulation Performance
- **Frontend**: Real-time updates with WebSocket connections
- **Backend**: Async FastAPI with background task processing
- **FRED Data**: Intelligent caching with configurable TTL
- **Simulation**: Mesa-based ABM with optimized agent scheduling

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **For LLM**: Additional GPU memory for local Nemotron

## 🧪 Testing

### Backend Tests
```bash
# Run all tests
python -m pytest src/tests/ -v

# Test FRED integration
python -m pytest src/tests/test_fred_client.py -v

# Test API endpoints
python -m pytest src/tests/test_api.py -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing
```bash
# Test FRED client directly
python -c "from src.data_integration.fred_client import FREDClient; client = FREDClient(); print(client.get_current_economic_snapshot())"

# Test API health
curl http://localhost:8000/api/health
```

## 📁 Project Structure

```
econagent-light/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── start_dev.py                # Development server startup
├── config.py                   # Configuration management
├── frontend/                   # React frontend
│   ├── src/                   # React source code
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   └── tailwind.config.js     # Tailwind CSS config
├── src/                       # Backend source code
│   ├── api/                   # FastAPI application
│   │   ├── main.py           # Main FastAPI app
│   │   ├── fred_endpoints.py  # FRED data endpoints
│   │   └── simulation_endpoints.py # Simulation API
│   ├── data_integration/      # FRED and data processing
│   │   ├── fred_client.py     # FRED API client
│   │   └── calibration_engine.py # Economic calibration
│   ├── mesa_model/           # Mesa ABM implementation
│   ├── lightagent_integration/ # LightAgent framework
│   └── tests/                # Test suite
└── data_cache/               # FRED data cache (created)
```

## 🔬 Economic Modeling

### FRED Data Integration
- Real-time economic indicators from Federal Reserve
- Automatic parameter calibration using historical trends
- Data quality validation and anomaly detection
- Comparison with simulation results for validation

### Economic Accuracy
The system maintains economic rigor through:
- Progressive taxation (2018 U.S. Federal brackets)
- Pareto skill distribution for agent heterogeneity
- Price and wage dynamics based on economic theory
- Taylor rule for interest rate adjustments
- Quarterly agent reflection and learning

### Validation Features
- Statistical comparison with FRED benchmarks
- Economic indicator trend analysis
- Simulation accuracy scoring
- Real-world data calibration reports

## 🚀 Getting Started

### Next Steps
1. **Explore the Dashboard**: Navigate through the React interface
2. **Create Your First Simulation**: Use the configuration panel
3. **Analyze FRED Data**: Check current economic conditions
4. **Monitor Progress**: Watch real-time simulation updates
5. **Export Results**: Download data for further analysis

### Troubleshooting
- **FRED API Issues**: Check your API key and internet connection
- **Frontend Not Loading**: Ensure Node.js dependencies are installed
- **Backend Errors**: Check Python dependencies and port availability
- **Simulation Failures**: Review configuration parameters and logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federal Reserve Economic Data (FRED)** for providing comprehensive economic data
- **Mesa Framework** for agent-based modeling capabilities
- **LightAgent Framework** for intelligent agent architecture
- **React and Tailwind CSS** for the modern web interface
- **FastAPI** for the robust backend API

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: See the `/docs` directory
- **API Docs**: http://localhost:8000/api/docs (when running)

---

**EconAgent-Light**: Modern economic simulation with real-time data integration 📊✨