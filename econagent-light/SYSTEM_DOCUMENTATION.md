# EconAgent-Light System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Economic Model](#economic-model)
4. [Data Integration](#data-integration)
5. [Frontend Application](#frontend-application)
6. [API Endpoints](#api-endpoints)
7. [Simulation Workflow](#simulation-workflow)
8. [Economic Indicators & Calculations](#economic-indicators--calculations)
9. [Visualization & Charts](#visualization--charts)
10. [Configuration & Setup](#configuration--setup)

---

## System Overview

**EconAgent-Light** is an agent-based economic simulation platform that integrates real-time Federal Reserve Economic Data (FRED) with computational economic modeling. The system simulates economic behavior through autonomous agents making work and consumption decisions, while incorporating real-world economic data to calibrate and validate the simulation.

### Key Features
- **Agent-Based Modeling**: 100+ autonomous economic agents with heterogeneous skills
- **Real-Time FRED Integration**: Live economic data from the Federal Reserve
- **Interactive Visualizations**: Multiple chart types showing economic relationships
- **REST API**: Full-featured API for simulation management
- **React Frontend**: Modern, responsive web interface
- **Economic Analysis**: Phillips Curve, Okun's Law, correlation analysis, and more

### Technology Stack
- **Backend**: Python 3.13, FastAPI, Mesa (Agent-Based Modeling)
- **Frontend**: React 18, TypeScript, Chart.js, TailwindCSS
- **Data**: FRED API, Pandas, NumPy
- **API**: RESTful with OpenAPI documentation

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dashboard   │  │   Charts     │  │  Controls    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ API Routes   │  │ Simulation   │  │ FRED Client  │      │
│  │              │  │ Manager      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Economic Model (Mesa)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ EconModel    │  │ EconAgent    │  │ Data         │      │
│  │              │  │ (100+ agents)│  │ Collector    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  External Data Sources                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  FRED API    │  │  Cache       │  │  Results     │      │
│  │  (St. Louis  │  │  Storage     │  │  Storage     │      │
│  │   Fed)       │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
econagent-light/
├── frontend/                    # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── Dashboard.tsx   # Main dashboard
│   │   │   ├── EconomicCharts.tsx  # Chart visualizations
│   │   │   ├── SimulationControls.tsx
│   │   │   ├── FREDDataPanel.tsx
│   │   │   └── EconomicInsights.tsx
│   │   ├── services/           # API client services
│   │   ├── types/              # TypeScript type definitions
│   │   └── App.tsx             # Root component
│   └── package.json
│
├── src/                        # Python backend
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API entry point
│   │   ├── fred_endpoints.py  # FRED data endpoints
│   │   └── simulation_endpoints.py  # Simulation endpoints
│   │
│   ├── mesa_model/            # Economic simulation model
│   │   ├── model.py           # EconModel class
│   │   ├── agents.py          # EconAgent class
│   │   └── utils.py           # Economic calculations
│   │
│   └── data_integration/      # FRED data integration
│       ├── fred_client.py     # FRED API client
│       └── real_data_manager.py  # Data management
│
├── config.py                   # Configuration settings
├── start_backend.py           # Backend startup script
└── requirements.txt           # Python dependencies
```

---

## Economic Model

### Model Overview

The **EconModel** is a Mesa-based agent-based model that simulates a simplified economy with the following components:

#### Core Components

1. **Agents (EconAgent)**
   - Heterogeneous economic agents with varying skills
   - Make autonomous work and consumption decisions
   - Accumulate wealth, pay taxes, receive redistribution

2. **Markets**
   - Labor market (employment/unemployment)
   - Goods market (production/consumption)
   - Financial market (savings/interest)

3. **Government**
   - Progressive income taxation
   - Redistribution of tax revenue
   - Monetary policy (interest rates)

### Agent Behavior

#### Agent Attributes
```python
class EconAgent:
    - unique_id: int           # Agent identifier
    - skill: float             # Productivity level (Pareto distributed)
    - wealth: float            # Accumulated savings
    - age: int                 # Agent age (18-65)
    - city: str                # Location
    - job: str                 # Current occupation
    - monthly_wage: float      # Potential earnings
```

#### Decision-Making Process

**1. Work Decision**
```python
def _make_work_decision():
    # Base propensity to work
    if unemployed:
        base_propensity = 0.8  # High motivation
    else:
        base_propensity = wage / (wealth * 0.1 + 100)
    
    # Adjust for market conditions
    propensity += unemployment_rate * 0.2
    
    # Add randomness
    propensity += random(-0.1, 0.1)
    
    return clamp(propensity, 0, 1)
```

**2. Consumption Decision**
```python
def _make_consumption_decision():
    # Base consumption by wealth level
    if wealth > 1000:
        base = 0.3  # Rich save more
    elif wealth > 200:
        base = 0.4  # Middle class
    else:
        base = 0.6  # Poor consume more
    
    # Adjust for prices
    consumption = base / (1 + price_level - 1)
    
    # Adjust for interest rates
    consumption *= (1 - interest_rate * 2)
    
    return clamp(consumption, 0, 1)
```

### Economic Dynamics

#### Monthly Simulation Step

```
Step 1: Agent Decisions
├── Each agent decides whether to work
├── Each agent decides consumption level
└── Decisions influenced by economic conditions

Step 2: Production Phase
├── Working agents produce goods
├── Production = hours × skill × productivity
└── Goods added to inventory

Step 3: Consumption Phase
├── Agents attempt to consume
├── Consumption limited by inventory
└── Wealth transferred for goods

Step 4: Market Updates
├── Wages adjusted based on employment
├── Prices adjusted based on inventory
├── Inflation calculated
└── Unemployment rate updated

Step 5: Monetary Policy
├── Interest rates updated (Taylor Rule)
├── Interest paid on savings
└── Real FRED data influence applied

Step 6: Fiscal Policy
├── Income taxes collected
├── Revenue redistributed equally
└── Government budget balanced

Step 7: Data Collection
└── All metrics recorded for analysis
```

### Economic Equations

#### Wage Adjustment
```
Δw = α_w × (employment_rate - target_employment)
new_wage = current_wage × (1 + Δw)
```

#### Price Adjustment
```
Δp = α_p × (inventory_level - target_inventory)
new_price = current_price × (1 + Δp)
```

#### Taylor Rule (Interest Rates)
```
r = r* + α_π(π - π*) + α_u(u - u*)

Where:
r = nominal interest rate
r* = natural rate (from FRED)
π = inflation rate
π* = inflation target (2%)
u = unemployment rate
u* = natural unemployment (from FRED)
α_π, α_u = policy response coefficients
```

#### Income Tax
```python
def compute_income_tax(income, brackets, rates):
    tax = 0
    for i, bracket in enumerate(brackets):
        if income > bracket:
            taxable = min(income, next_bracket) - bracket
            tax += taxable * rates[i]
    return tax
```

---

## Data Integration

### FRED API Integration

The system integrates real-time economic data from the Federal Reserve Economic Data (FRED) API.

#### Core Economic Series

| Category | Series ID | Description | Frequency |
|----------|-----------|-------------|-----------|
| **GDP** | GDP | Gross Domestic Product | Quarterly |
| | GDPC1 | Real GDP | Quarterly |
| | GDPPOT | Potential GDP | Quarterly |
| **Employment** | UNRATE | Unemployment Rate | Monthly |
| | CIVPART | Labor Force Participation | Monthly |
| | PAYEMS | Nonfarm Employment | Monthly |
| | AHETPI | Average Hourly Earnings | Monthly |
| **Inflation** | CPIAUCSL | Consumer Price Index | Monthly |
| | CPILFESL | Core CPI | Monthly |
| | PCEPI | PCE Price Index | Monthly |
| **Interest Rates** | FEDFUNDS | Federal Funds Rate | Monthly |
| | DGS10 | 10-Year Treasury | Daily |
| | DGS3MO | 3-Month Treasury | Daily |

#### Data Flow

```
1. API Request
   └── FREDClient.get_series(series_id, start_date, end_date)

2. Cache Check
   ├── Check local cache (24-hour TTL)
   ├── If fresh: return cached data
   └── If stale: fetch from API

3. Data Validation
   ├── Check for missing values
   ├── Detect anomalies (outliers, jumps)
   ├── Calculate quality score
   └── Generate quality report

4. Data Processing
   ├── Convert to pandas DataFrame
   ├── Handle missing values
   ├── Normalize to simulation scale
   └── Calculate derived metrics

5. Model Integration
   ├── Update simulation parameters
   ├── Calibrate economic conditions
   ├── Influence market dynamics
   └── Validate simulation results
```

#### Real-Time Updates

The simulation periodically updates with real FRED data:

```python
def _update_real_data():
    if step % update_frequency == 0:
        # Fetch current indicators
        indicators = fred_client.get_current_snapshot()
        
        # Gradually adjust simulation
        interest_rate += (real_rate - interest_rate) * 0.1
        inflation_target = real_inflation
        unemployment_target = real_unemployment
```

### Economic Snapshot

Current economic conditions are captured in an `EconomicSnapshot`:

```python
@dataclass
class EconomicSnapshot:
    timestamp: datetime
    unemployment_rate: float      # %
    inflation_rate: float          # % YoY
    fed_funds_rate: float          # %
    gdp_growth: float              # % YoY
    wage_growth: float             # % YoY
    labor_participation: float     # %
    consumer_sentiment: Optional[float]
```

---

## Frontend Application

### Component Architecture

#### Dashboard (Main Container)
```typescript
<Dashboard>
  ├── <Header> - Title and refresh controls
  ├── <MetricCards> - Key economic indicators
  ├── <EconomicCharts> - Main visualization area
  ├── <SimulationControls> - Create/manage simulations
  ├── <FREDDataPanel> - Real-time FRED data
  ├── <EconomicInsights> - Analysis and insights
  └── <SimulationsList> - Active/completed simulations
</Dashboard>
```

#### EconomicCharts Component

The charts component provides multiple visualization types:

**1. Economic Overview**
- Multi-line chart with dual y-axes
- Shows unemployment, inflation, GDP growth over time
- Moving averages for smoother trends
- Time period labels (quarters)

**2. Phillips Curve**
- Scatter plot: Unemployment vs Inflation
- Time-based color gradient (early → late periods)
- Theoretical Phillips Curve overlay
- Current FRED position highlighted

**3. Okun's Law**
- Scatter plot: GDP Growth vs Unemployment Change
- Theoretical Okun's Law line
- Filters extreme outliers
- Shows economic efficiency

**4. Correlation Analysis**
- Bar chart comparing observed vs expected correlations
- Color-coded by strength (positive/negative)
- Shows economic relationships

**5. Statistical Distribution**
- Grouped bar chart
- Mean, median, std dev, min, max
- Comparison with FRED data

### Chart Enhancements

#### Visual Features
```typescript
// Enhanced tooltips
tooltip: {
  backgroundColor: 'rgba(17, 24, 39, 0.95)',
  cornerRadius: 12,
  padding: 16,
  callbacks: {
    title: (context) => `Period: ${context[0].label}`,
    label: (context) => {
      const value = context.parsed.y;
      return `${dataset}: ${value.toFixed(2)}%`;
    },
    afterBody: (context) => [
      '',
      'Economic insight text...',
      'Additional context...'
    ]
  }
}

// Color schemes
colors: {
  unemployment: '#3B82F6',  // Blue
  inflation: '#F59E0B',     // Amber
  gdp: '#10B981',           // Green
  fred: '#EF4444'           // Red
}

// Interactive elements
- Hover effects
- Click to highlight
- Zoom and pan
- Export functionality
```

---

## API Endpoints

### FRED Data Endpoints

#### GET /api/fred/current
Get current economic snapshot from FRED.

**Response:**
```json
{
  "timestamp": "2025-11-07T00:00:00",
  "unemployment_rate": 3.8,
  "inflation_rate": 2.4,
  "fed_funds_rate": 5.33,
  "gdp_growth": 2.8,
  "wage_growth": 4.2,
  "labor_participation": 62.7
}
```

#### GET /api/fred/series/{series_id}
Fetch specific FRED series data.

**Parameters:**
- `series_id`: FRED series identifier
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

**Response:**
```json
{
  "series_id": "UNRATE",
  "observations": [
    {"date": "2024-01-01", "value": 3.7},
    {"date": "2024-02-01", "value": 3.9}
  ]
}
```

### Simulation Endpoints

#### POST /api/simulations
Create a new simulation.

**Request Body:**
```json
{
  "name": "Post-COVID Recovery",
  "num_agents": 100,
  "num_years": 20,
  "use_fred_calibration": true,
  "economic_scenario": "baseline",
  "productivity": 1.0,
  "skill_change": 0.02,
  "price_change": 0.02
}
```

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "status": "created",
  "message": "Simulation created successfully"
}
```

#### POST /api/simulations/{simulation_id}/start
Start a simulation.

#### GET /api/simulations/{simulation_id}/status
Get simulation status and progress.

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "name": "Post-COVID Recovery",
  "status": "running",
  "current_step": 120,
  "total_steps": 240,
  "progress_percent": 50.0,
  "current_metrics": {
    "unemployment_rate": 4.2,
    "inflation_rate": 2.1,
    "gdp_growth": 2.5
  }
}
```

#### GET /api/simulations/{simulation_id}/results
Get complete simulation results.

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "economic_indicators": {
    "unemployment_rates": [4.0, 4.1, 4.2, ...],
    "inflation_rates": [2.0, 2.1, 2.0, ...],
    "gdp_growth": [2.5, 2.6, 2.7, ...]
  },
  "final_metrics": {
    "avg_unemployment": 4.15,
    "avg_inflation": 2.05,
    "avg_gdp_growth": 2.58
  }
}
```

---

## Simulation Workflow

### Complete Simulation Lifecycle

```
1. Configuration
   ├── User sets parameters via UI
   ├── Choose economic scenario
   ├── Enable/disable FRED calibration
   └── Set simulation length

2. Initialization
   ├── Create EconModel instance
   ├── Initialize agents with skills
   ├── Load FRED data (if enabled)
   ├── Calibrate parameters
   └── Set initial conditions

3. Execution (Monthly Steps)
   ├── For each month (step):
   │   ├── Agent decisions
   │   ├── Production
   │   ├── Consumption
   │   ├── Market updates
   │   ├── Policy updates
   │   └── Data collection
   └── Continue until complete

4. Results Processing
   ├── Calculate summary statistics
   ├── Generate time series data
   ├── Compute correlations
   └── Create visualizations

5. Analysis & Visualization
   ├── Display charts
   ├── Show insights
   ├── Compare with FRED
   └── Export results
```

### Simulation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num_agents` | Number of economic agents | 100 | 10-1000 |
| `num_years` | Simulation duration (years) | 20 | 1-50 |
| `productivity` | Base productivity level | 1.0 | 0.5-2.0 |
| `skill_change` | Skill growth rate | 0.02 | 0-0.1 |
| `price_change` | Price adjustment rate | 0.02 | 0-0.1 |
| `max_price_inflation` | Maximum price inflation | 0.1 | 0-0.5 |
| `max_wage_inflation` | Maximum wage inflation | 0.05 | 0-0.3 |
| `pareto_param` | Skill distribution parameter | 8.0 | 2-20 |
| `base_interest_rate` | Initial interest rate | 0.01 | 0-0.1 |

---

## Economic Indicators & Calculations

### Primary Indicators

#### 1. Unemployment Rate
```python
employed = count(agents where worked_this_month == True)
unemployment_rate = 1 - (employed / total_agents)
```

#### 2. Inflation Rate
```python
inflation_rate = (current_price - previous_price) / previous_price
```

#### 3. GDP
```python
gdp = sum(agent.last_income for all agents)
```

#### 4. GDP Growth
```python
gdp_growth = (current_gdp - previous_gdp) / previous_gdp
```

### Derived Metrics

#### Gini Coefficient (Wealth Inequality)
```python
def calculate_gini(wealths):
    sorted_wealths = sort(wealths)
    n = len(wealths)
    cumsum = cumulative_sum(sorted_wealths)
    
    gini = (n + 1 - 2 * sum((n + 1 - i) * w 
            for i, w in enumerate(sorted_wealths, 1)) 
            / cumsum[-1]) / n
    
    return gini
```

#### Employment Rate
```python
employment_rate = employed_agents / total_agents
```

#### Average Consumption
```python
avg_consumption = sum(agent.actual_consumption) / total_agents
```

### Economic Relationships

#### Phillips Curve
Relationship between unemployment and inflation:
```
π = f(u)  # Typically negative relationship
```

#### Okun's Law
Relationship between GDP growth and unemployment change:
```
Δu = -β(g - g*)

Where:
Δu = change in unemployment
g = GDP growth
g* = potential GDP growth
β = Okun's coefficient (~0.5)
```

---

## Visualization & Charts

### Chart Types & Purposes

#### 1. Economic Overview (Line Chart)
**Purpose**: Track multiple indicators over time

**Features**:
- Dual y-axes (unemployment/inflation vs GDP)
- Moving averages for trend smoothing
- Time period labels (quarters)
- Fill areas for visual emphasis

**Insights**:
- Economic cycles
- Policy impacts
- Trend identification

#### 2. Phillips Curve (Scatter Plot)
**Purpose**: Analyze unemployment-inflation trade-off

**Features**:
- Time-based color gradient
- Theoretical curve overlay
- FRED current position marker
- Trend line

**Insights**:
- Policy trade-offs
- Economic stability
- Deviation from theory

#### 3. Okun's Law (Scatter Plot)
**Purpose**: Examine GDP-unemployment relationship

**Features**:
- Theoretical Okun line
- Outlier filtering
- Regression analysis
- Economic efficiency indicator

**Insights**:
- Economic efficiency
- Growth-employment link
- Policy effectiveness

#### 4. Correlation Matrix (Bar Chart)
**Purpose**: Show relationships between indicators

**Features**:
- Observed vs expected correlations
- Color-coded strength
- Statistical significance
- Comparison bars

**Insights**:
- Economic relationships
- Model validation
- Anomaly detection

#### 5. Statistical Distribution (Grouped Bar Chart)
**Purpose**: Summarize indicator statistics

**Features**:
- Mean, median, std dev, min, max
- FRED comparison
- Volatility measures
- Range visualization

**Insights**:
- Economic stability
- Volatility assessment
- Comparison with reality

### Chart Configuration

```typescript
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        padding: 20,
        font: { size: 14, weight: 'bold' },
        usePointStyle: true
      }
    },
    tooltip: {
      mode: 'nearest',
      intersect: false,
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      cornerRadius: 12,
      padding: 16
    }
  },
  scales: {
    x: {
      title: { display: true, text: 'X-Axis Label' },
      grid: { color: 'rgba(156, 163, 175, 0.3)' }
    },
    y: {
      title: { display: true, text: 'Y-Axis Label' },
      grid: { color: 'rgba(156, 163, 175, 0.3)' }
    }
  }
};
```

---

## Configuration & Setup

### Environment Variables

Create `.env` file:
```bash
# FRED API Configuration
FRED_API_KEY=your_api_key_here
FRED_CACHE_DIR=./data_cache
FRED_CACHE_HOURS=24

# Simulation Configuration
DEFAULT_NUM_AGENTS=100
DEFAULT_NUM_YEARS=20
DEFAULT_PRODUCTIVITY=1.0

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=3000

# Logging
LOG_LEVEL=INFO
LOG_FILE=econagent.log
```

### Installation

#### Backend Setup
```bash
# Create virtual environment
python3 -m venv venv_arm64
source venv_arm64/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python start_backend.py
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Running the Application

1. **Start Backend** (Terminal 1):
```bash
cd econagent-light
source venv_arm64/bin/activate
python start_backend.py
```

2. **Start Frontend** (Terminal 2):
```bash
cd econagent-light/frontend
npm start
```

3. **Access Application**:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/api/health

### Dependencies

#### Python (Backend)
```
fastapi>=0.104.0
uvicorn>=0.24.0
mesa>=3.0.0
pandas>=2.1.0
numpy>=1.26.0
requests>=2.31.0
python-dotenv>=1.0.0
```

#### JavaScript (Frontend)
```json
{
  "react": "^18.2.0",
  "typescript": "^5.0.0",
  "chart.js": "^4.4.0",
  "react-chartjs-2": "^5.2.0",
  "chartjs-plugin-annotation": "^3.0.1",
  "axios": "^1.6.0",
  "lucide-react": "^0.292.0",
  "react-hot-toast": "^2.4.1"
}
```

---

## Performance & Optimization

### Backend Optimization
- **Caching**: 24-hour cache for FRED data
- **Async Processing**: FastAPI async endpoints
- **Data Validation**: Early validation to prevent errors
- **Batch Processing**: Efficient agent updates

### Frontend Optimization
- **Code Splitting**: Lazy loading components
- **Memoization**: React.memo for expensive components
- **Debouncing**: API call throttling
- **Chart Optimization**: Data sampling for large datasets

### Scalability Considerations
- **Agent Count**: Tested up to 1000 agents
- **Simulation Length**: Up to 50 years (600 steps)
- **Concurrent Simulations**: Multiple simulations supported
- **Data Storage**: Efficient DataFrame operations

---

## Troubleshooting

### Common Issues

#### 1. FRED API Connection Failed
**Problem**: Cannot connect to FRED API
**Solution**:
- Check internet connection
- Verify API key in `.env`
- Check FRED API status
- Use cached data as fallback

#### 2. Charts Not Displaying
**Problem**: Charts show "No data available"
**Solution**:
- Ensure simulation has completed
- Check browser console for errors
- Verify API endpoints are responding
- Clear browser cache

#### 3. Simulation Hangs
**Problem**: Simulation stops progressing
**Solution**:
- Check backend logs
- Verify agent count is reasonable
- Ensure sufficient memory
- Restart backend server

#### 4. Import Errors
**Problem**: Python import errors
**Solution**:
- Activate correct virtual environment
- Reinstall dependencies
- Check Python version (3.13+)
- Verify PYTHONPATH

---

## Future Enhancements

### Planned Features
1. **Advanced Agent Behavior**
   - Machine learning-based decisions
   - Social network effects
   - Heterogeneous preferences

2. **Additional Economic Models**
   - International trade
   - Financial markets
   - Housing markets

3. **Enhanced Visualizations**
   - 3D visualizations
   - Network graphs
   - Real-time animations

4. **Policy Experiments**
   - Universal Basic Income
   - Carbon taxes
   - Trade policies

5. **Collaboration Features**
   - Multi-user simulations
   - Shared scenarios
   - Comparison tools

---

## References

### Academic Papers
- ACL24-EconAgent: Agent-Based Economic Modeling
- Mesa: Agent-Based Modeling in Python
- FRED API Documentation

### External Resources
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [Mesa Documentation](https://mesa.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Chart.js Documentation](https://www.chartjs.org/)

---

## License & Credits

**EconAgent-Light** - Economic Simulation Platform
Version 1.0.0

Built with:
- Mesa (Agent-Based Modeling Framework)
- FastAPI (Web Framework)
- React (Frontend Framework)
- FRED API (Economic Data)

---

*Last Updated: November 7, 2025*
