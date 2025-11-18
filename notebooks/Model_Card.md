# Model Card: EconAgent LLM Risk Forecasting System

**Date**: November 2025  
**Version**: 1.0  
**Status**: Development  

---

## Model Overview

This document provides a comprehensive audit trail of all machine learning models used in the EconAgent Risk Forecasting System, including forecasting models and LLM-based agents.

---

## 1. Forecasting Models

### 1.1 ARIMA (AutoRegressive Integrated Moving Average)

**Purpose**: Time-series forecasting for economic indicators

**Library**: `statsmodels`  
**Version**: 0.14.x  

**Configuration**:
- Auto-order selection via `pmdarima.auto_arima`
- Information criterion: AIC
- Max AR order: 5
- Max MA order: 5
- Max differencing: 2

**Training**:
- Data: FRED economic indicators (2018-2024)
- Frequency: Monthly
- Training window: Expanding window (min 36 months)
- Retraining: On-demand

**Performance Metrics** (Unemployment Rate 2018-2024):
- RMSE: ~0.42%
- MAE: ~0.35%
- 95% CI Coverage: ~91%

**Limitations**:
- Assumes linear relationships
- Struggles with structural breaks
- Requires stationary (or stationarizable) data

---

### 1.2 ETS (Exponential Smoothing)

**Purpose**: Time-series forecasting with trend/seasonal decomposition

**Library**: `statsmodels`  
**Version**: 0.14.x  

**Configuration**:
- Trend: Additive or None (auto-selected)
- Seasonal: None (for monthly macro data)
- Optimization: Maximum likelihood

**Training**:
- Same as ARIMA
- Damped trend available for long-horizon forecasts

**Performance Metrics** (Unemployment Rate 2018-2024):
- RMSE: ~0.45%
- MAE: ~0.37%
- 95% CI Coverage: ~89%

**Limitations**:
- Struggles with abrupt changes
- Assumes smooth trends

---

### 1.3 Ensemble (Weighted Average)

**Purpose**: Combine ARIMA + ETS forecasts with optimized weights

**Method**: Rolling cross-validation with trend adjustment

**Algorithm**:
1. Score models on 36-month sliding windows
2. Compute normalized inverse RMSE as base weight
3. Calculate recent trend: `(value_now - value_6mo_ago) / 6`
4. Apply ±10% weight adjustment for trend-capturing ability
5. Normalize weights to sum to 1.0

**Training**:
- Rolling CV: Up to 12 windows
- Window size: 36 months
- Trend window: 6 months

**Performance Metrics** (Unemployment Rate 2018-2024):
- RMSE: ~0.38%
- MAE: ~0.32%
- 95% CI Coverage: ~92%

**Advantages**:
- Robust to individual model failures
- Adapts to changing data patterns
- Confidence intervals via weighted combination

---

## 2. LLM-Based Agent Models

### 2.1 NeMo (Primary)

**Model**: `nvidia/llama-3.1-nemotron-70b-instruct`  
**Provider**: NVIDIA NeMo  
**Version**: Llama 3.1 (70B parameters)  
**Release**: August 2024  

**Deployment**:
- Inference server: NeMo Inference Microservice
- URL: `http://localhost:8000/v1` (configurable)
- Hardware: GPU required (A100/H100 recommended)
- Batch size: 32-64 prompts per call

**Usage in System**:
- Agent decision-making (work/consumption)
- Quarterly reflections
- Behavioral adjustments

**Prompt Format**:
- System prompt: "You are an expert economic agent..."
- User prompt: Structured perception + memory + task
- Output: Strict JSON schema

**Sampling Parameters**:
- Temperature: 0.3 (decisions), 0.4 (reflections)
- Max tokens: 200 (decisions), 300 (reflections)
- Top-p: 1.0

**Validation**:
- JSON schema enforcement
- Retry logic: Up to 3 attempts
- Fallback: Heuristic defaults on failure

**Known Issues**:
- Occasional JSON formatting errors (~2-5% of calls)
- Requires external GPU server
- Latency: ~500-1000ms per batch (64 agents)

---

### 2.2 Ollama (Fallback)

**Model**: `llama3.1` (8B or 70B)  
**Provider**: Ollama  
**Version**: Llama 3.1  
**Release**: July 2024  

**Deployment**:
- Local inference: Ollama server
- URL: `http://localhost:11434/v1` (configurable)
- Hardware: CPU or GPU (8B runs on CPU)
- Batch size: 16-32 prompts per call

**Usage**:
- Automatic fallback when NeMo unavailable
- Same prompt format as NeMo
- Slightly lower quality but more accessible

**Sampling Parameters**:
- Same as NeMo

**Performance**:
- Slightly higher JSON parsing errors (~5-8%)
- Faster on local hardware (8B variant)
- 70B variant comparable to NeMo

---

## 3. Agent Architecture

### 3.1 WorkerAgent

**Purpose**: Simulate worker behavior in economic scenarios

**Decision Variables**:
- `work`: Labor participation (0.0-1.0)
- `consumption`: Spending rate (0.0-1.0)

**State Variables**:
- Wage (monthly)
- Savings
- Loan balance
- Age
- Occupation

**Decision Frequency**: Every timestep (monthly)

**Reflection Frequency**: Every 3 timesteps (quarterly)

**Memory**:
- Window: 6 months
- Storage: Sliding deque
- Reflections: Persistent list

**Behavioral Model**:
- LLM mode: Perception → Memory → LLM decision → Action
- Heuristic mode: Rule-based (fallback)

---

### 3.2 FirmAgentLLM

**Purpose**: Simulate firm hiring/production decisions

**Decision Variables**:
- `production`: Production level (0.0-1.0)
- `hiring`: Hiring rate (0.0-1.0)

**State Variables**:
- Capital
- Revenue
- Costs

**Status**: Simplified heuristic (LLM extension available)

---

## 4. Batch Processing

### 4.1 BatchLLMClient

**Purpose**: Efficient multi-agent LLM inference

**Features**:
- Parallel requests: ThreadPoolExecutor (max 16 workers)
- Batch size: 32-64 agents (configurable)
- Shared context caching
- Automatic retry on failure
- NeMo → Ollama fallback

**Performance**:
- Throughput: ~50-100 agents/second (NeMo, GPU)
- Latency: ~500-1000ms per batch
- Token usage: ~150-250 tokens per agent

**Optimization**:
- Message compression: Shared system prompt
- Enumerated agent states
- JSON-only output (no prose)

---

## 5. Validation & Testing

### 5.1 Backtest Engine

**Methodology**:
- Time-series cross-validation
- Expanding window (min 36 months)
- Forecast horizon: 1-12 months
- Confidence interval testing (target: 90% coverage)

**Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- CI Coverage

**Test Data**:
- FRED indicators: UNRATE, CPIAUCSL, FEDFUNDS, BAA10Y
- Period: 2018-01-01 to 2024-01-01
- Frequency: Monthly

---

### 5.2 Scenario Testing

**Scenarios**:
1. Baseline (normal conditions)
2. Recession (unemployment spike, GDP contraction)
3. Interest Rate Shock (sudden rate increase)
4. Credit Crisis (spread widening)

**Validation**:
- KRI threshold changes vs baseline
- Default rate ranges
- Liquidity ratios
- Capital adequacy

**Test Results** (Example):
- Baseline default rate: 2-4%
- Recession default rate: 8-12%
- Rate shock default rate: 5-7%
- Credit crisis default rate: 10-15%

---

## 6. Ethical Considerations

### 6.1 Bias & Fairness

**Potential Biases**:
- LLM training data bias (pre-2023 economic patterns)
- Homogeneous agent assumptions (all "workers" similar)
- Geographic bias (US-centric data)

**Mitigation**:
- Diverse agent initialization (age, wage, savings)
- Regular model updates
- Scenario testing across economic conditions

---

### 6.2 Transparency

**Explainability**:
- All agent decisions logged
- Reflection reasoning captured
- Prompt templates documented (see ECONAGENT_PROMPTS.md)

**Auditability**:
- Version control for all code and prompts
- Deterministic seeding for reproducibility
- Metadata stored with every forecast/simulation

---

### 6.3 Limitations

**Forecasting Models**:
- Cannot predict "black swan" events
- Assume historical patterns continue
- Confidence intervals may underestimate true uncertainty

**LLM Agents**:
- Stochastic outputs (even with low temperature)
- JSON parsing failures require fallbacks
- No true understanding of economics (pattern matching)
- Limited by training data cutoff (~2023)

**System**:
- Simplified agent interactions (no explicit networks)
- Monthly timestep (cannot capture intra-month dynamics)
- US-centric data and scenarios

---

## 7. Usage Guidelines

### 7.1 Intended Use

**Primary**:
- Risk assessment and stress testing
- Scenario analysis
- Early warning indicators
- Educational/research purposes

**Not Intended For**:
- Real-time trading decisions
- Individual credit decisions
- Regulatory compliance (without validation)
- Production deployment without extensive testing

---

### 7.2 Recommended Practices

1. **Always backtest** on historical data before trusting forecasts
2. **Use ensemble** over single models
3. **Enable LLM agents** only with adequate GPU resources
4. **Monitor JSON parsing** success rates
5. **Regular retraining** (at least quarterly)
6. **Scenario testing** before production use
7. **Human oversight** for critical decisions

---

## 8. Model Lifecycle

### 8.1 Training/Retraining

**Frequency**:
- Forecasting models: On-demand or quarterly
- LLM agents: Pre-trained (no retraining required)

**Triggers**:
- New data available (FRED updates)
- Performance degradation detected
- Major economic regime change

**Process**:
1. Fetch latest FRED data
2. Retrain ARIMA + ETS
3. Re-optimize ensemble weights via rolling CV
4. Run backtest validation
5. Update production models if improved

---

### 8.2 Monitoring

**Metrics to Track**:
- Forecast accuracy (RMSE, MAE)
- CI coverage
- LLM JSON success rate
- Batch inference latency
- Agent decision distributions

**Alerts**:
- RMSE > 2x historical baseline
- CI coverage < 80%
- JSON failures > 10%
- Latency > 5 seconds per batch

---

## 9. References

### Papers
- **EconAgent**: [Citation needed - LLM-based economic agent paper]
- **LLM-ABM Survey**: "Large Language Models in Agent-Based Modeling" (2024)

### Libraries
- `statsmodels`: Seabold & Perktold (2010)
- `pmdarima`: Smith et al. (2017-)
- `Mesa`: Kazil et al. (2020-)
- `requests`: Reitz (2011-)

### Models
- **Llama 3.1**: Meta AI (2024)
- **Nemotron**: NVIDIA (2024)

---

## 10. Contact & Support

**Maintainer**: EconAgent Development Team  
**Repository**: [GitHub URL]  
**Documentation**: See README.md, HOW_IT_WORKS.md  
**Issues**: [GitHub Issues URL]  

---

**Last Updated**: November 17, 2025  
**Next Review**: Quarterly (February 2026)


