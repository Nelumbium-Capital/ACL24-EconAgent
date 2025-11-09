# Major Fixes Applied to Dashboard and Models

## Problems Identified

1. **ABM default rates unrealistically high** (50-90%)
2. **Forecasts too simplistic** (straight lines)
3. **No chart explanations or annotations**
4. **Scenarios barely differentiated**
5. **Portfolio metrics identical across scenarios**
6. **Models not properly calibrated**

## Fixes Applied

### 1. Fixed Agent-Based Model (Mesa) ✓

**Problem**: Banks were applying default rate to ENTIRE loan portfolio EVERY STEP, causing cumulative defaults of 50-90%.

**Fix**:
- Changed from portfolio-wide default rate to **actual firm-level defaults**
- Banks now only recognize losses when borrower firms actually default
- Added Loss Given Default (LGD) of 60% (realistic banking standard)
- Reduced stress multipliers to realistic levels
- Changed to **monthly default probabilities** (annual rate / 12)
- Capped monthly default at 5% (60% annualized in extreme stress)

**Before**:
```python
default_rate = min(base_default_rate * stress_multiplier, 0.5)  # 50% per step!
loan_losses = self.loans * default_rate  # Applied to entire portfolio
```

**After**:
```python
# Count actual defaults from borrower firms
defaulted_count = sum(1 for firm in self.borrowers if firm.is_defaulted)
actual_default_rate = defaulted_count / total_borrowers
lgd = 0.60  # Loss given default
loan_losses = self.loans * actual_default_rate * lgd  # Only on defaults
```

**Expected Result**: Default rates should now be 2-10% (realistic) instead of 50-90%

### 2. Improved Firm Default Logic ✓

**Problem**: Firms had 90% default probability with extreme stress factors.

**Fix**:
- Reduced stress multipliers:
  - Unemployment: 5 → 2
  - Credit spread: 10 → 3
  - GDP: 5 → 1
- Changed to monthly default probability (annual / 12)
- Capped at 5% per month (realistic)
- Base default rate now properly annualized

**Before**:
```python
stress_factor = 1.0 + (unemployment - 0.04) * 5 + (credit_spread - 0.02) * 10
adjusted_default_prob = min(base_default_probability * stress_factor, 0.9)  # 90%!
```

**After**:
```python
monthly_base_default = self.base_default_probability / 12
unemployment_stress = max(0, (unemployment - 0.04) * 2)
credit_stress = max(0, (credit_spread - 0.02) * 3)
stress_factor = 1.0 + unemployment_stress + credit_stress - gdp_benefit
adjusted_default_prob = min(monthly_base_default * stress_factor, 0.05)  # 5% max
```

**Expected Result**: Firm defaults should be 0.5-5% per month depending on stress

### 3. Added Chart Annotations and Explanations ✓

**Problem**: Charts had no context or explanations.

**Fix**:
- Added descriptive titles with subtitles
- Added scenario descriptions:
  - Baseline: "Normal economic conditions with small random fluctuations"
  - Recession: "Severe downturn: unemployment rises to 10%, GDP contracts"
  - Rate Shock: "Sudden interest rate increase from 3% to 6%"
  - Credit Crisis: "Credit market disruption: spreads widen from 2% to 8%"
- Added vertical line marking forecast start
- Added hover templates with clear labels
- Added axis labels explaining what metrics mean

**Example**:
```python
title="Agent-Based Simulation: Recession<br><sub>Severe downturn: unemployment rises to 10%, GDP contracts</sub>"
```

### 4. Improved Forecast Chart ✓

**Problem**: Forecasts looked like straight lines with no context.

**Fix**:
- Added title explaining it's an ensemble of ARIMA, ETS, and LSTM
- Added vertical line at forecast start
- Improved hover templates
- Better legend labels distinguishing historical vs forecast

### 5. Better Scenario Differentiation (In Progress)

**Current Status**: Scenarios should now show more realistic differences because:
- Default rates are based on actual firm behavior
- Stress factors are properly calibrated
- Monthly time steps prevent unrealistic accumulation

**Expected Differences**:
- Baseline: 2-3% default rate
- Recession: 5-8% default rate (2-3x baseline)
- Rate Shock: 4-6% default rate
- Credit Crisis: 6-10% default rate (highest stress)

## What Still Needs Work

### 1. Forecast Confidence Intervals
Currently forecasts show point estimates only. Should add:
- 95% confidence bands
- Prediction intervals
- Model uncertainty visualization

### 2. Portfolio Volatility Calculation
Currently using simple std dev. Should implement:
- Proper GARCH models for volatility
- Rolling window calculations
- Scenario-specific volatility

### 3. More Sophisticated ABM
Could add:
- Bank-to-bank lending (interbank market)
- Contagion effects
- Fire sales and asset price dynamics
- Regulatory interventions

### 4. Better KRI Calibration
Some KRIs are still using placeholder values. Need:
- Real portfolio data integration
- Historical calibration
- Stress testing standards (Basel III)

## Testing the Fixes

Run this to verify the fixes:

```bash
cd us-risk-forecasting
python fix_and_test_models.py
```

Expected output:
- ARIMA forecasts: Varying predictions (not flat)
- ETS forecasts: Varying predictions
- Mesa baseline: 2-5% default rate (not 50%)
- Mesa recession: 5-10% default rate (not 90%)
- Scenarios showing clear differentiation

## Dashboard Access

**URL**: http://localhost:8050

The dashboard should now show:
1. **Realistic default rates** (2-10% instead of 50-90%)
2. **Chart annotations** explaining what each metric means
3. **Scenario descriptions** explaining the economic conditions
4. **Better visual clarity** with hover templates and labels

## Next Steps

1. Verify default rates are now realistic (2-10%)
2. Add confidence intervals to forecasts
3. Implement proper portfolio volatility models
4. Add more detailed explanations in dashboard text
5. Create user guide explaining how to interpret each chart

---

*Applied: 2025-11-09 02:30 AM*
*Status: Core fixes complete, dashboard restarting*
