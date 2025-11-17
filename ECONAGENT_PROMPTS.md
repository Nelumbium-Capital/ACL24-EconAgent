  # EconAgent Prompt Templates

  This document provides the standardized prompt templates used in the EconAgent system, following the methodology from the EconAgent paper.

  ## Overview

  EconAgent uses structured prompts with three key components:
  1. **Perception**: Current state and observed economy
  2. **Memory**: Summarized history and past reflections
  3. **Output**: Strict JSON format for deterministic simulation

  ## System Prompts

  ### Decision-Making System Prompt

  ```
  You are an expert economic agent making rational decisions based on your financial situation and economic conditions. Output ONLY valid JSON with no additional text.
  ```

  ### Reflection System Prompt

  ```
  You are an expert economic agent reflecting on past experiences to improve future decision-making. Output ONLY valid JSON with no additional text.
  ```

  ## Worker Agent Prompts

  ### Perception + Action Prompt

  **Purpose**: Generate work/consumption decisions for current timestep

  **Template**:
  ```
  You are {name}, a {age}-year-old {occupation} in {location}.

  Your financial state:
  - Wage this month: ${wage:.2f}
  - Savings: ${savings:.2f}
  - Loan balance: ${loan_balance:.2f}

  Observed economy:
  - Inflation: {inflation*100:.1f}%
  - Interest rate: {interest_rate*100:.1f}%
  - Unemployment: {unemployment*100:.1f}%
  - Credit conditions (spread): {credit_spread*100:.1f}%

  Your memory:
  {memory_summary}

  TASK: Decide your economic behavior for this timestep. Output ONLY a valid JSON object:
  {{
    "work": 0.00-1.00  (fraction of maximum labor supply),
    "consumption": 0.00-1.00  (fraction of available resources)
  }}

  Example valid output:
  {{"work": 0.64, "consumption": 0.32}}

  Respond with JSON only, no other text.
  ```

  **Parameters**:
  - `name` (str): Agent name (e.g., "Worker_42")
  - `age` (int): Agent age (22-65)
  - `occupation` (str): Job type (e.g., "worker", "technician")
  - `location` (str): Geographic location (e.g., "City")
  - `wage` (float): Monthly wage in dollars
  - `savings` (float): Current savings in dollars
  - `loan_balance` (float): Outstanding debt in dollars
  - `inflation` (float): Inflation rate as decimal (e.g., 0.03 = 3%)
  - `interest_rate` (float): Interest rate as decimal
  - `unemployment` (float): Unemployment rate as decimal
  - `credit_spread` (float): Credit spread as decimal
  - `memory_summary` (str): Condensed memory text

  **Expected Output**:
  ```json
  {
    "work": 0.64,
    "consumption": 0.32
  }
  ```

  **Validation**:
  - `work`: Must be float in range [0.0, 1.0]
  - `consumption`: Must be float in range [0.0, 1.0]
  - JSON must be valid and parseable

  **Fallback on Failure**:
  ```json
  {
    "work": 0.6,
    "consumption": 0.4
  }
  ```

  ---

  ### Reflection Prompt (Quarterly)

  **Purpose**: Generate insights and behavioral adjustments based on recent experiences

  **Frequency**: Every 3 months (steps)

  **Template**:
  ```
  Over the past quarter, these events affected you:
  {event_history}

  Think step by step:
  1. What patterns did you notice?
  2. What did you learn?
  3. How will you adjust your behavior going forward?

  Return ONLY valid JSON:
  {{
    "lessons": [
      "Brief insight 1",
      "Brief insight 2",
      "Brief insight 3"
    ],
    "behavioral_adjustments": {{
      "work_delta": +/-0.10,
      "consumption_delta": +/-0.10
    }}
  }}

  Respond with JSON only, no other text.
  ```

  **Parameters**:
  - `event_history` (str): Formatted summary of last N months (typically 3)

  **Event History Format**:
  ```
  Month 12: worked 0.62, consumed 0.45 (high unemployment);
  Month 13: worked 0.58, consumed 0.40 (elevated inflation);
  Month 14: worked 0.60, consumed 0.42
  ```

  **Expected Output**:
  ```json
  {
    "lessons": [
      "Inflation reduced real spending power",
      "Unexpected rise in unemployment increased job insecurity",
      "Keeping high savings helps offset uncertainty"
    ],
    "behavioral_adjustments": {
      "work_delta": 0.04,
      "consumption_delta": -0.03
    }
  }
  ```

  **Validation**:
  - `lessons`: List of 1-3 strings (max 3 kept)
  - `work_delta`: Float in range [-0.2, 0.2]
  - `consumption_delta`: Float in range [-0.2, 0.2]

  **Fallback on Failure**:
  ```json
  {
    "lessons": ["Maintain current behavior"],
    "behavioral_adjustments": {
      "work_delta": 0.0,
      "consumption_delta": 0.0
    }
  }
  ```

  ---

  ## Memory Management

  ### Memory Window

  - **Size**: 6 months (configurable via `memory_window` setting)
  - **Structure**: Sliding window of `AgentObservation` objects
  - **Storage**: In-memory deque with max length

  ### Memory Summary Format

  **Template**:
  ```
  Over the past {N} months: avg labor participation={avg_work:.2f}, avg consumption={avg_consumption:.2f}. Last reflection: {last_lesson}
  ```

  **Example**:
  ```
  Over the past 6 months: avg labor participation=0.61, avg consumption=0.45. Last reflection: Inflation reduced real spending power
  ```

  ### Reflection Storage

  - **Frequency**: Quarterly (every 3 steps)
  - **Structure**: List of `AgentReflection` objects
  - **Persistence**: Kept indefinitely (no sliding window)
  - **Usage**: Most recent reflection included in memory summary

  ---

  ## Firm Agent Prompts

  ### Production + Hiring Prompt

  **Template** (simplified, can be extended):
  ```
  You are {name}, a firm with ${capital:.2f} in capital.

  Economic conditions:
  - Unemployment: {unemployment*100:.1f}%
  - Interest rate: {interest_rate*100:.1f}%
  - Credit spread: {credit_spread*100:.1f}%

  TASK: Decide production and hiring levels. Output JSON:
  {{
    "production": 0.00-1.00,
    "hiring": 0.00-1.00
  }}
  ```

  **Note**: Currently firms use heuristic rules; LLM integration is optional extension.

  ---

  ## Batch Prompting Optimization

  ### Message Compression

  To reduce token usage when batching many agent prompts:

  1. **Enumerate placeholders**:
    ```
    Agent #42: wage=$4000, savings=$5000, ...
    Agent #43: wage=$3500, savings=$7200, ...
    ```

  2. **Shared context caching**:
    - System prompt is shared across all agents in batch
    - Economic state (inflation, unemployment, etc.) is shared
    - Only agent-specific state varies

  3. **Batch size**: 32-64 agents per API call (configurable)

  ### JSON Schema Enforcement

  All prompts explicitly request:
  - "Output ONLY valid JSON"
  - "Respond with JSON only, no other text"
  - Example outputs provided in prompt

  Validation retries up to 3 times before using fallback.

  ---

  ## Temperature & Sampling Settings

  ### Decision Prompts
  - **Temperature**: 0.3 (relatively deterministic)
  - **Max tokens**: 200
  - **Top-p**: 1.0 (default)

  ### Reflection Prompts
  - **Temperature**: 0.4 (slightly more creative)
  - **Max tokens**: 300
  - **Top-p**: 1.0 (default)

  ---

  ## LLM Model Recommendations

  ### Primary: NeMo (nvidia/llama-3.1-nemotron-70b-instruct)
  - Best quality for economic reasoning
  - Requires GPU inference server
  - 70B parameter model

  ### Fallback: Ollama (llama3.1)
  - Local CPU/GPU inference
  - 8B or 70B variants
  - Easier deployment

  ### Alternative: Mistral 7B/8x7B
  - Good balance of speed and quality
  - Can run on modest hardware

  ---

  ## Prompt Engineering Best Practices

  1. **Be explicit about output format**: Always specify JSON schema
  2. **Provide examples**: Show valid output format in prompt
  3. **Constrain ranges**: Specify valid ranges (e.g., 0.0-1.0)
  4. **No additional text**: Explicitly request "JSON only"
  5. **Validate and retry**: Check JSON validity, retry on failure
  6. **Use fallbacks**: Always have default values for parse failures
  7. **Keep context short**: Summarize memory instead of full history
  8. **Batch when possible**: Group similar prompts to reduce latency

  ---

  ## Auditing & Compliance

  For regulatory compliance, all prompts are:
  - Version controlled in this document
  - Tagged with model name and version in metadata
  - Logged per-agent in simulation datacollector
  - Exportable for external audit

  See `Model_Card.md` for LLM model details and versioning.

