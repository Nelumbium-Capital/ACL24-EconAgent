"""
EconAgent implementation with perception, memory, and reflection.

Implements the EconAgent paper methodology for LLM-based economic agents
with bounded rationality, memory windows, and quarterly reflections.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
from mesa import Agent

from src.models.batch_llm_client import BatchLLMClient, LLMResponse, get_batch_client
from src.utils.logging_config import logger


@dataclass
class AgentObservation:
    """Single timestep observation for an agent."""
    timestep: int
    wage: float
    savings: float
    loan_balance: float
    work: float
    consumption: float
    inflation: float
    interest_rate: float
    unemployment: float
    credit_spread: float


@dataclass
class AgentReflection:
    """Quarterly reflection snapshot."""
    timestep: int
    lessons: List[str]
    behavioral_adjustments: Dict[str, float]


class EconAgentMemory:
    """
    Memory manager for EconAgent with sliding window + reflection storage.
    
    Maintains:
    - Last N months of observations (default 6)
    - Quarterly reflection snapshots (persistent)
    """
    
    def __init__(self, window_size: int = 6):
        """
        Initialize agent memory.
        
        Args:
            window_size: Number of recent observations to keep (months)
        """
        self.window_size = window_size
        self.observations: deque = deque(maxlen=window_size)
        self.reflections: List[AgentReflection] = []
    
    def add_observation(self, obs: AgentObservation):
        """Add new observation to memory."""
        self.observations.append(obs)
    
    def add_reflection(self, reflection: AgentReflection):
        """Store a reflection snapshot."""
        self.reflections.append(reflection)
    
    def get_memory_summary(self) -> str:
        """
        Generate compact memory summary for prompt.
        
        Returns:
            String summary of recent behavior and reflections
        """
        if not self.observations:
            return "No prior history."
        
        # Summarize recent observations
        recent_work = [obs.work for obs in self.observations]
        recent_consumption = [obs.consumption for obs in self.observations]
        
        avg_work = np.mean(recent_work)
        avg_consumption = np.mean(recent_consumption)
        
        summary = f"Over the past {len(self.observations)} months: "
        summary += f"avg labor participation={avg_work:.2f}, avg consumption={avg_consumption:.2f}. "
        
        # Include most recent reflection
        if self.reflections:
            last_reflection = self.reflections[-1]
            if last_reflection.lessons:
                summary += f"Last reflection: {last_reflection.lessons[0]}"
        
        return summary
    
    def get_event_history(self, last_n: int = 3) -> str:
        """
        Get recent event history for reflection.
        
        Args:
            last_n: Number of recent months to include
            
        Returns:
            String description of events
        """
        if not self.observations:
            return "No events recorded."
        
        recent = list(self.observations)[-last_n:]
        
        events = []
        for obs in recent:
            event = f"Month {obs.timestep}: worked {obs.work:.2f}, consumed {obs.consumption:.2f}"
            if obs.unemployment > 0.06:
                event += " (high unemployment)"
            if obs.inflation > 0.04:
                event += " (elevated inflation)"
            events.append(event)
        
        return "; ".join(events)


class EconAgentLLM:
    """
    LLM decision-maker with perception + memory prompt builder.
    
    Generates prompts following EconAgent paper format:
    - Perception: financial state + observed economy
    - Memory: summarized history
    - Output: JSON with decisions
    """
    
    def __init__(
        self,
        client: Optional[BatchLLMClient] = None,
        reflection_frequency: int = 3
    ):
        """
        Initialize EconAgent LLM.
        
        Args:
            client: Batch LLM client (uses global if None)
            reflection_frequency: Months between reflections
        """
        self.client = client or get_batch_client()
        self.reflection_frequency = reflection_frequency
    
    def build_decision_prompt(
        self,
        name: str,
        age: int,
        occupation: str,
        location: str,
        wage: float,
        savings: float,
        loan_balance: float,
        inflation: float,
        interest_rate: float,
        unemployment: float,
        credit_spread: float,
        memory_summary: str
    ) -> str:
        """
        Build perception + action prompt.
        
        Returns EconAgent-style prompt requesting JSON decision.
        """
        prompt = f"""You are {name}, a {age}-year-old {occupation} in {location}.

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

Respond with JSON only, no other text."""
        
        return prompt
    
    def build_reflection_prompt(
        self,
        event_history: str
    ) -> str:
        """
        Build quarterly reflection prompt.
        
        Returns EconAgent-style reflection prompt requesting JSON output.
        """
        prompt = f"""Over the past quarter, these events affected you:
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

Respond with JSON only, no other text."""
        
        return prompt
    
    def make_decision(
        self,
        agent_state: Dict[str, Any],
        memory: EconAgentMemory
    ) -> Dict[str, float]:
        """
        Make economic decision using LLM.
        
        Args:
            agent_state: Current agent state dict
            memory: Agent's memory
            
        Returns:
            Decision dict with 'work' and 'consumption' keys
        """
        prompt = self.build_decision_prompt(
            name=agent_state.get('name', 'Worker'),
            age=agent_state.get('age', 40),
            occupation=agent_state.get('occupation', 'worker'),
            location=agent_state.get('location', 'City'),
            wage=agent_state['wage'],
            savings=agent_state['savings'],
            loan_balance=agent_state['loan_balance'],
            inflation=agent_state['inflation'],
            interest_rate=agent_state['interest_rate'],
            unemployment=agent_state['unemployment'],
            credit_spread=agent_state['credit_spread'],
            memory_summary=memory.get_memory_summary()
        )
        
        # Use batch client for single inference
        responses = self.client.batch_inference(
            prompts=[prompt],
            system_prompt="You are an expert economic agent making rational decisions.",
            temperature=0.3,
            max_tokens=200,
            expected_json_keys=['work', 'consumption']
        )
        
        response = responses[0]
        
        if response.success and response.data:
            # Ensure values are in valid range
            work = np.clip(response.data.get('work', 0.5), 0.0, 1.0)
            consumption = np.clip(response.data.get('consumption', 0.5), 0.0, 1.0)
            return {'work': work, 'consumption': consumption}
        else:
            # Fallback: reasonable defaults
            logger.warning(f"LLM decision failed: {response.error}, using defaults")
            return {'work': 0.6, 'consumption': 0.4}
    
    def make_reflection(
        self,
        memory: EconAgentMemory
    ) -> AgentReflection:
        """
        Generate quarterly reflection.
        
        Args:
            memory: Agent's memory
            
        Returns:
            AgentReflection object
        """
        event_history = memory.get_event_history(last_n=self.reflection_frequency)
        prompt = self.build_reflection_prompt(event_history)
        
        responses = self.client.batch_inference(
            prompts=[prompt],
            system_prompt="You are an expert economic agent reflecting on past experiences.",
            temperature=0.4,
            max_tokens=300,
            expected_json_keys=['lessons', 'behavioral_adjustments']
        )
        
        response = responses[0]
        
        if response.success and response.data:
            lessons = response.data.get('lessons', ["Continue current strategy"])
            adjustments = response.data.get('behavioral_adjustments', {})
            
            return AgentReflection(
                timestep=len(memory.observations),
                lessons=lessons[:3],  # Max 3 lessons
                behavioral_adjustments={
                    'work_delta': np.clip(adjustments.get('work_delta', 0.0), -0.2, 0.2),
                    'consumption_delta': np.clip(adjustments.get('consumption_delta', 0.0), -0.2, 0.2)
                }
            )
        else:
            # Fallback reflection
            return AgentReflection(
                timestep=len(memory.observations),
                lessons=["Maintain current behavior"],
                behavioral_adjustments={'work_delta': 0.0, 'consumption_delta': 0.0}
            )


class WorkerAgent(Agent):
    """
    Mesa agent representing a worker with LLM-based decisions.
    
    Makes work/consumption decisions based on:
    - Financial state (wage, savings, debt)
    - Economic conditions (inflation, unemployment, etc.)
    - Memory of past decisions
    - Periodic reflections
    """
    
    def __init__(
        self,
        unique_id: int,
        model,
        name: str = None,
        age: int = None,
        occupation: str = "worker",
        location: str = "City",
        initial_savings: float = 5000.0,
        wage: float = 4000.0,
        use_llm: bool = True
    ):
        """
        Initialize worker agent.
        
        Args:
            unique_id: Unique agent ID
            model: Mesa model instance
            name: Agent name
            age: Agent age
            occupation: Occupation type
            location: Location
            initial_savings: Initial savings amount
            wage: Monthly wage
            use_llm: Whether to use LLM for decisions (vs heuristic)
        """
        super().__init__(unique_id, model)
        
        self.name = name or f"Worker_{unique_id}"
        self.age = age or 40  # Default age if not specified
        self.occupation = occupation
        self.location = location
        
        # Financial state
        self.savings = initial_savings
        self.wage = wage
        self.loan_balance = 0.0
        
        # Decision parameters
        self.work = 1.0  # Full-time initially
        self.consumption = 0.5  # Moderate consumption
        
        # Memory and LLM
        self.use_llm = use_llm
        self.memory = EconAgentMemory(window_size=6)
        self.llm = EconAgentLLM() if use_llm else None
        
        # Behavioral adjustments from reflection
        self.work_adjustment = 0.0
        self.consumption_adjustment = 0.0
        
        # Agent type for reporting
        self.agent_type = "worker"
        
        logger.debug(f"Created {self.name}, age {self.age}, LLM={use_llm}")
    
    def step(self):
        """Execute agent's decision-making for this period."""
        # Gather current state
        agent_state = {
            'name': self.name,
            'age': self.age,
            'occupation': self.occupation,
            'location': self.location,
            'wage': self.wage,
            'savings': self.savings,
            'loan_balance': self.loan_balance,
            'inflation': self.model.inflation_rate if hasattr(self.model, 'inflation_rate') else 0.03,
            'interest_rate': self.model.interest_rate,
            'unemployment': self.model.unemployment_rate,
            'credit_spread': self.model.credit_spread
        }
        
        # Make decision
        if self.use_llm and self.llm:
            decision = self.llm.make_decision(agent_state, self.memory)
            self.work = decision['work']
            self.consumption = decision['consumption']
        else:
            # Simple heuristic fallback
            self._heuristic_decision(agent_state)
        
        # Apply behavioral adjustments from reflections
        self.work = np.clip(self.work + self.work_adjustment, 0.0, 1.0)
        self.consumption = np.clip(self.consumption + self.consumption_adjustment, 0.0, 1.0)
        
        # Update financial state
        self._update_finances()
        
        # Store observation in memory
        obs = AgentObservation(
            timestep=self.model.current_step,
            wage=self.wage,
            savings=self.savings,
            loan_balance=self.loan_balance,
            work=self.work,
            consumption=self.consumption,
            inflation=agent_state['inflation'],
            interest_rate=agent_state['interest_rate'],
            unemployment=agent_state['unemployment'],
            credit_spread=agent_state['credit_spread']
        )
        self.memory.add_observation(obs)
        
        # Quarterly reflection
        if self.use_llm and self.model.current_step % 3 == 0 and self.model.current_step > 0:
            self._reflect()
    
    def _heuristic_decision(self, agent_state: Dict[str, Any]):
        """Simple heuristic fallback when LLM disabled."""
        # Work more if unemployment is low
        if agent_state['unemployment'] < 0.05:
            self.work = 0.9
        else:
            self.work = 0.6
        
        # Consume less if savings are low
        if self.savings < self.wage * 2:
            self.consumption = 0.3
        else:
            self.consumption = 0.5
    
    def _update_finances(self):
        """Update savings and consumption based on decisions."""
        # Income from work
        income = self.wage * self.work
        
        # Consumption spending
        available_to_consume = self.savings + income
        spending = available_to_consume * self.consumption
        
        # Update savings
        self.savings = self.savings + income - spending
        self.savings = max(0.0, self.savings)  # Can't go negative
    
    def _reflect(self):
        """Generate and apply quarterly reflection."""
        if not self.llm:
            return
        
        reflection = self.llm.make_reflection(self.memory)
        self.memory.add_reflection(reflection)
        
        # Apply behavioral adjustments
        adjustments = reflection.behavioral_adjustments
        self.work_adjustment = adjustments.get('work_delta', 0.0)
        self.consumption_adjustment = adjustments.get('consumption_delta', 0.0)
        
        logger.debug(f"{self.name} reflected: {reflection.lessons[0][:50]}...")


class FirmAgentLLM(Agent):
    """
    Mesa agent representing a firm with LLM-based hiring/production decisions.
    
    Similar to WorkerAgent but makes business decisions:
    - Production level
    - Hiring decisions
    - Investment
    """
    
    def __init__(
        self,
        unique_id: int,
        model,
        name: str = None,
        initial_capital: float = 50000.0,
        use_llm: bool = True
    ):
        """
        Initialize firm agent.
        
        Args:
            unique_id: Unique agent ID
            model: Mesa model instance
            name: Firm name
            initial_capital: Initial capital
            use_llm: Whether to use LLM
        """
        super().__init__(unique_id, model)
        
        self.name = name or f"Firm_{unique_id}"
        self.capital = initial_capital
        self.production = 1.0
        self.hiring = 0.5
        
        self.use_llm = use_llm
        self.memory = EconAgentMemory(window_size=6)
        self.agent_type = "firm"
    
    def step(self):
        """Execute firm's decision-making."""
        # Simple heuristic for firms (can be extended with LLM)
        if self.model.unemployment_rate > 0.07:
            self.hiring = 0.3
            self.production = 0.7
        else:
            self.hiring = 0.6
            self.production = 0.9
        
        # Update capital based on production
        revenue = self.production * 10000
        costs = self.hiring * 5000
        self.capital += revenue - costs
        self.capital = max(1000.0, self.capital)

