"""
LightAgent integration wrapper for economic agents.
Provides mem0 memory, tools, and Tree-of-Thought capabilities.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from LightAgent import LightAgent
    from mem0 import Memory
    LIGHTAGENT_AVAILABLE = True
except ImportError:
    LIGHTAGENT_AVAILABLE = False
    logging.warning("LightAgent not available, using fallback implementation")

from ..llm_integration import UnifiedLLMClient

logger = logging.getLogger(__name__)

@dataclass
class AgentProfile:
    """Agent profile for economic decision-making."""
    agent_id: str
    name: str
    age: int
    job: str
    city: str
    skill: float
    wealth: float
    monthly_wage: float
    last_work: bool
    last_consumption: float

@dataclass
class EnvironmentSnapshot:
    """Economic environment snapshot for agent context."""
    year: int
    month: int
    timestep: int
    price: float
    interest_rate: float
    unemployment_rate: float
    inflation_rate: float
    tax_brackets: List[float]
    tax_rates: List[float]
    offer_job: str
    offer_wage: float

class EconomicMemory:
    """Memory system for economic agents using mem0."""
    
    def __init__(self, enable_mem0: bool = True):
        self.enable_mem0 = enable_mem0
        self.memory = None
        self.local_memory = {}  # Fallback local memory
        
        if enable_mem0 and LIGHTAGENT_AVAILABLE:
            try:
                config = {
                    "version": "v1.1",
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": "econagent_memory",
                            "path": "./memory_db"
                        }
                    }
                }
                self.memory = Memory.from_config(config_dict=config)
                logger.info("Initialized mem0 memory system")
            except Exception as e:
                logger.warning(f"Failed to initialize mem0: {e}, using local memory")
                self.memory = None
    
    def store(self, user_id: str, data: str, metadata: Optional[Dict] = None):
        """Store memory for agent."""
        if self.memory:
            try:
                return self.memory.add(data, user_id=user_id, metadata=metadata)
            except Exception as e:
                logger.warning(f"mem0 store failed: {e}")
        
        # Fallback to local memory
        if user_id not in self.local_memory:
            self.local_memory[user_id] = []
        
        self.local_memory[user_id].append({
            "content": data,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        
        # Keep only last 10 memories per agent
        if len(self.local_memory[user_id]) > 10:
            self.local_memory[user_id] = self.local_memory[user_id][-10:]
    
    def retrieve(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories for agent."""
        if self.memory:
            try:
                results = self.memory.search(query, user_id=user_id, limit=limit)
                return [{"content": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
            except Exception as e:
                logger.warning(f"mem0 retrieve failed: {e}")
        
        # Fallback to local memory
        memories = self.local_memory.get(user_id, [])
        # Simple keyword matching for fallback
        relevant = []
        query_words = query.lower().split()
        
        for memory in memories[-5:]:  # Last 5 memories
            content = memory["content"].lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                relevant.append({
                    "content": memory["content"],
                    "score": score / len(query_words)
                })
        
        return sorted(relevant, key=lambda x: x["score"], reverse=True)

class LightAgentWrapper:
    """
    Wrapper integrating LightAgent framework with economic simulation.
    Provides memory, tools, and reasoning capabilities for economic agents.
    """
    
    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        enable_memory: bool = True,
        enable_tot: bool = True,
        memory_window: int = 1,
        reflection_frequency: int = 3
    ):
        self.llm_client = llm_client
        self.enable_memory = enable_memory
        self.enable_tot = enable_tot
        self.memory_window = memory_window
        self.reflection_frequency = reflection_frequency
        
        # Initialize memory system
        self.memory = EconomicMemory(enable_mem0=enable_memory)
        
        # Initialize LightAgent if available
        self.light_agent = None
        if LIGHTAGENT_AVAILABLE:
            try:
                self.light_agent = LightAgent(
                    role="You are an economic agent making work and consumption decisions based on market conditions and personal circumstances.",
                    model="custom",  # We'll override with our LLM client
                    memory=self.memory if enable_memory else None,
                    tree_of_thought=enable_tot,
                    tools=self._create_economic_tools()
                )
                logger.info("Initialized LightAgent framework")
            except Exception as e:
                logger.warning(f"Failed to initialize LightAgent: {e}")
                self.light_agent = None
        
        # Statistics
        self.stats = {
            "decisions_made": 0,
            "reflections_made": 0,
            "memory_stores": 0,
            "tool_calls": 0
        }
    
    def _create_economic_tools(self) -> List:
        """Create economic analysis tools for LightAgent."""
        tools = []
        
        def analyze_market_conditions(price: float, interest_rate: float, unemployment: float) -> str:
            """Analyze current market conditions for economic decision-making."""
            analysis = []
            
            if price > 1.0:
                analysis.append("Prices are high, consider reducing consumption")
            elif price < 0.8:
                analysis.append("Prices are low, good time for consumption")
            
            if interest_rate > 0.05:
                analysis.append("High interest rates favor saving over consumption")
            elif interest_rate < 0.02:
                analysis.append("Low interest rates favor consumption over saving")
            
            if unemployment > 0.1:
                analysis.append("High unemployment, job security is important")
            elif unemployment < 0.05:
                analysis.append("Low unemployment, good job market conditions")
            
            return "; ".join(analysis) if analysis else "Market conditions are stable"
        
        def calculate_tax_burden(income: float, tax_brackets: List[float], tax_rates: List[float]) -> Dict[str, float]:
            """Calculate tax burden for given income."""
            if not tax_brackets or not tax_rates:
                return {"tax_owed": 0.0, "effective_rate": 0.0}
            
            tax_owed = 0.0
            remaining_income = income
            
            for i, bracket in enumerate(tax_brackets[1:], 1):
                if remaining_income <= 0:
                    break
                
                bracket_size = bracket - tax_brackets[i-1]
                taxable_in_bracket = min(remaining_income, bracket_size)
                
                if i-1 < len(tax_rates):
                    tax_owed += taxable_in_bracket * tax_rates[i-1]
                
                remaining_income -= taxable_in_bracket
            
            effective_rate = tax_owed / income if income > 0 else 0.0
            
            return {
                "tax_owed": tax_owed,
                "effective_rate": effective_rate,
                "after_tax_income": income - tax_owed
            }
        
        # Add tool metadata for LightAgent
        analyze_market_conditions.tool_info = {
            "tool_name": "analyze_market_conditions",
            "tool_description": "Analyze current economic market conditions",
            "tool_params": [
                {"name": "price", "type": "float", "description": "Current price level"},
                {"name": "interest_rate", "type": "float", "description": "Current interest rate"},
                {"name": "unemployment", "type": "float", "description": "Current unemployment rate"}
            ]
        }
        
        calculate_tax_burden.tool_info = {
            "tool_name": "calculate_tax_burden",
            "tool_description": "Calculate tax burden for given income",
            "tool_params": [
                {"name": "income", "type": "float", "description": "Income to calculate taxes for"},
                {"name": "tax_brackets", "type": "list", "description": "Tax bracket thresholds"},
                {"name": "tax_rates", "type": "list", "description": "Tax rates for each bracket"}
            ]
        }
        
        tools.extend([analyze_market_conditions, calculate_tax_burden])
        return tools
    
    def decide(
        self,
        agent_profile: AgentProfile,
        env_snapshot: EnvironmentSnapshot,
        use_memory: bool = True
    ) -> Dict[str, float]:
        """
        Make economic decision using LightAgent framework.
        
        Args:
            agent_profile: Agent's personal information
            env_snapshot: Current economic environment
            use_memory: Whether to use memory for context
            
        Returns:
            Dict with 'work' and 'consumption' decisions
        """
        self.stats["decisions_made"] += 1
        
        try:
            # Build decision context
            messages = self._build_decision_messages(agent_profile, env_snapshot, use_memory)
            
            # Use LightAgent if available, otherwise direct LLM call
            if self.light_agent:
                # Override LightAgent's LLM with our unified client
                response = self._call_with_lightagent(messages, agent_profile.agent_id)
            else:
                # Direct LLM call with our unified client
                response = self.llm_client.make_economic_decision(messages)
            
            # Store interaction in memory
            if use_memory:
                self._store_decision_memory(agent_profile, env_snapshot, response)
            
            return {
                "work": response.get("work", 0.2),
                "consumption": response.get("consumption", 0.1),
                "_metadata": response.get("_metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Decision making failed for agent {agent_profile.agent_id}: {e}")
            return {
                "work": 0.2,
                "consumption": 0.1,
                "_metadata": {"error": str(e), "fallback_used": True}
            }
    
    def _build_decision_messages(
        self,
        profile: AgentProfile,
        env: EnvironmentSnapshot,
        use_memory: bool
    ) -> List[Dict[str, str]]:
        """Build conversation messages for economic decision."""
        messages = []
        
        # System message
        system_msg = f"""You are {profile.name}, a {profile.age}-year-old living in {profile.city}. 
You work as a {profile.job} and must make monthly economic decisions about work and consumption.
You will decide (1) your propensity to work this month and (2) the fraction of your available assets to spend on essential goods.
Return a JSON: {{"work": <0-1 step 0.02>, "consumption": <0-1 step 0.02>}}."""
        
        messages.append({"role": "system", "content": system_msg})
        
        # Add memory context if enabled
        if use_memory:
            memories = self.memory.retrieve(
                profile.agent_id,
                f"economic decision work consumption {env.month}",
                limit=3
            )
            if memories:
                memory_context = "Recent experiences: " + "; ".join([m["content"][:100] for m in memories[:2]])
                messages.append({"role": "system", "content": memory_context})
        
        # Economic context
        job_context = self._build_job_context(profile, env)
        economic_context = self._build_economic_context(profile, env)
        
        user_msg = f"""Economic context:
- This month: Year={env.year}, Month={env.month}
- Your last month: worked={profile.last_work}, consumption_spent={profile.last_consumption:.2f}
- Your current savings: ${profile.wealth:.2f}
- Expected income if working: ${profile.monthly_wage:.2f}

{job_context}

{economic_context}

Consider living costs, future aspirations, and economic trends. Give two numbers:
- "work": willingness to work this month (0.00–1.00 in steps of 0.02)
- "consumption": proportion of (savings + income) to spend on essential goods (0.00–1.00 in steps of 0.02)

Respond with valid JSON only."""
        
        messages.append({"role": "user", "content": user_msg})
        
        return messages
    
    def _build_job_context(self, profile: AgentProfile, env: EnvironmentSnapshot) -> str:
        """Build job-related context for decision."""
        if profile.job == "Unemployment":
            return f"You are unemployed and offered work as {env.offer_job} with monthly salary ${env.offer_wage:.2f}."
        else:
            wage_change = "increased" if profile.monthly_wage >= profile.monthly_wage else "decreased"
            return f"You work as {profile.job}. Your expected income is ${profile.monthly_wage:.2f}, {wage_change} from last month."
    
    def _build_economic_context(self, profile: AgentProfile, env: EnvironmentSnapshot) -> str:
        """Build economic environment context."""
        context_parts = []
        
        # Price context
        if env.timestep == 0:
            context_parts.append(f"The average price of essential goods is ${env.price:.2f}.")
        else:
            price_trend = "increased" if env.inflation_rate > 0 else "decreased"
            context_parts.append(f"Prices have {price_trend}, with essential goods at ${env.price:.2f}.")
        
        # Interest rate context
        context_parts.append(f"Interest rates are at {env.interest_rate*100:.2f}%.")
        
        # Tax context
        if env.tax_brackets and env.tax_rates:
            context_parts.append(f"Tax brackets: {[f'${b:.0f}' for b in env.tax_brackets[:3]]} with rates {[f'{r:.1%}' for r in env.tax_rates[:3]]}.")
        
        return " ".join(context_parts)
    
    def _call_with_lightagent(self, messages: List[Dict[str, str]], user_id: str) -> Dict[str, Any]:
        """Call LightAgent with custom LLM integration."""
        # Convert messages to single prompt for LightAgent
        prompt = messages[-1]["content"]  # Use the main user message
        
        try:
            # Use LightAgent's run method
            response_text = self.light_agent.run(prompt, user_id=user_id, stream=False)
            
            # Parse the response
            from ..llm_integration import validate_economic_decision
            decision = validate_economic_decision(response_text)
            
            return decision
            
        except Exception as e:
            logger.warning(f"LightAgent call failed: {e}, falling back to direct LLM")
            return self.llm_client.make_economic_decision(messages)
    
    def _store_decision_memory(
        self,
        profile: AgentProfile,
        env: EnvironmentSnapshot,
        decision: Dict[str, Any]
    ):
        """Store decision context in agent memory."""
        try:
            memory_content = f"Month {env.year}.{env.month:02d}: Decided work={decision['work']:.2f}, consumption={decision['consumption']:.2f}. "
            memory_content += f"Context: wealth=${profile.wealth:.0f}, price=${env.price:.2f}, job={profile.job}"
            
            metadata = {
                "month": env.month,
                "year": env.year,
                "decision_type": "economic",
                "work_decision": decision["work"],
                "consumption_decision": decision["consumption"]
            }
            
            self.memory.store(profile.agent_id, memory_content, metadata)
            self.stats["memory_stores"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to store decision memory: {e}")
    
    def reflect(
        self,
        agent_profile: AgentProfile,
        quarterly_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform quarterly reflection using LightAgent.
        
        Args:
            agent_profile: Agent's profile information
            quarterly_data: Economic data from last 3 months
            
        Returns:
            Dict with reflection insights and learning updates
        """
        self.stats["reflections_made"] += 1
        
        try:
            # Build reflection prompt
            messages = self._build_reflection_messages(agent_profile, quarterly_data)
            
            # Get reflection response
            if self.light_agent:
                prompt = messages[-1]["content"]
                response_text = self.light_agent.run(prompt, user_id=agent_profile.agent_id, stream=False)
            else:
                response = self.llm_client.call_model(messages, temperature=0.3, max_tokens=300)
                response_text = response["content"]
            
            # Parse reflection
            reflection = self._parse_reflection(response_text)
            
            # Store reflection in memory
            self._store_reflection_memory(agent_profile, reflection)
            
            return reflection
            
        except Exception as e:
            logger.error(f"Reflection failed for agent {agent_profile.agent_id}: {e}")
            return {
                "summary": "Unable to reflect on recent economic activity",
                "takeaways": ["Continue current economic strategy"],
                "error": str(e)
            }
    
    def _build_reflection_messages(
        self,
        profile: AgentProfile,
        quarterly_data: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build reflection conversation messages."""
        messages = []
        
        system_msg = f"You are {profile.name} reflecting on the last quarter's economic data and your decisions. Summarize key trends and lessons learned."
        messages.append({"role": "system", "content": system_msg})
        
        # Format quarterly data
        data_summary = []
        for i, month_data in enumerate(quarterly_data, 1):
            data_summary.append(f"Month {i}: Price=${month_data.get('price', 0):.2f}, "
                              f"Your work={month_data.get('work_decision', 0):.2f}, "
                              f"consumption={month_data.get('consumption_decision', 0):.2f}")
        
        user_msg = f"""Last quarter data:
{chr(10).join(data_summary)}

QUESTION:
1) Summarize labor market, consumption market, and financial market trends in 2-4 sentences each.
2) Provide 1-3 explicit actionable takeaways you will use when making work/consumption decisions next quarter.

Return JSON: {{"summary":"...", "takeaways":["...","..."]}}"""
        
        messages.append({"role": "user", "content": user_msg})
        
        return messages
    
    def _parse_reflection(self, response_text: str) -> Dict[str, Any]:
        """Parse reflection response."""
        try:
            # Try to parse as JSON
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
            else:
                # Extract JSON from text
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")
            
            return {
                "summary": data.get("summary", "Economic conditions were mixed"),
                "takeaways": data.get("takeaways", ["Continue monitoring market conditions"])
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse reflection: {e}")
            return {
                "summary": "Economic conditions were mixed with various market dynamics",
                "takeaways": ["Monitor price trends", "Adjust work-consumption balance", "Consider market volatility"]
            }
    
    def _store_reflection_memory(self, profile: AgentProfile, reflection: Dict[str, Any]):
        """Store quarterly reflection in memory."""
        try:
            memory_content = f"Quarterly reflection: {reflection['summary']} Key takeaways: {'; '.join(reflection['takeaways'])}"
            
            metadata = {
                "type": "reflection",
                "quarter": f"Q{(profile.agent_id.split('_')[-1] if '_' in profile.agent_id else '1')}",
                "takeaways_count": len(reflection["takeaways"])
            }
            
            self.memory.store(profile.agent_id, memory_content, metadata)
            
        except Exception as e:
            logger.warning(f"Failed to store reflection memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper usage statistics."""
        return {
            **self.stats,
            "memory_enabled": self.enable_memory,
            "tot_enabled": self.enable_tot,
            "lightagent_available": self.light_agent is not None
        }