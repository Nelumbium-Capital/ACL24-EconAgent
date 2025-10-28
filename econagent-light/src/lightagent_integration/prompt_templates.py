"""
Prompt templates for economic agent decision-making.
Ports original ACL24-EconAgent prompts to LightAgent framework.
"""

import re
from typing import Dict, List, Any, Optional
from .light_client import AgentProfile, EnvironmentSnapshot

def prettify_document(document: str) -> str:
    """
    Clean up document formatting (from original simulate_utils.py).
    Remove sequences of whitespace characters including newlines.
    """
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned

def format_numbers(numbers: List[float]) -> str:
    """Format list of numbers for display (from original)."""
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers: List[float]) -> str:
    """Format list of percentages for display (from original)."""
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'

def build_perception_prompt(
    profile: AgentProfile,
    env: EnvironmentSnapshot,
    memory_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build perception + decision prompt for economic agent.
    Ports original problem_prompt, job_prompt, and economic context.
    
    Args:
        profile: Agent profile information
        env: Economic environment snapshot
        memory_context: Optional memory context from previous interactions
        
    Returns:
        List of message dictionaries for LLM conversation
    """
    messages = []
    
    # System message (agent identity)
    system_prompt = f"""You are {profile.name}, a {profile.age}-year-old individual living in {profile.city}. 
As with all Americans, a portion of your monthly income is taxed by the federal government. 
This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: 
after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.

You will decide (1) your propensity to work this month and (2) the fraction of your available assets to spend on essential goods.
Return a JSON: {{"work": <0-1 step 0.02>, "consumption": <0-1 step 0.02>}}."""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add memory context if available
    if memory_context:
        messages.append({"role": "system", "content": f"Recent experiences: {memory_context}"})
    
    # Build main economic context prompt
    problem_prompt = f"Now it's {env.year}.{env.month:02d}."
    
    # Job context (from original job_prompt logic)
    if profile.job == "Unemployment":
        job_prompt = f"""In the previous month, you became unemployed and had no income. 
Now, you are invited to work as a(an) {env.offer_job} with monthly salary of ${env.offer_wage:.2f}."""
    else:
        # Determine wage change direction
        wage_change = "increased" if env.timestep > 0 else "stable"  # Simplified for now
        job_prompt = f"""In the previous month, you worked as a(an) {profile.job}. 
If you continue working this month, your expected income will be ${profile.monthly_wage:.2f}, 
which has {wage_change} compared to the last month due to market dynamics."""
    
    # Consumption context
    if profile.last_consumption <= 0 and env.timestep > 0:
        consumption_prompt = "Besides, you had no consumption due to shortage of goods."
    else:
        consumption_prompt = f"Besides, your consumption was ${profile.last_consumption * profile.wealth:.2f}."
    
    # Tax context
    tax_prompt = f"""Your tax deduction amounted to ${profile.wealth * 0.1:.2f}. 
However, as part of the government's redistribution program, you received a credit of ${profile.wealth * 0.05:.2f}.
In this month, the government sets the brackets: {format_numbers(env.tax_brackets[:4])} 
and their corresponding rates: {format_percentages(env.tax_rates[:4])}. 
Income earned within each bracket is taxed only at that bracket's rate."""
    
    # Price context
    if env.timestep == 0:
        price_prompt = f"Meanwhile, in the consumption market, the average price of essential goods is now at ${env.price:.2f}."
    else:
        if env.inflation_rate >= 0:
            price_prompt = f"""Meanwhile, inflation has led to a price increase in the consumption market, 
with the average price of essential goods now at ${env.price:.2f}."""
        else:
            price_prompt = f"""Meanwhile, deflation has led to a price decrease in the consumption market, 
with the average price of essential goods now at ${env.price:.2f}."""
    
    # Financial context
    financial_prompt = f"""Your current savings account balance is ${profile.wealth:.2f}. 
Interest rates, as set by your bank, stand at {env.interest_rate*100:.2f}%."""
    
    # Decision request
    decision_prompt = """With all these factors in play, and considering aspects like your living costs, 
any future aspirations, and the broader economic trends, how is your willingness to work this month? 
Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?

Please share your decisions in a JSON format. The format should have two keys: 
'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) 
and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your 
savings and income you intend to spend on essential goods)."""
    
    # Combine all prompts
    full_prompt = f"""{problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt} 
{financial_prompt} {decision_prompt}"""
    
    # Clean up formatting
    full_prompt = prettify_document(full_prompt)
    
    messages.append({"role": "user", "content": full_prompt})
    
    return messages

def build_reflection_prompt(
    profile: AgentProfile,
    quarterly_data: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Build quarterly reflection prompt for agent learning.
    Ports original reflection_prompt logic.
    
    Args:
        profile: Agent profile information
        quarterly_data: Economic data from last 3 months
        
    Returns:
        List of message dictionaries for LLM conversation
    """
    messages = []
    
    # System message
    system_prompt = f"""You are {profile.name} reflecting on the last quarter's economic data and your own actions. 
Summarize key trends and list up to 3 lessons that will change your decision-making next quarter."""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Format quarterly data
    data_summaries = []
    for i, month_data in enumerate(quarterly_data, 1):
        summary = f"""Month{i}: Price=${month_data.get('price', 0):.2f}, 
Interest={month_data.get('interest_rate', 0)*100:.1f}%, 
Your work decision={month_data.get('work_decision', 0):.2f}, 
consumption decision={month_data.get('consumption_decision', 0):.2f}, 
actual income=${month_data.get('income', 0):.2f}"""
        data_summaries.append(summary)
    
    # Main reflection prompt
    reflection_prompt = f"""Last quarter data:
{chr(10).join(data_summaries)}

Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, 
as well as their dynamics. What conclusions have you drawn?

QUESTION:
1) Summarize labor market, consumption market, and financial market trends in 2-4 sentences each.
2) Provide 1-3 explicit actionable takeaways (short bullet points) you will use when making 
   work/consumption decisions next quarter.

Return JSON: {{"summary":"...", "takeaways":["...","..."]}}

Your answer must be less than 200 words!"""
    
    reflection_prompt = prettify_document(reflection_prompt)
    
    messages.append({"role": "user", "content": reflection_prompt})
    
    return messages

def build_tool_context_prompt(
    market_analysis: str,
    tax_calculation: Dict[str, float],
    profile: AgentProfile
) -> str:
    """
    Build context from economic tools for enhanced decision-making.
    
    Args:
        market_analysis: Analysis from market conditions tool
        tax_calculation: Results from tax calculation tool
        profile: Agent profile
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    if market_analysis:
        context_parts.append(f"Market Analysis: {market_analysis}")
    
    if tax_calculation:
        tax_info = f"""Tax Analysis: If you earn income, you would owe ${tax_calculation.get('tax_owed', 0):.2f} 
in taxes (effective rate: {tax_calculation.get('effective_rate', 0):.1%}), 
leaving you with ${tax_calculation.get('after_tax_income', 0):.2f} after taxes."""
        context_parts.append(tax_info)
    
    return " ".join(context_parts)

def validate_decision_response(response_text: str) -> Dict[str, Any]:
    """
    Validate and parse decision response from LLM.
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Parsed and validated decision dictionary
    """
    import json
    
    try:
        # Try to parse as JSON
        if response_text.strip().startswith('{'):
            data = json.loads(response_text)
        else:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        
        # Validate and normalize values
        work = float(data.get('work', 0.5))
        consumption = float(data.get('consumption', 0.5))
        
        # Clamp to [0,1] range
        work = max(0.0, min(1.0, work))
        consumption = max(0.0, min(1.0, consumption))
        
        # Round to 0.02 step increments
        work = round(work * 50) / 50.0
        consumption = round(consumption * 50) / 50.0
        
        return {
            "work": work,
            "consumption": consumption,
            "valid": True,
            "raw_response": response_text[:200]
        }
        
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # Return fallback values
        return {
            "work": 0.2,
            "consumption": 0.1,
            "valid": False,
            "error": str(e),
            "raw_response": response_text[:200]
        }

def validate_reflection_response(response_text: str) -> Dict[str, Any]:
    """
    Validate and parse reflection response from LLM.
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Parsed reflection dictionary
    """
    import json
    
    try:
        # Try to parse as JSON
        if response_text.strip().startswith('{'):
            data = json.loads(response_text)
        else:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        
        summary = data.get("summary", "Economic conditions were mixed with various market dynamics")
        takeaways = data.get("takeaways", ["Monitor market conditions", "Adjust work-consumption balance"])
        
        # Ensure takeaways is a list
        if isinstance(takeaways, str):
            takeaways = [takeaways]
        elif not isinstance(takeaways, list):
            takeaways = ["Monitor market conditions", "Adjust work-consumption balance"]
        
        return {
            "summary": summary,
            "takeaways": takeaways[:3],  # Limit to 3 takeaways
            "valid": True,
            "raw_response": response_text[:300]
        }
        
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # Return fallback reflection
        return {
            "summary": "Economic conditions were mixed with various market dynamics affecting decision-making",
            "takeaways": [
                "Monitor price trends and inflation",
                "Adjust work-consumption balance based on market conditions", 
                "Consider long-term financial stability"
            ],
            "valid": False,
            "error": str(e),
            "raw_response": response_text[:300]
        }

# Template constants for easy customization
SYSTEM_ROLE_TEMPLATE = """You are {name}, a {age}-year-old individual living in {city}. 
As with all Americans, a portion of your monthly income is taxed by the federal government. 
This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: 
after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.

You will decide (1) your propensity to work this month and (2) the fraction of your available assets to spend on essential goods.
Return a JSON: {{"work": <0-1 step 0.02>, "consumption": <0-1 step 0.02>}}."""

DECISION_REQUEST_TEMPLATE = """With all these factors in play, and considering aspects like your living costs, 
any future aspirations, and the broader economic trends, how is your willingness to work this month? 
Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?

Please share your decisions in a JSON format. The format should have two keys: 
'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) 
and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your 
savings and income you intend to spend on essential goods)."""

REFLECTION_SYSTEM_TEMPLATE = """You are {name} reflecting on the last quarter's economic data and your own actions. 
Summarize key trends and list up to 3 lessons that will change your decision-making next quarter."""

REFLECTION_QUESTION_TEMPLATE = """QUESTION:
1) Summarize labor market, consumption market, and financial market trends in 2-4 sentences each.
2) Provide 1-3 explicit actionable takeaways (short bullet points) you will use when making 
   work/consumption decisions next quarter.

Return JSON: {{"summary":"...", "takeaways":["...","..."]}}

Your answer must be less than 200 words!"""