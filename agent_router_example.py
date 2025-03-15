"""
Example demonstrating the LLM Router as a decision-making component in the agent.

This example shows how the agent uses the LLM Router to determine the best
approach for handling different types of queries.
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent
from SimplerLLM.tools.serp import search_with_serper_api
from SimplerLLM.tools.python_func import execute_python_code

def calculator(expression: str) -> str:
    """
    Simple calculator tool that evaluates mathematical expressions.
    
    Args:
        expression: The mathematical expression to evaluate
        
    Returns:
        The result of the evaluation
    """
    try:
        # Use Python's eval function with a safe subset of operations
        code = f"""
# Safe math operations
import math

# Evaluate the expression
result = {expression}
print(result)
"""
        result = execute_python_code(code)
        return result.strip()
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.7,
        verbose=False,
    )
    
    # Create an agent with verbose output to see the decision-making process
    agent = Agent(llm=llm, verbose=True)
    
    # Add tools for the agent to use
    agent.add_tool(
        name="web_search",
        func=search_with_serper_api,
        description="Search the web for information",
        parameters={
            "query": "The search query string",
            "num_results": "Maximum number of results to return (default: 5)"
        }
    )
    
    agent.add_tool(
        name="calculator",
        func=calculator,
        description="Evaluate mathematical expressions",
        parameters={
            "expression": "The mathematical expression to evaluate (e.g., '2 + 2 * 3')"
        }
    )
    
    # Set a system prompt that emphasizes the decision-making process
    agent.set_system_prompt(
        "You are a helpful assistant that can answer questions directly or use tools when needed. "
        "For each query, first determine if you can answer directly from your knowledge, "
        "or if you need to use a tool to gather information."
    )
    
    # Example 1: Direct knowledge query (should be answered directly)
    print("\n--- Example 1: Direct Knowledge Query ---")
    query1 = "What is the capital of France?"
    print(f"User: {query1}")
    response1 = agent.run(query1)
    print(f"Agent: {response1}")
    
    # Example 2: Calculation query (should use calculator tool)
    print("\n--- Example 2: Calculation Query ---")
    query2 = "What is the square root of 144 plus 25?"
    print(f"User: {query2}")
    response2 = agent.run(query2)
    print(f"Agent: {response2}")
    
    # Example 3: Web search query (should use web_search tool)
    print("\n--- Example 3: Web Search Query ---")
    query3 = "What are the latest developments in quantum computing?"
    print(f"User: {query3}")
    response3 = agent.run(query3)
    print(f"Agent: {response3}")
    
    # Example 4: Ambiguous query (router should decide the best approach)
    print("\n--- Example 4: Ambiguous Query ---")
    query4 = "Tell me about the Pythagorean theorem and calculate 3^2 + 4^2"
    print(f"User: {query4}")
    response4 = agent.run(query4)
    print(f"Agent: {response4}")

if __name__ == "__main__":
    main()
