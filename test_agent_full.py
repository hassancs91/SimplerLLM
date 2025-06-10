from dotenv import load_dotenv
import os
from SimplerLLM.agents import Agent
from SimplerLLM.language.llm import LLM, LLMProvider

# Load environment variables from .env file
load_dotenv()

def get_llm_instance(verbose_llm=False):
    """Initializes and returns an LLM instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file. Please set it and ensure the .env file is in the root directory.")
        return None
    try:
        # Instantiate OpenAI_LLM directly
        llm_instance = LLM.create(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                verbose=False)
        # You can test with other providers/models by changing the class and model_name
        # from SimplerLLM.language.llm_providers import Anthropic_LLM
        # llm = Anthropic_LLM(api_key=os.getenv("ANTHROPIC_API_KEY"), model_name="claude-3-opus-20240229", verbose=verbose_llm)
        return llm_instance
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

# --- Mock Tool Definitions ---
def get_capital_city(country: str) -> str:
    print(f"TOOL CALLED: get_capital_city(country='{country}')")
    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "japan": "Tokyo",
        "united states": "Washington D.C."
    }
    return capitals.get(country.lower(), f"I don't know the capital of {country}.")

def get_population(city: str) -> str:
    print(f"TOOL CALLED: get_population(city='{city}')")
    populations = {
        "paris": "approximately 2.1 million",
        "berlin": "approximately 3.7 million",
        "tokyo": "approximately 14 million",
        "washington d.c.": "approximately 0.7 million"
    }
    return populations.get(city.lower(), f"I don't have population data for {city}.")

def simple_math(expression: str) -> str:
    print(f"TOOL CALLED: simple_math(expression='{expression}')")
    try:
        # Basic safety for eval - use a safer parser for production tools
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid characters in expression."
        # Sanitize expression further if necessary
        result = eval(expression) # Be cautious with eval
        return f"The result of '{expression}' is {result}."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# --- Main Test Execution ---
if __name__ == "__main__":
    # Initialize LLM (verbose_llm=False to keep LLM's own logs quiet, agent's verbose will show its steps)
    llm_instance = get_llm_instance(verbose_llm=False) 
    
    if not llm_instance:
        print("Exiting script as LLM could not be initialized.")
        exit()

    print("\n--- Scenario 1: Direct Question (No Tools) ---")
    # Agent initialized without any tools, verbose=True to see agent's thoughts
    agent_direct = Agent(llm=llm_instance, verbose=True) 
    query1 = "What is the result of 15 multiplied by 3?" 
    print(f"User Query: {query1}")
    response1 = agent_direct.run(query1)
    print(f"Agent Response: {response1}\n")

    # --- Tool Definitions for subsequent agents ---
    tools_available = {
        "get_country_capital": { # Renamed for clarity
            "func": get_capital_city,
            "description": "Finds the capital city of a given country. Example: 'France'.",
            "parameters": {"country": "The name of the country."}
        },
        "get_city_population": {
            "func": get_population,
            "description": "Gets the approximate population of a given city. Example: 'Paris'.",
            "parameters": {"city": "The name of the city."}
        },
        "calculate_math_expression": {
            "func": simple_math,
            "description": "Calculates a simple mathematical expression. Example: '5*2+10/2'. Only supports basic arithmetic.",
            "parameters": {"expression": "The mathematical expression string, like '100 + 50'."}
        }
    }

    print("\n--- Scenario 2: Single Tool Use (Capital City) ---")
    # Agent initialized with tools, agent verbose=True
    agent_with_tools = Agent(llm=llm_instance, tools=tools_available, verbose=True) 
    query2 = "What is the capital of Japan?"
    print(f"User Query: {query2}")
    response2 = agent_with_tools.run(query2, max_iterations=3) # Max iterations for safety
    print(f"Agent Response: {response2}\n")

    print("\n--- Scenario 3: Multi-Tool Use (Capital, then Population) ---")
    # Re-using agent_with_tools, it will have memory of the previous interaction
    query3 = "What is the capital of Germany and what is its population?"
    print(f"User Query: {query3}")
    response3 = agent_with_tools.run(query3, max_iterations=5) 
    print(f"Agent Response: {response3}\n")
    
    print("\n--- Scenario 4: Conversation & Memory ---")
    # agent_with_tools still has memory
    query4a = "My favorite programming language is Python."
    print(f"User Query: {query4a}")
    response4a = agent_with_tools.run(query4a, max_iterations=3)
    print(f"Agent Response: {response4a}\n")
    
    query4b = "What did I mention as my favorite programming language?"
    print(f"User Query: {query4b}")
    response4b = agent_with_tools.run(query4b, max_iterations=3)
    print(f"Agent Response: {response4b}\n")

    print("\n--- Scenario 5: Potentially Ambiguous Query (Hoping for Clarification or Tool) ---")
    query5 = "Tell me about that famous city in Italy and then add 50 to 100."
    # This is ambiguous ("that famous city") and combines tasks.
    print(f"User Query: {query5}")
    response5 = agent_with_tools.run(query5, max_iterations=5)
    print(f"Agent Response: {response5}\n")
    
    print("\n--- Scenario 6: Setting a System Prompt/Role ---")
    system_prompt_chef = "You are a world-class chef. You respond to all queries with culinary flair, even if using tools for non-food topics. You always start your response with 'Bonjour, mon ami!'."
    agent_chef = Agent(llm=llm_instance, tools=tools_available, verbose=True, 
                         system_prompt=system_prompt_chef)
    query6 = "What is the capital of France, chef?"
    print(f"User Query: {query6}")
    response6 = agent_chef.run(query6, max_iterations=3)
    print(f"Agent Response: {response6}\n")

    print("\n--- Test Script Finished ---")
    print("Please review the agent's thoughts, tool calls, and responses for each scenario.")
    print("Remember to have your OPENAI_API_KEY (or other provider's key) in a .env file in the root directory.")
