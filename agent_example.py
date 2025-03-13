"""
Example of using the SimplerLLM Agent Builder.

This example demonstrates how to create a simple agent that can use tools
from the SimplerLLM library.
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent
from SimplerLLM.tools.serp import search_with_serper_api

def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # You can change this to any supported model
        temperature=0.7,
        verbose=True
    )
    
    # Create an agent with verbose output
    agent = Agent(llm=llm, verbose=True)
    
    # Add a search tool
    agent.add_tool(
        name="web_search",
        func=search_with_serper_api,
        description="Search the web for information",
        parameters={
            "query": "The search query string",
            "num_results": "Maximum number of results to return (default: 5)"
        }
    )
    
    # Run the agent with a user query
    user_query = "what is the latest tech news today?"
    #print(f"\nUser Query: {user_query}\n")
    
    response = agent.run(user_query)
    
    print(f"\nFinal Response: {response}\n")

if __name__ == "__main__":
    main()
