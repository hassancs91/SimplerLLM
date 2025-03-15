"""
Example of using the SimplerLLM Agent Builder.

This example demonstrates how to create a simple agent that can use tools
from the SimplerLLM library and showcases the new modular architecture.
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent, AgentRole
from SimplerLLM.tools.serp import search_with_serper_api

def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # You can change this to any supported model
        temperature=0.7,
        verbose=False,
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
    
    # Example of setting a simple role (just a system prompt)
    agent.set_system_prompt("You are a helpful assistant that specializes in explaining programming concepts.")
    
    # Example of using the more advanced role system (commented out)
    """
    # Create a more detailed role
    researcher_role = AgentRole(
        name="Programming Educator",
        description="An expert in explaining programming concepts clearly",
        system_prompt="You are a programming educator who explains concepts in simple terms.",
        responsibilities=["Explain programming concepts", "Provide code examples", "Answer technical questions"],
        constraints=["Avoid overly technical jargon", "Focus on practical applications"],
        allowed_tools=["web_search"],
        priority_level=1,
        fallback_behavior="Provide general programming guidance"
    )
    
    # Assign the role to the agent
    agent.set_role(researcher_role)
    """
    
    # Run the agent with a user query
    user_query = "what is the current bitcoin price now?"
    
    
    # The agent will now:
    # 1. Use the LLM Router to determine if it needs tools or can answer directly
    # 2. If it needs tools, it will select and execute the appropriate tool
    # 3. Generate a response based on the tool results or direct knowledge
    response = agent.run(user_query)
    
    print(f"\nFinal Response: {response}\n")
    
    # Example of a query that would likely use the search tool
    # user_query = "What are the latest developments in quantum computing?"
    # response = agent.run(user_query)
    # print(f"\nFinal Response: {response}\n")

if __name__ == "__main__":
    main()
