"""
Advanced example of using the SimplerLLM Agent Builder.

This example demonstrates how to create a more complex agent that:
1. Uses ReliableLLM for fallback support
2. Incorporates multiple tools from the SimplerLLM library
3. Uses custom system prompts
"""

from SimplerLLM.language import OpenAILLM, GeminiLLM, ReliableLLM, LLMProvider
from SimplerLLM.agents import Agent, AgentMemory
from SimplerLLM.tools.serp import search_with_duck_duck_go
from SimplerLLM.tools.python_func import execute_python_code
from SimplerLLM.tools.file_functions import save_text_to_file

def main():
    # Create primary and secondary LLM instances
    primary_llm = OpenAILLM(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.7
    )
    
    secondary_llm = GeminiLLM(
        provider=LLMProvider.GEMINI,
        model_name="gemini-pro",
        temperature=0.7
    )
    
    # Create a ReliableLLM instance with fallback
    reliable_llm = ReliableLLM(primary_llm, secondary_llm, verbose=True)
    
    # Create a custom system prompt
    system_prompt = """You are an advanced AI assistant with access to various tools.
You can search the web, execute Python code, and save files.

When a user asks a question:
1. Determine if you need external information or computation
2. Use the appropriate tool to gather information or perform calculations
3. Provide a clear, concise response based on the results

Always think step by step and explain your reasoning."""

    # Create a memory instance with a larger token limit
    memory = AgentMemory(max_tokens=8000)
    
    # Create an agent with the reliable LLM, custom memory, and system prompt
    agent = Agent(
        llm=reliable_llm,
        memory=memory,
        system_prompt=system_prompt,
        verbose=True
    )
    
    # Add tools to the agent
    
    # 1. Web search tool
    agent.add_tool(
        name="web_search",
        func=search_with_duck_duck_go,
        description="Search the web for information",
        parameters={
            "query": "The search query string",
            "max_results": "Maximum number of results to return (default: 5)"
        }
    )
    
    # 2. Python execution tool
    agent.add_tool(
        name="execute_python",
        func=execute_python_code,
        description="Execute Python code and return the result",
        parameters={
            "input_code": "Python code to execute"
        }
    )
    
    # 3. File saving tool
    agent.add_tool(
        name="save_file",
        func=save_text_to_file,
        description="Save text to a file",
        parameters={
            "text": "Text content to save",
            "filename": "Name of the file to save to (default: output.txt)"
        }
    )
    
    # Example user queries to demonstrate different tool usage
    queries = [
        "What is the current price of Bitcoin?",
        "Calculate the factorial of 5 using Python",
        "Save the first 10 Fibonacci numbers to a file named fibonacci.txt"
    ]
    
    # Run the agent for each query
    for i, query in enumerate(queries):
        print(f"\n\n--- Query {i+1}: {query} ---\n")
        response = agent.run(query)
        print(f"\nFinal Response: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
