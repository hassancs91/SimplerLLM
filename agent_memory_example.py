"""
Example demonstrating the memory capabilities of the SimplerLLM Agent.

This example shows how the agent maintains conversation context and can
refer back to previously discussed information.
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent, AgentMemory

def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.7,
        verbose=False,
    )
    
    # Create an agent with verbose output
    agent = Agent(llm=llm, verbose=True)
    
    # Set a role that emphasizes memory capabilities
    agent.set_system_prompt(
        "You are a helpful assistant with excellent memory. When asked about previous "
        "parts of the conversation, refer back to what was discussed earlier."
    )
    
    # First interaction - introduce some information
    print("\n--- First Interaction ---")
    query1 = "My name is Alex and I work as a software engineer. I'm interested in machine learning."
    print(f"User: {query1}")
    response1 = agent.run(query1)
    print(f"Agent: {response1}")
    
    # Second interaction - introduce more information
    print("\n--- Second Interaction ---")
    query2 = "I'm currently working on a project using Python and TensorFlow."
    print(f"User: {query2}")
    response2 = agent.run(query2)
    print(f"Agent: {response2}")
    
    # Third interaction - ask about previously mentioned information
    print("\n--- Third Interaction (Memory Test) ---")
    query3 = "What do I do for work and what programming languages am I using?"
    print(f"User: {query3}")
    response3 = agent.run(query3)
    print(f"Agent: {response3}")
    
    # Fourth interaction - ask about something not mentioned
    print("\n--- Fourth Interaction (Memory Boundaries Test) ---")
    query4 = "What's my favorite food?"
    print(f"User: {query4}")
    response4 = agent.run(query4)
    print(f"Agent: {response4}")
    
    # Print memory contents for demonstration
    print("\n--- Memory Contents ---")
    for i, message in enumerate(agent.memory.get_messages()):
        if message["role"] != "system":
            print(f"{i}. {message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()
