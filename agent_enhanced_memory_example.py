"""
Example demonstrating the enhanced agent with multiple memory systems.

This example shows how the enhanced agent uses multiple memory systems
to improve context retention and response quality.
"""

import os
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import (
    EnhancedAgent, 
    MemoryManager, 
    ConversationMemory, 
    EntityMemory
)

def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # You can change this to any supported model
        temperature=0.7,
        verbose=False,
    )
    
    # Create memory systems with persistence
    os.makedirs("memory", exist_ok=True)
    conversation_memory = ConversationMemory(
        llm=llm,
        max_tokens=4000,
        summarize_threshold=3000
    )
    entity_memory = EntityMemory(
        llm=llm,
        file_path="memory/entities.json"
    )
    
    # Create memory manager
    memory_manager = MemoryManager(llm)
    memory_manager.add_memory("conversation", conversation_memory)
    memory_manager.add_memory("entity", entity_memory)
    
    # Create an enhanced agent
    agent = EnhancedAgent(
        llm=llm,
        memory_manager=memory_manager,
        verbose=True
    )
    
    # Set a role that emphasizes memory capabilities
    agent.set_system_prompt(
        "You are a helpful assistant with excellent memory. When asked about previous "
        "parts of the conversation, refer back to what was discussed earlier. "
        "When you identify important information like names, dates, or facts, "
        "remember them for future reference."
    )
    
    # First interaction - introduce personal information
    print("\n--- First Interaction ---")
    query1 = "My name is Alex Johnson and I work as a software engineer at TechCorp. I was born on May 15, 1985."
    print(f"User: {query1}")
    response1 = agent.run(query1)
    print(f"Agent: {response1}")
    
    # Second interaction - introduce project information
    print("\n--- Second Interaction ---")
    query2 = "I'm currently working on a machine learning project using Python and TensorFlow. The project deadline is December 10, 2025."
    print(f"User: {query2}")
    response2 = agent.run(query2)
    print(f"Agent: {response2}")
    
    # Third interaction - ask about previously mentioned personal information
    print("\n--- Third Interaction (Personal Information Test) ---")
    query3 = "What's my name and where do I work?"
    print(f"User: {query3}")
    response3 = agent.run(query3)
    print(f"Agent: {response3}")
    
    # # Fourth interaction - ask about previously mentioned project information
    # print("\n--- Fourth Interaction (Project Information Test) ---")
    # query4 = "What programming languages am I using for my project and when is the deadline?"
    # print(f"User: {query4}")
    # response4 = agent.run(query4)
    # print(f"Agent: {response4}")
    
    # # Fifth interaction - introduce a lot of information to trigger summarization
    # print("\n--- Fifth Interaction (Adding More Information) ---")
    # query5 = """
    # Let me tell you more about my background. I graduated from Stanford University in 2007 with a degree in Computer Science. 
    # Before joining TechCorp, I worked at Google for 5 years and at Microsoft for 3 years. 
    # I specialize in machine learning and distributed systems. I've published several papers on reinforcement learning.
    # My current project at TechCorp involves building a recommendation system for our e-commerce platform.
    # The system needs to process about 10 million user interactions daily and provide real-time recommendations.
    # We're using a hybrid approach combining collaborative filtering and content-based methods.
    # """
    # print(f"User: {query5}")
    # response5 = agent.run(query5)
    # print(f"Agent: {response5}")
    
    # # Sixth interaction - test if the agent remembers information after summarization
    # print("\n--- Sixth Interaction (Memory After Summarization Test) ---")
    # query6 = "Where did I work before TechCorp and what is my current project about?"
    # print(f"User: {query6}")
    # response6 = agent.run(query6)
    # print(f"Agent: {response6}")
    
    # # Print entity memory contents
    # print("\n--- Entity Memory Contents ---")
    # for name, entity in entity_memory.entities.items():
    #     attr_str = ", ".join([f"{k}: {v}" for k, v in entity.attributes.items()])
    #     print(f"Entity: {name} (Type: {entity.type}), Attributes: {attr_str}")
    
    # # Print conversation memory summary if available
    # if conversation_memory.summary:
    #     print("\n--- Conversation Summary ---")
    #     print(conversation_memory.summary)

if __name__ == "__main__":
    main()
