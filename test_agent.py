

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.agents.agent import Agent
from SimplerLLM.tools.serp import search_with_serper_api

# Initialize the LLM
llm_instance = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        verbose=False)



    # Initialize the EnhancedAgent
agent = Agent(llm=llm_instance, verbose=True)


agent.add_tool(
        name="search",
        func=search_with_serper_api,
        description="A tool that searches the web for information.",
        parameters={"query": "The search query"}
    )


    # Run the agent with different inputs
print("Running agent with direct answer query...")
response1 = agent.run("What is the capital of France?")
print(f"Response 1: {response1}")

print("\nRunning agent with tool usage query...")
response2 = agent.run("Search for the population of New York City.")
print("Response 2: {}".format(response2))

print("\nRunning agent with memory retrieval query...")
response3 = agent.run("What did I ask you before?")
print("Response 3: {}".format(response3))

print("\nRunning agent with entity extraction query...")
response4 = agent.run("I spoke with John Smith, the CEO of Acme Corp, yesterday.")
print("Response 4: {}".format(response4))

print("\nRunning agent with follow up entity query...")
response5 = agent.run("What is John Smith's role?")
print("Response 5: {}".format(response5))


