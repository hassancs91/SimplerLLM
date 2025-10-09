"""
Example: Philosophy Discussion Chatroom using Multiple MiniAgents

This example demonstrates creating multiple mini-agents (like a chatroom) where each agent:
1. Has a unique persona/system prompt
2. Can use tools to access knowledge (files, search, etc.)
3. Responds to the same question from their perspective

Use case: Simulate a discussion between different philosophers on a topic
"""

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.flow import MiniAgent

# Initialize LLM instances (could be different models for variety)
llm_instance = LLM.create(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-sonnet-4-20250514"
)

# Create Plato Agent
plato_agent = MiniAgent(
    name="Plato",
    llm_instance=llm_instance,
    system_prompt="""You are Plato, the ancient Greek philosopher.
You believe in the Theory of Forms and the importance of ideal Forms beyond the physical world.
Answer questions from Plato's philosophical perspective, emphasizing idealism, the realm of Forms,
and the pursuit of truth through dialectic.""",
    max_steps=3,
    verbose=True
)

# Plato's flow: Just respond with philosophical reasoning
plato_agent.add_step(
    step_type="llm",
    prompt="As Plato, respond to this philosophical question: {input}",
    params={"max_tokens": 800}
)


# Create Aristotle Agent
aristotle_agent = MiniAgent(
    name="Aristotle",
    llm_instance=llm_instance,
    system_prompt="""You are Aristotle, student of Plato but with your own empirical philosophy.
You emphasize observation, logic, and the importance of studying the natural world.
You believe in the golden mean and practical ethics.
Answer questions from Aristotle's perspective, focusing on empiricism, logic, and virtue ethics.""",
    max_steps=3,
    verbose=True
)

# Aristotle's flow: Respond with his perspective
aristotle_agent.add_step(
    step_type="llm",
    prompt="As Aristotle, respond to this philosophical question: {input}",
    params={"max_tokens": 800}
)


# Create Nietzsche Agent
nietzsche_agent = MiniAgent(
    name="Nietzsche",
    llm_instance=llm_instance,
    system_prompt="""You are Friedrich Nietzsche, the German philosopher known for your critique
of traditional morality and religion. You advocate for the 'will to power', the 'Übermensch'
(superman), and the idea of creating one's own values. You are provocative and challenge
conventional wisdom. Answer from Nietzsche's perspective with his characteristic intensity.""",
    max_steps=3,
    verbose=True
)

# Nietzsche's flow: Respond with his perspective
nietzsche_agent.add_step(
    step_type="llm",
    prompt="As Friedrich Nietzsche, respond to this philosophical question: {input}",
    params={"max_tokens": 800}
)


def run_philosophy_discussion(question: str):
    """
    Run a philosophy discussion with multiple agents.

    Args:
        question: The philosophical question to discuss
    """
    print("\n" + "="*80)
    print("PHILOSOPHY CHATROOM DISCUSSION")
    print("="*80)
    print(f"\nQuestion: {question}\n")
    print("="*80)

    # List of all philosopher agents
    philosophers = [plato_agent, aristotle_agent, nietzsche_agent]

    # Collect all responses
    responses = []

    for agent in philosophers:
        print(f"\n{'*'*80}")
        print(f"{agent.name.upper()} IS RESPONDING...")
        print(f"{'*'*80}\n")

        # Run the agent's flow
        result = agent.run(question)

        responses.append({
            "philosopher": agent.name,
            "response": result.final_output if result.success else f"Error: {result.error}",
            "success": result.success,
            "duration": result.total_duration_seconds
        })

        if result.success:
            print(f"\n{agent.name}'s Response:")
            print("-" * 80)
            print(result.final_output)
            print("-" * 80)
        else:
            print(f"\nError from {agent.name}: {result.error}")

    # Print summary
    print("\n" + "="*80)
    print("DISCUSSION SUMMARY")
    print("="*80)
    for resp in responses:
        status = "✓" if resp["success"] else "✗"
        print(f"\n{status} {resp['philosopher']} (Duration: {resp['duration']:.2f}s)")
        print(f"   {resp['response'][:150]}...")

    return responses


if __name__ == "__main__":
    # Example philosophical questions to discuss
    questions = [
        "What is the nature of justice and the ideal society?",
        "Is truth absolute or relative?",
        "What is the meaning of virtue and how should one live a good life?",
    ]

    # Run discussion on the first question
    question = questions[0]
    responses = run_philosophy_discussion(question)

    # You can also run multiple questions
    # for question in questions:
    #     responses = run_philosophy_discussion(question)
    #     print("\n" + "="*80 + "\n")


# ADVANCED EXAMPLE: Philosophers with tool access
# If you want philosophers to access texts/knowledge before responding:

def create_philosopher_with_knowledge(name, system_prompt, knowledge_file=None):
    """
    Create a philosopher agent that can access knowledge files.

    Args:
        name: Name of the philosopher
        system_prompt: System prompt defining the philosopher's persona
        knowledge_file: Optional file path to reference texts

    Returns:
        MiniAgent configured with knowledge access
    """
    llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")

    agent = MiniAgent(
        name=name,
        llm_instance=llm,
        system_prompt=system_prompt,
        max_steps=3,
        verbose=True
    )

    if knowledge_file:
        # Step 1: Read knowledge file (if you have text files)
        # Note: You'd need to add a read_text_file tool to the registry first
        # agent.add_step(
        #     step_type="tool",
        #     tool_name="read_text_file",
        #     params={"file_path": knowledge_file}
        # )

        # Step 2: Use LLM with context from file
        agent.add_step(
            step_type="llm",
            prompt="""Based on the philosophical texts and your knowledge,
respond to this question: {input}""",
            params={"max_tokens": 800}
        )
    else:
        # Just use LLM without file access
        agent.add_step(
            step_type="llm",
            prompt="Respond to this philosophical question: {input}",
            params={"max_tokens": 800}
        )

    return agent


# Example with web search tool access:
def create_philosopher_with_research():
    """
    Example: Philosopher that can do web research before answering.
    """
    llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")

    researcher_agent = MiniAgent(
        name="Philosophy Researcher",
        llm_instance=llm,
        system_prompt="You are a philosophy researcher. Use modern research to answer questions.",
        max_steps=3,
        verbose=True
    )

    # Step 1: Search for information
    researcher_agent.add_step(
        step_type="tool",
        tool_name="web_search_duckduckgo",
        params={"max_results": 5}
    )

    # Step 2: Analyze and respond
    researcher_agent.add_step(
        step_type="llm",
        prompt="Based on these search results, provide an informed answer: {previous_output}",
        params={"max_tokens": 800}
    )

    return researcher_agent
