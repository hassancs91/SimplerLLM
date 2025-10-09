"""
Example: Async MiniAgent Flows for Concurrent Execution

This example demonstrates the async capabilities of MiniAgent, showing how to:
1. Run a single agent asynchronously
2. Run multiple agents concurrently for dramatic speed improvements
3. Compare sync vs async execution times

Use cases:
- Running multiple agents in parallel (philosophy chatroom example)
- Non-blocking I/O operations
- Better resource utilization for API-heavy workflows
"""

import asyncio
import time
from pydantic import BaseModel, Field
from typing import List
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.flow import MiniAgent


# Define Pydantic model for structured output
class PhilosophicalResponse(BaseModel):
    """Structured philosophical response."""
    philosopher_name: str = Field(description="Name of the philosopher")
    main_argument: str = Field(description="Main philosophical argument")
    key_principles: List[str] = Field(description="2-3 key philosophical principles")
    conclusion: str = Field(description="Final conclusion or takeaway")


# Example 1: Simple Async Execution
async def example_simple_async():
    """Run a single agent asynchronously."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Async Execution")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Simple Async Agent",
        llm_instance=llm,
        system_prompt="You are a helpful assistant.",
        max_steps=2,
        verbose=True
    )

    agent.add_step(
        step_type="llm",
        prompt="What are the 3 main benefits of async programming in Python?",
        params={"max_tokens": 500}
    )

    # Run asynchronously
    result = await agent.run_async("Explain async programming")

    if result.success:
        print("\n" + "-"*80)
        print("RESULT:")
        print("-"*80)
        print(result.final_output)
    else:
        print(f"\nError: {result.error}")


# Example 2: Concurrent Execution - Philosophy Chatroom
async def example_concurrent_philosophers():
    """
    Run multiple philosopher agents concurrently.
    This is MUCH faster than running them sequentially!
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Concurrent Philosophy Discussion")
    print("="*80 + "\n")

    # Create LLM instance
    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    # Define the philosophical question
    question = "What is the relationship between knowledge and virtue?"

    # Create three philosopher agents
    plato_agent = MiniAgent(
        name="Plato",
        llm_instance=llm,
        system_prompt="You are Plato. Answer from the perspective of Platonic philosophy.",
        max_steps=2,
        verbose=False  # Set to False for cleaner output
    )
    plato_agent.add_step(
        step_type="llm",
        prompt="As Plato, respond to: {input}",
        params={"max_tokens": 600}
    )

    aristotle_agent = MiniAgent(
        name="Aristotle",
        llm_instance=llm,
        system_prompt="You are Aristotle. Answer from the perspective of Aristotelian philosophy.",
        max_steps=2,
        verbose=False
    )
    aristotle_agent.add_step(
        step_type="llm",
        prompt="As Aristotle, respond to: {input}",
        params={"max_tokens": 600}
    )

    kant_agent = MiniAgent(
        name="Kant",
        llm_instance=llm,
        system_prompt="You are Immanuel Kant. Answer from the perspective of Kantian philosophy.",
        max_steps=2,
        verbose=False
    )
    kant_agent.add_step(
        step_type="llm",
        prompt="As Kant, respond to: {input}",
        params={"max_tokens": 600}
    )

    print(f"Question: {question}\n")
    print("-"*80)

    # Run all agents CONCURRENTLY using asyncio.gather
    start_time = time.time()

    results = await asyncio.gather(
        plato_agent.run_async(question),
        aristotle_agent.run_async(question),
        kant_agent.run_async(question)
    )

    concurrent_duration = time.time() - start_time

    # Display results
    print("\n" + "="*80)
    print("CONCURRENT EXECUTION RESULTS")
    print("="*80)

    for result in results:
        if result.success:
            print(f"\n{result.agent_name}'s Response:")
            print("-"*80)
            print(result.final_output)
            print(f"\nExecution time: {result.total_duration_seconds:.2f}s")
        else:
            print(f"\nError from {result.agent_name}: {result.error}")

    print("\n" + "="*80)
    print(f"TOTAL CONCURRENT TIME: {concurrent_duration:.2f}s")
    print("="*80)

    return results, concurrent_duration


# Example 3: Sync vs Async Comparison
async def example_sync_vs_async_comparison():
    """
    Compare sync vs async execution to see the performance difference.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Sync vs Async Performance Comparison")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    questions = [
        "What is consciousness?",
        "What is free will?",
        "What is the nature of reality?"
    ]

    # Create agents
    agents = []
    for i, question in enumerate(questions):
        agent = MiniAgent(
            name=f"Agent {i+1}",
            llm_instance=llm,
            system_prompt="You are a philosophy expert.",
            max_steps=2,
            verbose=False
        )
        agent.add_step(
            step_type="llm",
            prompt="Provide a concise answer to: {input}",
            params={"max_tokens": 400}
        )
        agents.append(agent)

    # SYNCHRONOUS EXECUTION
    print("Running SYNCHRONOUSLY (one after another)...")
    sync_start = time.time()

    sync_results = []
    for i, (agent, question) in enumerate(zip(agents, questions)):
        print(f"  [{i+1}/3] Processing: {question[:50]}...")
        result = agent.run(question)
        sync_results.append(result)

    sync_duration = time.time() - sync_start

    print(f"\n✓ Sync execution completed in: {sync_duration:.2f}s")

    # ASYNCHRONOUS EXECUTION
    print("\nRunning ASYNCHRONOUSLY (all at once)...")
    async_start = time.time()

    async_results = await asyncio.gather(
        *[agent.run_async(question) for agent, question in zip(agents, questions)]
    )

    async_duration = time.time() - async_start

    print(f"✓ Async execution completed in: {async_duration:.2f}s")

    # Comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Synchronous:  {sync_duration:.2f}s")
    print(f"Asynchronous: {async_duration:.2f}s")
    speedup = sync_duration / async_duration if async_duration > 0 else 0
    print(f"Speedup:      {speedup:.2f}x faster")
    print(f"Time saved:   {sync_duration - async_duration:.2f}s")
    print("="*80)


# Example 4: Async with JSON Output
async def example_async_with_json_output():
    """
    Demonstrate async execution with structured JSON output.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Async with Structured JSON Output")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    philosophers = ["Socrates", "Descartes", "Nietzsche"]
    question = "What is the meaning of life?"

    # Create agents with JSON output
    agents = []
    for philosopher in philosophers:
        agent = MiniAgent(
            name=f"{philosopher} Agent",
            llm_instance=llm,
            system_prompt=f"You are {philosopher}. Respond from his philosophical perspective.",
            max_steps=2,
            verbose=False
        )
        agent.add_step(
            step_type="llm",
            prompt=f"As {philosopher}, answer: {{input}}",
            output_model=PhilosophicalResponse,
            max_retries=3,
            params={"max_tokens": 800}
        )
        agents.append(agent)

    print(f"Question: {question}\n")
    print("Running all agents concurrently with JSON output...")

    # Run concurrently
    start_time = time.time()
    results = await asyncio.gather(
        *[agent.run_async(question) for agent in agents]
    )
    duration = time.time() - start_time

    # Display structured results
    print("\n" + "="*80)
    print("STRUCTURED JSON RESPONSES")
    print("="*80)

    for result in results:
        if result.success:
            response = result.final_output
            print(f"\n{response.philosopher_name}:")
            print(f"  Main Argument: {response.main_argument}")
            print(f"  Key Principles:")
            for principle in response.key_principles:
                print(f"    - {principle}")
            print(f"  Conclusion: {response.conclusion}")
        else:
            print(f"\nError from {result.agent_name}: {result.error}")

    print(f"\n{'='*80}")
    print(f"All responses generated in: {duration:.2f}s")
    print("="*80)


# Example 5: Mixed Sync/Async Operations
async def example_mixed_operations():
    """
    Demonstrate mixing async agents with synchronous tools.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Mixed Async Operations with Tools")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Mixed Agent",
        llm_instance=llm,
        system_prompt="You are a helpful assistant.",
        max_steps=3,
        verbose=True
    )

    # Add steps including tool usage
    agent.add_step(
        step_type="llm",
        prompt="Generate a simple Python function to add two numbers",
        params={"max_tokens": 300}
    )

    agent.add_step(
        step_type="tool",
        tool_name="execute_python_code"  # This tool is sync but runs in executor
    )

    agent.add_step(
        step_type="llm",
        prompt="Summarize the execution result: {previous_output}",
        params={"max_tokens": 200}
    )

    # Run async - tools will automatically run in executor
    result = await agent.run_async("Create and test an add function")

    if result.success:
        print("\n" + "-"*80)
        print("FINAL RESULT:")
        print("-"*80)
        print(result.final_output)
    else:
        print(f"\nError: {result.error}")


# Main execution
async def main():
    """Run all async examples."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  ASYNC MINIAGENT EXAMPLES".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    # Example 1: Simple async
    # await example_simple_async()

    # Example 2: Concurrent philosophers (most impressive!)
    await example_concurrent_philosophers()

    # Example 3: Sync vs async comparison
    # await example_sync_vs_async_comparison()

    # Example 4: Async with JSON output
    # await example_async_with_json_output()

    # Example 5: Mixed operations
    # await example_mixed_operations()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Benefits of Async MiniAgent:")
    print("  ✓ Much faster when running multiple agents concurrently")
    print("  ✓ Better resource utilization")
    print("  ✓ Non-blocking I/O operations")
    print("  ✓ Perfect for chatrooms, multi-agent systems")
    print("  ✓ Works with both text and JSON output modes")
    print("  ✓ Seamless integration with asyncio ecosystem")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
