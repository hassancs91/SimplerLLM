"""
Demo script for SelfConsistency feature.
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_consistency import SelfConsistency, AnswerType


def demo_math_question():
    """Demo with a math question (uses EXACT comparison)."""
    print("=" * 60)
    print("DEMO 1: Math Question (EXACT comparison)")
    print("=" * 60)

    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    consistency = SelfConsistency(
        llm=llm,
        num_samples=5,
        temperature=0.7,
        verbose=True,
    )

    result = consistency.generate(
        prompt="What is 17 x 24? Think step by step.",
        system_prompt="You are a math tutor. Solve the problem step by step. End your response with 'Answer: X' where X is the final number.",
    )

    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Agreement: {result.num_agreeing}/{result.num_samples}")
    print(f"Answer Type: {result.answer_type.value}")
    print(f"Execution Time: {result.execution_time:.2f}s")

    if result.is_tie:
        print(f"TIE between: {result.tied_answers}")

    print("\nAnswer Groups:")
    for group in result.answer_groups:
        print(f"  - '{group.answer}': {group.count} votes ({group.percentage:.0f}%)")

    return result


def demo_reasoning_question():
    """Demo with a reasoning question (uses SEMANTIC comparison)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Reasoning Question (SEMANTIC comparison)")
    print("=" * 60)

    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    consistency = SelfConsistency(
        llm=llm,
        num_samples=5,
        temperature=0.8,
        verbose=True,
    )

    result = consistency.generate(
        prompt="What is the capital of Australia and why was it chosen over Sydney or Melbourne?",
        system_prompt="Give a concise answer in 1-2 sentences.",
    )

    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Answer Type: {result.answer_type.value}")

    if result.is_tie:
        print(f"TIE between: {result.tied_answers}")

    print("\nAnswer Groups:")
    for group in result.answer_groups:
        print(f"  - '{group.answer[:50]}...': {group.count} votes ({group.percentage:.0f}%)")

    return result


def demo_logic_puzzle():
    """Demo with a logic puzzle."""
    print("\n" + "=" * 60)
    print("DEMO 3: Logic Puzzle")
    print("=" * 60)

    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    consistency = SelfConsistency(
        llm=llm,
        num_samples=7,
        temperature=0.9,
        verbose=True,
    )

    result = consistency.generate(
        prompt="""A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?

Think carefully before answering.""",
        system_prompt="You are a logical thinker. Read the problem carefully. State only the final number as your answer.",
    )

    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Correct answer should be: 9")

    print("\nAnswer Groups:")
    for group in result.answer_groups:
        print(f"  - '{group.answer}': {group.count} votes ({group.percentage:.0f}%)")

    return result


if __name__ == "__main__":
    print("\n🎯 Self-Consistency Demo\n")

    # Run demos
    demo_math_question()
    demo_reasoning_question()
    demo_logic_puzzle()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
