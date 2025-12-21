"""
Self-Consistency Demo for Video
Majority voting for more reliable answers
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_consistency import SelfConsistency


def main():
    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY DEMO")
    print("Ask multiple times, trust the consensus")
    print("=" * 60)

    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    consistency = SelfConsistency(
        llm=llm,
        num_samples=5,
        temperature=0.7,
        verbose=True,
    )

    # =========================================================
    # Test 1: Math Problem
    # =========================================================
    print("\n" + "-" * 60)
    print("TEST 1: MATH PROBLEM")
    print("-" * 60)
    print("Question: What is 17 x 24?")

    result = consistency.generate(
        prompt="What is 17 × 24? Show your work briefly, then state the answer.",
    )

    print(f"\n{'='*40}")
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Agreement: {result.num_agreeing}/{result.num_samples}")

    print("\nAnswer Distribution:")
    for group in result.answer_groups:
        bar = "█" * group.count
        print(f"  '{group.answer}': {bar} ({group.count} votes, {group.percentage:.0f}%)")

    # =========================================================
    # Test 2: Logic Puzzle (Trick Question)
    # =========================================================
    print("\n" + "-" * 60)
    print("TEST 2: LOGIC PUZZLE (Trick Question)")
    print("-" * 60)
    print("Question: A farmer has 17 sheep. All but 9 run away. How many are left?")

    result = consistency.generate(
        prompt="A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Think carefully.",
    )

    print(f"\n{'='*40}")
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Correct answer: 9 (it's a trick question!)")

    print("\nAnswer Distribution:")
    for group in result.answer_groups:
        bar = "█" * group.count
        print(f"  '{group.answer}': {bar} ({group.count} votes, {group.percentage:.0f}%)")

    if result.is_tie:
        print(f"\n⚠️  TIE detected between: {result.tied_answers}")

    # =========================================================
    # Test 3: Factual Question
    # =========================================================
    print("\n" + "-" * 60)
    print("TEST 3: FACTUAL QUESTION")
    print("-" * 60)
    print("Question: What year was the Eiffel Tower completed?")

    result = consistency.generate(
        prompt="What year was the Eiffel Tower completed? Just give the year.",
    )

    print(f"\n{'='*40}")
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Agreement: {result.num_agreeing}/{result.num_samples}")

    print("\nAnswer Distribution:")
    for group in result.answer_groups:
        bar = "█" * group.count
        print(f"  '{group.answer}': {bar} ({group.count} votes, {group.percentage:.0f}%)")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSamples per question: {consistency.num_samples}")
    print(f"Temperature: {consistency.temperature}")
    print(f"Total execution time: {result.execution_time:.2f}s")
    print("\nKey insight: Instead of trusting a single response,")
    print("Self-Consistency gives you the consensus across multiple attempts!")


if __name__ == "__main__":
    main()
