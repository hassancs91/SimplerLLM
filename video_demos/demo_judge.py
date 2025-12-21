"""
LLM Judge Demo for Video
Multiple models compete, judge picks/combines best
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_judge import LLMJudge


def main():
    print("\n" + "=" * 60)
    print("LLM JUDGE DEMO")
    print("Multiple models compete, judge picks the best")
    print("=" * 60)

    # Create providers (contestants)
    print("\nSetting up contestants: GPT-4o vs Claude...")
    providers = [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
    ]

    # Create judge
    print("Setting up judge: GPT-4o")
    judge_llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    judge = LLMJudge(
        providers=providers,
        judge_llm=judge_llm,
        parallel=True,
        default_criteria=["clarity", "accuracy", "beginner-friendly"],
        verbose=True,
    )

    prompt = "Explain what an API is using a real-world analogy that anyone can understand"

    print(f"\nPrompt: \"{prompt}\"")

    # =========================================================
    # Mode 1: Select Best
    # =========================================================
    print("\n" + "=" * 60)
    print("MODE 1: SELECT BEST")
    print("Pick the single best answer")
    print("=" * 60)

    result = judge.generate(prompt=prompt, mode="select_best")

    print(f"\n📊 SCORES:")
    winner = None
    for eval in result.evaluations:
        print(f"   {eval.provider_name}: {eval.overall_score}/10 (Rank #{eval.rank})")
        if eval.strengths:
            print(f"      Strengths: {', '.join(eval.strengths[:2])}")
        if eval.weaknesses:
            print(f"      Weaknesses: {', '.join(eval.weaknesses[:2])}")
        if eval.rank == 1:
            winner = eval.provider_name

    print(f"\n🏆 WINNER: {winner or 'See evaluations'}")
    print(f"\n📝 BEST ANSWER:")
    print("-" * 40)
    # Truncate for display
    answer = result.final_answer
    if len(answer) > 400:
        answer = answer[:400] + "..."
    print(answer)

    # =========================================================
    # Mode 2: Synthesize
    # =========================================================
    print("\n" + "=" * 60)
    print("MODE 2: SYNTHESIZE")
    print("Combine best parts of all answers")
    print("=" * 60)

    result = judge.generate(prompt=prompt, mode="synthesize")

    print(f"\n✨ SYNTHESIZED ANSWER (best of all providers):")
    print("-" * 40)
    answer = result.final_answer
    if len(answer) > 500:
        answer = answer[:500] + "..."
    print(answer)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nProviders used: {len(providers)}")
    print(f"Total execution time: {result.total_execution_time:.2f}s")
    print("\nKey insight: Instead of manually asking each AI and comparing,")
    print("LLM Judge does it all in parallel and gives you the best result!")


if __name__ == "__main__":
    main()
