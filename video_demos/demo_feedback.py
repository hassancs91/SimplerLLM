"""
LLM Feedback Loop Demo for Video
Shows iterative improvement through critique cycles
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_feedback import LLMFeedbackLoop


def main():
    print("\n" + "=" * 60)
    print("LLM FEEDBACK LOOP DEMO")
    print("=" * 60)

    # Create LLM
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    # Create feedback loop
    feedback = LLMFeedbackLoop(
        llm=llm,
        max_iterations=3,
        quality_threshold=8.5,  # Stop early if we hit this score
        verbose=True,
    )

    # Run improvement loop
    result = feedback.improve(
        prompt="Explain what an API is to a beginner with no programming experience",
        focus_on=["clarity", "beginner-friendly", "real-world examples"]
    )

    # Display results
    print("\n" + "=" * 60)
    print("IMPROVEMENT TRAJECTORY")
    print("=" * 60)

    for iteration in result.all_iterations:
        critique = iteration.critique
        print(f"\n--- Iteration {iteration.iteration_number} ---")
        print(f"Score: {critique.quality_score}/10")
        print(f"\nStrengths:")
        for s in critique.strengths[:2]:
            print(f"  + {s}")
        print(f"\nWeaknesses:")
        for w in critique.weaknesses[:2]:
            print(f"  - {w}")
        print(f"\nSuggestions:")
        for sug in critique.improvement_suggestions[:2]:
            print(f"  > {sug}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nInitial Score: {result.initial_score}/10")
    print(f"Final Score: {result.final_score}/10")
    print(f"Improvement: +{result.final_score - result.initial_score:.1f} points")
    print(f"Iterations: {result.total_iterations}")
    print(f"Stopped because: {result.stopped_reason}")
    print(f"Time: {result.total_execution_time:.1f}s")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(f"\n{result.final_answer}")


if __name__ == "__main__":
    main()
