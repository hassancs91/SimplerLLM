"""
LLM Brainstorm Demo for Video
Simple example showing recursive idea generation
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm


def main():
    print("\n" + "=" * 60)
    print("LLM BRAINSTORM DEMO")
    print("=" * 60)

    # Create LLM
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    # Create brainstormer
    brainstorm = RecursiveBrainstorm(
        llm=llm,
        max_depth=2,              # Go 2 levels deep
        ideas_per_level=3,        # 3 ideas at each level
        mode="tree",              # Expand ALL ideas
        min_quality_threshold=5,  # Only expand ideas scoring 5+
        verbose=True,
    )

    # Run brainstorm
    result = brainstorm.brainstorm(
        prompt="YouTube video ideas for a tech channel focused on AI tools",
        context="Target audience: developers and tech enthusiasts"
    )

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal ideas generated: {result.total_ideas}")
    print(f"Execution time: {result.execution_time:.1f}s")
    print(f"Deepest level reached: {result.max_depth_reached}")

    print("\n--- IDEA TREE ---")
    for level in result.levels:
        print(f"\nDepth {level.depth} ({level.total_ideas} ideas):")
        for idea in sorted(level.ideas, key=lambda x: x.quality_score, reverse=True):
            parent_info = f" <- {idea.parent_id}" if idea.parent_id else " (ROOT)"
            print(f"  [{idea.quality_score:.1f}] {idea.text[:55]}...{parent_info}")

    print("\n--- TOP 5 IDEAS OVERALL ---")
    top_ideas = sorted(result.all_ideas, key=lambda x: x.quality_score, reverse=True)[:5]
    for i, idea in enumerate(top_ideas, 1):
        print(f"\n{i}. [{idea.quality_score:.1f}/10] {idea.text}")
        if idea.reasoning:
            print(f"   Why: {idea.reasoning[:80]}...")

    print("\n--- BEST IDEA ---")
    best = result.overall_best_idea
    print(f"Title: {best.text}")
    print(f"Score: {best.quality_score}/10")
    print(f"Depth: {best.depth}")
    if best.criteria_scores:
        print(f"Criteria: {best.criteria_scores}")


if __name__ == "__main__":
    main()
