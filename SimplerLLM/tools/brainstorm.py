"""
Brainstorm tools for MiniAgent flows.

Provides wrapper functions to use RecursiveBrainstorm in MiniAgent workflows.
"""

from typing import Dict, Any, Optional, List
from SimplerLLM.language.llm import LLM
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm, BrainstormResult


def recursive_brainstorm_tool(
    prompt: str,
    llm_instance: LLM,
    max_depth: int = 2,
    ideas_per_level: int = 5,
    mode: str = "hybrid",
    top_n: int = 3,
    evaluation_criteria: Optional[List[str]] = None,
    min_quality_threshold: float = 5.0,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Tool wrapper for recursive brainstorming in MiniAgent flows.

    This function wraps RecursiveBrainstorm for use as a tool in MiniAgent
    workflows. It returns a simplified dictionary representation suitable
    for flow processing.

    Args:
        prompt: The brainstorming prompt (can be from flow input)
        llm_instance: LLM instance to use (must be passed from flow)
        max_depth: Maximum recursion depth (default: 2 for tools)
        ideas_per_level: Number of ideas per expansion (default: 5)
        mode: Generation mode - "tree", "linear", or "hybrid" (default: "hybrid")
        top_n: For hybrid mode, number of top ideas to expand (default: 3)
        evaluation_criteria: List of criteria to evaluate against
        min_quality_threshold: Minimum quality score to continue (1-10)
        verbose: Print progress information
        **kwargs: Additional parameters (e.g., context)

    Returns:
        Dictionary with brainstorming results:
        - best_idea: The highest-scoring idea
        - top_ideas: List of top 5 ideas with scores
        - total_ideas: Total number of ideas generated
        - all_ideas: Full list of all ideas
        - execution_time: Time taken in seconds
        - tree_structure: Hierarchical representation

    Example in MiniAgent:
        >>> from SimplerLLM.language import LLM, LLMProvider
        >>> from SimplerLLM.language.flow import MiniAgent
        >>>
        >>> llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
        >>> agent = MiniAgent("Brainstorm Agent", llm, max_steps=2)
        >>>
        >>> agent.add_step(
        ...     step_type="tool",
        ...     tool_name="recursive_brainstorm",
        ...     params={
        ...         "llm_instance": llm,
        ...         "max_depth": 2,
        ...         "mode": "hybrid"
        ...     }
        ... )
        >>>
        >>> result = agent.run("Ways to improve remote team collaboration")
    """
    # Create brainstormer instance
    brainstormer = RecursiveBrainstorm(
        llm=llm_instance,
        max_depth=max_depth,
        ideas_per_level=ideas_per_level,
        mode=mode,
        top_n=top_n,
        evaluation_criteria=evaluation_criteria,
        min_quality_threshold=min_quality_threshold,
        verbose=verbose,
    )

    # Execute brainstorming
    context = kwargs.get("context", None)
    result: BrainstormResult = brainstormer.brainstorm(
        prompt=prompt,
        mode=mode,
        context=context,
    )

    # Convert to simplified dict for flow processing
    all_ideas_sorted = sorted(result.all_ideas, key=lambda x: x.quality_score, reverse=True)
    top_5_ideas = all_ideas_sorted[:5]

    return {
        "best_idea": {
            "text": result.overall_best_idea.text if result.overall_best_idea else "",
            "score": result.overall_best_idea.quality_score if result.overall_best_idea else 0.0,
            "reasoning": result.overall_best_idea.reasoning if result.overall_best_idea else "",
            "depth": result.overall_best_idea.depth if result.overall_best_idea else 0,
        },
        "top_ideas": [
            {
                "text": idea.text,
                "score": idea.quality_score,
                "reasoning": idea.reasoning,
                "depth": idea.depth,
            }
            for idea in top_5_ideas
        ],
        "total_ideas": result.total_ideas,
        "total_iterations": result.total_iterations,
        "max_depth_reached": result.max_depth_reached,
        "execution_time": result.execution_time,
        "mode_used": result.mode,
        "all_ideas": [
            {
                "id": idea.id,
                "text": idea.text,
                "score": idea.quality_score,
                "depth": idea.depth,
                "parent_id": idea.parent_id,
            }
            for idea in result.all_ideas
        ],
        "tree_structure": result.to_tree_dict(),
    }


def simple_brainstorm(
    prompt: str,
    llm_instance: LLM,
    num_ideas: int = 5,
    verbose: bool = False,
) -> List[str]:
    """
    Simplified brainstorming tool that returns just a list of ideas.

    This is a simpler version of recursive_brainstorm_tool that doesn't
    do recursive expansion, just generates a flat list of ideas.

    Args:
        prompt: The brainstorming prompt
        llm_instance: LLM instance to use
        num_ideas: Number of ideas to generate (default: 5)
        verbose: Print progress information

    Returns:
        List of idea strings

    Example:
        >>> ideas = simple_brainstorm(
        ...     prompt="Ways to reduce carbon emissions",
        ...     llm_instance=llm,
        ...     num_ideas=5
        ... )
        >>> for idea in ideas:
        ...     print(f"- {idea}")
    """
    # Use RecursiveBrainstorm with max_depth=1 (no recursion)
    brainstormer = RecursiveBrainstorm(
        llm=llm_instance,
        max_depth=1,
        ideas_per_level=num_ideas,
        mode="linear",
        verbose=verbose,
    )

    result = brainstormer.brainstorm(prompt=prompt)

    # Return just the idea texts
    return [idea.text for idea in result.all_ideas]


def create_brainstorm_tool(llm_instance: LLM, **default_params):
    """
    Factory function to create a brainstorm tool with pre-configured LLM.

    This is useful if you want to create a tool with a specific LLM instance
    and default parameters baked in.

    Args:
        llm_instance: The LLM instance to use
        **default_params: Default parameters to pass to RecursiveBrainstorm

    Returns:
        A function that can be called with just a prompt

    Example:
        >>> llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
        >>> my_brainstorm = create_brainstorm_tool(
        ...     llm,
        ...     max_depth=3,
        ...     mode="tree",
        ...     verbose=True
        ... )
        >>> result = my_brainstorm("Ways to improve education")
    """
    def brainstorm_tool(prompt: str, context: Optional[str] = None, **override_params):
        # Merge default params with overrides
        params = {**default_params, **override_params}

        return recursive_brainstorm_tool(
            prompt=prompt,
            llm_instance=llm_instance,
            context=context,
            **params
        )

    return brainstorm_tool
