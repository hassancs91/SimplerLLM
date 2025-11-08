# Recursive Brainstorm

A powerful recursive brainstorming system for SimplerLLM that generates and expands ideas using LLMs with three different expansion strategies.

## Features

- **Three Generation Modes**:
  - **Tree Mode**: Exponential expansion of all qualifying ideas
  - **Linear Mode**: Focused refinement of the best ideas at each level
  - **Hybrid Mode**: Selective expansion of top-N ideas (best of both worlds)

- **Flexible Configuration**:
  - Configurable recursion depth
  - Custom evaluation criteria
  - Quality threshold filtering
  - Temperature and scoring control

- **Rich Output**:
  - Structured Pydantic models for all results
  - Hierarchical tree representation
  - Detailed tracking of all iterations
  - Quality scores and criteria evaluations

- **Multiple Integration Options**:
  - Standalone class for direct use
  - Tool wrapper for MiniAgent flows
  - Async support for better performance

## Installation

The feature is built into SimplerLLM. Just ensure you have the latest version:

```bash
pip install simplerllm
```

## Quick Start

### Standalone Usage

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm

# Create LLM instance
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o-mini"
)

# Create brainstorm instance
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=3,
    ideas_per_level=5,
    mode="hybrid",  # or "tree" or "linear"
    top_n=3,  # for hybrid mode
)

# Run brainstorming
result = brainstorm.brainstorm(
    prompt="Innovative ways to reduce carbon emissions"
)

# Access results
print(f"Generated {result.total_ideas} ideas")
print(f"Best idea: {result.overall_best_idea.text}")
print(f"Score: {result.overall_best_idea.quality_score}/10")
```

### MiniAgent Integration

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.flow import MiniAgent

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini")
agent = MiniAgent("Brainstorm Agent", llm, max_steps=2)

# Add brainstorm step
agent.add_step(
    step_type="tool",
    tool_name="recursive_brainstorm",
    params={
        "llm_instance": llm,
        "max_depth": 2,
        "mode": "hybrid",
        "top_n": 3,
    }
)

# Add analysis step
agent.add_step(
    step_type="llm",
    prompt="Analyze the brainstorm results and pick the top 3 ideas: {previous_output}"
)

result = agent.run("Ways to improve remote team collaboration")
```

## Generation Modes

### Tree Mode (Exponential Expansion)

Generates ideas at each level and expands **all** qualifying ideas recursively.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=5,
    mode="tree",
)
```

**When to use**: When you want comprehensive exploration and maximum idea diversity.

**Example output**:
```
Depth 0: 5 ideas
Depth 1: 25 ideas (5 expanded, each generating 5)
Depth 2: 125 ideas (25 expanded, each generating 5)
Total: 155 ideas
```

### Linear Mode (Focused Refinement)

Generates ideas, picks the **best one**, refines it, and repeats.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=3,
    ideas_per_level=5,
    mode="linear",
)
```

**When to use**: When you want to deeply refine a single idea path.

**Example output**:
```
Depth 0: 5 ideas → pick best
Depth 1: 5 refinements of best → pick best
Depth 2: 5 refinements of best → pick best
Total: 15 ideas (single focused path)
```

### Hybrid Mode (Selective Expansion)

Generates ideas and expands only the **top-N** highest-scoring ideas.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=5,
    mode="hybrid",
    top_n=3,  # Expand top 3 ideas
)
```

**When to use**: Balance between breadth and depth (RECOMMENDED for most use cases).

**Example output**:
```
Depth 0: 5 ideas → pick top 3
Depth 1: 15 ideas (3 expanded, each generating 5) → pick top 3
Depth 2: 15 ideas (3 expanded, each generating 5)
Total: 35 ideas
```

## Configuration Options

```python
RecursiveBrainstorm(
    llm=llm,                          # LLM instance (required)
    max_depth=3,                      # Maximum recursion depth
    ideas_per_level=5,                # Ideas to generate per expansion
    mode="hybrid",                    # "tree", "linear", or "hybrid"
    top_n=3,                          # For hybrid: number of top ideas to expand
    evaluation_criteria=[             # Custom criteria for scoring
        "feasibility",
        "innovation",
        "impact",
        "scalability"
    ],
    min_quality_threshold=5.0,        # Minimum score (1-10) to expand idea
    verbose=True,                     # Print progress information
)
```

## Working with Results

### Accessing Ideas

```python
result = brainstorm.brainstorm("Your prompt")

# Get all ideas
all_ideas = result.all_ideas

# Get ideas at specific depth
depth_0_ideas = result.get_ideas_at_depth(0)
depth_1_ideas = result.get_ideas_at_depth(1)

# Get children of an idea
children = result.get_children(idea.id)

# Get best ideas per level
best_per_level = result.best_ideas_per_level

# Get overall best
best_idea = result.overall_best_idea
```

### Analyzing the Tree

```python
# Get path from root to best idea
path = result.get_path_to_best()
for idea in path:
    print(f"Depth {idea.depth}: {idea.text} (score: {idea.quality_score})")

# Export as hierarchical dict
tree_dict = result.to_tree_dict()
import json
print(json.dumps(tree_dict, indent=2))
```

### Iteration Details

```python
# Access all iterations
for iteration in result.all_iterations:
    print(f"Iteration {iteration.iteration_number}")
    print(f"  Depth: {iteration.depth}")
    print(f"  Generated: {len(iteration.generated_ideas)} ideas")
    print(f"  Time: {iteration.execution_time:.2f}s")
```

## Custom Evaluation Criteria

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    evaluation_criteria=[
        "technical_feasibility",
        "market_potential",
        "time_to_market",
        "competitive_advantage",
        "user_value"
    ],
)

result = brainstorm.brainstorm("SaaS product ideas for remote teams")

# Access criteria scores
for idea in result.all_ideas:
    print(f"{idea.text}")
    for criterion, score in idea.criteria_scores.items():
        print(f"  {criterion}: {score}/10")
```

## Async Support

```python
import asyncio

async def brainstorm_async():
    result = await brainstorm.brainstorm_async(
        prompt="Your prompt here"
    )
    return result

result = asyncio.run(brainstorm_async())
```

## Examples

See [examples/llm_brainstorm_example.py](../../../examples/llm_brainstorm_example.py) for comprehensive examples including:

1. Basic tree mode usage
2. Linear refinement
3. Hybrid mode
4. Custom evaluation criteria
5. MiniAgent integration
6. Async brainstorming
7. Result navigation and analysis

Run examples:
```bash
# Run all examples
python examples/llm_brainstorm_example.py

# Run specific example
python examples/llm_brainstorm_example.py 1
```

## Best Practices

### Mode Selection

- **Tree mode**: Use when you need comprehensive exploration (e.g., initial ideation, research)
- **Linear mode**: Use when refining a specific concept (e.g., product development, optimization)
- **Hybrid mode**: Use for balanced exploration (RECOMMENDED for most cases)

### Depth and Ideas Configuration

- Start with **max_depth=2-3** to avoid exponential LLM costs
- Use **ideas_per_level=3-5** for balance between diversity and cost
- For hybrid mode, set **top_n=2-3** (about 50-60% of ideas_per_level)

### Quality Filtering

- Set **min_quality_threshold=6.0-7.0** to focus on high-quality ideas
- Lower threshold (4.0-5.0) for exploratory brainstorming
- Higher threshold (8.0+) for selective refinement

### Evaluation Criteria

- Use **3-5 criteria** for balanced evaluation
- Make criteria specific to your domain
- Examples:
  - Product ideas: feasibility, market_fit, innovation, scalability
  - Research: novelty, impact, feasibility, clarity
  - Marketing: creativity, target_audience_fit, actionability, budget

## Architecture

### File Structure

```
SimplerLLM/language/llm_brainstorm/
├── __init__.py              # Module exports
├── recursive_brainstorm.py  # Main RecursiveBrainstorm class
├── models.py                # Pydantic models for structured data
└── README.md                # This file

SimplerLLM/tools/
└── brainstorm.py           # Tool wrappers for MiniAgent

examples/
└── llm_brainstorm_example.py  # Usage examples

tests/
└── test_llm_brainstorm.py     # Unit tests
```

### Data Models

- **BrainstormIdea**: Single idea with metadata
- **IdeaGeneration**: LLM output for generating ideas
- **IdeaEvaluation**: LLM output for evaluating ideas
- **BrainstormLevel**: All ideas at a specific depth
- **BrainstormIteration**: Single iteration metadata
- **BrainstormResult**: Complete brainstorm session results

## Testing

Run tests:
```bash
pytest tests/test_llm_brainstorm.py -v
```

## Contributing

To add similar iterative/recursive features:

1. Create module under `SimplerLLM/language/your_feature/`
2. Define Pydantic models in `models.py`
3. Implement main class with sync and async methods
4. Create tool wrappers in `SimplerLLM/tools/`
5. Register tools in `ToolRegistry`
6. Add examples and tests

## License

Part of SimplerLLM. See main repository license.
