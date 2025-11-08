---
sidebar_position: 1
---

# Recursive Brainstorm

A powerful recursive brainstorming system that generates and expands ideas using LLMs with three different expansion strategies. Perfect for generating hundreds of creative ideas, domain names, product concepts, and more!

## Why Use Recursive Brainstorm?

Traditional brainstorming with LLMs gives you a flat list of ideas. Recursive Brainstorm takes it further by:

- **Generating ideas recursively** - Each good idea spawns more refined variations
- **Three flexible modes** - Choose between breadth (tree), depth (linear), or balanced (hybrid) exploration
- **Automatic evaluation** - Every idea gets scored against custom criteria
- **Structured output** - All ideas organized with quality scores, tree relationships, and metadata
- **CSV export** - Easily export all ideas for analysis in Excel, pandas, or any data tool
- **MiniAgent integration** - Use brainstorming as a step in complex workflows

## Quick Start

Generate 100+ creative ideas in seconds:

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
    max_depth=2,           # How many levels deep
    ideas_per_level=5,     # Ideas to generate per expansion
    mode="hybrid"          # Balanced exploration
)

# Run brainstorming
result = brainstorm.brainstorm(
    prompt="Innovative ways to reduce plastic waste"
)

# Access results
print(f"Generated {result.total_ideas} ideas!")
print(f"Best idea: {result.overall_best_idea.text}")
print(f"Score: {result.overall_best_idea.quality_score}/10")
```

## Three Generation Modes

Choose the mode that fits your needs:

### Tree Mode - Maximum Diversity

Expands **all** qualifying ideas recursively for comprehensive exploration.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=5,
    mode="tree"  # Exponential expansion
)

result = brainstorm.brainstorm("Product ideas for remote teams")
# Generates: 5 + 25 + 125 = 155 ideas
```

**Best for**: Initial ideation, research, comprehensive exploration

### Linear Mode - Deep Refinement

Picks the **best** idea at each level and refines it further.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=3,
    ideas_per_level=5,
    mode="linear"  # Single focused path
)

result = brainstorm.brainstorm("Design a habit-tracking app")
# Generates: 5 + 5 + 5 = 15 ideas (best path only)
```

**Best for**: Refining a specific concept, deep exploration, optimization

### Hybrid Mode - Balanced Approach (Recommended)

Expands only the **top-N** highest-scoring ideas at each level.

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=5,
    mode="hybrid",
    top_n=3  # Expand top 3 ideas only
)

result = brainstorm.brainstorm("AI startup ideas")
# Generates: 5 + 15 + 45 = 65 ideas (selective expansion)
```

**Best for**: Most use cases - balances breadth and depth

## Configuration Options

Customize brainstorming to your needs:

```python
RecursiveBrainstorm(
    llm=llm,                    # LLM instance (required)
    max_depth=3,                # Maximum recursion depth
    ideas_per_level=5,          # Ideas per expansion
    mode="hybrid",              # "tree", "linear", or "hybrid"
    top_n=3,                    # For hybrid: top N to expand
    evaluation_criteria=[       # Custom scoring criteria
        "feasibility",
        "innovation",
        "impact",
        "scalability"
    ],
    min_quality_threshold=6.0,  # Min score (1-10) to expand
    verbose=True                # Print progress
)
```

### Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 3 | Maximum recursion levels (0-indexed) |
| `ideas_per_level` | 5 | Ideas to generate per expansion |
| `mode` | "tree" | Generation strategy |
| `top_n` | 3 | For hybrid: number of top ideas to expand |
| `evaluation_criteria` | ["quality", "feasibility", "impact"] | Criteria for scoring |
| `min_quality_threshold` | 5.0 | Minimum score to continue expanding (1-10) |
| `verbose` | False | Print progress information |

## Working with Results

Access and analyze generated ideas:

```python
result = brainstorm.brainstorm("Generate marketing strategies")

# Get all ideas
all_ideas = result.all_ideas  # List of BrainstormIdea objects
print(f"Total ideas: {result.total_ideas}")

# Get best idea
best = result.overall_best_idea
print(f"Best: {best.text}")
print(f"Score: {best.quality_score}/10")
print(f"Reasoning: {best.reasoning}")

# Get ideas at specific depth
root_ideas = result.get_ideas_at_depth(0)  # Initial ideas
level_1_ideas = result.get_ideas_at_depth(1)  # First expansion

# Get children of an idea
children = result.get_children(idea.id)

# Get path to best idea (from root to best)
path = result.get_path_to_best()
for i, idea in enumerate(path):
    print(f"Level {i}: {idea.text} (score: {idea.quality_score})")
```

### Idea Object Properties

Each `BrainstormIdea` contains:

```python
idea.id                 # Unique identifier
idea.text              # The idea content
idea.quality_score     # Overall score (1-10)
idea.depth             # Level in tree
idea.parent_id         # Parent idea ID (None for root)
idea.reasoning         # Why this idea was generated
idea.criteria_scores   # Dict of individual criterion scores
```

## CSV Export

Export ideas to CSV for analysis in Excel, pandas, or other tools:

### Auto-save During Brainstorming

```python
result = brainstorm.brainstorm(
    "Generate domain names for my startup",
    save_csv=True,                    # Auto-save to CSV
    csv_path="domain_names.csv",      # File path
    csv_expand_criteria=True          # Separate columns per criterion
)
```

### Manual Export After Brainstorming

```python
# With expanded criteria (each criterion gets its own column)
result.to_csv("ideas_detailed.csv", expand_criteria=True)

# Compact format (criteria as JSON string)
result.to_csv("ideas_compact.csv", expand_criteria=False)
```

### CSV Output Format

**Expanded format** (default):
```csv
id,text,quality_score,depth,parent_id,reasoning,feasibility_score,impact_score
idea_1_0,Smart recycling bins,8.5,0,,Uses AI,8.0,9.0
idea_1_1,Plastic-free marketplace,7.2,0,,Reduces at source,7.5,8.0
```

**Compact format**:
```csv
id,text,quality_score,depth,parent_id,reasoning,criteria_scores
idea_1_0,Smart recycling bins,8.5,0,,Uses AI,"{""feasibility"": 8.0}"
```

## Custom Evaluation Criteria

Define your own criteria for scoring ideas:

```python
brainstorm = RecursiveBrainstorm(
    llm=llm,
    evaluation_criteria=[
        "technical_feasibility",
        "market_potential",
        "time_to_market",
        "competitive_advantage",
        "user_value"
    ]
)

result = brainstorm.brainstorm("SaaS product ideas")

# Access detailed scores
for idea in result.all_ideas[:5]:
    print(f"\n{idea.text}")
    print(f"Overall: {idea.quality_score}/10")
    for criterion, score in idea.criteria_scores.items():
        print(f"  {criterion}: {score}/10")
```

## Real-World Example: Domain Name Generator

Generate 100+ domain name ideas:

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini")

brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=4,  # 4 + 16 + 64 = 84 domains
    mode="tree",
    evaluation_criteria=[
        "memorability",
        "brandability",
        "simplicity",
        "availability_likelihood"
    ],
    min_quality_threshold=5.0,
    verbose=True
)

result = brainstorm.brainstorm(
    prompt="""Generate creative domain names for an AI-powered
    productivity app for remote teams. Include extensions like
    .com, .io, .app""",
    save_csv=True,
    csv_path="domain_names.csv"
)

# Show top 10 domains
top_domains = sorted(
    result.all_ideas,
    key=lambda x: x.quality_score,
    reverse=True
)[:10]

print("\nTop 10 Domain Names:")
for i, domain in enumerate(top_domains, 1):
    print(f"{i}. {domain.text} - Score: {domain.quality_score}/10")
```

## MiniAgent Integration

Use brainstorming as a tool in multi-step workflows:

```python
from SimplerLLM.language.flow import MiniAgent

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini")
agent = MiniAgent("Product Development Agent", llm, max_steps=3)

# Step 1: Brainstorm ideas
agent.add_step(
    step_type="tool",
    tool_name="recursive_brainstorm",
    params={
        "llm_instance": llm,
        "max_depth": 2,
        "mode": "hybrid",
        "top_n": 3
    }
)

# Step 2: Analyze best idea
agent.add_step(
    step_type="llm",
    prompt="""Analyze the best brainstormed idea and create
    an action plan with 5 concrete steps.

    Brainstorm results: {previous_output}"""
)

# Step 3: Create product brief
agent.add_step(
    step_type="llm",
    prompt="""Based on the analysis, create a one-page product brief.

    Analysis: {previous_output}"""
)

# Run the workflow
result = agent.run("Productivity tools for developers")
print(result.final_output)
```

## Async Support

For better performance with large brainstorming sessions:

```python
import asyncio

async def brainstorm_async():
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini")

    brainstorm = RecursiveBrainstorm(llm, max_depth=2, mode="hybrid")

    result = await brainstorm.brainstorm_async(
        "Innovative blockchain use cases"
    )

    return result

# Run async
result = asyncio.run(brainstorm_async())
```

## Best Practices

### Mode Selection

- **Tree mode**: Initial ideation, research, comprehensive coverage
- **Linear mode**: Refining specific concepts, deep exploration
- **Hybrid mode**: Most use cases (recommended starting point)

### Depth and Ideas Configuration

- Start with `max_depth=2-3` to control costs
- Use `ideas_per_level=3-5` for balance between diversity and cost
- For hybrid mode, set `top_n` to 50-60% of `ideas_per_level`

```python
# Cost-effective configuration
brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,
    ideas_per_level=4,
    mode="hybrid",
    top_n=2
)
# Generates ~20-30 ideas with manageable API costs
```

### Quality Filtering

- Use `min_quality_threshold=6.0-7.0` to focus on high-quality ideas
- Lower threshold (4.0-5.0) for exploratory brainstorming
- Higher threshold (8.0+) for selective refinement

### Evaluation Criteria

Use **3-5 criteria** for balanced evaluation:

```python
# Product ideas
evaluation_criteria=["feasibility", "market_fit", "innovation", "scalability"]

# Research ideas
evaluation_criteria=["novelty", "impact", "feasibility", "clarity"]

# Marketing ideas
evaluation_criteria=["creativity", "target_audience_fit", "actionability"]
```

## Cost Estimation

Understanding LLM API calls:

**Tree mode** (exponential):
- `max_depth=2`, `ideas_per_level=5`
- Calls: 1 + 5 + 25 = **31 LLM calls**

**Linear mode** (linear):
- `max_depth=2`, `ideas_per_level=5`
- Calls: 1 + 1 + 1 = **3 LLM calls**

**Hybrid mode** (controlled):
- `max_depth=2`, `ideas_per_level=5`, `top_n=3`
- Calls: 1 + 3 + 9 = **13 LLM calls**

## Common Use Cases

### 1. Domain Name Generation
```python
brainstorm.brainstorm("Generate domain names for [business description]")
```

### 2. Product Ideation
```python
brainstorm.brainstorm("Product features for [target audience]")
```

### 3. Content Ideas
```python
brainstorm.brainstorm("Blog post topics about [subject]")
```

### 4. Problem Solving
```python
brainstorm.brainstorm("Solutions to reduce [specific problem]")
```

### 5. Marketing Strategies
```python
brainstorm.brainstorm("Marketing campaigns for [product/service]")
```

## Tips & Tricks

**1. Add context for better ideas:**
```python
result = brainstorm.brainstorm(
    prompt="Generate app ideas",
    context="Target: college students, Budget: $10k, Timeline: 3 months"
)
```

**2. Sort by specific criteria:**
```python
# Sort by a specific criterion
top_by_novelty = sorted(
    result.all_ideas,
    key=lambda x: x.criteria_scores.get("novelty", 0),
    reverse=True
)[:10]
```

**3. Filter by depth:**
```python
# Only get top-level ideas
root_ideas = [idea for idea in result.all_ideas if idea.depth == 0]
```

**4. Export tree structure to JSON:**
```python
import json

tree = result.to_tree_dict()
with open("idea_tree.json", "w") as f:
    json.dump(tree, f, indent=2)
```

---

That's how you can benefit from SimplerLLM to make brainstorming simpler and more powerful! Generate hundreds of high-quality ideas in seconds, organize them hierarchically, and export for further analysis. 🚀
