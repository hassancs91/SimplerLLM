# LLMFeedbackLoop - Iterative Self-Improvement System

LLMFeedbackLoop is a powerful tool for iteratively refining LLM responses through critique and improvement cycles. It supports multiple architectural patterns and provides complete control over the refinement process with full history tracking.

## Features

- **Three Architectural Patterns**: Single provider self-critique, dual provider (generator + critic), multi-provider rotation
- **Iterative Refinement**: Automatically improve answers through multiple cycles
- **Structured Critique**: Pydantic models for consistent, structured feedback
- **Smart Stopping**: Max iterations, quality thresholds, convergence detection
- **Temperature Scheduling**: Fixed, decreasing, or custom temperature per iteration
- **Full History Tracking**: Access all iterations with scores and critiques
- **Focus Criteria**: Target specific aspects for improvement
- **Convergence Detection**: Automatically stop when improvements plateau
- **Custom Prompts**: Override default critique and improvement templates

## Installation

LLMFeedbackLoop is included in SimplerLLM. Just import it:

```python
from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop
```

## Quick Start

```python
from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

# Create LLM instance
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")

# Initialize feedback loop
feedback = LLMFeedbackLoop(llm=llm, max_iterations=3)

# Improve an answer
result = feedback.improve("Explain machine learning in simple terms")

# Access results
print(f"Improvement: {result.initial_score} → {result.final_score}")
print(result.final_answer)
```

## Three Architectural Patterns

### 1. Single Provider Self-Critique

The same LLM generates, critiques, and improves its own answers.

```python
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")

feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=3
)

result = feedback.improve("Explain quantum computing")
```

**Use Case**: Simplest pattern, good for quick iterations with a single model.

**How it works**:
1. LLM generates initial answer
2. LLM critiques its own answer
3. LLM improves based on its critique
4. Repeat

### 2. Dual Provider (Generator + Critic)

One LLM generates answers, another provides critiques.

```python
generator = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")
critic = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4")

feedback = LLMFeedbackLoop(
    generator_llm=generator,
    critic_llm=critic,
    max_iterations=3
)

result = feedback.improve("Write a professional email")
```

**Use Case**: Leverage different models' strengths - one for generation, one for evaluation.

**How it works**:
1. Generator creates answer
2. Critic evaluates and suggests improvements
3. Generator improves based on critic's feedback
4. Repeat

### 3. Multi-Provider Rotation

Multiple providers rotate through generate/critique roles.

```python
providers = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4"),
    LLM.create(LLMProvider.GEMINI, model_name="gemini-pro"),
]

feedback = LLMFeedbackLoop(
    providers=providers,
    max_iterations=3
)

result = feedback.improve("Explain neural networks")
```

**Use Case**: Maximum diversity - each model builds on the previous one's work.

**How it works**:
1. Provider A generates initial answer
2. Provider B critiques Provider A's answer
3. Provider B improves the answer
4. Provider C critiques Provider B's answer
5. And so on...

## Stopping Criteria

LLMFeedbackLoop supports multiple stopping criteria (can be combined):

### 1. Max Iterations (Required)

```python
feedback = LLMFeedbackLoop(llm=llm, max_iterations=5)
# Will stop after 5 iterations maximum
```

### 2. Quality Threshold (Optional)

```python
feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=10,
    quality_threshold=9.0  # Stop when score >= 9.0
)
```

### 3. Convergence Detection (Optional, Default: On)

```python
feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=10,
    convergence_threshold=0.1,  # Stop if improvement < 10%
    check_convergence=True
)
```

Convergence is detected when:
- Score improvement falls below threshold (e.g., < 10%)
- Answer text becomes very similar (> 95% similar)

## Focused Improvement

Target specific criteria for improvement:

```python
result = feedback.improve(
    prompt="Explain blockchain",
    focus_on=["simplicity", "clarity", "conciseness"]
)
# The loop will focus improvements on these specific criteria
```

Default criteria: `["accuracy", "clarity", "completeness"]`

## Temperature Scheduling

Control creativity/randomness across iterations:

### Fixed Temperature (Default)

```python
feedback = LLMFeedbackLoop(
    llm=llm,
    temperature=0.7  # Same temperature for all iterations
)
```

### Decreasing Temperature

```python
feedback = LLMFeedbackLoop(
    llm=llm,
    temperature=0.9,
    temperature_schedule="decreasing"  # 0.9 → 0.63 → 0.44 → 0.31
)
```

More creative at start, more conservative later.

### Custom Schedule

```python
feedback = LLMFeedbackLoop(
    llm=llm,
    temperature_schedule=[0.9, 0.7, 0.5, 0.3]
)
```

Precise control over each iteration's temperature.

## Result Structure

The `FeedbackResult` object provides comprehensive information:

```python
result = feedback.improve(...)

# Final output
result.final_answer              # str - Final refined answer
result.final_score               # float - Quality score (1-10)
result.initial_score             # float - First iteration score

# History and trajectory
result.all_iterations            # List[IterationResult] - Complete history
result.improvement_trajectory    # List[float] - Scores: [6.5, 7.8, 8.9, 9.2]

# Metadata
result.total_iterations          # int - Actual iterations run
result.stopped_reason            # str - "max_iterations", "converged", "threshold_met"
result.convergence_detected      # bool
result.total_execution_time      # float - Total time in seconds
result.architecture_used         # str - "single", "dual", "multi_rotation"

# Access specific iteration
iteration = result.all_iterations[0]
iteration.answer                 # Answer at this iteration
iteration.critique               # Full Critique object
iteration.critique.strengths     # List[str]
iteration.critique.weaknesses    # List[str]
iteration.critique.improvement_suggestions  # List[str]
iteration.critique.quality_score # float (1-10)
iteration.improvement_from_previous  # float - Improvement percentage
```

## Critique Structure

Each iteration's critique is a structured Pydantic model:

```python
critique = iteration.critique

critique.strengths               # List[str] - What's good
critique.weaknesses              # List[str] - What needs improvement
critique.improvement_suggestions # List[str] - Specific suggestions
critique.quality_score           # float (1-10) - Overall score
critique.specific_issues         # Dict[str, str] - Issues per criterion
critique.reasoning               # str - Detailed reasoning
```

## Advanced Usage

### Starting with an Existing Answer

```python
initial_answer = "Machine learning is when computers learn stuff."

result = feedback.improve(
    prompt="Explain machine learning",
    initial_answer=initial_answer  # Start here instead of generating new
)
# The loop will improve this existing answer
```

### Custom Critique Prompt Template

```python
custom_critique = """
Evaluate this answer strictly on {criteria}.

Question: {original_prompt}
Answer: {current_answer}

Provide harsh, specific critique.
"""

result = feedback.improve(
    prompt="Explain AI",
    critique_prompt_template=custom_critique
)
```

Template variables available:
- `{original_prompt}` - The original question
- `{current_answer}` - Current answer being critiqued
- `{criteria}` - Comma-separated criteria list

### Custom Improvement Prompt Template

```python
custom_improvement = """
Improve this answer:

Original Question: {original_prompt}
Current Answer: {current_answer}

Problems identified:
{weaknesses}

Suggestions:
{suggestions}

{focus_instruction}

Write an improved version.
"""

result = feedback.improve(
    prompt="Explain databases",
    improvement_prompt_template=custom_improvement
)
```

Template variables available:
- `{original_prompt}` - The original question
- `{current_answer}` - Current answer
- `{strengths}` - Newline-separated strengths
- `{weaknesses}` - Newline-separated weaknesses
- `{suggestions}` - Newline-separated suggestions
- `{issues}` - Formatted specific issues
- `{focus_instruction}` - Focus criteria instruction (auto-generated)

### Async Usage

```python
result = await feedback.improve_async(prompt="Explain async")
```

Note: Currently runs synchronously. Full async support coming when LLM wrappers support it.

## Use Cases

### 1. Maximum Quality Answers

```python
# Get the absolute best answer through iterative refinement
feedback = LLMFeedbackLoop(
    llm=gpt4,
    max_iterations=5,
    quality_threshold=9.5
)
result = feedback.improve(critical_prompt)
```

### 2. Research & Analysis

```python
# Track improvement process with full history
feedback = LLMFeedbackLoop(llm=llm, max_iterations=5, verbose=True)
result = feedback.improve(prompt)

# Analyze trajectory
for i, iteration in enumerate(result.all_iterations):
    print(f"Iteration {i+1}:")
    print(f"  Score: {iteration.critique.quality_score}")
    print(f"  Weaknesses: {iteration.critique.weaknesses}")
```

### 3. Multi-Model Collaboration

```python
# Leverage strengths of different models
providers = [gpt4, claude, gemini]
feedback = LLMFeedbackLoop(providers=providers, max_iterations=3)
result = feedback.improve(complex_question)
# Each model builds on the previous one's work
```

### 4. Cost-Effective Quality

```python
# Use cheap model with multiple iterations instead of expensive model once
cheap_llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini")
feedback = LLMFeedbackLoop(llm=cheap_llm, max_iterations=5)
result = feedback.improve(prompt)
# Final quality may match expensive model at lower cost
```

### 5. Automated Editing

```python
# Improve existing content
feedback = LLMFeedbackLoop(llm=llm, max_iterations=3)

result = feedback.improve(
    prompt="Make this more professional",
    initial_answer=draft_email,
    focus_on=["professionalism", "clarity", "conciseness"]
)
```

## Comparison with LLMJudge

| Feature | LLMJudge | LLMFeedbackLoop |
|---------|----------|-----------------|
| **Purpose** | Compare multiple providers | Iteratively improve single answer |
| **Input** | Multiple providers answer once | One answer improved multiple times |
| **Output** | Best/synthesized from multiple | Refined single answer |
| **Iterations** | 1 round | Multiple rounds (configurable) |
| **Use Case** | Get best answer fast | Maximize answer quality |
| **History** | All provider responses | All improvement iterations |

## Combining LLMJudge and LLMFeedbackLoop

Use both for ultimate quality:

```python
# Step 1: Use FeedbackLoop to refine answer
feedback = LLMFeedbackLoop(llm=gpt4, max_iterations=3)
refined = feedback.improve(prompt)

# Step 2: Use Judge to compare with other providers
judge = LLMJudge(providers=[gpt4, claude, gemini], judge_llm=opus)
result = judge.generate(prompt, mode="synthesize")

# Compare results
print(f"FeedbackLoop score: {refined.final_score}")
print(f"Judge best score: {max(e.overall_score for e in result.evaluations)}")
```

## Best Practices

1. **Start Small**: Begin with 3 iterations, increase if needed

2. **Enable Convergence**: Use `check_convergence=True` to avoid wasted iterations

3. **Set Quality Thresholds**: If you have a target quality, set `quality_threshold`

4. **Use Focus Criteria**: Specify `focus_on` to guide improvements effectively

5. **Dual Provider for Critical Tasks**: Use a strong critic model for important content

6. **Monitor Costs**: Each iteration = 2 API calls (1 for critique, 1 for improvement)

7. **Verbose for Debugging**: Use `verbose=True` during development

8. **Temperature Scheduling**: Use decreasing temperature for convergence to optimal answers

## Performance Considerations

- **Iterations**: More iterations = better quality but higher cost and latency
- **Convergence**: Usually converges in 2-4 iterations for most tasks
- **Parallel**: Currently sequential (critique → improve → critique...)
- **Cost**: N iterations = ~2N API calls (critique + improvement each iteration)
- **Time**: Expect 2-15 seconds per iteration depending on model

## Troubleshooting

### Import Error

```python
# Error: cannot import name 'LLMFeedbackLoop'
# Solution: Update SimplerLLM
pip install --upgrade SimplerLLM
```

### Convergence Too Early

```python
# Lower convergence threshold
feedback = LLMFeedbackLoop(
    llm=llm,
    convergence_threshold=0.05  # Only stop if improvement < 5%
)
```

### Quality Not Improving

```python
# Try different pattern or focus criteria
feedback = LLMFeedbackLoop(
    generator_llm=gpt4,
    critic_llm=claude,  # Use different critic
    max_iterations=5
)

result = feedback.improve(
    prompt=prompt,
    focus_on=["specific_issue_1", "specific_issue_2"]
)
```

### Too Many Iterations

```python
# Enable convergence detection
feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=10,
    check_convergence=True,
    quality_threshold=9.0  # Stop when good enough
)
```

## API Reference

### LLMFeedbackLoop Class

```python
class LLMFeedbackLoop:
    def __init__(
        llm: Optional[LLM] = None,              # Single provider
        generator_llm: Optional[LLM] = None,    # Dual: generator
        critic_llm: Optional[LLM] = None,       # Dual: critic
        providers: Optional[List[LLM]] = None,  # Multi-provider
        max_iterations: int = 3,
        convergence_threshold: float = 0.1,
        quality_threshold: Optional[float] = None,
        check_convergence: bool = True,
        default_criteria: Optional[List[str]] = None,
        temperature: float = 0.7,
        temperature_schedule: Optional[Union[str, List[float]]] = None,
        verbose: bool = False
    )

    def improve(
        prompt: str,
        initial_answer: Optional[str] = None,
        focus_on: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improvement_prompt_template: Optional[str] = None
    ) -> FeedbackResult

    async def improve_async(...) -> FeedbackResult
```

## Examples

See:
- [examples/llm_feedback_example.py](../../../examples/llm_feedback_example.py) - Simple usage
- [tests/test_llm_feedback.py](../../../tests/test_llm_feedback.py) - Comprehensive tests

## Contributing

Contributions welcome! Areas for improvement:
- True async execution when LLM wrappers support it
- Parallel critique+improvement for faster iterations
- Integration with MiniAgent as flow steps
- Visual trajectory plotting
- A/B testing between different configurations

## License

Same as SimplerLLM
