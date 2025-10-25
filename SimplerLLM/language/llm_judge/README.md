# LLMJudge - Multi-Provider Orchestration & Evaluation

LLMJudge is a powerful tool for orchestrating multiple LLM providers, evaluating their responses, and generating comparative analyses or synthesized answers. It's perfect for production use (getting the best answer) and research/benchmarking (comparing model performance).

## Features

- **Three Evaluation Modes**: Select best, synthesize, or compare responses
- **Multi-Provider Support**: Works with any LLM providers in SimplerLLM
- **Parallel Execution**: Run providers simultaneously for faster results
- **Structured Output**: Pydantic models for all responses and evaluations
- **Custom Criteria**: Define your own evaluation criteria
- **Batch Evaluation**: Benchmark multiple prompts at once
- **Statistical Reports**: Generate comprehensive evaluation reports
- **Router Training**: Export data for LLMRouter configuration
- **Detailed Scoring**: 1-10 scores per criterion with rankings and reasoning

## Installation

LLMJudge is included in SimplerLLM. Just import it:

```python
from SimplerLLM.language import LLM, LLMProvider, LLMJudge
```

## Quick Start

```python
from SimplerLLM.language import LLM, LLMProvider, LLMJudge

# Create provider instances
providers = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4"),
    LLM.create(LLMProvider.GEMINI, model_name="gemini-pro"),
]

# Create judge instance
judge_llm = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-opus-4")

# Initialize LLMJudge
judge = LLMJudge(providers=providers, judge_llm=judge_llm, parallel=True)

# Get best answer
result = judge.generate(
    prompt="Explain quantum computing",
    mode="synthesize"  # Combines best parts from all answers
)

# Access results
print(result.final_answer)
print(result.confidence_scores)
```

## Three Evaluation Modes

### 1. Select Best (`mode="select_best"`)

Picks the single best response from all providers.

```python
result = judge.generate(
    prompt="What is machine learning?",
    mode="select_best",
    criteria=["accuracy", "clarity", "conciseness"]
)

# The final_answer is the complete text of the winning response
print(f"Winner: {result.evaluations[0].provider_name}")
print(f"Score: {result.evaluations[0].overall_score}/10")
```

**Use Case**: When you want the best single answer and trust one provider's complete response.

### 2. Synthesize (`mode="synthesize"`)

Combines the best elements from all responses into a new, improved answer.

```python
result = judge.generate(
    prompt="Benefits of Python for data science",
    mode="synthesize",
    criteria=["completeness", "accuracy", "clarity"]
)

# The final_answer is a NEW synthesized response
print(result.final_answer)
```

**Use Case**: Production applications where you want the highest quality answer by leveraging multiple models.

### 3. Compare (`mode="compare"`)

Provides detailed comparative analysis without selecting a winner.

```python
result = judge.generate(
    prompt="Supervised vs unsupervised learning",
    mode="compare",
    criteria=["accuracy", "clarity", "depth"]
)

# The final_answer is a comparative summary
for eval in result.evaluations:
    print(f"{eval.provider_name}: {eval.overall_score}/10")
    print(f"Strengths: {eval.strengths}")
    print(f"Weaknesses: {eval.weaknesses}")
```

**Use Case**: Research, benchmarking, and model evaluation.

## Evaluation Criteria

### Default Criteria

- `accuracy`: Factual correctness
- `clarity`: How clear and understandable the response is
- `completeness`: Whether all aspects are covered

### Custom Criteria

Define your own criteria based on your needs:

```python
result = judge.generate(
    prompt="Write a haiku about coding",
    mode="select_best",
    criteria=[
        "creativity",
        "adherence_to_format",
        "emotional_impact",
        "technical_accuracy"
    ]
)
```

## Batch Evaluation & Benchmarking

Evaluate multiple prompts to compare provider performance:

```python
# Define test prompts
prompts = [
    "What is AI?",
    "Explain neural networks",
    "Applications of ML in healthcare",
    # ... more prompts
]

# Batch evaluate
results = judge.evaluate_batch(prompts, mode="compare")

# Generate statistical report
report = judge.generate_evaluation_report(results, export_format="json")

print(f"Best Provider: {report.best_provider_overall}")
print(f"Win Counts: {report.provider_win_counts}")
print(f"Average Scores: {report.average_scores}")
print(f"Best by Criteria: {report.best_provider_by_criteria}")
```

### Export Formats

- **JSON**: Structured data for analysis
- **CSV**: Spreadsheet-friendly format

```python
# Export as JSON
report = judge.generate_evaluation_report(results, export_format="json")
# Creates: llm_judge_report_YYYYMMDD_HHMMSS.json

# Export as CSV
report = judge.generate_evaluation_report(results, export_format="csv")
# Creates: llm_judge_report_YYYYMMDD_HHMMSS.csv
```

## Router Training Data

Generate insights for configuring LLMRouter:

```python
result = judge.generate(
    prompt="Write Python code for binary search",
    mode="select_best",
    generate_summary=True  # Enable router summary
)

# Access router summary
summary = judge._router_summary
print(summary.query_type)           # e.g., "coding"
print(summary.winning_provider)     # e.g., "OPENAI"
print(summary.recommendation)       # e.g., "Use OPENAI for coding tasks"
print(summary.criteria_winners)     # Which provider won each criterion
```

## Configuration Options

### Parallel vs Sequential Execution

```python
# Parallel (faster)
judge = LLMJudge(providers=providers, judge_llm=judge_llm, parallel=True)

# Sequential (one at a time)
judge = LLMJudge(providers=providers, judge_llm=judge_llm, parallel=False)
```

### Default Criteria

```python
judge = LLMJudge(
    providers=providers,
    judge_llm=judge_llm,
    default_criteria=["accuracy", "clarity", "creativity"]
)

# Uses default criteria if not specified in generate()
result = judge.generate("Write a story")
```

### Verbose Logging

```python
judge = LLMJudge(
    providers=providers,
    judge_llm=judge_llm,
    verbose=True  # Enable detailed logging
)
```

## Result Structure

The `JudgeResult` object contains:

```python
result = judge.generate(...)

# Final answer (selected or synthesized)
result.final_answer          # str

# All provider responses
result.all_responses         # List[ProviderResponse]
# Each with: provider_name, model_name, response_text, execution_time, error

# Evaluations for each provider
result.evaluations           # List[ProviderEvaluation]
# Each with: provider_name, overall_score, rank, criterion_scores, reasoning

# Judge's overall reasoning
result.judge_reasoning       # str

# Confidence scores (0-1 scale)
result.confidence_scores     # Dict[str, float]

# Metadata
result.mode                  # JudgeMode
result.criteria_used         # List[str]
result.total_execution_time  # float
result.timestamp             # datetime
```

## Use Cases

### 1. Production - Get Best Answer

```python
# Get highest quality answer for user queries
judge = LLMJudge(providers=[gpt4, claude, gemini], judge_llm=opus)
result = judge.generate(user_query, mode="synthesize")
return result.final_answer
```

### 2. Research - Model Comparison

```python
# Compare different models on your specific domain
test_prompts = load_domain_specific_prompts()
results = judge.evaluate_batch(test_prompts)
report = judge.generate_evaluation_report(results)
analyze_which_model_is_best(report)
```

### 3. Cost Optimization

```python
# Test cheaper models vs expensive ones
providers = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4"),        # Expensive
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini"),  # Cheap
]
# See if cheaper model is good enough
```

### 4. Quality Assurance

```python
# Ensure your production prompts generate high-quality answers
critical_prompts = ["How to handle PII?", "Security best practices"]
results = judge.evaluate_batch(critical_prompts, mode="compare")
# Review and verify quality before deployment
```

### 5. A/B Testing

```python
# Compare old vs new prompts
judge = LLMJudge(providers=[same_model, same_model], judge_llm=judge)
# Use different system prompts for each provider
# Compare which prompt engineering approach works better
```

## Advanced Examples

### Custom System Prompts

```python
result = judge.generate(
    prompt="Explain databases",
    mode="select_best",
    system_prompt="You are a database expert with 20 years of experience."
)
```

### Async Usage

```python
result = await judge.generate_async(
    prompt="Explain async programming",
    mode="synthesize"
)
```

Note: Currently runs synchronously. Full async support coming when LLM wrappers support it.

### Handling Errors

```python
result = judge.generate(prompt="Test")

# Check for provider errors
for response in result.all_responses:
    if response.error:
        print(f"{response.provider_name} failed: {response.error}")

# Filter failed providers
successful = [r for r in result.all_responses if not r.error]
```

## Best Practices

1. **Choose the Right Judge**: Use a strong model (e.g., Claude Opus, GPT-4) as judge for best evaluation quality

2. **Parallel by Default**: Enable `parallel=True` for faster execution unless you have API rate limits

3. **Relevant Criteria**: Choose evaluation criteria that match your use case

4. **Synthesize for Production**: Use `mode="synthesize"` in production for highest quality answers

5. **Compare for Research**: Use `mode="compare"` for benchmarking and model evaluation

6. **Batch for Efficiency**: Use `evaluate_batch()` instead of multiple `generate()` calls

7. **Monitor Costs**: Be aware that LLMJudge makes N+1 API calls (N providers + 1 judge)

## Performance Considerations

- **Parallel Execution**: Significantly faster but may hit rate limits
- **API Costs**: Judge adds one extra API call (usually to a stronger/more expensive model)
- **Time**: Expect 2-10 seconds per evaluation depending on providers and parallelization

## Troubleshooting

### Import Error

```python
# Error: cannot import name 'LLMJudge'
# Solution: Make sure SimplerLLM is up to date
pip install --upgrade SimplerLLM
```

### All Providers Failing

```python
# Check API keys are set
import os
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"
```

### Judge Evaluation Fails

```python
# Use verbose mode to see detailed errors
judge = LLMJudge(providers=..., judge_llm=..., verbose=True)
```

## API Reference

### LLMJudge Class

```python
class LLMJudge:
    def __init__(
        providers: List[LLM],           # Provider instances
        judge_llm: LLM,                 # Judge instance
        parallel: bool = True,          # Parallel execution
        default_criteria: List[str] = None,  # Default criteria
        verbose: bool = False           # Logging
    )

    def generate(
        prompt: str,                    # Question/prompt
        mode: str = "synthesize",       # Mode
        criteria: List[str] = None,     # Criteria
        system_prompt: str = None,      # System prompt
        generate_summary: bool = False  # Router summary
    ) -> JudgeResult

    async def generate_async(...) -> JudgeResult

    def evaluate_batch(
        prompts: List[str],
        mode: str = "compare",
        criteria: List[str] = None,
        system_prompt: str = None
    ) -> List[JudgeResult]

    def generate_evaluation_report(
        results: List[JudgeResult],
        export_format: str = None  # "json" or "csv"
    ) -> EvaluationReport
```

## Examples

See:
- [examples/llm_judge_example.py](../../../examples/llm_judge_example.py) - Simple usage example
- [tests/test_llm_judge.py](../../../tests/test_llm_judge.py) - Comprehensive test suite

## Contributing

Contributions welcome! Areas for improvement:
- True async execution when LLM wrappers support it
- More sophisticated query type inference for router training
- Additional export formats (Excel, databases)
- Visualization tools for evaluation reports
- Integration with MiniAgent as flow steps

## License

Same as SimplerLLM
