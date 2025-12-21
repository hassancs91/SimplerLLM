# LLM Validator - Manual vs Automated

## What is LLM Validator?
Using multiple AI models to validate/verify AI-generated content. Instead of trusting one model's output, you get multiple "opinions" and aggregate them.

Think of it as having multiple experts review the same work and reaching a consensus.

---

## WHY VALIDATE?

LLMs can:
- Hallucinate facts
- Miss important details
- Be confidently wrong
- Give inconsistent quality

**Solution**: Have other LLMs check the work!

---

## MANUAL APPROACH (ChatGPT/Claude/Gemini UIs)

### Scenario: Validate a Generated Article

You asked GPT-4 to write an article about "Benefits of Meditation". Now you want to verify it's accurate.

### Step 1: Generate Content (GPT-4)
```
Write a short article about the health benefits of meditation.
```

**Response:**
```
Meditation has been shown to reduce stress, lower blood pressure,
and improve focus. Studies from Harvard Medical School indicate
that regular meditation can reduce anxiety by up to 60%...
```

### Step 2: Ask Claude to Validate (Open Claude, new chat)
**Prompt:**
```
You are a fact-checker. Rate this content from 0 to 1 on accuracy,
and explain any issues:

CONTENT:
"Meditation has been shown to reduce stress, lower blood pressure,
and improve focus. Studies from Harvard Medical School indicate
that regular meditation can reduce anxiety by up to 60%..."

Provide:
- Score (0-1)
- Confidence (0-1)
- Explanation
```

**Response:**
```
Score: 0.7
Confidence: 0.8
Explanation: The general claims about stress and blood pressure are
accurate. However, the "60% reduction" claim needs verification -
I cannot confirm this specific statistic from Harvard.
```

### Step 3: Ask Gemini to Validate (Open Gemini, new chat)
**Same prompt, different model...**

**Response:**
```
Score: 0.75
Confidence: 0.85
Explanation: The meditation benefits are well-documented. The Harvard
claim is partially accurate but the percentage may be exaggerated...
```

### Step 4: Manually Calculate Consensus
Now you:
- Open a spreadsheet or calculator
- Average the scores: (0.7 + 0.75) / 2 = 0.725
- Compare explanations
- Decide if content is valid

---

## THE PROBLEMS

1. **Multiple browser tabs**: Switch between ChatGPT, Claude, Gemini
2. **Copy/paste content**: Into each chat
3. **Inconsistent prompts**: Easy to phrase differently
4. **Manual calculations**: Average scores yourself
5. **No structured output**: Parse text responses
6. **Time consuming**: 10+ minutes for one validation
7. **Can't automate**: No way to script this

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_validator import LLMValidator

# Create multiple validators (different models = different perspectives)
validators = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
    LLM.create(LLMProvider.GEMINI, model_name="gemini-2.0-flash"),
]

# Create validator
validator = LLMValidator(
    validators=validators,
    parallel=True,  # Run all at once
    default_threshold=0.7,  # Pass if score >= 0.7
)

# Content to validate
content = """
Meditation has been shown to reduce stress, lower blood pressure,
and improve focus. Studies from Harvard Medical School indicate
that regular meditation can reduce anxiety by up to 60%...
"""

# Validate!
result = validator.validate(
    content=content,
    validation_prompt="Check if the health claims are accurate and well-supported",
    original_question="What are the health benefits of meditation?",
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Valid: {result.is_valid}")
print(f"Consensus: {result.consensus}")
```

**Output:**
```
Overall Score: 0.73
Valid: True
Consensus: True (all validators within 0.15 of each other)
```

---

## HOW IT WORKS

```
Content to Validate
        ↓
┌───────────────────────────────────────────────────────────┐
│                    LLM VALIDATOR                          │
├───────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                 │
│  │  GPT-4  │   │ Claude  │   │ Gemini  │   (parallel)    │
│  │ Score:  │   │ Score:  │   │ Score:  │                 │
│  │  0.70   │   │  0.75   │   │  0.74   │                 │
│  └─────────┘   └─────────┘   └─────────┘                 │
│         ↓            ↓            ↓                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              AGGREGATION                             │ │
│  │  • Average: (0.70 + 0.75 + 0.74) / 3 = 0.73        │ │
│  │  • Consensus: All within 0.15 range? ✓             │ │
│  │  • Valid: 0.73 >= 0.70 threshold? ✓                │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
        ↓
ValidationResult {
    overall_score: 0.73,
    is_valid: true,
    consensus: true,
    validators: [...]
}
```

---

## AGGREGATION METHODS

```python
# Simple average (default)
result = validator.validate(content, prompt, aggregation="average")

# Weighted by provider (trust some more)
result = validator.validate(
    content, prompt,
    aggregation="weighted",
    weights={"OPENAI": 1.5, "ANTHROPIC": 1.0, "GEMINI": 0.8}
)

# Median (resistant to outliers)
result = validator.validate(content, prompt, aggregation="median")

# Consensus-based
result = validator.validate(content, prompt, aggregation="consensus")
```

---

## REAL EXAMPLE: Fact-Checking an AI Response

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_validator import LLMValidator

# Setup validators
validators = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
]

validator = LLMValidator(validators=validators, verbose=True)

# AI-generated content to check
ai_response = """
The Eiffel Tower was built in 1889 for the World's Fair in Paris.
It stands 330 meters tall and was designed by Gustave Eiffel.
The tower receives approximately 7 million visitors annually and
was originally intended to be dismantled after 20 years.
"""

# Validate factual accuracy
result = validator.validate(
    content=ai_response,
    validation_prompt="""Verify these facts:
    1. Construction date and purpose
    2. Height accuracy
    3. Designer attribution
    4. Visitor numbers
    5. Original dismantling plan

    Score based on factual accuracy.""",
    original_question="Tell me about the Eiffel Tower",
)

print(f"\n{'='*50}")
print(f"VALIDATION RESULT")
print(f"{'='*50}")
print(f"Overall Score: {result.overall_score:.2f}/1.0")
print(f"Is Valid: {result.is_valid}")
print(f"Consensus: {result.consensus}")
print(f"\nIndividual Scores:")
for v in result.validators:
    print(f"  {v.provider_name}: {v.score:.2f} - {v.explanation[:60]}...")
```

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| 3+ browser tabs | 1 function call |
| Copy/paste to each | Automatic distribution |
| Calculate average manually | Automatic aggregation |
| Inconsistent prompts | Consistent evaluation |
| 10+ minutes | 5 seconds |
| Can't automate | Fully scriptable |
| No consensus detection | Built-in consensus check |

---

## USE CASES

1. **Fact-checking**: Verify AI-generated articles before publishing
2. **Quality assurance**: Score content before using in production
3. **Homework checking**: Validate AI tutoring responses
4. **Medical/Legal**: Extra validation for sensitive content
5. **Translation**: Multiple models verify translation accuracy

---

## SIMPLE DEMO SCRIPT

```python
"""
LLM Validator Demo for Video
Multiple AI models validate content accuracy
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_validator import LLMValidator


def main():
    print("\n" + "=" * 60)
    print("LLM VALIDATOR DEMO")
    print("Multiple AI models checking content accuracy")
    print("=" * 60)

    # Create validators (multiple perspectives)
    validators = [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
    ]

    validator = LLMValidator(
        validators=validators,
        parallel=True,
        default_threshold=0.7,
        verbose=True,
    )

    # Content to validate
    content = """
    Python was created by Guido van Rossum and first released in 1991.
    It is named after Monty Python's Flying Circus. Python is known for
    its simple syntax and is the most popular programming language for
    AI and machine learning. The latest version is Python 4.0.
    """

    print("\nContent to validate:")
    print("-" * 40)
    print(content)
    print("-" * 40)

    # Validate
    result = validator.validate(
        content=content,
        validation_prompt="Check all factual claims for accuracy",
        original_question="Tell me about Python programming language",
    )

    # Results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Score: {result.overall_score:.2f}/1.0")
    print(f"Is Valid: {result.is_valid}")
    print(f"Consensus: {result.consensus}")
    print(f"Consensus Details: {result.consensus_details}")

    print("\n--- Individual Validator Scores ---")
    for v in result.validators:
        status = "✓" if v.is_valid else "✗"
        print(f"\n{status} {v.provider_name} ({v.model_name})")
        print(f"  Score: {v.score:.2f}")
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Explanation: {v.explanation[:100]}...")


if __name__ == "__main__":
    main()
```
