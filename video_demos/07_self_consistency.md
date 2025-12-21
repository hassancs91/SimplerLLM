# Self-Consistency - Manual vs Automated

## What is Self-Consistency?
Ask the same question multiple times, collect all answers, and pick the one that appears most often (majority voting).

The idea: If the LLM arrives at the same answer through different reasoning paths, that answer is more likely to be correct.

---

## WHY USE SELF-CONSISTENCY?

LLMs can be inconsistent:
- Same question → Different answers each time
- Sometimes right, sometimes wrong
- Temperature adds randomness

**Solution**: Ask multiple times, trust the consensus!

Best for:
- Math problems
- Logic puzzles
- Factual questions
- Any question with a definitive answer

---

## MANUAL APPROACH (ChatGPT UI)

### Scenario: Solve a Tricky Math Problem

**Question:** "What is 17 × 24?"

### Step 1: Ask Once
**Response 1:**
```
Let me calculate: 17 × 24
17 × 20 = 340
17 × 4 = 68
340 + 68 = 408

Answer: 408
```

### Step 2: Ask Again (New Chat or "Regenerate")
**Response 2:**
```
17 × 24:
= 17 × (25 - 1)
= 17 × 25 - 17
= 425 - 17
= 408

Answer: 408
```

### Step 3: Ask Again
**Response 3:**
```
17 × 24 = 17 × 24 = 408

Answer: 408
```

### Step 4: Ask Again
**Response 4:**
```
Let's break it down:
17 × 24
= (20 - 3) × 24
= 480 - 72
= 408

Answer: 408
```

### Step 5: Ask Again
**Response 5:**
```
17 × 24 = 412

Wait, let me recalculate...
17 × 24 = 408

Answer: 408
```

### Step 6: Count Manually
- 408: ✓✓✓✓✓ (5 times)
- 412: mentioned once but corrected

**Consensus: 408** (100% after correction)

---

## THE PROBLEMS

1. **Click "Regenerate" 5+ times**: Tedious
2. **Copy each answer**: Manual tracking
3. **Count occurrences**: Spreadsheet or pen & paper
4. **Extract the actual answer**: Parse through reasoning
5. **Handle different formats**: "408", "The answer is 408", "= 408"
6. **Can't automate**: Manual process every time

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_consistency import SelfConsistency

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

consistency = SelfConsistency(
    llm=llm,
    num_samples=5,      # Ask 5 times
    temperature=0.7,    # Some randomness for diversity
    parallel=True,      # All at once
)

result = consistency.generate(
    prompt="What is 17 × 24? Think step by step.",
    system_prompt="Solve the problem. End with 'Answer: X'"
)

print(f"Answer: {result.final_answer}")      # "408"
print(f"Confidence: {result.confidence:.0%}") # "100%"
print(f"Agreement: {result.num_agreeing}/{result.num_samples}")
```

**Output:**
```
Answer: 408
Confidence: 100%
Agreement: 5/5
```

---

## HOW IT WORKS

```
Same Prompt (with temperature > 0)
        │
        ├──▶ Sample 1: "408"
        ├──▶ Sample 2: "408"
        ├──▶ Sample 3: "408"
        ├──▶ Sample 4: "412"  ← outlier
        └──▶ Sample 5: "408"
                │
                ▼
        ┌─────────────────┐
        │ ANSWER GROUPING │
        │                 │
        │ "408": 4 votes  │ ◀── Winner!
        │ "412": 1 vote   │
        └────────┬────────┘
                 │
                 ▼
        ConsistencyResult {
            final_answer: "408",
            confidence: 0.80,  (4/5)
            is_tie: false
        }
```

---

## TWO COMPARISON MODES

### EXACT Mode (Numbers, Booleans, Short Answers)
For answers that should match exactly.

```python
# Auto-detected for numeric/short answers
result = consistency.generate(
    prompt="What is 17 × 24?",
)
# Compares: "408" == "408" ✓
```

### SEMANTIC Mode (Text, Explanations)
For longer answers where meaning matters, not exact words.

```python
result = consistency.generate(
    prompt="Why is the sky blue?",
    answer_type=AnswerType.SEMANTIC,
)
# Uses LLM to check if answers mean the same thing
```

**Auto-Detection:**
- Numbers, bools, short answers → EXACT
- Longer text → SEMANTIC

---

## HANDLING TIES

What if two answers are equally popular?

```python
result = consistency.generate(prompt="...")

if result.is_tie:
    print(f"Tie between: {result.tied_answers}")
    # ["Paris", "Lyon"] - both got 2 votes
else:
    print(f"Winner: {result.final_answer}")
```

You decide how to handle ties in your application!

---

## REAL EXAMPLE: Logic Puzzle

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_consistency import SelfConsistency

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

consistency = SelfConsistency(
    llm=llm,
    num_samples=7,      # More samples for tricky questions
    temperature=0.9,    # Higher diversity
    verbose=True,
)

# Tricky logic puzzle
result = consistency.generate(
    prompt="""A farmer has 17 sheep. All but 9 run away.
    How many sheep does the farmer have left?

    Think carefully before answering.""",
)

print(f"\nAnswer: {result.final_answer}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Correct answer: 9 (it's a trick question!)")

# Show the vote distribution
print("\nVote Distribution:")
for group in result.answer_groups:
    print(f"  '{group.answer}': {group.count} votes ({group.percentage:.0f}%)")
```

**Output:**
```
Answer: 9
Confidence: 100%
Correct answer: 9 (it's a trick question!)

Vote Distribution:
  '9': 7 votes (100%)
```

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| Click regenerate 5x | 1 function call |
| Copy each response | Automatic collection |
| Extract answers manually | Smart extraction |
| Count in spreadsheet | Automatic grouping |
| Handle ties yourself | Tie detection built-in |
| 5+ minutes | Seconds |

---

## WHEN TO USE SELF-CONSISTENCY

✅ **Good for:**
- Math problems
- Logic puzzles
- Factual questions
- Multiple choice
- Yes/No questions
- Any question with a "right" answer

❌ **Not ideal for:**
- Creative writing (no "correct" answer)
- Opinion questions
- Open-ended exploration

---

## SIMPLE DEMO SCRIPT

```python
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

    # Test 1: Math
    print("\n--- TEST 1: Math Problem ---")
    result = consistency.generate(
        prompt="What is 17 × 24? Show your work.",
    )
    print(f"\nAnswer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Agreement: {result.num_agreeing}/{result.num_samples}")

    # Test 2: Logic Puzzle
    print("\n--- TEST 2: Logic Puzzle ---")
    result = consistency.generate(
        prompt="A farmer has 17 sheep. All but 9 run away. How many are left?",
    )
    print(f"\nAnswer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"(Correct answer is 9 - it's a trick question!)")

    # Show groups
    print("\nAnswer Groups:")
    for group in result.answer_groups:
        print(f"  '{group.answer}': {group.count} votes ({group.percentage:.0f}%)")

    if result.is_tie:
        print(f"\n⚠️  TIE detected between: {result.tied_answers}")


if __name__ == "__main__":
    main()
```
