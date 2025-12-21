# LLM Judge - Manual vs Automated

## What is LLM Judge?
Ask multiple AI models the same question, then have another AI judge which answer is best (or combine them into a superior answer).

Think of it as a talent competition where multiple contestants perform, and a judge picks the winner or creates a "best of" mashup.

---

## WHY USE MULTIPLE MODELS?

Different LLMs have different strengths:
- **GPT-4**: Great at coding and structured tasks
- **Claude**: Excellent at nuanced writing and reasoning
- **Gemini**: Strong at factual knowledge and research

**Why not get the best of all worlds?**

---

## MANUAL APPROACH (ChatGPT/Claude/Gemini UIs)

### Scenario: Get the Best Explanation of Blockchain

### Step 1: Ask GPT-4
**Prompt:** "Explain blockchain technology in simple terms"

**Response:**
```
Blockchain is like a digital ledger that's shared across many computers.
Each "block" contains transaction data, and once added, it can't be
changed. Think of it as a chain of secure, linked records...
```

### Step 2: Ask Claude (Open new tab)
**Same prompt...**

**Response:**
```
Imagine a Google Doc that everyone can see but no one can edit
past entries. That's essentially blockchain - a shared record book
where new pages are added but old ones are locked forever...
```

### Step 3: Ask Gemini (Open another tab)
**Same prompt...**

**Response:**
```
Blockchain is a decentralized database technology. Each block contains
a cryptographic hash of the previous block, creating an immutable chain.
It's the foundation of cryptocurrencies like Bitcoin...
```

### Step 4: Compare Them Yourself (NEW PROMPT to any model)
```
I have 3 explanations of blockchain. Which is best for beginners?

Response A: "Blockchain is like a digital ledger..."
Response B: "Imagine a Google Doc..."
Response C: "Blockchain is a decentralized database..."

Pick the best one and explain why.
```

**Response:**
```
Response B is best for beginners because it uses a familiar analogy
(Google Doc) that most people understand. Response A is also good but
slightly more abstract. Response C is too technical for beginners.

Winner: Response B (Claude)
```

---

## THE PROBLEMS

1. **3+ browser tabs**: Ask each model separately
2. **Copy/paste chaos**: Collect all responses manually
3. **Formatting nightmare**: Present responses to judge
4. **Time consuming**: 15+ minutes for one question
5. **No synthesis**: Can't easily combine best parts
6. **Can't automate**: Manual process every time

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_judge import LLMJudge

# Create multiple providers (the contestants)
providers = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
    LLM.create(LLMProvider.GEMINI, model_name="gemini-2.0-flash"),
]

# Create the judge (evaluator)
judge_llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

# Create LLM Judge
judge = LLMJudge(
    providers=providers,
    judge_llm=judge_llm,
    parallel=True,  # Ask all providers at once
)

# Get judged answer
result = judge.generate(
    prompt="Explain blockchain technology in simple terms",
    mode="select_best",  # Pick the best answer
)

# Get winner (rank 1)
winner = next((e.provider_name for e in result.evaluations if e.rank == 1), None)
print(f"Winner: {winner}")
print(f"Best Answer: {result.final_answer}")
```

---

## THREE MODES

### Mode 1: Select Best
Pick the single best answer from all providers.

```python
result = judge.generate(prompt, mode="select_best")
```
```
GPT-4 Answer  ──┐
Claude Answer ──┼──▶ Judge ──▶ "Claude's answer is best because..."
Gemini Answer ──┘
```

### Mode 2: Synthesize
Combine the best parts of all answers into one superior response.

```python
result = judge.generate(prompt, mode="synthesize")
```
```
GPT-4 Answer  ──┐
Claude Answer ──┼──▶ Judge ──▶ "Taking the analogy from Claude,
Gemini Answer ──┘              the structure from GPT-4, and
                               the facts from Gemini..."
```

### Mode 3: Compare
Detailed comparison of all answers without picking a winner.

```python
result = judge.generate(prompt, mode="compare")
```
```
GPT-4 Answer  ──┐
Claude Answer ──┼──▶ Judge ──▶ Detailed comparison report
Gemini Answer ──┘              with pros/cons of each
```

---

## HOW IT WORKS

```
                    Your Prompt
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  GPT-4  │    │ Claude  │    │ Gemini  │
   │ Answer  │    │ Answer  │    │ Answer  │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ↓
              ┌─────────────────┐
              │      JUDGE      │
              │   (Another LLM) │
              │                 │
              │ • Evaluates all │
              │ • Scores each   │
              │ • Picks/combines│
              └────────┬────────┘
                       ↓
               ┌──────────────┐
               │ Final Answer │
               │ + Evaluations│
               │ + Reasoning  │
               └──────────────┘
```

---

## REAL EXAMPLE: Best Coding Explanation

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_judge import LLMJudge

# Setup
providers = [
    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
]

judge_llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

judge = LLMJudge(
    providers=providers,
    judge_llm=judge_llm,
    default_criteria=["clarity", "accuracy", "beginner-friendly"],
    verbose=True,
)

# Synthesize best answer
result = judge.generate(
    prompt="Explain recursion in programming with a simple example",
    mode="synthesize",
)

print("=" * 50)
print("SYNTHESIZED ANSWER (Best of all providers)")
print("=" * 50)
print(result.final_answer)

print("\n" + "=" * 50)
print("INDIVIDUAL SCORES")
print("=" * 50)
for eval in result.evaluations:
    print(f"{eval.provider_name}: {eval.overall_score}/10")
    print(f"  Strengths: {', '.join(eval.strengths[:2])}")
```

---

## VALIDATOR vs JUDGE

| LLM Validator | LLM Judge |
|---------------|-----------|
| Validates ONE answer | Gets MULTIPLE answers |
| Multiple models check same content | Multiple models create content |
| "Is this correct?" | "Which is best?" |
| Scores: 0-1 pass/fail | Scores: comparative ranking |
| Output: validation report | Output: best/synthesized answer |

**Use Validator**: When you have content and want to verify it
**Use Judge**: When you want the best possible answer from multiple sources

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| 4+ browser tabs | 1 function call |
| Ask each model separately | Parallel execution |
| Copy/paste to compare | Automatic collection |
| Manual comparison | AI-powered judging |
| Pick one answer | Synthesize best parts |
| 15+ minutes | Seconds |

---

## USE CASES

1. **Important questions**: Get the best possible answer
2. **Creative writing**: Combine different writing styles
3. **Technical explanations**: Best accuracy + clarity combo
4. **Brainstorming**: Multiple perspectives → synthesized best
5. **Quality assurance**: Compare AI outputs objectively

---

## SIMPLE DEMO SCRIPT

```python
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
    judge_llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    judge = LLMJudge(
        providers=providers,
        judge_llm=judge_llm,
        parallel=True,
        verbose=True,
    )

    prompt = "Explain what an API is using a real-world analogy that anyone can understand"

    # Mode 1: Select Best
    print("\n" + "=" * 60)
    print("MODE: SELECT BEST")
    print("=" * 60)

    result = judge.generate(prompt=prompt, mode="select_best")

    # Get winner from evaluations (rank 1)
    winner = next((e.provider_name for e in result.evaluations if e.rank == 1), None)
    print(f"\nWinner: {winner or 'See evaluations'}")
    print(f"\nBest Answer:\n{result.final_answer[:300]}...")

    print("\nScores:")
    for eval in result.evaluations:
        print(f"  {eval.provider_name}: {eval.overall_score}/10")

    # Mode 2: Synthesize
    print("\n" + "=" * 60)
    print("MODE: SYNTHESIZE")
    print("=" * 60)

    result = judge.generate(prompt=prompt, mode="synthesize")

    print(f"\nSynthesized Answer (best of both):\n{result.final_answer[:400]}...")


if __name__ == "__main__":
    main()
```
