# LLM Feedback Loop - Manual vs Automated

## What is LLM Feedback?
An iterative improvement system where the LLM critiques its own answer and refines it until it reaches quality standards. Like having a built-in editor that keeps improving your work.

---

## MANUAL APPROACH (ChatGPT/Claude UI)

### Step 1: Generate Initial Answer
**Prompt:**
```
Explain what an API is to a beginner with no programming experience.
```

**Response:**
```
An API (Application Programming Interface) is a set of protocols and tools
that allows different software applications to communicate with each other.
It defines the methods and data formats that applications can use to request
and exchange information...
```

### Step 2: Critique the Answer (NEW PROMPT)
**Prompt:**
```
You are an expert critic. Evaluate this explanation:

ANSWER TO EVALUATE:
"An API (Application Programming Interface) is a set of protocols and tools
that allows different software applications to communicate with each other..."

Rate it 1-10 on:
- Clarity for beginners
- Use of jargon
- Helpfulness of examples

Provide:
- Strengths
- Weaknesses
- Specific improvement suggestions
```

**Response:**
```
Score: 5/10

Strengths:
- Technically accurate
- Mentions communication between apps

Weaknesses:
- Uses jargon ("protocols", "data formats")
- No real-world examples
- Too abstract for beginners

Suggestions:
- Use a restaurant analogy (waiter as API)
- Remove technical terms
- Add concrete example
```

### Step 3: Generate Improved Version (NEW PROMPT)
**Prompt:**
```
Improve this explanation based on the feedback:

ORIGINAL ANSWER:
"An API is a set of protocols and tools..."

FEEDBACK:
- Use a restaurant analogy
- Remove technical terms
- Add concrete example

Write a better version for beginners.
```

**Response:**
```
Imagine you're at a restaurant. You don't go into the kitchen to cook -
you tell the waiter what you want, and they bring it to you.

An API works the same way. When you use an app like Uber, the app doesn't
know where all the cars are. It asks Google Maps (through an API)
"where are cars near this location?" and gets an answer back.

The API is like the waiter - it takes your request to the kitchen (another
service) and brings back what you need.
```

### Step 4: Critique Again... (NEW PROMPT)
```
Rate this improved explanation 1-10...
(repeat the whole process)
```

### Step 5: Improve Again... (NEW PROMPT)
```
Based on the new feedback, improve...
```

---

## THE PROBLEM

For 3 iterations of improvement:
- Prompt 1: Generate initial answer
- Prompt 2: Critique answer 1
- Prompt 3: Improve based on critique
- Prompt 4: Critique answer 2
- Prompt 5: Improve based on critique
- Prompt 6: Critique answer 3 (final evaluation)

**That's 6 prompts minimum!**

And you have to:
- Copy/paste answers between prompts
- Track scores manually
- Decide when to stop
- Format critique requests consistently
- Remember all the feedback

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_feedback import LLMFeedbackLoop

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=3,
    quality_threshold=8.0,  # Stop if score reaches 8/10
)

result = feedback.improve(
    prompt="Explain what an API is to a beginner with no programming experience",
    focus_on=["clarity", "beginner-friendly", "real-world examples"]
)

print(f"Initial score: {result.initial_score}/10")
print(f"Final score: {result.final_score}/10")
print(f"Iterations: {result.total_iterations}")
print(f"\nFinal answer:\n{result.final_answer}")
```

**Output:**
```
Initial score: 5.0/10
Final score: 8.5/10
Iterations: 3
Stopped: quality_threshold_met

Final answer:
Imagine you're at a restaurant...
```

---

## 3 ARCHITECTURES

### 1. Self-Critique (Single LLM)
Same LLM generates AND critiques itself.
```python
feedback = LLMFeedbackLoop(llm=gpt4)
```
```
GPT-4 generates → GPT-4 critiques → GPT-4 improves → repeat
```

### 2. Dual Provider (Generator + Critic)
One LLM writes, another critiques (different perspective).
```python
feedback = LLMFeedbackLoop(
    generator_llm=gpt4,      # Writes answers
    critic_llm=claude,        # Critiques them
)
```
```
GPT-4 generates → Claude critiques → GPT-4 improves → Claude critiques → ...
```

### 3. Multi-Provider Rotation
Multiple LLMs take turns (maximum diversity).
```python
feedback = LLMFeedbackLoop(
    providers=[gpt4, claude, gemini],
    max_iterations=3,
)
```
```
GPT-4 generates → Claude critiques → Gemini improves → GPT-4 critiques → ...
```

---

## STOPPING CRITERIA

The loop automatically stops when:

| Condition | Description |
|-----------|-------------|
| `max_iterations` | Reached iteration limit |
| `quality_threshold` | Score meets target (e.g., 8/10) |
| `convergence` | Answer stopped improving (<10% gain) |
| `text_similarity` | New answer is 95%+ similar to previous |

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| 6+ prompts | 1 function call |
| Track scores yourself | Automatic score tracking |
| Decide when to stop | Smart stopping criteria |
| Copy/paste chaos | Seamless iteration |
| One perspective | Multi-provider options |
| No history | Full improvement trajectory |

---

## SIMPLE DEMO SCRIPT

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_feedback import LLMFeedbackLoop

# Setup
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=3,
    verbose=True,
)

# Run improvement loop
result = feedback.improve(
    prompt="Explain what an API is to a beginner with no programming experience",
    focus_on=["clarity", "beginner-friendly", "examples"]
)

# Show results
print("\n" + "="*50)
print("IMPROVEMENT TRAJECTORY")
print("="*50)

for i, iteration in enumerate(result.all_iterations, 1):
    print(f"\nIteration {i}: {iteration.critique.quality_score}/10")
    print(f"  Strengths: {', '.join(iteration.critique.strengths[:2])}")
    print(f"  Weaknesses: {', '.join(iteration.critique.weaknesses[:2])}")

print(f"\n{'='*50}")
print(f"Started at: {result.initial_score}/10")
print(f"Ended at: {result.final_score}/10")
print(f"Improvement: +{result.final_score - result.initial_score:.1f} points")
print(f"Stopped because: {result.stopped_reason}")
```
