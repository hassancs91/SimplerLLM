# LLM Brainstorm - Manual vs Automated

## What is LLM Brainstorm?
A technique to generate ideas recursively - you generate initial ideas, then expand on each one to create sub-ideas, going deeper and deeper.

---

## MANUAL APPROACH (ChatGPT/Claude UI)

### Step 1: Generate Initial Ideas
**Prompt:**
```
Generate 3 YouTube video ideas for a tech channel focused on AI tools.

For each idea, provide:
- The video title
- Brief description
- Why it would work
```

**Example Response:**
```
1. "5 AI Tools That Will Replace Your Entire Workflow"
   - Showcase productivity AI tools
   - Works because people want efficiency

2. "I Built an App in 24 Hours Using Only AI"
   - Challenge format, build something with AI coding tools
   - Works because of entertainment + education value

3. "The Truth About AI Replacing Jobs in 2025"
   - Analysis of which jobs are actually at risk
   - Works because of fear/curiosity factor
```

### Step 2: Evaluate Each Idea (NEW PROMPT)
**Prompt:**
```
Rate these 3 video ideas from 1-10 on:
- Viral potential
- Ease to produce
- Value to viewers

Ideas:
1. "5 AI Tools That Will Replace Your Entire Workflow"
2. "I Built an App in 24 Hours Using Only AI"
3. "The Truth About AI Replacing Jobs in 2025"
```

**Example Response:**
```
1. AI Tools video: 7/10 (common topic, moderate effort)
2. 24-Hour Build: 9/10 (unique, high effort but engaging)
3. Jobs Analysis: 6/10 (clickbait-y, hard to be original)
```

### Step 3: Expand the Best Idea (NEW PROMPT)
**Prompt:**
```
Take this video idea and generate 3 specific variations or angles:

"I Built an App in 24 Hours Using Only AI"

Make each variation unique and more specific.
```

**Example Response:**
```
1. "I Built a $1000/Month SaaS in 24 Hours Using Only AI"
   - Focus on monetizable product

2. "Can AI Build a Better App Than Me? 24-Hour Challenge"
   - AI vs Human angle, more dramatic

3. "I Let ChatGPT Control My Entire Coding Session"
   - Full surrender to AI, entertaining format
```

### Step 4: Evaluate Again... (NEW PROMPT)
And you keep going... manually... for each branch...

---

## THE PROBLEM

To properly brainstorm with 3 ideas, 2 levels deep:
- Prompt 1: Generate 3 initial ideas
- Prompt 2: Evaluate all 3
- Prompt 3: Expand idea 1 → 3 sub-ideas
- Prompt 4: Expand idea 2 → 3 sub-ideas
- Prompt 5: Expand idea 3 → 3 sub-ideas
- Prompt 6-8: Evaluate each batch...
- And so on...

**That's 8+ manual prompts just for a basic brainstorm!**

And you have to:
- Copy/paste between prompts
- Keep track of which ideas were best
- Remember the tree structure
- Manually organize everything

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

brainstorm = RecursiveBrainstorm(
    llm=llm,
    max_depth=2,           # Go 2 levels deep
    ideas_per_level=3,     # 3 ideas at each level
    mode="tree",           # Expand ALL ideas
    evaluation_criteria=["viral_potential", "ease_to_produce", "value"],
    verbose=True,
)

result = brainstorm.brainstorm(
    prompt="YouTube video ideas for a tech channel focused on AI tools"
)

# Get results
print(f"Total ideas generated: {result.total_ideas}")
print(f"Best overall idea: {result.overall_best_idea.text}")
print(f"Score: {result.overall_best_idea.quality_score}/10")
```

**Output:**
```
Total ideas generated: 12  (3 initial + 9 expansions)
Best overall idea: "I Built a $1000/Month SaaS in 24 Hours Using Only AI"
Score: 8.5/10
```

---

## 3 MODES EXPLAINED

### Tree Mode (Expand Everything)
```
Initial Ideas → ALL get expanded → ALL get expanded again
     ├── Idea 1 → Sub-idea 1.1, 1.2, 1.3
     ├── Idea 2 → Sub-idea 2.1, 2.2, 2.3
     └── Idea 3 → Sub-idea 3.1, 3.2, 3.3
```
**Use when:** You want maximum exploration

### Linear Mode (Focus on Best)
```
Initial Ideas → Pick BEST → Expand BEST only → Pick BEST → Expand...
     ├── Idea 1 (best, score 9) → Sub-idea 1.1 (best) → Final refinement
     ├── Idea 2 (score 7) ✗
     └── Idea 3 (score 6) ✗
```
**Use when:** You want to deeply refine ONE idea

### Hybrid Mode (Top N)
```
Initial Ideas → Pick TOP 2 → Expand those → Pick TOP 2 → ...
     ├── Idea 1 (score 9) → Sub-ideas...
     ├── Idea 2 (score 8) → Sub-ideas...
     └── Idea 3 (score 5) ✗
```
**Use when:** Balance between exploration and focus

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| 8+ prompts | 1 function call |
| Copy/paste chaos | Automatic tracking |
| Lose track of tree | Full tree structure |
| Manual scoring | Auto-evaluation |
| No persistence | Save to CSV |
| 30+ minutes | 30 seconds |

---

## SIMPLE DEMO SCRIPT

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm

# Setup
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
brainstorm = RecursiveBrainstorm(llm=llm, max_depth=2, ideas_per_level=3, verbose=True)

# Run
result = brainstorm.brainstorm("YouTube video ideas for a tech channel focused on AI")

# Results
print(f"\n{'='*50}")
print(f"Generated {result.total_ideas} ideas in {result.execution_time:.1f}s")
print(f"\nBest idea: {result.overall_best_idea.text}")
print(f"Score: {result.overall_best_idea.quality_score}/10")

# Show tree
print("\nIdea Tree:")
for level in result.levels:
    print(f"\nDepth {level.depth}:")
    for idea in level.ideas:
        parent = f" (from: {idea.parent_id})" if idea.parent_id else ""
        print(f"  [{idea.quality_score:.1f}] {idea.text[:60]}...{parent}")
```
