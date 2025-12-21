# Semantic Routing - Manual vs Automated

## What is Semantic Routing?
Using an LLM to intelligently choose the right path/handler/option based on the *meaning* of a request, not just keywords.

Think of it as a smart receptionist who understands what you need and directs you to the right department.

---

## USE CASES

1. **Chatbot routing**: Route user questions to the right agent (sales, support, billing)
2. **Content classification**: Categorize articles, tickets, emails
3. **Workflow automation**: Decide which process to trigger
4. **Multi-skill agents**: Choose which tool/skill to use
5. **Dynamic prompts**: Select the right prompt template for the task

---

## MANUAL APPROACH (ChatGPT/Claude UI)

### Scenario: Customer Support Router
You have 4 departments: Sales, Technical Support, Billing, General

### Step 1: For EACH incoming message, ask the LLM
**Prompt:**
```
Classify this customer message into one of these categories:
- sales: Questions about buying, pricing, plans
- technical: Problems with product, bugs, how-to
- billing: Payment issues, invoices, refunds
- general: Everything else

Customer message: "I can't login to my account and I need to download my invoice"

Which category? Explain your reasoning.
```

**Response:**
```
Category: billing

Reasoning: While "can't login" could be technical, the primary need is
downloading an invoice which is billing-related. The login issue is
secondary to the billing task.
```

### Step 2: Route to the appropriate handler
Now you manually take that response and route accordingly...

### Step 3: Repeat for EVERY message
- New message → New prompt → Read response → Route manually

---

## THE PROBLEM

For a chatbot handling 100 messages:
- 100 classification prompts
- Manually parse each response
- Copy/paste routing logic
- No caching (same questions reclassified)
- Inconsistent format handling

**It's not scalable!**

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_provider_router import QueryClassifier

llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

# Define your routes
classifier = QueryClassifier(
    classifier_llm=llm,
    method="hybrid",  # Try patterns first, LLM for complex cases
    enable_cache=True,  # Don't reclassify identical queries
    custom_patterns={
        "sales": [r"pricing", r"buy", r"upgrade", r"plans?", r"demo"],
        "technical": [r"bug", r"error", r"not working", r"crash", r"how to"],
        "billing": [r"invoice", r"payment", r"refund", r"charge", r"receipt"],
    }
)

# Route a message
result = classifier.classify(
    "I can't login to my account and I need to download my invoice"
)

print(f"Route to: {result.query_type}")    # "billing"
print(f"Confidence: {result.confidence}")   # 0.85
print(f"Reasoning: {result.reasoning}")
```

---

## HOW IT WORKS

```
User Query
    ↓
┌─────────────────────────────────────────┐
│           SEMANTIC ROUTER               │
├─────────────────────────────────────────┤
│ 1. Check cache (seen this before?)      │
│ 2. Try pattern matching (fast)          │
│ 3. If unsure → Ask LLM (accurate)       │
│ 4. Return: category + confidence        │
└─────────────────────────────────────────┘
    ↓
Route to correct handler
```

### Three Methods:

**Pattern Only** (Fast but rigid)
```python
classifier = QueryClassifier(method="pattern", ...)
```
- Uses regex patterns
- Instant classification
- Misses nuanced queries

**LLM Only** (Accurate but slow)
```python
classifier = QueryClassifier(method="llm", ...)
```
- LLM classifies every query
- Understands context and nuance
- Slower, uses API calls

**Hybrid** (Best of both)
```python
classifier = QueryClassifier(method="hybrid", ...)
```
- Try patterns first
- If low confidence → use LLM
- Fast for obvious cases, accurate for tricky ones

---

## REAL EXAMPLES

### Example 1: Simple (Pattern matches)
```
Query: "How much does the pro plan cost?"
Method: Pattern matched "cost" → "sales"
Confidence: 0.85
Time: <1ms
```

### Example 2: Ambiguous (LLM needed)
```
Query: "I was charged twice but the feature still doesn't work"
Method: LLM classification
Result: "billing" (primary issue is the charge)
Confidence: 0.75
Reasoning: "While there's a technical component, the double charge
           is the actionable issue requiring billing intervention"
```

### Example 3: Cached
```
Query: "How much does the pro plan cost?" (asked again)
Method: Cache hit
Time: <1ms
```

---

## BUILDING A COMPLETE ROUTER

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_provider_router import QueryClassifier

# Setup
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

classifier = QueryClassifier(
    classifier_llm=llm,
    method="hybrid",
    enable_cache=True,
    custom_patterns={
        "sales": [r"price", r"cost", r"buy", r"subscribe"],
        "technical": [r"error", r"bug", r"broken", r"help me"],
        "billing": [r"invoice", r"refund", r"payment"],
        "general": [],  # Catch-all
    }
)

# Define handlers
def handle_sales(query):
    return "Connecting you to sales team..."

def handle_technical(query):
    return "Creating a support ticket..."

def handle_billing(query):
    return "Pulling up your billing history..."

def handle_general(query):
    return "How can I help you today?"

# Route handler map
handlers = {
    "sales": handle_sales,
    "technical": handle_technical,
    "billing": handle_billing,
    "general": handle_general,
}

# Process incoming messages
def process_message(user_message):
    # Classify
    classification = classifier.classify(user_message)

    # Route
    handler = handlers.get(classification.query_type, handle_general)
    response = handler(user_message)

    return {
        "response": response,
        "routed_to": classification.query_type,
        "confidence": classification.confidence,
    }

# Example
result = process_message("I need a refund for my last purchase")
print(result)
# {'response': 'Pulling up your billing history...',
#  'routed_to': 'billing',
#  'confidence': 0.85}
```

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| Prompt per message | One setup, infinite queries |
| Parse responses manually | Structured classification |
| No caching | Built-in caching |
| Slow for every query | Fast pattern + smart LLM |
| Copy/paste routing | Direct function routing |

---

## SIMPLE DEMO SCRIPT

```python
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_provider_router import QueryClassifier

# Setup
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

classifier = QueryClassifier(
    classifier_llm=llm,
    method="hybrid",
    enable_cache=True,
    verbose=True,
)

# Test queries
test_queries = [
    "How much does the enterprise plan cost?",
    "My app keeps crashing when I click save",
    "I was charged twice last month",
    "Can you help me understand how the API works?",
    "I want to cancel my subscription and get a refund",
]

print("SEMANTIC ROUTING DEMO")
print("=" * 50)

for query in test_queries:
    result = classifier.classify(query)
    print(f"\nQuery: {query[:50]}...")
    print(f"  → Route: {result.query_type}")
    print(f"  → Confidence: {result.confidence:.0%}")
    print(f"  → Method: {result.matched_by}")
```
