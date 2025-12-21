# Web Search + JSON Output - Manual vs Automated

## What is This?
Combining two powerful techniques:
1. **Web Search**: Get real-time, up-to-date information from the internet
2. **Structured JSON Output**: Force the LLM to return data in a specific, usable format

Perfect for: news aggregation, price monitoring, research automation, data collection.

---

## THE GOAL: Get Latest AI News

We want to get the latest AI news and have it in a structured format we can use in our app (not just raw text).

**Desired Output:**
```json
{
  "news": [
    {
      "title": "OpenAI Releases GPT-5",
      "source": "TechCrunch",
      "date": "2024-12-20",
      "summary": "OpenAI announced...",
      "url": "https://..."
    },
    ...
  ]
}
```

---

## MANUAL APPROACH (ChatGPT/Claude UI)

### Step 1: Search and Gather Information
**Prompt:**
```
Search the web for the latest AI news from the past week.
Find the top 5 most important stories.
```

**Response:**
```
Here are the latest AI news stories:

1. OpenAI has released a new update to ChatGPT that includes...
   According to TechCrunch, this update focuses on...

2. Google DeepMind announced a breakthrough in...
   The Verge reports that scientists are excited about...

3. ...
```

### Step 2: Ask for Structured Format (NEW PROMPT)
**Prompt:**
```
Take that news and format it as JSON with this structure:
{
  "news": [
    {
      "title": "string",
      "source": "string",
      "date": "YYYY-MM-DD",
      "summary": "string",
      "url": "string"
    }
  ]
}
```

**Response:**
```json
{
  "news": [
    {
      "title": "OpenAI Releases ChatGPT Update",
      "source": "TechCrunch",
      "date": "2024-12-20",
      ...
    }
  ]
}
```

### Step 3: Copy & Parse the JSON
Now you manually:
- Copy the JSON from the chat
- Paste into your code/app
- Hope the format is correct
- Parse and use it

---

## THE PROBLEMS

1. **Two-step process**: Search, then format separately
2. **Copy/paste**: Manual data transfer
3. **Format inconsistency**: LLM might return slightly different JSON structure each time
4. **Validation**: No guarantee the JSON is valid or matches your schema
5. **No retry logic**: If format is wrong, start over
6. **Not automatable**: Can't run this in a script/app

---

## AUTOMATED APPROACH (SimplerLLM)

```python
from pydantic import BaseModel
from typing import List
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

# Step 1: Define your data structure
class NewsItem(BaseModel):
    title: str
    source: str
    date: str
    summary: str
    url: str

class AINewsResponse(BaseModel):
    news: List[NewsItem]

# Step 2: Create LLM with web search capability
llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

# Step 3: One call - search web + get structured JSON
result = generate_pydantic_json_model(
    model_class=AINewsResponse,
    prompt="Get the top 5 latest AI news stories from the past week",
    llm_instance=llm,
    web_search=True,  # Enable real-time web search
)

# Step 4: Use the structured data directly
for item in result.news:
    print(f"📰 {item.title}")
    print(f"   Source: {item.source} | Date: {item.date}")
    print(f"   {item.summary[:100]}...")
    print(f"   🔗 {item.url}\n")
```

**Output:**
```
📰 OpenAI Releases GPT-5 with Revolutionary Capabilities
   Source: TechCrunch | Date: 2024-12-20
   OpenAI has announced the release of GPT-5, featuring significant...
   🔗 https://techcrunch.com/2024/12/20/openai-gpt5

📰 Google DeepMind Achieves Breakthrough in Protein Folding
   Source: Nature | Date: 2024-12-19
   DeepMind's AlphaFold 3 has solved previously impossible protein...
   🔗 https://nature.com/articles/deepmind-alphafold3

...
```

---

## HOW IT WORKS

```
┌─────────────────────────────────────────────────────────────┐
│  1. Define Pydantic Model (your desired data structure)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. LLM searches the web (web_search=True)                  │
│     → Gets current, real-time information                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. LLM formats response to match your schema               │
│     → Automatic JSON mode enabled                           │
│     → Validates against Pydantic model                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Retry logic if JSON is invalid (up to 3 attempts)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Return typed Python object (not raw string!)            │
│     → result.news[0].title  ← Direct access                 │
└─────────────────────────────────────────────────────────────┘
```

---

## MORE EXAMPLES

### Example 1: Stock Price Tracker
```python
class StockInfo(BaseModel):
    symbol: str
    price: float
    change_percent: float
    volume: int
    last_updated: str

class StockData(BaseModel):
    stocks: List[StockInfo]

result = generate_pydantic_json_model(
    model_class=StockData,
    prompt="Get current prices for AAPL, GOOGL, MSFT, NVDA",
    llm_instance=llm,
    web_search=True,
)

for stock in result.stocks:
    emoji = "📈" if stock.change_percent > 0 else "📉"
    print(f"{emoji} {stock.symbol}: ${stock.price} ({stock.change_percent:+.2f}%)")
```

### Example 2: Weather Forecast
```python
class DayForecast(BaseModel):
    date: str
    high_temp: int
    low_temp: int
    condition: str  # "sunny", "rainy", etc.
    precipitation_chance: int

class WeatherData(BaseModel):
    location: str
    forecasts: List[DayForecast]

result = generate_pydantic_json_model(
    model_class=WeatherData,
    prompt="Get the 5-day weather forecast for San Francisco",
    llm_instance=llm,
    web_search=True,
)
```

### Example 3: Product Research
```python
class Product(BaseModel):
    name: str
    price: float
    rating: float
    review_count: int
    pros: List[str]
    cons: List[str]

class ProductComparison(BaseModel):
    query: str
    products: List[Product]
    recommendation: str

result = generate_pydantic_json_model(
    model_class=ProductComparison,
    prompt="Compare the top 3 noise-canceling headphones under $300",
    llm_instance=llm,
    web_search=True,
)
```

---

## KEY BENEFITS

| Manual | Automated |
|--------|-----------|
| 2+ prompts | 1 function call |
| Copy/paste JSON | Direct Python object |
| Hope format is right | Pydantic validation |
| No retry on error | Automatic retry (3x) |
| Can't automate | Fully scriptable |
| Static info | Real-time web search |

---

## SIMPLE DEMO SCRIPT

```python
"""
Web Search + JSON Output Demo
Get latest AI news in structured format
"""

from pydantic import BaseModel
from typing import List
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model


# Define our data structure
class NewsItem(BaseModel):
    title: str
    source: str
    date: str
    summary: str
    category: str  # "breakthrough", "product", "research", "business"


class AINewsResponse(BaseModel):
    news: List[NewsItem]
    search_date: str


def main():
    print("\n" + "=" * 60)
    print("WEB SEARCH + JSON OUTPUT DEMO")
    print("Getting Latest AI News...")
    print("=" * 60)

    # Create LLM
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    # Fetch and structure news
    result = generate_pydantic_json_model(
        model_class=AINewsResponse,
        prompt="""Search for the top 5 most important AI news stories from the past week.
        Include breakthroughs, product launches, research papers, and business news.
        Get the actual current news, not examples.""",
        llm_instance=llm,
        web_search=True,
    )

    # Display results
    print(f"\nSearch Date: {result.search_date}")
    print(f"Found {len(result.news)} stories\n")

    for i, item in enumerate(result.news, 1):
        category_emoji = {
            "breakthrough": "🔬",
            "product": "📦",
            "research": "📚",
            "business": "💼",
        }.get(item.category.lower(), "📰")

        print(f"{category_emoji} [{i}] {item.title}")
        print(f"    Source: {item.source} | Date: {item.date}")
        print(f"    {item.summary}")
        print()


if __name__ == "__main__":
    main()
```
