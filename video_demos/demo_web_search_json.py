"""
Web Search + JSON Output Demo for Video
Get latest AI news in structured format
"""

from pydantic import BaseModel, Field
from typing import List
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model


# Define our data structure
class NewsItem(BaseModel):
    title: str = Field(description="The headline of the news article")
    source: str = Field(description="The publication or website name")
    date: str = Field(description="Publication date in YYYY-MM-DD format")
    summary: str = Field(description="A 1-2 sentence summary of the news")
    category: str = Field(description="One of: breakthrough, product, research, business")


class AINewsResponse(BaseModel):
    news: List[NewsItem] = Field(description="List of AI news items")
    search_date: str = Field(description="Today's date when search was performed")


def main():
    print("\n" + "=" * 60)
    print("WEB SEARCH + JSON OUTPUT DEMO")
    print("Getting Latest AI News...")
    print("=" * 60)

    # Create LLM
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    # Fetch and structure news
    print("\nSearching the web for latest AI news...\n")

    result = generate_pydantic_json_model(
        model_class=AINewsResponse,
        prompt="""Search for the top 5 most important AI news stories from the past week.

        Include a mix of:
        - Breakthroughs (new capabilities, scientific discoveries)
        - Products (new AI tools, updates, launches)
        - Research (papers, studies, findings)
        - Business (funding, acquisitions, partnerships)

        Get actual current news with real sources and dates.""",
        llm_instance=llm,
        web_search=True,
        temperature=0.3,  # Lower for more factual output
    )

    # Check if we got a valid result
    if isinstance(result, str):
        print(f"Error: {result}")
        return

    # Display results
    print(f"Search Date: {result.search_date}")
    print(f"Found {len(result.news)} stories")
    print("=" * 60)

    category_emojis = {
        "breakthrough": "🔬",
        "product": "📦",
        "research": "📚",
        "business": "💼",
    }

    for i, item in enumerate(result.news, 1):
        emoji = category_emojis.get(item.category.lower(), "📰")

        print(f"\n{emoji} [{i}] {item.title}")
        print(f"    Source: {item.source}")
        print(f"    Date: {item.date}")
        print(f"    Category: {item.category}")
        print(f"    Summary: {item.summary}")

    # Show the structured data aspect
    print("\n" + "=" * 60)
    print("STRUCTURED DATA ACCESS")
    print("=" * 60)
    print("\nYou can access data programmatically:")
    print(f"  result.news[0].title  → \"{result.news[0].title[:40]}...\"")
    print(f"  result.news[0].source → \"{result.news[0].source}\"")
    print(f"  result.news[0].date   → \"{result.news[0].date}\"")
    print(f"  len(result.news)      → {len(result.news)}")

    # Example: Filter by category
    print("\n" + "=" * 60)
    print("FILTERING EXAMPLE")
    print("=" * 60)

    breakthroughs = [n for n in result.news if n.category.lower() == "breakthrough"]
    products = [n for n in result.news if n.category.lower() == "product"]

    print(f"\nBreakthroughs found: {len(breakthroughs)}")
    print(f"Products found: {len(products)}")


if __name__ == "__main__":
    main()
