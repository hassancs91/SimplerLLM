"""
Semantic Routing Demo for Video
Shows intelligent query classification and routing
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_provider_router import QueryClassifier


def main():
    print("\n" + "=" * 60)
    print("SEMANTIC ROUTING DEMO")
    print("=" * 60)

    # Create LLM
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

    # Create classifier with custom patterns
    classifier = QueryClassifier(
        classifier_llm=llm,
        method="hybrid",  # Try patterns first, LLM for complex cases
        enable_cache=True,
        custom_patterns={
            "sales": [r"price", r"cost", r"buy", r"upgrade", r"plans?", r"demo", r"subscribe"],
            "technical": [r"error", r"bug", r"broken", r"crash", r"not working", r"how to", r"help"],
            "billing": [r"invoice", r"refund", r"payment", r"charge", r"receipt", r"cancel"],
        },
        verbose=True,
    )

    # Test queries - mix of easy and tricky ones
    test_queries = [
        # Easy - Pattern should match
        "How much does the enterprise plan cost?",
        "My app keeps crashing when I click save",
        "I need a copy of my invoice from last month",

        # Tricky - LLM needed for nuance
        "I was charged twice but the feature still doesn't work",
        "Can you help me understand your pricing before I decide?",
        "I want to cancel my subscription and get a refund",
    ]

    print("\nRouting test queries...\n")

    for query in test_queries:
        result = classifier.classify(query)
        print("-" * 60)
        print(f"Query: \"{query}\"")
        print(f"  Route to: {result.query_type.upper()}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Method: {result.matched_by}")
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning[:80]}...")

    # Show cache stats
    print("\n" + "=" * 60)
    print("CACHE STATS")
    print("=" * 60)
    stats = classifier.get_cache_stats()
    print(f"Cached entries: {stats['total_entries']}")
    print(f"Cache hits: {stats['total_hits']}")

    # Demo: Query the same thing again (should hit cache)
    print("\n" + "=" * 60)
    print("CACHING DEMO - Same query again")
    print("=" * 60)
    result = classifier.classify("How much does the enterprise plan cost?")
    print(f"Query: \"How much does the enterprise plan cost?\"")
    print(f"  Method: {result.matched_by}")  # Should be "cache"


if __name__ == "__main__":
    main()
