from SimplerLLM.language.llm_router import LLMRouter
from SimplerLLM.language.llm import LLM, LLMProvider

def main():
    # Initialize LLM
    llm_instance = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        verbose=False
    )

    # Initialize router
    router = LLMRouter(llm_instance=llm_instance)

    # Add template choices
    template_1 = """
    Most [***Topic People] spend their entire careers striving for [Goal*].**
    But here's something they don't teach you in school:
    Achieving [***Goal*] is way easier than you think.**
    Here's how to save yourself a decade and unlock [***Outcome*] this year:
    """

    template_2 = """
    They say it takes 10 years to master a craft.**
    But using this framework, I mastered [***Topic*] in 1/10th the time.**
    [***Step #1*]**
    [***Step #2*]**
    [***Step #3*]**
    [***Step #4*]**
    """

    template_3 = """
    Most people think learning [***Topic*] takes 10,000 hours.**
    But I can explain it to you in 30 seconds.
    A quick breakdown on:
    [***Main Point #1*]**
    [***Main Point #2*]**
    [***Main Point #3*]**
    """

    # Test bulk choice management
    print("Testing choice management:\n")
    
    # Add choices in bulk
    choices = [
        (template_1, {"type": "tweet", "style": "achievement"}),
        (template_2, {"type": "thread", "style": "educational"}),
        (template_3, {"type": "thread", "style": "quick-tips"})
    ]
    indices = router.add_choices(choices)
    print("Added choices with indices:", indices)

    # Get all choices
    print("\nAll choices:")
    print("-" * 50)
    all_choices = router.get_choices()
    for i, (content, metadata) in enumerate(all_choices):
        print(f"\nChoice {i+1}:")
        print(f"Type: {metadata.get('type')}")
        print(f"Style: {metadata.get('style')}")
        print(f"Content preview: {content[:100]}...")

    # Get specific choice
    print("\nGetting specific choice:")
    print("-" * 50)
    choice = router.get_choice(1)  # Get second choice
    if choice:
        content, metadata = choice
        print(f"Choice 2 type: {metadata.get('type')}")
        print(f"Choice 2 style: {metadata.get('style')}")
        print(f"Content preview: {content[:100]}...")

    print("\nTesting metadata filtering:\n")

    # Test 1: Filter by type
    print("\nTest 1: Finding tweet templates")
    print("-" * 50)
    result = router.route_with_metadata(
        "I want to create a tweet about my success story",
        metadata_filter={"type": "tweet"}
    )
    if result:
        print(f"Found tweet template (index {result.selected_index + 1}):")
        print(f"Confidence: {result.confidence_score}")
        print(f"Reasoning: {result.reasoning}")

    # Test 2: Filter threads by style
    print("\nTest 2: Finding educational thread templates")
    print("-" * 50)
    result = router.route_with_metadata(
        "Need to explain a complex topic step by step",
        metadata_filter={"type": "thread", "style": "educational"}
    )
    if result:
        print(f"Found educational thread template (index {result.selected_index + 1}):")
        print(f"Confidence: {result.confidence_score}")
        print(f"Reasoning: {result.reasoning}")

    # Test 3: Get top 2 thread templates
    print("\nTest 3: Top 2 thread templates")
    print("-" * 50)
    results = router.route_top_k_with_metadata(
        "I want to share some quick tips about productivity",
        metadata_filter={"type": "thread"},
        k=2
    )
    for i, result in enumerate(results):
        print(f"\nThread template {i+1} (index {result.selected_index + 1}):")
        print(f"Confidence: {result.confidence_score}")
        print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    main()
