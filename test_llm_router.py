from SimplerLLM.language.llm_router import LLMRouter
from SimplerLLM.language.llm import LLM, LLMProvider

def main():
    # Initialize LLM
    llm_instance = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        verbose=True
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

    # Add choices to router
    router.add_choice(template_1, metadata={"type": "tweet"})
    router.add_choice(template_2, metadata={"type": "thread"})
    router.add_choice(template_3, metadata={"type": "thread"})

    # Test inputs
    inputs = [
        "I want to create a tweet about my top marketing tips",
        "Need a thread format to explain complex coding concepts",
        "Looking for a quick way to share productivity hacks"
    ]


    #result = router.route(inputs[0])
    #print(result)


    results = router.route_top_k(inputs[0], k=3)
        
        
    for i, result in enumerate(results):
            print(f"\nMatch {i+1}:")
            print(f"Template: {result.selected_index + 1}")
            print(f"Confidence: {result.confidence_score}")
            print(f"Reasoning: {result.reasoning}")

    #print("Testing route_top_k with different inputs:\n")
    # for input_text in inputs:
    #     print(f"\nInput: {input_text}")
    #     print("-" * 50)
        
    #     # Get top 3 matches
    #     results = router.route_top_k(input_text, k=3)
        
    #     # Print results
    #     for i, result in enumerate(results):
    #         print(f"\nMatch {i+1}:")
    #         print(f"Template: {result.selected_index + 1}")
    #         print(f"Confidence: {result.confidence_score}")
    #         print(f"Reasoning: {result.reasoning}")
    #     print("-" * 50)

if __name__ == "__main__":
    main()
