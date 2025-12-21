"""
LLM Validator Demo for Video
Multiple AI models validate content accuracy
"""

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_validator import LLMValidator


def main():
    print("\n" + "=" * 60)
    print("LLM VALIDATOR DEMO")
    print("Multiple AI models checking content accuracy")
    print("=" * 60)

    # Create validators (multiple perspectives)
    print("\nSetting up validators: GPT-4o + Claude...")
    validators = [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514"),
    ]

    validator = LLMValidator(
        validators=validators,
        parallel=True,
        default_threshold=0.7,
        verbose=True,
    )

    # Content with a deliberate error (Python 4.0 doesn't exist)
    content = """
    Python was created by Guido van Rossum and first released in 1991.
    It is named after Monty Python's Flying Circus. Python is known for
    its simple syntax and is the most popular programming language for
    AI and machine learning. The latest version is Python 4.0.
    """

    print("\n" + "-" * 60)
    print("CONTENT TO VALIDATE:")
    print("-" * 60)
    print(content)
    print("-" * 60)
    print("(Note: There's a factual error - Python 4.0 doesn't exist!)")

    # Validate
    print("\nRunning validation with multiple models...")
    result = validator.validate(
        content=content,
        validation_prompt="Check all factual claims for accuracy. Pay attention to version numbers, dates, and attributions.",
        original_question="Tell me about Python programming language",
    )

    # Results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Overall metrics
    print(f"\nOverall Score: {result.overall_score:.2f}/1.0")
    print(f"Overall Confidence: {result.overall_confidence:.2f}")
    print(f"Is Valid (>= 0.7): {'Yes ✓' if result.is_valid else 'No ✗'}")
    print(f"Validators Agree: {'Yes ✓' if result.consensus else 'No ✗'}")
    print(f"Consensus Details: {result.consensus_details}")

    # Individual scores
    print("\n" + "-" * 60)
    print("INDIVIDUAL VALIDATOR SCORES")
    print("-" * 60)

    for v in result.validators:
        status = "✓ PASS" if v.is_valid else "✗ FAIL"
        print(f"\n{v.provider_name} ({v.model_name}): {status}")
        print(f"  Score: {v.score:.2f}")
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Time: {v.execution_time:.2f}s")
        print(f"  Explanation:")

        # Word wrap the explanation
        explanation = v.explanation
        words = explanation.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal execution time: {result.total_execution_time:.2f}s")
    print(f"Aggregation method: {result.aggregation_method.value}")

    if result.is_valid:
        print("\n✓ Content PASSED validation (but check the explanations!)")
    else:
        print("\n✗ Content FAILED validation - needs revision")


if __name__ == "__main__":
    main()
