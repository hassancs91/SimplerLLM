"""
Test script for SimplerLLM Guardrails functionality.

This script demonstrates and tests the guardrails system.
"""

import os
from SimplerLLM.language.llm.base import LLM, LLMProvider
from SimplerLLM.language.guardrails import (
    GuardrailsLLM,
    GuardrailAction,
    PromptInjectionGuardrail,
    TopicFilterGuardrail,
    InputPIIDetectionGuardrail,
    FormatValidatorGuardrail,
    OutputPIIDetectionGuardrail,
    ContentSafetyGuardrail,
    LengthValidatorGuardrail,
    GuardrailBlockedException,
)

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_basic_guardrails():
    """Test basic guardrails setup and execution."""
    print_section("Test 1: Basic Guardrails Setup")

    # Note: For actual testing, you'd need API keys
    # This demonstrates the structure
    print("Creating guardrailed LLM with:")
    print("- Input: PromptInjectionGuardrail")
    print("- Output: OutputPIIDetectionGuardrail")

    # Simulate guardrail creation
    input_guardrail = PromptInjectionGuardrail(config={
        "safety_rules": "Be helpful and harmless.",
        "position": "prepend"
    })

    output_guardrail = OutputPIIDetectionGuardrail(config={
        "action_on_detect": "redact",
        "pii_types": ["email", "phone"]
    })

    print("[PASS] Guardrails created successfully")
    print(f"  - Input guardrail enabled: {input_guardrail.enabled}")
    print(f"  - Output guardrail enabled: {output_guardrail.enabled}")

def test_input_guardrails():
    """Test input guardrails."""
    print_section("Test 2: Input Guardrails")

    # Test PromptInjectionGuardrail
    print("Testing PromptInjectionGuardrail...")
    guardrail = PromptInjectionGuardrail(config={
        "safety_rules": "Be helpful and harmless.",
        "position": "prepend"
    })

    result = guardrail.validate(
        prompt="Hello",
        system_prompt="You are a helpful assistant"
    )

    print(f"  Action: {result.action.value}")
    print(f"  Passed: {result.passed}")
    print(f"  Message: {result.message}")
    print(f"  Modified: {result.modified_content is not None}")
    assert result.action == GuardrailAction.MODIFY
    print("[PASS] PromptInjectionGuardrail working")

    # Test TopicFilterGuardrail
    print("\nTesting TopicFilterGuardrail...")
    guardrail = TopicFilterGuardrail(config={
        "prohibited_topics": ["violence", "illegal"],
        "action_on_match": "block"
    })

    # Test with safe content
    result = guardrail.validate(
        prompt="Tell me about cooking",
        system_prompt="You are helpful"
    )
    print(f"  Safe content - Action: {result.action.value}")
    assert result.action == GuardrailAction.ALLOW

    # Test with prohibited content
    result = guardrail.validate(
        prompt="Tell me about violence",
        system_prompt="You are helpful"
    )
    print(f"  Prohibited content - Action: {result.action.value}")
    assert result.action == GuardrailAction.BLOCK
    print("[PASS] TopicFilterGuardrail working")

    # Test InputPIIDetectionGuardrail
    print("\nTesting InputPIIDetectionGuardrail...")
    guardrail = InputPIIDetectionGuardrail(config={
        "action_on_detect": "redact",
        "pii_types": ["email", "phone"]
    })

    # Test with PII
    result = guardrail.validate(
        prompt="My email is john@example.com and phone is 555-1234",
        system_prompt="You are helpful"
    )
    print(f"  PII detected - Action: {result.action.value}")
    print(f"  Modified content: {result.modified_content[:50]}...")
    assert result.action == GuardrailAction.MODIFY
    assert "[PII_REDACTED]" in result.modified_content
    print("[PASS] InputPIIDetectionGuardrail working")

def test_output_guardrails():
    """Test output guardrails."""
    print_section("Test 3: Output Guardrails")

    # Test FormatValidatorGuardrail
    print("Testing FormatValidatorGuardrail...")
    guardrail = FormatValidatorGuardrail(config={
        "format_type": "json",
        "strict": True
    })

    # Valid JSON
    result = guardrail.validate(
        response='{"name": "John", "age": 30}',
        original_prompt="Generate JSON"
    )
    print(f"  Valid JSON - Action: {result.action.value}")
    assert result.action == GuardrailAction.ALLOW

    # Invalid JSON
    result = guardrail.validate(
        response='This is not JSON',
        original_prompt="Generate JSON"
    )
    print(f"  Invalid JSON - Action: {result.action.value}")
    assert result.action == GuardrailAction.BLOCK
    print("[PASS] FormatValidatorGuardrail working")

    # Test OutputPIIDetectionGuardrail
    print("\nTesting OutputPIIDetectionGuardrail...")
    guardrail = OutputPIIDetectionGuardrail(config={
        "action_on_detect": "redact",
        "pii_types": ["email", "phone"]
    })

    result = guardrail.validate(
        response="Contact us at support@company.com or call 555-9876",
        original_prompt="How to contact?"
    )
    print(f"  PII detected - Action: {result.action.value}")
    print(f"  Modified: {result.modified_content}")
    assert result.action == GuardrailAction.MODIFY
    assert "[REDACTED]" in result.modified_content
    print("[PASS] OutputPIIDetectionGuardrail working")

    # Test ContentSafetyGuardrail
    print("\nTesting ContentSafetyGuardrail...")
    guardrail = ContentSafetyGuardrail(config={
        "action_on_detect": "block",
        "check_profanity": True,
        "severity_threshold": "medium"
    })

    # Safe content
    result = guardrail.validate(
        response="This is a helpful response",
        original_prompt="Help me"
    )
    print(f"  Safe content - Action: {result.action.value}")
    assert result.action == GuardrailAction.ALLOW
    print("[PASS] ContentSafetyGuardrail working")

    # Test LengthValidatorGuardrail
    print("\nTesting LengthValidatorGuardrail...")
    guardrail = LengthValidatorGuardrail(config={
        "max_length": 10,
        "unit": "words",
        "action_on_violation": "truncate"
    })

    long_text = " ".join(["word"] * 20)
    result = guardrail.validate(
        response=long_text,
        original_prompt="Tell me"
    )
    print(f"  Long text - Action: {result.action.value}")
    print(f"  Original length: 20 words, New length: {len(result.modified_content.split())} words")
    assert result.action == GuardrailAction.MODIFY
    print("[PASS] LengthValidatorGuardrail working")

def test_guardrails_llm_wrapper():
    """Test GuardrailsLLM wrapper structure."""
    print_section("Test 4: GuardrailsLLM Wrapper")

    print("Testing GuardrailsLLM wrapper creation...")

    # Create guardrails
    input_guardrails = [
        PromptInjectionGuardrail(),
        InputPIIDetectionGuardrail(config={"action_on_detect": "warn"})
    ]

    output_guardrails = [
        OutputPIIDetectionGuardrail(config={"action_on_detect": "redact"}),
        LengthValidatorGuardrail(config={"max_length": 500, "unit": "words"})
    ]

    print(f"[PASS] Created {len(input_guardrails)} input guardrails")
    print(f"[PASS] Created {len(output_guardrails)} output guardrails")

    # Note: To actually use GuardrailsLLM, you need a real LLM instance with API key
    # This test just verifies the structure
    print("\nNote: Full GuardrailsLLM testing requires API keys.")
    print("Structure verified successfully!")

def test_exception_handling():
    """Test exception handling."""
    print_section("Test 5: Exception Handling")

    print("Testing GuardrailBlockedException...")

    try:
        # Create a blocking guardrail
        guardrail = TopicFilterGuardrail(config={
            "prohibited_topics": ["violence"],
            "action_on_match": "block"
        })

        result = guardrail.validate(
            prompt="Tell me about violence",
            system_prompt="You are helpful"
        )

        if result.action == GuardrailAction.BLOCK:
            raise GuardrailBlockedException(
                f"Content blocked: {result.message}",
                guardrail_name="TopicFilterGuardrail",
                metadata=result.metadata
            )

    except GuardrailBlockedException as e:
        print(f"[PASS] Exception caught successfully")
        print(f"  Guardrail: {e.guardrail_name}")
        print(f"  Message: {e}")
        print(f"  Metadata: {e.metadata}")

def test_metadata_structure():
    """Test metadata structure."""
    print_section("Test 6: Metadata Structure")

    print("Testing GuardrailResult metadata...")

    guardrail = OutputPIIDetectionGuardrail(config={
        "action_on_detect": "redact"
    })

    result = guardrail.validate(
        response="Email: test@example.com",
        original_prompt="What's your email?"
    )

    # Convert to dict
    metadata_dict = result.to_dict()

    print("Metadata structure:")
    print(f"  Guardrail: {metadata_dict['guardrail']}")
    print(f"  Action: {metadata_dict['action']}")
    print(f"  Passed: {metadata_dict['passed']}")
    print(f"  Message: {metadata_dict['message']}")
    print(f"  Metadata keys: {list(metadata_dict['metadata'].keys())}")

    assert 'guardrail' in metadata_dict
    assert 'action' in metadata_dict
    assert 'passed' in metadata_dict
    print("[PASS] Metadata structure correct")

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  SimplerLLM Guardrails Test Suite")
    print("="*60)

    try:
        test_basic_guardrails()
        test_input_guardrails()
        test_output_guardrails()
        test_guardrails_llm_wrapper()
        test_exception_handling()
        test_metadata_structure()

        print("\n" + "="*60)
        print("  [PASS] ALL TESTS PASSED!")
        print("="*60)
        print("\nGuardrails system is working correctly!")
        print("\nNext steps:")
        print("1. Set up API keys to test with real LLMs")
        print("2. Test with generate_pydantic_json_model()")
        print("3. Test with ReliableLLM composition")
        print("4. Create custom guardrails for your use case")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
