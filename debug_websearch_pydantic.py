"""
Debug Script: Web Search + Nested Pydantic Model Validation Issue

This script reproduces the validation error that occurs when using
generate_pydantic_json_model with web_search=True and a nested model structure.

Issue: The LLM returns individual objects instead of wrapping them in the
expected container field, causing Pydantic validation to fail.

Expected: {"competitors": [{"name": "...", "url": "...", "one_liner": "..."}]}
Actual:   Multiple separate objects: {"name": "A", ...}, {"name": "B", ...}

Run: python debug_websearch_pydantic.py
"""

from typing import List
from pydantic import BaseModel, Field
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model


# =============================================================================
# PYDANTIC MODELS (same as in business_idea_enrichment_service.py)
# =============================================================================

class CompetitorInfo(BaseModel):
    """Single competitor information from web search."""
    name: str = Field(description="Product/company name")
    url: str = Field(description="Homepage URL")
    one_liner: str = Field(description="What it does in one sentence")


class CompetitorSearchResult(BaseModel):
    """Structured output for competitor search."""
    competitors: List[CompetitorInfo] = Field(
        min_length=1,
        max_length=5,
        description="List of 1-5 competitors"
    )


# =============================================================================
# TEST PROMPT (same as COMPETITOR_SEARCH_PROMPT)
# =============================================================================

COMPETITOR_SEARCH_PROMPT = """You are a market research analyst. Given a business idea, find 3-5 existing tools or competitors in the same space.

For each competitor return:
- name: Product/company name
- url: Homepage URL
- one_liner: What it does in one sentence

Rules:
- Find REAL products that exist today (not hypothetical)
- Include direct competitors AND adjacent tools solving similar problems
- Prefer established tools with actual users over brand new ones
- If fewer than 3 exist, return what you find

Idea: {idea_title} - {description}"""


# =============================================================================
# FLAT MODEL FOR COMPARISON (this works)
# =============================================================================

class ResearchSummary(BaseModel):
    """Flat model - works correctly with web_search."""
    topic: str = Field(description="The research topic")
    key_findings: List[str] = Field(description="List of key findings")
    summary: str = Field(description="Brief summary")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_nested_model_with_websearch():
    """
    Test the nested model (CompetitorSearchResult) with web_search=True.
    This reproduces the validation error.
    """
    print("\n" + "="*70)
    print("TEST 1: Nested Model (CompetitorSearchResult) + web_search=True")
    print("="*70)

    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
    )

    prompt = COMPETITOR_SEARCH_PROMPT.format(
        idea_title="White-label Trading Platform",
        description="A SaaS platform that allows businesses to launch their own branded trading platform for stocks, forex, and crypto."
    )

    print(f"\nPrompt:\n{prompt[:200]}...")
    print("\nCalling generate_pydantic_json_model with web_search=True...")

    result = generate_pydantic_json_model(
        model_class=CompetitorSearchResult,
        prompt=prompt,
        llm_instance=llm,
        web_search=True,
        max_tokens=4096,
        full_response=True,
    )

    if isinstance(result, str):
        print(f"\n❌ ERROR: {result}")
        return False
    else:
        print(f"\n✅ SUCCESS!")
        print(f"Competitors found: {len(result.model_object.competitors)}")
        for comp in result.model_object.competitors:
            print(f"  - {comp.name}: {comp.one_liner[:50]}...")
        return True


def test_flat_model_with_websearch():
    """
    Test the flat model (ResearchSummary) with web_search=True.
    This should work correctly.
    """
    print("\n" + "="*70)
    print("TEST 2: Flat Model (ResearchSummary) + web_search=True")
    print("="*70)

    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
    )

    prompt = "Research the latest developments in white-label trading platforms in 2025"

    print(f"\nPrompt: {prompt}")
    print("\nCalling generate_pydantic_json_model with web_search=True...")

    result = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt=prompt,
        llm_instance=llm,
        web_search=True,
        max_tokens=4096,
        full_response=True,
    )

    if isinstance(result, str):
        print(f"\n❌ ERROR: {result}")
        return False
    else:
        print(f"\n✅ SUCCESS!")
        print(f"Topic: {result.model_object.topic}")
        print(f"Findings: {len(result.model_object.key_findings)}")
        return True


def test_nested_model_without_websearch():
    """
    Test the nested model (CompetitorSearchResult) WITHOUT web_search.
    This should work because json_mode is enforced.
    """
    print("\n" + "="*70)
    print("TEST 3: Nested Model (CompetitorSearchResult) WITHOUT web_search")
    print("="*70)

    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
    )

    # Simpler prompt without needing real-time data
    prompt = """List 3 well-known project management tools as competitors.

For each competitor return:
- name: Product/company name
- url: Homepage URL
- one_liner: What it does in one sentence"""

    print(f"\nPrompt: {prompt[:100]}...")
    print("\nCalling generate_pydantic_json_model WITHOUT web_search...")

    result = generate_pydantic_json_model(
        model_class=CompetitorSearchResult,
        prompt=prompt,
        llm_instance=llm,
        web_search=False,  # No web search
        max_tokens=4096,
        full_response=True,
    )

    if isinstance(result, str):
        print(f"\n❌ ERROR: {result}")
        return False
    else:
        print(f"\n✅ SUCCESS!")
        print(f"Competitors found: {len(result.model_object.competitors)}")
        for comp in result.model_object.competitors:
            print(f"  - {comp.name}: {comp.one_liner[:50]}...")
        return True


def debug_json_extraction():
    """
    Debug the JSON extraction process to see what's happening.
    """
    print("\n" + "="*70)
    print("DEBUG: Raw Response Inspection")
    print("="*70)

    from SimplerLLM.tools.json_helpers import (
        extract_json_from_text,
        generate_json_example_from_pydantic,
    )

    # Show what the expected JSON format looks like
    print("\n1. Expected JSON format from Pydantic model:")
    example = generate_json_example_from_pydantic(CompetitorSearchResult)
    print(f"   {example}")

    # Simulate a problematic LLM response
    print("\n2. Simulated problematic LLM response (multiple separate objects):")
    problematic_response = """Here are the competitors I found:

{"name": "Vulcan Point Crypto Exchange", "url": "https://vulcanpoint.com/", "one_liner": "White-label crypto exchange platform."}

{"name": "AlphaPoint", "url": "https://alphapoint.com/", "one_liner": "Digital asset exchange technology."}

{"name": "Modulus", "url": "https://modulus.io/", "one_liner": "Exchange and brokerage technology."}
"""
    print(f"   {problematic_response[:200]}...")

    print("\n3. What extract_json_from_text returns:")
    extracted = extract_json_from_text(problematic_response)
    print(f"   Number of objects extracted: {len(extracted) if extracted else 0}")
    if extracted:
        for i, obj in enumerate(extracted):
            print(f"   Object {i+1}: {obj}")

    # Now show what a correct response looks like
    print("\n4. Correct LLM response (single wrapper object):")
    correct_response = """{"competitors": [{"name": "Vulcan Point", "url": "https://vulcanpoint.com/", "one_liner": "White-label crypto exchange."}, {"name": "AlphaPoint", "url": "https://alphapoint.com/", "one_liner": "Digital asset exchange."}]}"""
    print(f"   {correct_response[:100]}...")

    print("\n5. What extract_json_from_text returns for correct response:")
    extracted_correct = extract_json_from_text(correct_response)
    print(f"   Number of objects extracted: {len(extracted_correct) if extracted_correct else 0}")
    if extracted_correct:
        print(f"   Object: {extracted_correct[0]}")


def test_forced_failure_scenario():
    """
    Force the problematic scenario by simulating what happens when
    extract_json_from_text returns multiple unwrapped objects.

    This bypasses the LLM and directly tests the validation logic.
    """
    print("\n" + "="*70)
    print("TEST 4: FORCED FAILURE - Simulate Problematic LLM Response")
    print("="*70)

    from SimplerLLM.tools.json_helpers import (
        extract_json_from_text,
        validate_json_with_pydantic_model,
        convert_json_to_pydantic_model,
    )

    # This is what the LLM sometimes returns (the problematic case)
    problematic_llm_response = """Based on my research, here are the top competitors in the white-label trading platform space:

{"name": "Vulcan Point Crypto Exchange Platform", "url": "https://vulcanpoint.com/products/crypto-products/crypto-exchange-platform/", "one_liner": "Enterprise-grade white-label crypto exchange SaaS offering spot trading, order-book matching, wallets, KYC/AML, analytics, branding, and fiat gateways under your own brand."}

{"name": "FxTrusts MT5 White-Label", "url": "https://www.fxtrusts.com/", "one_liner": "MetaTrader 5-based white-label brokerage platform launching in weeks with full MT5 stack (desktop, web, mobile), APIs, and familiar interface for rapid deployment."}

{"name": "Marginware White-Label Trading Platform", "url": "https://www.marginware.com/products.php", "one_liner": "Cloud-based white-label prop-firm and forex/CFD broker platform providing fast, scalable multi-asset trading infrastructure with turnkey bridge solutions and risk management."}

{"name": "Quadcode Brokerage Solutions", "url": "https://quadcode.com/", "one_liner": "All-in-one white-label brokerage SaaS offering trading platform, back office, liquidity, payments, and compliance support for launching fully operational broker brands."}

These platforms represent the main competitors in this space."""

    print("\n1. Simulating problematic LLM response...")
    print(f"   Response preview: {problematic_llm_response[:150]}...")

    # Step 1: Extract JSON (this is what SimplerLLM does)
    print("\n2. Calling extract_json_from_text()...")
    json_objects = extract_json_from_text(problematic_llm_response)
    print(f"   Extracted {len(json_objects)} JSON objects")
    for i, obj in enumerate(json_objects):
        print(f"   Object {i+1} keys: {list(obj.keys())}")

    # Step 2: Validate (this is where it fails)
    print("\n3. Calling validate_json_with_pydantic_model()...")
    print(f"   Model: CompetitorSearchResult (expects 'competitors' field)")

    validated, errors = validate_json_with_pydantic_model(
        CompetitorSearchResult, json_objects
    )

    if errors:
        print(f"\n❌ VALIDATION FAILED with {len(errors)} errors:")
        for i, error in enumerate(errors):
            print(f"\n   Error {i+1}:")
            print(f"   Data: {error['data']}")
            print(f"   Error: {error['error'][:200]}...")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED")
        print(f"   Validated objects: {len(validated)}")
        return True


def test_proposed_fix():
    """
    Test a proposed fix: detect unwrapped objects and wrap them
    in the expected container field before validation.
    """
    print("\n" + "="*70)
    print("TEST 5: PROPOSED FIX - Auto-wrap Unwrapped Objects")
    print("="*70)

    from typing import get_type_hints, get_origin, get_args
    from SimplerLLM.tools.json_helpers import (
        extract_json_from_text,
        validate_json_with_pydantic_model,
    )

    # Same problematic response
    problematic_llm_response = """Here are competitors:

{"name": "Vulcan Point", "url": "https://vulcanpoint.com/", "one_liner": "White-label crypto exchange."}

{"name": "AlphaPoint", "url": "https://alphapoint.com/", "one_liner": "Digital asset exchange."}

{"name": "Modulus", "url": "https://modulus.io/", "one_liner": "Exchange technology."}
"""

    json_objects = extract_json_from_text(problematic_llm_response)
    print(f"\n1. Extracted {len(json_objects)} JSON objects")

    # PROPOSED FIX: Detect if we need to wrap objects
    def try_wrap_for_nested_model(model_class, json_objects):
        """
        If validation fails and the model has a single List field,
        try wrapping the extracted objects in that field.
        """
        # First, try normal validation
        validated, errors = validate_json_with_pydantic_model(model_class, json_objects)

        if not errors:
            return json_objects, validated, errors  # Already valid

        # Check if model has a single List field we can wrap into
        type_hints = get_type_hints(model_class)
        list_fields = []

        for field_name, field_type in type_hints.items():
            origin = get_origin(field_type)
            if origin is list:
                args = get_args(field_type)
                if args and hasattr(args[0], '__annotations__'):
                    # This is a List[SomeModel] field
                    inner_model = args[0]
                    list_fields.append((field_name, inner_model))

        if len(list_fields) == 1:
            field_name, inner_model = list_fields[0]

            # Check if extracted objects match the inner model structure
            inner_fields = set(get_type_hints(inner_model).keys())

            all_match = all(
                isinstance(obj, dict) and set(obj.keys()) == inner_fields
                for obj in json_objects
            )

            if all_match:
                # Wrap the objects in the expected structure
                wrapped = [{field_name: json_objects}]
                print(f"   AUTO-WRAP: Wrapping {len(json_objects)} objects into '{field_name}' field")

                # Re-validate with wrapped structure
                validated, errors = validate_json_with_pydantic_model(model_class, wrapped)
                return wrapped, validated, errors

        return json_objects, validated, errors

    print("\n2. Applying proposed fix (auto-wrap detection)...")
    wrapped_objects, validated, errors = try_wrap_for_nested_model(
        CompetitorSearchResult, json_objects
    )

    if errors:
        print(f"\n❌ FIX FAILED - Still {len(errors)} errors")
        return False
    else:
        print(f"\n✅ FIX WORKED!")
        print(f"   Validated: {validated}")

        # Convert to Pydantic model
        model = CompetitorSearchResult(**wrapped_objects[0])
        print(f"   Model: {model}")
        print(f"   Competitors: {len(model.competitors)}")
        for comp in model.competitors:
            print(f"     - {comp.name}")
        return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("SimplerLLM Debug: Web Search + Nested Pydantic Model")
    print("="*70)

    # First, debug the JSON extraction to understand the issue
    debug_json_extraction()

    # Run the forced failure test (no LLM needed)
    results = {}

    # Test 4: Force the failure scenario
    print("\n\n" + "="*70)
    print("FORCED FAILURE TESTS (No LLM API calls)")
    print("="*70)

    try:
        results["forced_failure"] = test_forced_failure_scenario()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["forced_failure"] = False

    # Test 5: Test the proposed fix
    try:
        results["proposed_fix"] = test_proposed_fix()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["proposed_fix"] = False

    # Run live tests (requires API)
    print("\n\n" + "="*70)
    print("LIVE TESTS (Requires OpenAI API)")
    print("="*70)

    run_live_tests = input("\nRun live API tests? (y/n): ").strip().lower() == 'y'

    if run_live_tests:
        # Test 1: This may FAIL or PASS depending on LLM response
        try:
            results["nested_with_websearch"] = test_nested_model_with_websearch()
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            results["nested_with_websearch"] = False

        # Test 2: This should PASS (flat model works)
        try:
            results["flat_with_websearch"] = test_flat_model_with_websearch()
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            results["flat_with_websearch"] = False

        # Test 3: This should PASS (no web_search = json_mode works)
        try:
            results["nested_without_websearch"] = test_nested_model_without_websearch()
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            results["nested_without_websearch"] = False

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if not results.get("forced_failure") and results.get("proposed_fix"):
        print("""
CONFIRMED: The issue is reproducible and the fix works!

The problem:
- When LLM returns separate JSON objects instead of wrapped structure
- extract_json_from_text() extracts each object separately
- validate_json_with_pydantic_model() fails because each object
  doesn't have the expected wrapper field

The fix:
- Detect when extracted objects match the inner model structure
- Auto-wrap them in the expected container field before validation
- This should be implemented in json_generation.py

Location to fix in SimplerLLM:
- SimplerLLM/language/llm_addons/json_generation.py
- After extract_json_from_text() returns, before validation
- Add logic similar to test_proposed_fix() above
""")
