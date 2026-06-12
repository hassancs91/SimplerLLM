"""
OpenRouter Comprehensive Test Script

Tests multiple OpenRouter models with:
  1. Basic text generation
  2. Full response with metadata (tokens, timing)
  3. JSON mode
  4. Pydantic model generation (generate_pydantic_json_model)
  5. Nested Pydantic model generation
  6. Async Pydantic generation
  7. ReliableLLM failover (OpenRouter -> OpenRouter fallback)
  8. ReliableLLM + Pydantic generation (generate_pydantic_json_model_reliable)

Usage:
    # Test the default model list
    python test_openrouter_comprehensive.py

    # Test specific models
    python test_openrouter_comprehensive.py openai/gpt-4o-mini anthropic/claude-3.5-haiku

Requires OPENROUTER_API_KEY in your environment or .env file.
A summary table of all results is printed at the end.
"""

import asyncio
import sys
import time
from typing import List

from pydantic import BaseModel

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable,
)

# =============================================================================
# CONFIGURATION - Edit this list to test different models
# =============================================================================
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

# Models used for the ReliableLLM tests
RELIABLE_PRIMARY = "openai/gpt-4o-mini"
RELIABLE_SECONDARY = "google/gemini-2.5-flash"
RELIABLE_BROKEN_PRIMARY = "fake-provider/nonexistent-model"

MAX_TOKENS = 1024

# Collected results: list of (model, test name, status, seconds, detail)
RESULTS = []


# =============================================================================
# Pydantic models used for structured generation tests
# =============================================================================
class BlogPostIdea(BaseModel):
    title: str
    hook: str
    keywords: List[str]


class Ingredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    name: str
    cuisine: str
    ingredients: List[Ingredient]
    steps: List[str]


# =============================================================================
# Helpers
# =============================================================================
def safe_print(text):
    """Print text safely, handling Unicode characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).encode('ascii', 'replace').decode('ascii'))


def record(model, test_name, status, elapsed, detail=""):
    RESULTS.append((model, test_name, status, elapsed, detail))
    marker = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[SKIP]")
    safe_print(f"  {marker} {test_name} ({elapsed:.2f}s) {detail}")


def run_test(model, test_name, func):
    """Run a single test function, record PASS/FAIL with timing."""
    start = time.time()
    try:
        detail = func() or ""
        record(model, test_name, "PASS", time.time() - start, detail)
    except Exception as e:
        record(model, test_name, "FAIL", time.time() - start, str(e)[:120])


# =============================================================================
# Per-model test suite
# =============================================================================
def test_model(model_name: str):
    safe_print(f"\n{'='*70}")
    safe_print(f"MODEL: {model_name}")
    safe_print(f"{'='*70}")

    llm = LLM.create(provider=LLMProvider.OPENROUTER, model_name=model_name)

    # -------------------------------------------------------------------
    # TEST 1: Basic text generation
    # -------------------------------------------------------------------
    def basic_generation():
        response = llm.generate_response(
            prompt="In one sentence, what is an LLM?",
            max_tokens=MAX_TOKENS,
        )
        assert isinstance(response, str) and len(response) > 0, "Empty response"
        return f"-> {response[:60]}..."

    run_test(model_name, "Basic generation", basic_generation)

    # -------------------------------------------------------------------
    # TEST 2: Full response with metadata
    # -------------------------------------------------------------------
    def full_response_metadata():
        response = llm.generate_response(
            prompt="Say hello in French.",
            max_tokens=MAX_TOKENS,
            full_response=True,
        )
        assert response.generated_text, "No generated text"
        assert response.input_token_count is not None, "No input token count"
        assert response.output_token_count is not None, "No output token count"
        return (f"tokens in/out: {response.input_token_count}/"
                f"{response.output_token_count}")

    run_test(model_name, "Full response metadata", full_response_metadata)

    # -------------------------------------------------------------------
    # TEST 3: JSON mode
    # -------------------------------------------------------------------
    def json_mode():
        import json
        response = llm.generate_response(
            prompt='Return a JSON object with keys "city" and "country" for Paris.',
            max_tokens=MAX_TOKENS,
            json_mode=True,
        )
        # Some open models wrap JSON in markdown code fences despite json_mode
        cleaned = response.strip()
        fenced = cleaned.startswith("```")
        if fenced:
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        data = json.loads(cleaned)
        assert isinstance(data, dict), "Not a JSON object"
        note = " (markdown-fenced)" if fenced else ""
        return f"-> {json.dumps(data)[:50]}{note}"

    run_test(model_name, "JSON mode", json_mode)

    # -------------------------------------------------------------------
    # TEST 4: Pydantic generation (simple model)
    # -------------------------------------------------------------------
    def pydantic_simple():
        result = generate_pydantic_json_model(
            model_class=BlogPostIdea,
            prompt="Generate a blog post idea about AI tools for content creators.",
            llm_instance=llm,
            max_tokens=MAX_TOKENS,
        )
        assert isinstance(result, BlogPostIdea), f"Got {type(result).__name__}: {result}"
        assert result.title and result.keywords, "Missing fields"
        return f"-> \"{result.title[:50]}\" ({len(result.keywords)} keywords)"

    run_test(model_name, "Pydantic simple model", pydantic_simple)

    # -------------------------------------------------------------------
    # TEST 5: Pydantic generation (nested model)
    # -------------------------------------------------------------------
    def pydantic_nested():
        result = generate_pydantic_json_model(
            model_class=Recipe,
            prompt="Generate a simple pasta recipe with 3-5 ingredients.",
            llm_instance=llm,
            max_tokens=MAX_TOKENS,
        )
        assert isinstance(result, Recipe), f"Got {type(result).__name__}: {result}"
        assert len(result.ingredients) > 0, "No ingredients"
        assert all(isinstance(i, Ingredient) for i in result.ingredients), "Bad nesting"
        return f"-> \"{result.name[:40]}\" ({len(result.ingredients)} ingredients, {len(result.steps)} steps)"

    run_test(model_name, "Pydantic nested model", pydantic_nested)

    # -------------------------------------------------------------------
    # TEST 6: Async Pydantic generation
    # -------------------------------------------------------------------
    def pydantic_async():
        async def _run():
            return await generate_pydantic_json_model_async(
                model_class=BlogPostIdea,
                prompt="Generate a blog post idea about prompt engineering.",
                llm_instance=llm,
                max_tokens=MAX_TOKENS,
            )
        result = asyncio.run(_run())
        assert isinstance(result, BlogPostIdea), f"Got {type(result).__name__}: {result}"
        return f"-> \"{result.title[:50]}\""

    run_test(model_name, "Async Pydantic model", pydantic_async)


# =============================================================================
# ReliableLLM test suite
# =============================================================================
def test_reliable_llm():
    safe_print(f"\n{'='*70}")
    safe_print(f"RELIABLE LLM TESTS (OpenRouter -> OpenRouter failover)")
    safe_print(f"{'='*70}")

    label = "ReliableLLM"

    # -------------------------------------------------------------------
    # TEST 7: Normal operation - primary handles the request
    # -------------------------------------------------------------------
    def reliable_primary():
        primary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_PRIMARY)
        secondary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_SECONDARY)
        reliable = ReliableLLM(primary, secondary, skip_validation=True)

        response, provider, model = reliable.generate_response(
            prompt="In one sentence, what is failover?",
            max_tokens=MAX_TOKENS,
            return_provider=True,
        )
        assert isinstance(response, str) and len(response) > 0, "Empty response"
        assert model == RELIABLE_PRIMARY, f"Expected primary, got {model}"
        return f"-> answered by {model}"

    run_test(label, f"Primary path ({RELIABLE_PRIMARY})", reliable_primary)

    # -------------------------------------------------------------------
    # TEST 8: Failover - broken primary model falls back to secondary
    # -------------------------------------------------------------------
    def reliable_failover():
        broken = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_BROKEN_PRIMARY)
        secondary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_SECONDARY)
        reliable = ReliableLLM(broken, secondary, skip_validation=True)

        response, provider, model = reliable.generate_response(
            prompt="In one sentence, what is a fallback system?",
            max_tokens=MAX_TOKENS,
            return_provider=True,
        )
        assert isinstance(response, str) and len(response) > 0, "Empty response"
        assert model == RELIABLE_SECONDARY, f"Expected fallback to secondary, got {model}"
        return f"-> failed over to {model}"

    run_test(label, f"Failover (broken -> {RELIABLE_SECONDARY})", reliable_failover)

    # -------------------------------------------------------------------
    # TEST 9: ReliableLLM + Pydantic generation
    # -------------------------------------------------------------------
    def reliable_pydantic():
        primary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_PRIMARY)
        secondary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_SECONDARY)
        reliable = ReliableLLM(primary, secondary, skip_validation=True)

        result = generate_pydantic_json_model_reliable(
            model_class=BlogPostIdea,
            prompt="Generate a blog post idea about building reliable AI pipelines.",
            reliable_llm=reliable,
            max_tokens=MAX_TOKENS,
        )
        # Returns (model_object, provider, model_name) on success, error string on failure
        assert isinstance(result, tuple), f"Generation failed: {result}"
        instance, provider, model = result
        assert isinstance(instance, BlogPostIdea), f"Got {type(instance).__name__}: {instance}"
        assert model == RELIABLE_PRIMARY, f"Expected primary, got {model}"
        return f"-> \"{instance.title[:40]}\" (by {model})"

    run_test(label, "Reliable + Pydantic", reliable_pydantic)

    # -------------------------------------------------------------------
    # TEST 10: ReliableLLM + Pydantic with broken primary (full failover path)
    # -------------------------------------------------------------------
    def reliable_pydantic_failover():
        broken = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_BROKEN_PRIMARY)
        secondary = LLM.create(provider=LLMProvider.OPENROUTER, model_name=RELIABLE_SECONDARY)
        reliable = ReliableLLM(broken, secondary, skip_validation=True)

        result = generate_pydantic_json_model_reliable(
            model_class=BlogPostIdea,
            prompt="Generate a blog post idea about API redundancy.",
            reliable_llm=reliable,
            max_tokens=MAX_TOKENS,
        )
        assert isinstance(result, tuple), f"Generation failed: {result}"
        instance, provider, model = result
        assert isinstance(instance, BlogPostIdea), f"Got {type(instance).__name__}: {instance}"
        assert model == RELIABLE_SECONDARY, f"Expected fallback to secondary, got {model}"
        return f"-> \"{instance.title[:40]}\" (failed over to {model})"

    run_test(label, "Reliable + Pydantic failover", reliable_pydantic_failover)


# =============================================================================
# Summary
# =============================================================================
def print_summary():
    safe_print(f"\n{'='*70}")
    safe_print("SUMMARY")
    safe_print(f"{'='*70}")

    name_width = max(len(m) for m, *_ in RESULTS) if RESULTS else 20
    test_width = max(len(t) for _, t, *_ in RESULTS) if RESULTS else 30

    passed = failed = 0
    current_model = None
    for model, test_name, status, elapsed, _ in RESULTS:
        if model != current_model:
            safe_print(f"\n{model}")
            current_model = model
        marker = "PASS" if status == "PASS" else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        safe_print(f"  {test_name:<{test_width}}  {marker}  {elapsed:>6.2f}s")

    total = passed + failed
    safe_print(f"\n{'-'*70}")
    safe_print(f"TOTAL: {total} tests | PASSED: {passed} | FAILED: {failed}")
    safe_print(f"{'-'*70}")

    if failed:
        safe_print("\nFailed tests:")
        for model, test_name, status, _, detail in RESULTS:
            if status == "FAIL":
                safe_print(f"  - [{model}] {test_name}: {detail}")


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    safe_print("="*70)
    safe_print("OpenRouter Comprehensive Test Suite")
    safe_print("="*70)
    safe_print(f"\nModels to test: {', '.join(models)}")
    safe_print(f"ReliableLLM pair: {RELIABLE_PRIMARY} -> {RELIABLE_SECONDARY}")
    safe_print("\nMake sure OPENROUTER_API_KEY is set (env or .env).\n")

    overall_start = time.time()

    for model in models:
        test_model(model)

    test_reliable_llm()

    print_summary()
    safe_print(f"\nTotal run time: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
