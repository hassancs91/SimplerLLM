"""
CometAPI Comprehensive Test Script - SimplerLLM

Tests multiple models through CometAPI with:
  1. Basic generation (per model)
  2. Pydantic structured generation - generate_pydantic_json_model (per model)
  3. Async Pydantic generation
  4. ReliableLLM fallback (including a forced-failure fallback test)
  5. ReliableLLM + Pydantic - generate_pydantic_json_model_reliable

Usage:
    python test_cometapi_comprehensive.py                 # run with default model list
    python test_cometapi_comprehensive.py gpt-4o-mini gemini-2.5-flash   # custom models

Requires COMETAPI_API_KEY (or COMETAPI_KEY) in your environment / .env file.
"""
import asyncio
import sys
import time

from pydantic import BaseModel, Field
from typing import List

from SimplerLLM.language import (
    LLM,
    LLMProvider,
    generate_pydantic_json_model,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable,
)
from SimplerLLM.language.llm.reliable import ReliableLLM

# =============================================================================
# Models to test through CometAPI (edit this list as needed)
# =============================================================================
DEFAULT_MODELS = [
    "gpt-4o-mini",          # OpenAI
    "gpt-5-mini",           # OpenAI (reasoning)
    "claude-sonnet-4-6",    # Anthropic
    "gemini-2.5-flash",     # Google
    "deepseek-v3",          # DeepSeek
    "qwen3-30b-a3b",        # Qwen
]


# =============================================================================
# Pydantic models for structured generation tests
# =============================================================================
class BookRecommendation(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author of the book")
    year: int = Field(description="Year of publication")
    genres: List[str] = Field(description="List of genres")
    one_line_pitch: str = Field(description="One sentence pitch for the book")


class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(description="Exactly 4 answer options")
    correct_answer: str
    difficulty: str = Field(description="easy, medium, or hard")


def safe_print(text):
    """Print text safely, handling Unicode characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).encode('ascii', 'replace').decode('ascii'))


def header(title):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)


# =============================================================================
# Test 1 + 2: Basic generation and Pydantic generation per model
# =============================================================================
def test_model(model_name, results):
    """Run basic + pydantic tests against a single CometAPI model."""
    llm = LLM.create(provider=LLMProvider.COMETAPI, model_name=model_name)

    # --- Basic generation ---
    header(f"[{model_name}] Basic Generation")
    try:
        start = time.time()
        response = llm.generate_response(
            prompt="What is the capital of Japan? Answer in one short sentence.",
            max_tokens=2000,
            full_response=True,
        )
        elapsed = time.time() - start
        safe_print(f"Response: {response.generated_text[:150]}")
        print(f"Tokens: {response.input_token_count}->{response.output_token_count} | Time: {elapsed:.2f}s")
        if response.reasoning_tokens:
            print(f"Reasoning tokens: {response.reasoning_tokens}")
        results[model_name]["basic"] = "PASS"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results[model_name]["basic"] = f"FAIL: {str(e)[:60]}"

    # --- Pydantic structured generation ---
    header(f"[{model_name}] Pydantic Generation (BookRecommendation)")
    try:
        start = time.time()
        book = generate_pydantic_json_model(
            model_class=BookRecommendation,
            prompt="Recommend a classic science fiction book.",
            llm_instance=llm,
            max_tokens=2000,
        )
        elapsed = time.time() - start
        if isinstance(book, BookRecommendation):
            safe_print(f"Title: {book.title} ({book.year}) by {book.author}")
            safe_print(f"Genres: {', '.join(book.genres)}")
            safe_print(f"Pitch: {book.one_line_pitch}")
            print(f"Time: {elapsed:.2f}s")
            results[model_name]["pydantic"] = "PASS"
        else:
            # On repeated validation failure the helper returns an error string
            safe_print(f"[FAIL] Did not return a model: {str(book)[:120]}")
            results[model_name]["pydantic"] = "FAIL: no valid model returned"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results[model_name]["pydantic"] = f"FAIL: {str(e)[:60]}"


# =============================================================================
# Test 3: Async Pydantic generation
# =============================================================================
def test_async_pydantic(model_name, results):
    header(f"[{model_name}] Async Pydantic Generation (QuizQuestion)")
    try:
        llm = LLM.create(provider=LLMProvider.COMETAPI, model_name=model_name)
        quiz = asyncio.run(generate_pydantic_json_model_async(
            model_class=QuizQuestion,
            prompt="Create a quiz question about the solar system.",
            llm_instance=llm,
            max_tokens=2000,
        ))
        if isinstance(quiz, QuizQuestion):
            safe_print(f"Q: {quiz.question}")
            safe_print(f"Options: {quiz.options}")
            safe_print(f"Answer: {quiz.correct_answer} | Difficulty: {quiz.difficulty}")
            results["_async_pydantic"] = "PASS"
        else:
            safe_print(f"[FAIL] Did not return a model: {str(quiz)[:120]}")
            results["_async_pydantic"] = "FAIL: no valid model returned"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results["_async_pydantic"] = f"FAIL: {str(e)[:60]}"


# =============================================================================
# Test 4: ReliableLLM fallback between two CometAPI models
# =============================================================================
def test_reliable_llm(primary_model, secondary_model, results):
    header(f"ReliableLLM: {primary_model} (primary) -> {secondary_model} (secondary)")
    try:
        primary = LLM.create(provider=LLMProvider.COMETAPI, model_name=primary_model)
        secondary = LLM.create(provider=LLMProvider.COMETAPI, model_name=secondary_model)
        reliable = ReliableLLM(primary, secondary, skip_validation=True)

        response, provider, model_used = reliable.generate_response(
            prompt="Say hello in exactly three words.",
            max_tokens=2000,
            return_provider=True,
        )
        safe_print(f"Response: {str(response)[:100]}")
        print(f"Served by: {provider.name} ({model_used})")
        results["_reliable_basic"] = "PASS"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results["_reliable_basic"] = f"FAIL: {str(e)[:60]}"


def test_reliable_llm_forced_fallback(secondary_model, results):
    """Primary uses a nonexistent model so the request must fall back."""
    header(f"ReliableLLM Forced Fallback: nonexistent-model -> {secondary_model}")
    try:
        primary = LLM.create(provider=LLMProvider.COMETAPI, model_name="nonexistent-model-xyz-123")
        secondary = LLM.create(provider=LLMProvider.COMETAPI, model_name=secondary_model)
        reliable = ReliableLLM(primary, secondary, skip_validation=True)

        response, provider, model_used = reliable.generate_response(
            prompt="What is 5 + 5? Answer with just the number.",
            max_tokens=2000,
            return_provider=True,
        )
        safe_print(f"Response: {str(response)[:100]}")
        print(f"Served by: {provider.name} ({model_used})")
        if model_used == secondary_model:
            print("[OK] Fallback to secondary worked as expected")
            results["_reliable_fallback"] = "PASS"
        else:
            print("[WARN] Primary unexpectedly succeeded with a fake model name")
            results["_reliable_fallback"] = "WARN: primary did not fail"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results["_reliable_fallback"] = f"FAIL: {str(e)[:60]}"


# =============================================================================
# Test 5: ReliableLLM + Pydantic structured generation
# =============================================================================
def test_reliable_pydantic(primary_model, secondary_model, results):
    header(f"ReliableLLM + Pydantic: {primary_model} -> {secondary_model}")
    try:
        primary = LLM.create(provider=LLMProvider.COMETAPI, model_name=primary_model)
        secondary = LLM.create(provider=LLMProvider.COMETAPI, model_name=secondary_model)
        reliable = ReliableLLM(primary, secondary, skip_validation=True)

        book, provider, model_used = generate_pydantic_json_model_reliable(
            model_class=BookRecommendation,
            prompt="Recommend a famous fantasy novel.",
            reliable_llm=reliable,
            max_tokens=2000,
        )
        if isinstance(book, BookRecommendation):
            safe_print(f"Title: {book.title} ({book.year}) by {book.author}")
            print(f"Served by: {provider.name} ({model_used})")
            results["_reliable_pydantic"] = "PASS"
        else:
            safe_print(f"[FAIL] Did not return a model: {str(book)[:120]}")
            results["_reliable_pydantic"] = "FAIL: no valid model returned"
    except Exception as e:
        safe_print(f"[FAIL] {str(e)[:120]}")
        results["_reliable_pydantic"] = f"FAIL: {str(e)[:60]}"


# =============================================================================
# Summary
# =============================================================================
def print_summary(models, results):
    header("SUMMARY")
    col = max(len(m) for m in models) + 2
    print(f"{'Model'.ljust(col)} {'Basic':<10} {'Pydantic'}")
    print("-" * (col + 30))
    for m in models:
        basic = results[m].get("basic", "-")
        pydantic = results[m].get("pydantic", "-")
        status_b = basic if basic == "PASS" else basic[:28]
        status_p = pydantic if pydantic == "PASS" else pydantic[:28]
        print(f"{m.ljust(col)} {status_b:<10} {status_p}")

    print()
    for key, label in [
        ("_async_pydantic", "Async Pydantic"),
        ("_reliable_basic", "ReliableLLM basic"),
        ("_reliable_fallback", "ReliableLLM forced fallback"),
        ("_reliable_pydantic", "ReliableLLM + Pydantic"),
    ]:
        print(f"{label:<30} {results.get(key, '-')}")

    failures = [v for v in results.values() if isinstance(v, str) and v.startswith("FAIL")]
    failures += [v for m in models for v in results[m].values() if v.startswith("FAIL")]
    print(f"\n{'!'*5} {len(failures)} failure(s)" if failures else "\nAll tests passed!")


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    print("="*70)
    print("CometAPI Comprehensive Test - SimplerLLM")
    print(f"Models under test: {', '.join(models)}")
    print("="*70)

    results = {m: {} for m in models}

    # Per-model tests: basic generation + pydantic generation
    for model in models:
        test_model(model, results)

    # Async pydantic (first model in the list)
    test_async_pydantic(models[0], results)

    # ReliableLLM tests (first two models; reuse first if only one given)
    primary = models[0]
    secondary = models[1] if len(models) > 1 else models[0]
    test_reliable_llm(primary, secondary, results)
    test_reliable_llm_forced_fallback(secondary, results)
    test_reliable_pydantic(primary, secondary, results)

    print_summary(models, results)


if __name__ == "__main__":
    main()
