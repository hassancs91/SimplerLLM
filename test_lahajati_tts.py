"""
SimplerLLM Lahajati TTS Test Script
Tests Lahajati TTS features: Arabic voices, dialects, performance styles, and async.
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

from SimplerLLM.voice.tts import (
    TTS,
    TTSProvider,
    TTSResponse,
    TTSValidationError,
    LahajatiTTS,
    LAHAJATI_FORMATS,
    LAHAJATI_INPUT_MODES,
    Dialect,
    Performance,
)

# Test output directory
OUTPUT_DIR = "tts_test_output"


def safe_print(text):
    """Print text safely, handling Unicode characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def print_result(test_name, response: TTSResponse):
    """Helper to print test results cleanly."""
    print(f"\n{'='*60}")
    print(f"[OK] {test_name}")
    print(f"{'='*60}")

    if isinstance(response.audio_data, bytes):
        size_kb = len(response.audio_data) / 1024
        print(f"Audio: {size_kb:.1f} KB in memory")
    else:
        print(f"Audio: saved to {response.audio_data}")

    print(f"Model: {response.model} | Voice: {response.voice} | Format: {response.format}")

    if response.process_time:
        print(f"Process time: {response.process_time:.2f}s")

    if response.file_path:
        print(f"File: {response.file_path}")

    if response.dialect_id:
        print(f"Dialect ID: {response.dialect_id}")

    if response.performance_id:
        print(f"Performance ID: {response.performance_id}")

    if response.custom_prompt:
        print(f"Custom Prompt: {response.custom_prompt[:50]}...")


def ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def main():
    ensure_output_dir()

    # Check for API key
    if not os.getenv("LAHAJATI_API_KEY"):
        print("=" * 60)
        print("LAHAJATI TTS TESTS - SKIPPED")
        print("=" * 60)
        print("\n[SKIP] LAHAJATI_API_KEY not set in environment")
        print("Set LAHAJATI_API_KEY in your .env file to run these tests.")
        return

    print("\n" + "=" * 60)
    print("LAHAJATI TTS TESTS")
    print("=" * 60)

    # Create Lahajati TTS instance
    tts = TTS.create(TTSProvider.LAHAJATI)

    # =========================================================================
    # TEST 1: List Available Voices
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: List Available Voices")
    print("=" * 60)
    try:
        voices = tts.list_voices()
        print(f"Found {len(voices)} voices:")
        for v in voices[:5]:
            safe_print(f"  - {v.voice_id}: {v.name} ({v.gender})")
        if len(voices) > 5:
            print(f"  ... and {len(voices) - 5} more")

        # Get first voice ID for subsequent tests
        test_voice_id = voices[0].voice_id if voices else None
        print(f"\nUsing voice: {test_voice_id}")
    except Exception as e:
        print(f"[ERROR] Failed to list voices: {e}")
        test_voice_id = None

    if not test_voice_id:
        print("\n[SKIP] Cannot continue without a valid voice ID")
        return

    # =========================================================================
    # TEST 2: List Available Dialects
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: List Available Dialects")
    print("=" * 60)
    test_dialect_id = None
    try:
        dialects = tts.list_dialects()
        print(f"Found {len(dialects)} dialects:")
        for d in dialects[:5]:
            safe_print(f"  - {d.dialect_id}: {d.display_name}")
        if len(dialects) > 5:
            print(f"  ... and {len(dialects) - 5} more")
        test_dialect_id = dialects[0].dialect_id if dialects else None
    except Exception as e:
        print(f"[ERROR] Failed to list dialects: {e}")

    # =========================================================================
    # TEST 3: List Available Performances
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: List Available Performance Styles")
    print("=" * 60)
    test_performance_id = None
    try:
        performances = tts.list_performances()
        print(f"Found {len(performances)} performance styles:")
        for p in performances[:5]:
            safe_print(f"  - {p.performance_id}: {p.display_name}")
        if len(performances) > 5:
            print(f"  ... and {len(performances) - 5} more")
        test_performance_id = performances[0].performance_id if performances else None
    except Exception as e:
        print(f"[ERROR] Failed to list performances: {e}")

    # =========================================================================
    # TEST 4: Basic Generation (Structured Mode)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Basic Generation (Structured Mode)")
    print("=" * 60)
    try:
        response = tts.generate_speech(
            text="مرحبا بكم في SimplerLLM",  # "Welcome to SimplerLLM" in Arabic
            voice=test_voice_id,
            input_mode=0,  # Structured mode
            dialect_id=test_dialect_id,
            performance_id=test_performance_id,
        )
        print_result("Basic Generation", response)
        assert isinstance(response.audio_data, bytes), "Audio data should be bytes"
        assert len(response.audio_data) > 0, "Audio data should not be empty"
    except Exception as e:
        print(f"[ERROR] Basic generation failed: {e}")

    # =========================================================================
    # TEST 5: Generation with Dialect
    # =========================================================================
    if test_dialect_id and test_performance_id:
        print("\n" + "=" * 60)
        print(f"TEST 5: Generation with Dialect ({test_dialect_id})")
        print("=" * 60)
        try:
            response = tts.generate_speech(
                text="هذا اختبار للهجة العربية",  # "This is an Arabic dialect test"
                voice=test_voice_id,
                input_mode=0,
                dialect_id=test_dialect_id,
                performance_id=test_performance_id,
            )
            print_result("Dialect Generation", response)
            assert response.dialect_id == test_dialect_id
        except Exception as e:
            print(f"[ERROR] Dialect generation failed: {e}")

    # =========================================================================
    # TEST 6: Generation with Performance Style
    # =========================================================================
    if test_dialect_id and test_performance_id:
        print("\n" + "=" * 60)
        print(f"TEST 6: Generation with Performance Style ({test_performance_id})")
        print("=" * 60)
        try:
            response = tts.generate_speech(
                text="هذا اختبار لأسلوب الأداء",  # "This is a performance style test"
                voice=test_voice_id,
                input_mode=0,
                dialect_id=test_dialect_id,
                performance_id=test_performance_id,
            )
            print_result("Performance Generation", response)
            assert response.performance_id == test_performance_id
        except Exception as e:
            print(f"[ERROR] Performance generation failed: {e}")

    # =========================================================================
    # TEST 7: Generation with Both Dialect and Performance
    # =========================================================================
    if test_dialect_id and test_performance_id:
        print("\n" + "=" * 60)
        print("TEST 7: Generation with Dialect + Performance")
        print("=" * 60)
        try:
            response = tts.generate_speech(
                text="اختبار شامل مع اللهجة والأداء",  # "Comprehensive test with dialect and performance"
                voice=test_voice_id,
                input_mode=0,
                dialect_id=test_dialect_id,
                performance_id=test_performance_id,
            )
            print_result("Combined Generation", response)
        except Exception as e:
            print(f"[ERROR] Combined generation failed: {e}")

    # =========================================================================
    # TEST 8: Custom Prompt Mode
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 8: Custom Prompt Mode")
    print("=" * 60)
    try:
        response = tts.generate_speech(
            text="مرحبا",  # "Hello"
            voice=test_voice_id,
            input_mode=1,  # Custom mode
            custom_prompt="Speak with a warm and friendly tone, like greeting an old friend.",
        )
        print_result("Custom Prompt Mode", response)
        assert response.input_mode == 1
        assert response.custom_prompt is not None
    except Exception as e:
        print(f"[ERROR] Custom prompt generation failed: {e}")

    # =========================================================================
    # TEST 9: Save to File
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 9: Save to File")
    print("=" * 60)
    output_path = os.path.join(OUTPUT_DIR, "lahajati_test.mp3")
    try:
        response = tts.generate_speech(
            text="حفظ الملف الصوتي",  # "Saving audio file"
            voice=test_voice_id,
            input_mode=0,
            dialect_id=test_dialect_id,
            performance_id=test_performance_id,
            output_path=output_path,
        )
        print_result("Save to File", response)
        assert os.path.exists(output_path), f"File should exist at {output_path}"
        assert response.file_path is not None, "file_path should be set"
    except Exception as e:
        print(f"[ERROR] Save to file failed: {e}")

    # =========================================================================
    # TEST 10: Async Generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 10: Async Generation")
    print("=" * 60)

    async def test_async():
        response = await tts.generate_speech_async(
            text="اختبار التوليد غير المتزامن",  # "Async generation test"
            voice=test_voice_id,
            input_mode=0,
            dialect_id=test_dialect_id,
            performance_id=test_performance_id,
        )
        return response

    try:
        response = asyncio.run(test_async())
        print_result("Async Generation", response)
        assert isinstance(response.audio_data, bytes), "Async should return audio bytes"
    except Exception as e:
        print(f"[ERROR] Async generation failed: {e}")

    # =========================================================================
    # Error Handling Tests
    # =========================================================================
    print("\n" + "=" * 60)
    print("ERROR HANDLING TESTS")
    print("=" * 60)

    # TEST 11: Empty Text Validation
    print(f"\n{'='*60}")
    print("TEST 11: Empty Text Validation")
    print(f"{'='*60}")
    try:
        tts.generate_speech(text="", voice=test_voice_id)
        print("[FAIL] Should have raised TTSValidationError")
    except TTSValidationError as e:
        print(f"[OK] Correctly raised TTSValidationError: {e}")

    # TEST 12: Missing Voice Validation
    print(f"\n{'='*60}")
    print("TEST 12: Missing Voice Validation")
    print(f"{'='*60}")
    try:
        # Create TTS without default voice
        tts_no_voice = TTS.create(TTSProvider.LAHAJATI)
        tts_no_voice.generate_speech(text="Test")
        print("[FAIL] Should have raised TTSValidationError")
    except TTSValidationError as e:
        print(f"[OK] Correctly raised TTSValidationError: {e}")

    # TEST 13: Invalid Input Mode Validation
    print(f"\n{'='*60}")
    print("TEST 13: Invalid Input Mode Validation")
    print(f"{'='*60}")
    try:
        tts.generate_speech(text="Test", voice=test_voice_id, input_mode=5)
        print("[FAIL] Should have raised TTSValidationError")
    except TTSValidationError as e:
        print(f"[OK] Correctly raised TTSValidationError: {e}")

    # TEST 14: Custom Mode without Prompt Validation
    print(f"\n{'='*60}")
    print("TEST 14: Custom Mode without Prompt Validation")
    print(f"{'='*60}")
    try:
        tts.generate_speech(text="Test", voice=test_voice_id, input_mode=1)  # No custom_prompt
        print("[FAIL] Should have raised TTSValidationError")
    except TTSValidationError as e:
        print(f"[OK] Correctly raised TTSValidationError: {e}")

    # TEST 14b: Structured Mode Missing Parameters Validation
    print(f"\n{'='*60}")
    print("TEST 14b: Structured Mode Missing Parameters Validation")
    print(f"{'='*60}")
    try:
        tts.generate_speech(text="Test", voice=test_voice_id, input_mode=0)  # No dialect_id/performance_id
        print("[FAIL] Should have raised TTSValidationError")
    except TTSValidationError as e:
        print(f"[OK] Correctly raised TTSValidationError: {e}")

    # =========================================================================
    # Factory Pattern Tests
    # =========================================================================
    print("\n" + "=" * 60)
    print("FACTORY PATTERN TESTS")
    print("=" * 60)

    # TEST 15: Create via Factory with Custom Defaults
    print(f"\n{'='*60}")
    print("TEST 15: Factory with Custom Defaults")
    print(f"{'='*60}")
    custom_tts = TTS.create(
        TTSProvider.LAHAJATI,
        voice=test_voice_id,
    )
    print(f"Created TTS with default voice: {custom_tts.default_voice}")
    assert custom_tts.default_voice == test_voice_id, f"Voice should be {test_voice_id}"
    print("[OK] Factory with custom defaults works correctly")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL LAHAJATI TTS TESTS COMPLETED!")
    print("=" * 60)
    print(f"\nAudio files saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
