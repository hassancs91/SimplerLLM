"""
GPT-5 Reliability Test Script
Tests the GPT-5 model with multiple calls to check for empty responses
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any
from SimplerLLM.language.llm import LLMProvider, LLM

# Configuration
NUM_CALLS = 100  # Number of test calls to make
DELAY_BETWEEN_CALLS = 1  # Seconds to wait between calls (to avoid rate limiting)
VERBOSE = True  # Set to True to see each response in real-time
SAVE_LOG = True  # Save results to a log file

# Test prompts - you can add more variations to test different scenarios
TEST_PROMPTS = [
    "Give me a sentence of 5 words",
    "Write a simple greeting",
    "Count from 1 to 5",
    "Name three colors",
    "What is 2 + 2?"
]

class GPT5ReliabilityTester:
    def __init__(self):
        """Initialize the tester with GPT-5 model"""
        self.llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5")
        self.results = {
            "successful": 0,
            "empty": 0,
            "errors": 0,
            "responses": [],
            "error_messages": [],
            "response_times": [],
            "empty_indices": [],
            "error_indices": []
        }

    def test_single_call(self, prompt: str, call_index: int) -> Dict[str, Any]:
        """
        Make a single call to GPT-5 and record the result

        Args:
            prompt: The prompt to send
            call_index: The index of this call in the test sequence

        Returns:
            Dictionary containing the result of the call
        """
        start_time = time.time()
        result = {
            "index": call_index,
            "prompt": prompt,
            "response": None,
            "status": "unknown",
            "error": None,
            "response_time": 0
        }

        try:
            response = self.llm_instance.generate_response(
                prompt=prompt,
                max_tokens=100  # Limiting tokens for faster testing
            )

            response_time = time.time() - start_time
            result["response_time"] = response_time

            # Check if response is empty, None, or just whitespace
            if response is None:
                result["status"] = "empty"
                result["response"] = None
            elif isinstance(response, str) and response.strip() == "":
                result["status"] = "empty"
                result["response"] = ""
            else:
                result["status"] = "success"
                result["response"] = response

        except Exception as e:
            response_time = time.time() - start_time
            result["response_time"] = response_time
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def update_statistics(self, result: Dict[str, Any]):
        """Update statistics based on the result of a call"""
        if result["status"] == "success":
            self.results["successful"] += 1
            self.results["responses"].append(result["response"])
        elif result["status"] == "empty":
            self.results["empty"] += 1
            self.results["empty_indices"].append(result["index"])
        elif result["status"] == "error":
            self.results["errors"] += 1
            self.results["error_messages"].append(result["error"])
            self.results["error_indices"].append(result["index"])

        self.results["response_times"].append(result["response_time"])

    def display_progress(self, current: int, total: int, result: Dict[str, Any]):
        """Display progress and current result"""
        percentage = (current / total) * 100
        status_symbol = "✓" if result["status"] == "success" else "✗" if result["status"] == "error" else "○"

        print(f"\r[{current}/{total}] ({percentage:.1f}%) {status_symbol} ", end="")
        print(f"Success: {self.results['successful']} | Empty: {self.results['empty']} | Errors: {self.results['errors']}", end="")

        if VERBOSE and current % 10 == 0:  # Show details every 10 calls
            print()  # New line for verbose output
            if result["status"] == "success":
                print(f"  Call #{current}: SUCCESS - Response: {result['response'][:50]}...")
            elif result["status"] == "empty":
                print(f"  Call #{current}: EMPTY RESPONSE")
            elif result["status"] == "error":
                print(f"  Call #{current}: ERROR - {result['error'][:50]}...")

    def run_test(self):
        """Run the complete reliability test"""
        print("=" * 60)
        print("GPT-5 RELIABILITY TEST")
        print(f"Testing with {NUM_CALLS} calls")
        print(f"Delay between calls: {DELAY_BETWEEN_CALLS}s")
        print("=" * 60)
        print()

        start_time = datetime.now()

        for i in range(1, NUM_CALLS + 1):
            # Select a prompt (cycling through the available prompts)
            prompt = TEST_PROMPTS[(i - 1) % len(TEST_PROMPTS)]

            # Make the call
            result = self.test_single_call(prompt, i)

            # Update statistics
            self.update_statistics(result)

            # Display progress
            self.display_progress(i, NUM_CALLS, result)

            # Delay between calls (except for the last call)
            if i < NUM_CALLS:
                time.sleep(DELAY_BETWEEN_CALLS)

        end_time = datetime.now()
        self.results["total_time"] = str(end_time - start_time)

        print("\n")  # Clear the progress line
        self.generate_report()

    def generate_report(self):
        """Generate and display the final report"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        total_calls = NUM_CALLS
        success_rate = (self.results["successful"] / total_calls) * 100
        empty_rate = (self.results["empty"] / total_calls) * 100
        error_rate = (self.results["errors"] / total_calls) * 100

        print(f"\nTotal Calls: {total_calls}")
        print(f"Test Duration: {self.results['total_time']}")

        print(f"\n✓ Successful: {self.results['successful']} ({success_rate:.1f}%)")
        print(f"○ Empty: {self.results['empty']} ({empty_rate:.1f}%)")
        print(f"✗ Errors: {self.results['errors']} ({error_rate:.1f}%)")

        if self.results["response_times"]:
            avg_time = sum(self.results["response_times"]) / len(self.results["response_times"])
            min_time = min(self.results["response_times"])
            max_time = max(self.results["response_times"])

            print(f"\nResponse Times:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")

        if self.results["empty_indices"]:
            print(f"\nEmpty responses at calls: {self.results['empty_indices'][:10]}")
            if len(self.results["empty_indices"]) > 10:
                print(f"  ... and {len(self.results['empty_indices']) - 10} more")

        if self.results["error_indices"]:
            print(f"\nErrors at calls: {self.results['error_indices'][:10]}")
            if len(self.results["error_indices"]) > 10:
                print(f"  ... and {len(self.results['error_indices']) - 10} more")

        if self.results["error_messages"]:
            unique_errors = list(set(self.results["error_messages"]))
            print(f"\nUnique error messages ({len(unique_errors)}):")
            for error in unique_errors[:3]:  # Show first 3 unique errors
                print(f"  - {error[:100]}...")

        # Sample successful responses
        if self.results["responses"]:
            print(f"\nSample successful responses:")
            for i, response in enumerate(self.results["responses"][:3], 1):
                print(f"  {i}. {response[:100]}...")

        print("\n" + "=" * 60)

        # Save to log file if configured
        if SAVE_LOG:
            self.save_log()

    def save_log(self):
        """Save detailed results to a log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt5_reliability_test_{timestamp}.json"

        log_data = {
            "timestamp": timestamp,
            "configuration": {
                "num_calls": NUM_CALLS,
                "delay_between_calls": DELAY_BETWEEN_CALLS,
                "test_prompts": TEST_PROMPTS
            },
            "summary": {
                "total_calls": NUM_CALLS,
                "successful": self.results["successful"],
                "empty": self.results["empty"],
                "errors": self.results["errors"],
                "success_rate": (self.results["successful"] / NUM_CALLS) * 100,
                "empty_rate": (self.results["empty"] / NUM_CALLS) * 100,
                "error_rate": (self.results["errors"] / NUM_CALLS) * 100,
                "total_time": self.results["total_time"]
            },
            "details": {
                "empty_indices": self.results["empty_indices"],
                "error_indices": self.results["error_indices"],
                "error_messages": list(set(self.results["error_messages"])),
                "response_times": self.results["response_times"],
                "sample_responses": self.results["responses"][:10]  # Save first 10 responses
            }
        }

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\nDetailed log saved to: {filename}")


if __name__ == "__main__":
    # Run the test
    tester = GPT5ReliabilityTester()
    try:
        tester.run_test()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print(f"Completed {tester.results['successful'] + tester.results['empty'] + tester.results['errors']} calls")
        tester.generate_report()
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        print("Partial results:")
        tester.generate_report()