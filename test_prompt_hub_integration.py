import os
from dotenv import load_dotenv
from SimplerLLM.prompts.hub import (
    fetch_prompt_from_hub,
    list_prompts_from_hub,
    fetch_prompt_version_from_hub,
    ManagedPrompt,
    PromptSummaryData,
    PromptManagerError,
    MissingAPIKeyError,
    PromptNotFoundError,
    VariableError,
    AuthenticationError,
    NetworkError
)

# --- Configuration ---
# Load environment variables from .env file in the current directory
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("SIMPLERLLM_API_KEY")

# Define prompt IDs and version known to exist and be accessible
# Using the examples provided in the requests
TEST_PROMPT_ID_LATEST = "68176c8e43f8635dfc05753e" # For fetching latest version test
TEST_PROMPT_ID_VERSION = "6818502c9ee5dc68a95f7900" # For fetching specific version test
TEST_VERSION_NUMBER = 1
TEST_VARIABLE_NAME = "TARGET_AUDIENCE" # Variable name expected in both test prompts

# Emojis for output
SUCCESS_EMOJI = "✅"
ERROR_EMOJI = "❌"

# --- Helper Function ---
def print_separator(title=""):
    """Prints a separator line with an optional title."""
    print("\n" + "=" * 10 + f" {title} " + "=" * (60 - len(title)))

# --- Main Test Execution ---
if __name__ == "__main__":
    print("🚀 Starting SimplerLLM Prompt Hub Integration Tests...")

    if not API_KEY:
        print(f"\n{ERROR_EMOJI} ERROR: SIMPLERLLM_API_KEY environment variable not found.")
        print("Please ensure it is set in your .env file in the project root.")
        exit(1)
    else:
        # Mask part of the key for display
        masked_key = API_KEY[:15] + "..." + API_KEY[-4:]
        print(f"Using API Key (masked): {masked_key}")

    # 1. Test list_prompts_from_hub
    print_separator("Testing list_prompts_from_hub")
    try:
        print("Fetching list of owned and shared prompts...")
        prompts_list = list_prompts_from_hub(api_key=API_KEY, include_shared=True)
        print(f"{SUCCESS_EMOJI} SUCCESS: Found {len(prompts_list)} prompts.")
        if prompts_list:
            print("First few prompts:")
            for i, p in enumerate(prompts_list[:5]):
                 print(f"  - ID: {p.id}, Name: {p.name}, Version: {p.version}")
    except MissingAPIKeyError as e:
        print(f"{ERROR_EMOJI} ERROR (List Prompts): {e}")
    except AuthenticationError as e:
        print(f"{ERROR_EMOJI} ERROR (List Prompts): Authentication failed. Check your API key. Details: {e}")
    except NetworkError as e:
        print(f"{ERROR_EMOJI} ERROR (List Prompts): Network error occurred. Details: {e}")
    except PromptManagerError as e:
        print(f"{ERROR_EMOJI} ERROR (List Prompts): An API or validation error occurred. Details: {e}")
    except Exception as e:
        print(f"{ERROR_EMOJI} ERROR (List Prompts): An unexpected error occurred: {e}")

    # 2. Test fetch_prompt_from_hub (Latest Version)
    print_separator(f"Testing fetch_prompt_from_hub (ID: {TEST_PROMPT_ID_LATEST})")
    try:
        print(f"Fetching latest version of prompt ID: {TEST_PROMPT_ID_LATEST}...")
        managed_prompt_latest = fetch_prompt_from_hub(prompt_id=TEST_PROMPT_ID_LATEST, api_key=API_KEY)
        print(f"{SUCCESS_EMOJI} SUCCESS: Fetched prompt '{managed_prompt_latest.name}' (ID: {managed_prompt_latest.id}, Version: {managed_prompt_latest.version})")

        # Test variable setting and formatting (if applicable)
        if any(var.name == TEST_VARIABLE_NAME for var in managed_prompt_latest.variables):
            print(f"Attempting to set variable '{TEST_VARIABLE_NAME}'...")
            try:
                managed_prompt_latest.set_variable(TEST_VARIABLE_NAME, "Test Audience Latest")
                formatted_prompt = managed_prompt_latest.get_formatted_prompt()
                print(f"{SUCCESS_EMOJI} SUCCESS: Variable '{TEST_VARIABLE_NAME}' set and prompt formatted.")
                # print("\nFormatted Prompt Snippet:")
                # print(formatted_prompt[:200] + "...") # Print a snippet
            except VariableError as e:
                print(f"{ERROR_EMOJI} ERROR (Variable Handling - Latest): {e}")
        else:
            print(f"NOTE: Test prompt {TEST_PROMPT_ID_LATEST} does not contain the variable '{TEST_VARIABLE_NAME}' for formatting test.")

    except PromptNotFoundError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): Prompt not found. Details: {e}")
    except MissingAPIKeyError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): {e}")
    except AuthenticationError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): Authentication failed. Check your API key. Details: {e}")
    except NetworkError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): Network error occurred. Details: {e}")
    except PromptManagerError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): An API or validation error occurred. Details: {e}")
    except Exception as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Latest): An unexpected error occurred: {e}")

    # 3. Test fetch_prompt_version_from_hub
    print_separator(f"Testing fetch_prompt_version_from_hub (ID: {TEST_PROMPT_ID_VERSION}, Version: {TEST_VERSION_NUMBER})")
    try:
        print(f"Fetching version {TEST_VERSION_NUMBER} of prompt ID: {TEST_PROMPT_ID_VERSION}...")
        specific_version_prompt = fetch_prompt_version_from_hub(
            prompt_id=TEST_PROMPT_ID_VERSION,
            version=TEST_VERSION_NUMBER,
            api_key=API_KEY
        )
        print(f"{SUCCESS_EMOJI} SUCCESS: Fetched prompt '{specific_version_prompt.name}' (ID: {specific_version_prompt.id}, Version: {specific_version_prompt.version})")

        # Basic check to ensure version matches
        if specific_version_prompt.version != TEST_VERSION_NUMBER:
             print(f"⚠️ WARNING: Fetched version ({specific_version_prompt.version}) does not match requested version ({TEST_VERSION_NUMBER}). This might indicate an API behavior or test setup issue.")

        # Test variable setting and formatting for this specific version
        if any(var.name == TEST_VARIABLE_NAME for var in specific_version_prompt.variables):
            print(f"Attempting to set variable '{TEST_VARIABLE_NAME}' for specific version...")
            try:
                specific_version_prompt.set_variable(TEST_VARIABLE_NAME, "Test Audience Versioned")
                formatted_prompt_versioned = specific_version_prompt.get_formatted_prompt()
                print(f"{SUCCESS_EMOJI} SUCCESS: Variable '{TEST_VARIABLE_NAME}' set and prompt formatted for specific version.")
                # print("\nFormatted Versioned Prompt Snippet:")
                # print(formatted_prompt_versioned[:200] + "...") # Print a snippet
            except VariableError as e:
                print(f"{ERROR_EMOJI} ERROR (Variable Handling - Versioned): {e}")
        else:
             print(f"NOTE: Test prompt {TEST_PROMPT_ID_VERSION} version {TEST_VERSION_NUMBER} does not contain the variable '{TEST_VARIABLE_NAME}' for formatting test.")

    except PromptNotFoundError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): Prompt version not found. Details: {e}")
    except MissingAPIKeyError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): {e}")
    except AuthenticationError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): Authentication failed. Check your API key. Details: {e}")
    except NetworkError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): Network error occurred. Details: {e}")
    except PromptManagerError as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): An API or validation error occurred. Details: {e}")
    except Exception as e:
        print(f"{ERROR_EMOJI} ERROR (Fetch Version): An unexpected error occurred: {e}")

    print_separator("🏁 Tests Complete")
