#add streaming
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import time

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY in .env file.")

MAX_RETRIES = 3  # Set a default max retries
RETRY_DELAY = 2  # Set a default retry delay
STREAMING_DELAY = 0.1

# Initialize the OpenAI client
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate(user_prompt, model="gpt-3.5-turbo", max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY, enable_streaming=False,print_response = False,streaming_delay = STREAMING_DELAY):
    """
    Generates a text response for a given user prompt using the specified model.
    Optionally enables streaming and prints the output.

    Args:
        user_prompt (str): The prompt to send to the model.
        model (str): The model to use for the response. Default is "gpt-3.5-turbo".
        max_retries (int): Maximum number of retries on failure. Default is MAX_RETRIES.
        retry_delay (int): Delay between retries in seconds. Default is RETRY_DELAY.
        enable_streaming (bool): Enable streaming of the output. Default is False.

    Returns:
        str: The generated response, or None if it fails to generate.
    """
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")

    for attempt in range(max_retries):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                stream=enable_streaming
            )
        
            if enable_streaming:
                responses = []
                for message in completion:
                    responses.append(message.choices[0].delta.content)
                # After collecting all the responses into the 'responses' list:
                responses = [response for response in responses if response is not None and response != '']
                if print_response:
                   print_responses(responses,streaming_delay)
                return responses
            else:
                response = [completion.choices[0].message.content][0]
                if print_response:
                   print_responses(response)
                return response

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {max_retries} attempts due to: {e}")
                return None

async def async_generate(user_prompt, model="gpt-3.5-turbo", max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """
    Generates a text response for a given user prompt using the specified model.

    Args:
        user_prompt (str): The prompt to send to the model.
        model (str): The model to use for the response. Default is "text-davinci-003".
        max_retries (int): Maximum number of retries on failure. Default is MAX_RETRIES.
        retry_delay (int): Delay between retries in seconds. Default is RETRY_DELAY.

    Returns:
        str: The generated response, or None if it fails to generate.
    """
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")

    for attempt in range(max_retries):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return completion.choices[0].message.content

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {max_retries} attempts due to: {e}")
                return None

def print_responses(responses,streaming_delay = STREAMING_DELAY):
    """
    Prints each value. If the input is a list, it prints each item. If it's a single value, it just prints that value.

    Args:
        values (list or any): A list of values or a single value to print.
    """
    if isinstance(responses, list):
        for value in responses:
            print(value, end='', flush=True)  # Flush ensures the output is immediately printed
            time.sleep(streaming_delay)  # Wait for 0.1 seconds before printing the next item
        print()  # Print a newline character at the end
    else:
        print(responses)