#add streaming
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import time
import instructor

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


MAX_RETRIES = os.getenv('MAX_RETRIES')
if MAX_RETRIES is not None:
    MAX_RETRIES = int(MAX_RETRIES)
else:
    MAX_RETRIES = 3  # Default value


RETRY_DELAY = os.getenv('RETRY_DELAY')  
if RETRY_DELAY is not None:
    RETRY_DELAY = int(RETRY_DELAY)
else:
    RETRY_DELAY = 2  # Default value

STREAMING_DELAY = os.getenv('STREAMING_DELAY')
if STREAMING_DELAY is not None:
    STREAMING_DELAY = float(RETRY_DELAY)
else:
    STREAMING_DELAY = 0.1  # Default value

# Initialize the OpenAI clients
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
instructor.apatch(async_openai_client)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
instructor.patch(openai_client)


def generate_text_basic(user_prompt, model):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

async def generate_text_basic_async(user_prompt, model):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

def generate_text(user_prompt,system_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None
            
async def generate_text_async(user_prompt,system_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7 ):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model,
                messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

def generate_full_response(user_prompt,system_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return completion
        
        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None
            
async def generate_full_response_async(user_prompt,system_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model,
                messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return completion

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

def generate_json_with_pydantic(user_prompt, pydantic_model,model_name):

    for attempt in range(MAX_RETRIES):
        try:
            resut: pydantic_model =  openai_client.chat.completions.create(
                model=model_name,
                response_model=pydantic_model,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return resut
        except Exception as e:
            if attempt < MAX_RETRIES - 1:  # don't wait after the last attempt
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                print(f"Response generation exception after max retries: {e}")
                return None

async def generate_json_with_pydantic_async(user_prompt, pydantic_model,model_name):

    for attempt in range(MAX_RETRIES):
        try:
            resut: pydantic_model = await async_openai_client.chat.completions.create(
                model=model_name,
                response_model=pydantic_model,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return resut
        except Exception as e:
            if attempt < MAX_RETRIES - 1:  # don't wait after the last attempt
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                print(f"Response generation exception after max retries: {e}")
                return None

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

#GPT-Vision posponded to V2
# def analyze_image_with_vision(image_url, prompt, model="gpt-4-vision-preview",max_tokens=300):
#     response = openai_client.chat.completions.create(
#     model="model",
#     messages=[
#         {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {
#             "type": "image_url",
#             "image_url": {
#                 "url": image_url,
#             },
#             },
#         ],
#         }
#     ],
#     max_tokens=max_tokens,
#     )

    return response.choices[0].message.content

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