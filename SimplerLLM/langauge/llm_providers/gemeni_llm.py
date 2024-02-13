import google.generativeai as genai
import os
from dotenv import load_dotenv
import os
import asyncio
import time

# Load environment variables
load_dotenv()

# Constants
GEMENI_API_KEY = os.getenv('GEMENI_API_KEY')
if GEMENI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY in .env file.")

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


genai.configure(api_key=GEMENI_API_KEY)


def generate_text_basic(user_prompt, model):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string.")
    
    model = genai.GenerativeModel(model)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                        user_prompt
            )
            return response.text

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

def generate_text(user_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string.")
    
    model = genai.GenerativeModel(model)


    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                        user_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    temperature=temperature,top_p=top_p)
            )
            return response.text

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None

def generate_full_response(user_prompt, model, max_tokens=2000, top_p=1.0, temperature=0.7):
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string.")
    
    model = genai.GenerativeModel(model)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                        user_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    temperature=temperature,top_p=top_p)
            )
            return response
        
        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}")
                return None






