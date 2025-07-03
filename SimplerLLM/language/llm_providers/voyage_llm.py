import time
import asyncio
import os
from dotenv import load_dotenv
from .llm_response_models import LLMEmbeddingsResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# Import voyageai with error handling
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    voyageai = None


def generate_embeddings(
    model_name,
    user_input=None,
    full_response=False,
    api_key=None,
    input_type=None,
    output_dimension=None,
    output_dtype="float"
):
    """
    Generate embeddings using Voyage AI API.
    
    Args:
        model_name (str): The Voyage AI model name
        user_input (str or list): Text(s) to embed
        full_response (bool): Whether to return full response object
        api_key (str): Voyage AI API key
        input_type (str): Optional input type ("query" or "document")
        output_dimension (int): Optional output dimension (256, 512, 1024, 2048)
        output_dtype (str): Output data type ("float", "int8", "uint8", "binary", "ubinary")
    """
    if not VOYAGE_AVAILABLE:
        raise ImportError("voyageai package is not installed. Install it with: pip install voyageai")
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    
    # Initialize Voyage AI client
    voyage_client = voyageai.Client(api_key=api_key)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Prepare embed parameters
            embed_params = {
                "texts": user_input if isinstance(user_input, list) else [user_input],
                "model": model_name,
                "output_dtype": output_dtype
            }
            
            # Add optional parameters
            if input_type:
                embed_params["input_type"] = input_type
            if output_dimension:
                embed_params["output_dimension"] = output_dimension
            
            # Generate embeddings
            response = voyage_client.embed(**embed_params)
            
            # Extract embeddings
            embeddings = response.embeddings
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generated_embeddings = embeddings[0] if embeddings else []
            else:
                generated_embeddings = embeddings
            
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generated_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return generated_embeddings
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_embeddings_async(
    model_name,
    user_input=None,
    full_response=False,
    api_key=None,
    input_type=None,
    output_dimension=None,
    output_dtype="float"
):
    """
    Asynchronously generate embeddings using Voyage AI API.
    
    Args:
        model_name (str): The Voyage AI model name
        user_input (str or list): Text(s) to embed
        full_response (bool): Whether to return full response object
        api_key (str): Voyage AI API key
        input_type (str): Optional input type ("query" or "document")
        output_dimension (int): Optional output dimension (256, 512, 1024, 2048)
        output_dtype (str): Output data type ("float", "int8", "uint8", "binary", "ubinary")
    """
    if not VOYAGE_AVAILABLE:
        raise ImportError("voyageai package is not installed. Install it with: pip install voyageai")
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    
    # Initialize Voyage AI client
    voyage_client = voyageai.Client(api_key=api_key)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Prepare embed parameters
            embed_params = {
                "texts": user_input if isinstance(user_input, list) else [user_input],
                "model": model_name,
                "output_dtype": output_dtype
            }
            
            # Add optional parameters
            if input_type:
                embed_params["input_type"] = input_type
            if output_dimension:
                embed_params["output_dimension"] = output_dimension
            
            # Note: Voyage AI client doesn't have async methods yet, so we run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: voyage_client.embed(**embed_params)
            )
            
            # Extract embeddings
            embeddings = response.embeddings
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generated_embeddings = embeddings[0] if embeddings else []
            else:
                generated_embeddings = embeddings
            
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generated_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return generated_embeddings
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)