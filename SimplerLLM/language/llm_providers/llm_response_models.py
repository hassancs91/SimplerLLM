from pydantic import BaseModel
from typing import Any, Optional


class LLMFullResponse(BaseModel):
    generated_text: str
    model: str
    process_time: float
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    llm_provider_response: Any


class LLMEmbeddingsResponse(BaseModel):
    generated_embedding: Any
    model: str
    process_time: float
    llm_provider_response: Any
