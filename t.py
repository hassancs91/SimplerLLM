import asyncio
from SimplerLLM.langauge.llm import LLM, LLMProvider

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


class TitlesModel(BaseModel):
    Titles: List[str]


llm_instance = LLM.create(model=LLMProvider.OPENAI)


prompt = " generate 5 random catchy titles in a json format"

result : TitlesModel = llm_instance.generate_json_with_pydantic(prompt,TitlesModel,"gpt-3.5-turbo-1106")
print(result.Titles[0])




# async def generate_text():
#     llm_instance = LLM.create(model=LLMProvider.OPENAI)
#     result = await llm_instance.generate_text_basic_async("What is a dolphin in 2 sentences?")
#     print(result)

# Running the async function
#asyncio.run(generate_text())