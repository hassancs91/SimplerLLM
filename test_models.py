from dotenv import load_dotenv
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from pydantic import BaseModel

load_dotenv()


class Person(BaseModel):
        Name: str 
        Age: int

#llm = LLM.create(provider=LLMProvider.OPENAI,model_name="gpt-4o")
#llm = LLM.create(provider=LLMProvider.GEMINI,model_name="gemini-3-pro-preview")
llm = LLM.create(provider=LLMProvider.ANTHROPIC,model_name="claude-opus-4-5-20251101")


response = llm.generate_response(prompt="Generate a Random Person name")

#response = generate_pydantic_json_model(model_class=Person,llm_instance=llm,prompt="Generate a Random Person")
#print(response.model_dump())


print(response)