import SimplerLLM.langauge.llm_providers.openai_llm as openai_llm
import SimplerLLM.langauge.llm_providers.cohere_llm as cohere_llm
import SimplerLLM.langauge.llm_providers.gemeni_llm as gemeni_llm
import SimplerLLM.langauge.llm_local.gpt2_llm as gpt2_llm
from enum import Enum


class LLMProvider(Enum):
    OPENAI = 1
    GEMENI = 2
    COHERE = 3
    LOCAL_GPT2 = 4


class LLM:
    def __init__(self, provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", temperature=0.7,top_p=1.0):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p



    @staticmethod
    def create(model=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", temperature=0.7,top_p=1.0):
        if model == LLMProvider.OPENAI:
            return OpenAILLM(model, model_name, temperature, top_p)
        if model == LLMProvider.GEMENI:
            return LLM(model, model_name, temperature)
        if model == LLMProvider.COHERE:
            return LLM(model, model_name, temperature)
        if model == LLMProvider.LOCAL_GPT2:
            return LOCALGPT2LLM(model, model_name, temperature)
        else:
            return LLM(model, model_name, temperature)

    def set_model(self, provider):
        if not isinstance(provider, LLMProvider):
            raise ValueError("Provider must be an instance of LLMProvider Enum")
        self.provider = provider


    def generate_text(self, input_text):
        if self.provider == LLMProvider.OPENAI:
            return openai_llm.generate(input_text)
        elif self.provider == LLMProvider.GEMENI:
            return "generated with Gemeni"
        elif self.provider == LLMProvider.COHERE:
            return "generated with coehere"
        else:
            raise ValueError("Unsupported model")



 
 

class OpenAILLM(LLM):
    def __init__(self, model, model_name, temperature,top_p):
        super().__init__(model, model_name, temperature,top_p)



    def generate_text(self, user_prompt, system_prompt="", model_name=None, temperature=None, top_p=None, max_tokens=500):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        return openai_llm.generate_text(user_prompt=user_prompt, system_prompt=system_prompt, 
                                        model=model_name, temperature=temperature, 
                                        top_p=top_p, max_tokens=max_tokens)
    

    async def generate_text_async(self, user_prompt, system_prompt="", model_name=None, temperature=None, top_p=None, max_tokens=500):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        return await openai_llm.generate_text_async(user_prompt=user_prompt, system_prompt=system_prompt, 
                                        model=model_name, temperature=temperature, 
                                        top_p=top_p, max_tokens=max_tokens)
    


    def generate_full_response(self, user_prompt, system_prompt="", model_name=None, temperature=None, top_p=None, max_tokens=500):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        return openai_llm.generate_full_response(user_prompt=user_prompt, system_prompt=system_prompt, 
                                        model=model_name, temperature=temperature, 
                                        top_p=top_p, max_tokens=max_tokens)
    

    async def generate_full_response_async(self, user_prompt, system_prompt="", model_name=None, temperature=None, top_p=None, max_tokens=500):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        return await openai_llm.generate_full_response_async(user_prompt=user_prompt, system_prompt=system_prompt, 
                                        model=model_name, temperature=temperature, 
                                        top_p=top_p, max_tokens=max_tokens)
    
    def generate_json_with_pydantic(self, user_prompt, pydantic_model,model_name):
        return openai_llm.generate_json_with_pydantic(user_prompt=user_prompt,pydantic_model = pydantic_model,model_name=model_name)
    

    async def generate_json_with_pydantic_async(self, user_prompt, pydantic_model,model_name):
        return await openai_llm.generate_json_with_pydantic_async(user_prompt=user_prompt,pydantic_model = pydantic_model,model_name=model_name)




    
    







class LOCALGPT2LLM(LLM):
    def __init__(self, model, model_name, temperature):
        super().__init__(model, model_name, temperature)

    def generate_text(self, input_text):
        return "Generated with GPT2 Local"
