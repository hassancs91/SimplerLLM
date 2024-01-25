import SimplerLLM.langauge.openai_llm as openai_llm
import SimplerLLM.langauge.cohere_llm as cohere_llm
import SimplerLLM.langauge.gemeni_llm as gemeni_llm
from enum import Enum


class LLMProvider(Enum):
    OPENAI = 1
    GEMENI = 2
    COHERE = 3
    LOCAL_GPT2 = 4


class LLM:
    def __init__(self, provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", temperature=0.7):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

    @staticmethod
    def create(model=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", temperature=0.7):
        if model == LLMProvider.OPENAI:
            return OpenAILLM(model, model_name, temperature)
        if model == LLMProvider.GEMENI:
            return LLM(model, model_name, temperature)
        if model == LLMProvider.COHERE:
            return LLM(model, model_name, temperature)
        if model == LLMProvider.LOCAL_GPT2:
            return LOCALGPT2LLM(model, model_name, temperature)
        else:
            return LLM(model, model_name, temperature)

    def set_model(self, model):
        if not isinstance(model, LLMProvider):
            raise ValueError("model must be an instance of LLMModel enum")
        self.model = model


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
    def __init__(self, model, model_name, temperature):
        super().__init__(model, model_name, temperature)

    def generate_text(self, input_text):
        return self._generate_with_openai(input_text)

    def _generate_with_openai(self, input_text):
        return openai_llm.generate(input_text, self.model_name, self.temperature)
    
    # OpenAI specific methods
    def generate_with_json_model(self, input_text,pydantic_model):
        return "Json result"
    



class LOCALGPT2LLM(LLM):
    def __init__(self, model, model_name, temperature):
        super().__init__(model, model_name, temperature)

    def generate_text(self, input_text):
        return "Generated with GPT2 Local"
