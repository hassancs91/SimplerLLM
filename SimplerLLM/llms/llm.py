import SimplerLLM.llms.openai_llm as openai_llm
from enum import Enum

class LLMModel(Enum):
    OPENAI = 1
    GEMENI = 2
    CLUADE = 3



class LLM:
    def __init__(self, model=LLMModel.OPENAI, temperature=0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.other_params = kwargs

    def set_model(self, model):
        if not isinstance(model, LLMModel):
            raise ValueError("model must be an instance of LLMModel enum")
        self.model = model

    def generate_text(self, input_text):
        if self.model == LLMModel.OPENAI:
            return self._generate_with_openai(input_text)
        elif self.model == LLMModel.OTHERAPI:
            return self._generate_with_otherapi(input_text)
        else:
            raise ValueError("Unsupported model")

    def _generate_with_openai(self, input_text):
        return openai_llm.generate(input_text)
        # Use self.temperature and self.other_params as needed
        pass

    def _generate_with_otherapi(self, input_text):
        # Code to generate text using another API
        # Use self.temperature and self.other_params as needed
        pass