
from pydantic import BaseModel



# import time
# from typing import Tuple, Union
from typing import List
# from typing import Tuple, Type, Union



# from SimplerLLM.tools.json_helpers import (
#     convert_pydantic_to_json,
#     extract_json_from_text,
#     convert_json_to_pydantic_model,
#     validate_json_with_pydantic_model,
#     generate_json_example_from_pydantic
#     )

from SimplerLLM.langauge.llm import LLM, LLMProvider
from SimplerLLM.langauge.llm_addons import generate_basic_pydantic_json_model


llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")

llm_gemini = LLM.create(provider=LLMProvider.GEMENI, model_name="gemini-pro")


base_prompt= """
I'm working on exploring the subtopics under a specific top-level topic. T
he topic I'm interested in is [{topic}]. Could you help me by generating a comprehensive list of 3 
child topics that fall under this main topic? Please consider various aspects, 
related fields, subfields, and any specific areas that are commonly associated with it. 
"""



#prompt = base_prompt.format(topic="Physics")


# #Manually Set
class Topics(BaseModel):
    list: List[str]


class YouTubeTitles(BaseModel):
    list: List[str]
    topic: str

base_pr = "Generate 5 youtube titles for a video about {topic}"

prompt = base_pr.format(topic="SEO")

model_response = generate_basic_pydantic_json_model(model_class=YouTubeTitles, prompt=prompt, llm_instance=llm_gemini)

print(model_response)

# json_model = generate_json_example_from_pydantic(Topics)


# optimized_prompt = prompt + f'\n\n.The response should me a structured JSON format that matches the following JSON: {json_model}'


# ai_response = llm_instance.generate_text(optimized_prompt)


# json_object = extract_json_from_text(ai_response)


# validated, errors = validate_json_with_pydantic_model(Topics, json_object)

# if not errors:
#     model_object = convert_json_to_pydantic_model(Topics, json_object[0])
#     return model_object



# tiny_list = [
#     "Technology",
#     "Health",
#     "Business",
# ]

# sub_level_topics = []
# for parent_topic in tiny_list:
#     prompt = base_prompt.format(topic=parent_topic)
#     optimized_prompt = prompt + f'.Please provide a response in a structured JSON format that matches the following Pydantic model: {json_model}'

#     # Generate content using the modified prompt
#     genmeni_response = llm_instance.generate_text(optimized_prompt)

#     # Extract and validate the JSON from the LLM's response
#     json_object = extract_json_from_text(genmeni_response)

#     #validate the response
#     validated, errors = validate_json_with_pydantic_model(Topics, json_object)
#     if not errors:
#         model_object = convert_json_to_pydantic_model(Topics, json_object[0])
#         #play with json
#         for topic in model_object.list:
#             sub_level_topics.append(topic)



# print(len(sub_level_topics))
# print(sub_level_topics)