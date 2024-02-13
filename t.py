import asyncio
from SimplerLLM.langauge.llm import LLM, LLMProvider
import SimplerLLM.prompts.prompt_builder as pr
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


class QuesionsModel(BaseModel):
    Questions: List[str]


llm_instance = LLM.create(model=LLMProvider.OPENAI)


#prompt = " generate 5 random catchy titles in a json format"

#result : TitlesModel = llm_instance.generate_json_with_pydantic(prompt,TitlesModel,"gpt-3.5-turbo-1106")
#print(result.Titles[0])
prompt_template_plain = '''
I am building a dataset of questions to 
use in a machine learning project,the questions 
are general knowledge questions and the answers of 
these questions must be simple and can be extracted 
easily from a context. let's start with {topic}, 
generate a list of 100 questions without numbering, just questions without answers, and return the list as a valid json format'''

topics = [
    "The Solar System and Planets",
    "Life Cycle of a Butterfly",
    "Simple Machines: Levers, Wheels, and Pulleys",
    "Basics of the Human Body",
    "Dinosaurs and Prehistoric Life",
    "States of Matter: Solids, Liquids, Gases",
    "Weather Patterns and Phenomena",
    "Plant Life: Photosynthesis and Growth",
    "The Water Cycle",
    "Volcanoes and Earthquakes",
    "Introduction to Ecosystems",
    "Insects and Their Habitats",
    "Underwater Life and Oceans",
    "Types of Rocks and Minerals",
    "The Moon and its Phases",
    "Energy Sources: Renewable and Non-renewable",
    "Magnetism and Electricity",
    "The Five Senses",
    "Birds and Their Characteristics",
    "The Science of Rainbows",
    "Astronauts and Space Exploration",
    "Recycling and Conservation",
    "Food Chains and Food Webs",
    "The Human Brain and Learning",
    "Microscopic Organisms",
    "Simple Chemical Reactions",
    "Animal Adaptations",
    "The Seasons and Earth's Tilt",
    "Light and Sound Waves",
    "Trees and Forest Ecosystems",
    "Gravity and How it Works",
    "The Human Skeletal System",
    "Endangered Animals and Protection",
    "Stars and Constellations",
    "Reptiles and Amphibians",
    "Fossils and Paleontology",
    "Bacteria and Viruses",
    "Renewable Energy: Solar and Wind Power",
    "Nutrition and Healthy Eating",
    "The Science of Flight",
    "The Human Circulatory System",
    "Deserts and Their Features",
    "Polar Regions and Wildlife",
    "Rainforests and Biodiversity",
    "Life on a Farm",
    "Physical and Chemical Changes",
    "The Human Respiratory System",
    "Circuits and Electronics",
    "Metamorphosis in Frogs and Insects",
    "The Universe and Galaxies"
]

with open('output.txt', 'w', encoding='utf-8') as file:
    for topic in topics:
        prompt = pr.create_template(prompt_template_plain)
        prompt.assign_parms(topic=topic)
        result : QuesionsModel = llm_instance.generate_json_with_pydantic(prompt.content, QuesionsModel, "gpt-3.5-turbo-1106")

        for q in result.Questions:
            file.write(q + '\n')







# async def generate_text():
#     llm_instance = LLM.create(model=LLMProvider.OPENAI)
#     result = await llm_instance.generate_text_basic_async("What is a dolphin in 2 sentences?")
#     print(result)

# Running the async function
#asyncio.run(generate_text())