from SimplerLLM.langauge.llm import LLM, LLMProvider
import SimplerLLM.prompts.prompt_builder as pr
from SimplerLLM.tools.content_loader import read_content_from_url


# Example usage:
llm_instance = LLM.create(provider=LLMProvider.OPENAI,model_name="gpt-3.5-turbo-1106")
#gemeni_instanat = LLM.create(provider=LLMProvider.GEMENI,model_name="gemini-pro")

#print(gemeni_instanat.generate_full_response("generate a 5 words sentence"))


prompt_template_plain = '''
I am building a dataset of questions to 
use in a machine learning project,the questions 
are general knowledge questions and the answers of 
these questions must be simple and can be extracted 
easily from a context. let's start with {topic}, 
generate a list of 100 questions without numbering, just questions without answers.'''


##prompt = pr.create_template(prompt_template_plain)
#prompt.assign_parms(topic = "geography")


#reponse = llm_instance.generate_text(prompt.content,max_tokens=4096,model_name="gpt-3.5-turbo-1106")
##print(reponse)



# Use the template directly
# print(prompt.content)  # Prints: "Hello, my name is Alice and I enjoy painting."

# print(type(prompt.content)) 

# reponses = generator.generate(prompt.content)

# generator.print_responses(reponses)


#few shot prompts
# Assuming the FewShotPrompt class is already defined as per the previous example

# Step 1: Define the template
# multi_value_prompt_template = "Hello {name}, your next meeting is on {date}."

# # Step 2: Create parameter sets
# params_list = [
#     {"name": "Alice", "date": "January 10th"},
#     {"name": "Bob", "date": "January 12th"},
#     {"name": "Charlie", "date": "January 15th"}
# ]

# # Step 3: Generate prompts
# multi_value_prompt = pr.create_multi_value_prompts(multi_value_prompt_template)
# generated_prompts = multi_value_prompt.generate_prompts(params_list)

# # Step 4: Display the generated prompts
# for prompt in generated_prompts:
#     print(prompt)
#results = search_with_value_serp_api("baseball",5)
#print(results[0]["link"])
url = "https://learnwithhasan.com/keyword-research-with-ai/"

content = read_content_from_url(url)

print (content.title)