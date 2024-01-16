import SimplerLLM.llms.openai_llm as generator
import SimplerLLM.prompts.prompt_builder as pr
from SimplerLLM.tools.blogs import read_content_from_url

# Example usage:
prompt_template_plain = "Hello, my name is {name} and I enjoy {activity}."
prompt = pr.create_template(prompt_template_plain)
prompt.assign_parms(name = "Hasan", activity="Baseball")

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


url = "https://learnwithhasan.com/keyword-research-with-ai/"

content = read_content_from_url(url)

print ("ok")