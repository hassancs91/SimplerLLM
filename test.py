import SimplerLLM.text_openai as generator
import SimplerLLM.prompt_builder as pr



# Example usage:
prompt_template_plain = "Hello, my name is {name} and I enjoy {activity}."
prompt = pr.create_template(prompt_template_plain)
prompt.assign_parms(name = "Hasan", activity="Baseball")

# Use the template directly
print(prompt.content)  # Prints: "Hello, my name is Alice and I enjoy painting."

print(type(prompt.content)) 

reponses = generator.generate(prompt)

generator.print_responses(reponses)