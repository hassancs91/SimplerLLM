import SimplerLLM.text_openai as generator
import SimplerLLM.prompt_templates as pr


# Example usage:
template = pr.create_template("Hello, my name is {name} and I enjoy {activity}.")
template.name = "Alice"
template.activity = "painting"

# Use the template directly
print(template)  # Prints: "Hello, my name is Alice and I enjoy painting."
#final_prompt = str(template)
# List current placeholders
print("Current placeholders:", template.list_placeholders())

print(type(template)) 

#reponses = generator.generate(template)

#generator.print_responses(reponses)