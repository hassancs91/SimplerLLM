class SimplePrompt:
    """
    A class for creating and manipulating simple prompt templates.
    """

    def __init__(self, template: str):
        if not isinstance(template, str):
            raise ValueError("Template must be a string")
        self.template = template
        self.content = ''  # Holds the latest filled template

    def assign_parms(self, **kwargs) -> str:
        """
        Assigns parameters to the template and returns the filled template.
        """
        try:
            self.content = self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing a required key in the template: {e}")
        except Exception as e:
            # Catch-all for other exceptions related to string formatting
            raise ValueError(f"Error processing the template: {e}")
        return self.content

    def update_template(self, new_template: str):
        """
        Updates the template and clears the latest content.
        """
        if not isinstance(new_template, str):
            raise ValueError("New template must be a string")
        self.template = new_template
        self.content = ''

    def __str__(self) -> str:
        return self.content

def create_prompt_template(template_string: str) -> SimplePrompt:
    """
    Factory function to create a SimpleTemplate instance.
    """
    if not isinstance(template_string, str):
        raise ValueError("Template string must be a string")
    return SimplePrompt(template_string)



class MultiValuePrompt:
    """
    A class for creating and manipulating prompt templates with multiple sets of parameters.
    """

    def __init__(self, template: str):
        if not isinstance(template, str):
            raise ValueError("Template must be a string")
        self.template = template
        self.generated_prompts = []  # Holds the generated prompts

    def generate_prompts(self, params_list: list) -> list:
        """
        Generates prompts for each set of parameters in the params_list.
        """
        if not all(isinstance(params, dict) for params in params_list):
            raise ValueError("Each item in params_list must be a dictionary")

        self.generated_prompts = []
        for params in params_list:
            try:
                filled_prompt = self.template.format(**params)
                self.generated_prompts.append(filled_prompt)
            except KeyError as e:
                raise KeyError(f"Missing a required key in the template: {e}")
            except Exception as e:
                raise ValueError(f"Error processing the template: {e}")

        return self.generated_prompts

    def __str__(self) -> str:
        return "\n".join(self.generated_prompts)

def create_multi_value_prompts(template_string: str) -> MultiValuePrompt:
    """
    Factory function to create a FewShotPrompt instance.
    """
    if not isinstance(template_string, str):
        raise ValueError("Template string must be a string")
    return MultiValuePrompt(template_string)