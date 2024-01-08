class SimpleTemplate:
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

def create_template(template_string: str) -> SimpleTemplate:
    """
    Factory function to create a SimpleTemplate instance.
    """
    if not isinstance(template_string, str):
        raise ValueError("Template string must be a string")
    return SimpleTemplate(template_string)
