class PromptTemplate:
    def __init__(self, template):
        self._template = template
        self._placeholders = {}
        self._final_prompt = template  # Initialize with the template itself

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._placeholders[key] = value
            self._update_prompt()

    def _update_prompt(self):
        """Updates the internal representation of the final prompt."""
        self._final_prompt = self._template
        for placeholder, value in self._placeholders.items():
            self._final_prompt = self._final_prompt.replace(f"{{{placeholder}}}", str(value))

    def list_placeholders(self):
        """Lists the current placeholders set in the template."""
        return list(self._placeholders.keys())

    def __str__(self):
        """Returns the final prompt when the object is used in a string context."""
        return self._final_prompt

    def __repr__(self):
        """Also returns the final prompt for consistent representation in all contexts."""
        return self._final_prompt





def create_template(prompt_with_placeholders):
    """Create a new PromptTemplate object with the given prompt."""
    return PromptTemplate(prompt_with_placeholders)