class MessagesTemplate:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def validate_alternation(self):
        if not self.messages:
            return False, "MessageTemplate is empty."

        if self.messages[0]["role"] != "user":
            return False, "MessageTemplate must start with a user message."

        if self.messages[-1]["role"] != "user":
            return False, "MessageTemplate must end with a user message."

        for i in range(len(self.messages) - 1):
            if self.messages[i]["role"] == self.messages[i + 1]["role"]:
                return False, f"Consecutive messages found at index {i} and {i + 1}."

        return True, "MessageTemplate is valid."


    def get_messages(self):
        # Validate the message alternation
        is_valid, validation_message = self.validate_alternation()
        if is_valid:
            return self.messages
        else:
            raise ValueError(validation_message)
        

    def __repr__(self):
        return f"MessageTemplate({self.messages})"
    


