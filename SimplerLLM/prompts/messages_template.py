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

    def prepend_messages(self, messages_list):
        """Prepends the current messages with a list of messages, ensuring alternation."""
        if not all(isinstance(message, dict) and 'role' in message and 'content' in message for message in messages_list):
            raise ValueError("All items in the list must be dictionaries with 'role' and 'content' keys")

        if self.messages and messages_list:
            # Check if the first message of this template and the last message of the messages_list are from the same role
            if self.messages[0]['role'] == messages_list[-1]['role']:
                raise ValueError("Cannot merge messages due to role conflict at the boundary.")

        # Prepend by combining lists with messages_list coming first
        self.messages = messages_list + self.messages

    def get_messages(self):
        # Validate the message alternation
        is_valid, validation_message = self.validate_alternation()
        if is_valid:
            return self.messages
        else:
            raise ValueError(validation_message)
        
    def get_last_message(self):
        """Returns the last message if available, otherwise returns None."""
        if not self.messages:
            return None
        return self.messages[-1]
        
    def __repr__(self):
        return f"MessageTemplate({self.messages})"
    


