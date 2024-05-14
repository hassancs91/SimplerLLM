


def save_text_to_file(text, filename="output.txt"):
    """
    Saves the given text to a file specified by the filename.
    
    Parameters:
        text (str): The text to be saved to the file. This can be any length and include Unicode characters.
        filename (str): The name of the file where the text will be saved. Default is "output.txt".
        
    Returns:
        bool: True if the file is saved successfully, False otherwise.
        
    Raises:
        TypeError: If the provided text is not a string.
        OSError: If there are issues with writing to the file system, such as permissions.
        
    Examples:
        >>> save_text_to_file("Hello, world!", "greeting.txt")
        True
        
        >>> save_text_to_file(123, "numbers.txt")
        TypeError: Provided text must be a string.
    """
    # Check if the provided 'text' is indeed a string
    if not isinstance(text, str):
        raise TypeError("Provided text must be a string.")
    
    try:
        # Open the file in write mode with encoding set to utf-8
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        return True
    except OSError as e:
        # Handle exceptions that may occur during file writing
        print(f"An error occurred while writing to the file: {e}")
        return False
