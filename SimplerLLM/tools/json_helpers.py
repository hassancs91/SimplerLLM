import re
import json
from pydantic import BaseModel, ValidationError
from typing import get_type_hints
from pydantic import BaseModel
from typing import Type, get_type_hints, List, get_origin, get_args, Union, Dict, Any



def convert_pydantic_to_json(model_instance):
    """
    Converts a Pydantic model instance to a JSON string.

    Args:
        model_instance (YourModel): An instance of your Pydantic model.

    Returns:
        str: A JSON string representation of the model.
    """
    return model_instance.model_dump_json()


    

def extract_json_from_text(text_response):
    """
    Extracts JSON objects from text. First tries to parse the entire text as JSON,
    then falls back to regex-based extraction if that fails.
    
    Args:
        text_response (str): Text that may contain JSON objects
        
    Returns:
        list or None: List of extracted JSON objects, or None if no valid JSON found
    """
    # First try to parse the entire text as JSON directly
    try:
        json_obj = json.loads(text_response)
        return [json_obj]  # Return as a list to maintain compatibility
    except json.JSONDecodeError:
        # If direct parsing fails, fall back to extraction
        pattern = r'\{.*?\}'
        matches = re.finditer(pattern, text_response, re.DOTALL)
        json_objects = []

        for match in matches:
            json_str = __json_extend_search(text_response, match.span())
            try:
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Try with the deprecated method as a last resort
                try:
                    json_str = __extend_search_deprecated(text_response, match.span())
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue

        return json_objects if json_objects else None


def __json_extend_search(text, span):
    start, end = span
    nest_count = 1  # Starts with 1 since we know '{' is at the start position
    for i in range(end, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end] 



@DeprecationWarning
def __extend_search_deprecated(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            nest_count += 1
        elif text[i] == "}":
            nest_count -= 1
            if nest_count == 0:
                return text[start : i + 1]
    return text[start:end]


def validate_json_with_pydantic_model(model_class, json_data):
    """
    Validates JSON data against a specified Pydantic model.

    Args:
        model_class (BaseModel): The Pydantic model class to validate against.
        json_data (dict or list): JSON data to validate. Can be a dict for a single JSON object,
                                  or a list for multiple JSON objects.

    Returns:
        list: A list of validated JSON objects that match the Pydantic model.
        list: A list of errors for JSON objects that do not match the model.
    """
    validated_data = []
    validation_errors = []

    if isinstance(json_data, list):
        for item in json_data:
            try:
                model_instance = model_class(**item)
                validated_data.append(model_instance.model_dump())
            except ValidationError as e:
                validation_errors.append({"error": str(e), "data": item})
    elif isinstance(json_data, dict):
        try:
            model_instance = model_class(**json_data)
            validated_data.append(model_instance.model_dump())
        except ValidationError as e:
            validation_errors.append({"error": str(e), "data": json_data})
    else:
        raise ValueError("Invalid JSON data type. Expected dict or list.")

    return validated_data, validation_errors


def convert_json_to_pydantic_model(model_class, json_data):
    try:
        model_instance = model_class(**json_data)
        return model_instance
    except ValidationError as e:
        return None


# Define a function to provide example values based on type
def example_value_for_type(field_type: Type):
    # Handle Any type specifically
    if field_type is Any:
        return "example_value"  # Return a simple string for Any type
        
    origin = get_origin(field_type)
    if origin is None:  # Not a generic type
        if issubclass(field_type, BaseModel):  # Check if it's a custom Pydantic model
            # Generate an example using all fields of the model
            example_data = {field_name: example_value_for_type(field_type)
                            for field_name, field_type in get_type_hints(field_type).items()}
            return field_type(**example_data)
        elif field_type == str:
            return "example_string"
        elif field_type == int:
            return 0
        elif field_type == float:
            return 0.0
        elif field_type == bool:
            return True
        else:
            return "example_value"  # More generic fallback value
    elif origin == list:  # It's a List
        args = get_args(field_type)
        if not args:
            return []  # No type specified for elements, return empty list
        element_type = args[0]
        # Create a list with 3 elements of the specified type
        return [example_value_for_type(element_type) for _ in range(3)]
    elif origin == dict:  # It's a Dict
        args = get_args(field_type)
        if not args or len(args) < 2:
            return {}  # No type specified for keys/values, return empty dict
        key_type, value_type = args[0], args[1]
        # Create a dict with example key-value pairs
        example_dict = {}
        for i in range(1, 3):  # Create 2 example key-value pairs
            example_dict[f"example_key_{i}"] = example_value_for_type(value_type)
        return example_dict
    elif origin == Union:  # Handle Optional (Union[Type, None])
        args = get_args(field_type)
        # If one of the args is NoneType, it's an Optional
        if type(None) in args:
            # Find the non-None type
            for arg in args:
                if arg is not type(None):
                    return example_value_for_type(arg)
        # For other Union types, use the first type
        return example_value_for_type(args[0]) if args else None

def generate_json_example_from_pydantic(model_class: Type[BaseModel]) -> str:
    example_data = {}
    for field_name, field_type in get_type_hints(model_class).items():
        example_data[field_name] = example_value_for_type(field_type)

    model_instance = model_class(**example_data)
    # Use standard json module instead of model_dump_json to ensure proper formatting
    import json
    return json.dumps(model_instance.model_dump())
