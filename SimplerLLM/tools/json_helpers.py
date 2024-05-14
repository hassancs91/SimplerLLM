import re
import json
from pydantic import BaseModel, ValidationError
from typing import get_type_hints
from pydantic import BaseModel
from typing import Type, get_type_hints, List, get_origin, get_args



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
    pattern = r'\{.*?\}'
    matches = re.finditer(pattern, text_response, re.DOTALL)
    json_objects = []

    for match in matches:
        json_str = __json_extend_search(text_response, match.span())
        try:
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
def __extract_json_from_text_deprecated(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r"\{[^{}]*\}"

    matches = re.finditer(pattern, text_response)
    json_objects = []

    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue

    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer


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
                validated_data.append(model_instance.dict())
            except ValidationError as e:
                validation_errors.append({"error": str(e), "data": item})
    elif isinstance(json_data, dict):
        try:
            model_instance = model_class(**json_data)
            validated_data.append(model_instance.dict())
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
            return "Unsupported type"
    elif origin == list:  # It's a List
        args = get_args(field_type)
        if not args:
            return []  # No type specified for elements, return empty list
        element_type = args[0]
        # Create a list with 3 elements of the specified type
        return [example_value_for_type(element_type) for _ in range(3)]

def generate_json_example_from_pydantic(model_class: Type[BaseModel]) -> str:
    example_data = {}
    for field_name, field_type in get_type_hints(model_class).items():
        example_data[field_name] = example_value_for_type(field_type)

    model_instance = model_class(**example_data)
    return model_instance.json()