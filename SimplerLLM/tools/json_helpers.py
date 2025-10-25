import re
import json
from pydantic import BaseModel, ValidationError
from typing import get_type_hints
from pydantic import BaseModel
from typing import Type, get_type_hints, List, get_origin, get_args, Union, Dict, Any, Literal
from enum import Enum
from datetime import datetime, date, time
from uuid import UUID



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


def get_field_constraints(field_info):
    """
    Extract constraints from a Pydantic FieldInfo object.

    Args:
        field_info: Pydantic FieldInfo object containing field metadata

    Returns:
        dict: Dictionary of constraints (ge, le, gt, lt, min_length, max_length, min_items, max_items, pattern)
    """
    constraints = {}

    # Check if field_info has constraints attribute (Pydantic v2)
    if hasattr(field_info, 'constraints'):
        for constraint in field_info.constraints:
            constraint_type = type(constraint).__name__
            if hasattr(constraint, 'ge'):
                constraints['ge'] = constraint.ge
            if hasattr(constraint, 'le'):
                constraints['le'] = constraint.le
            if hasattr(constraint, 'gt'):
                constraints['gt'] = constraint.gt
            if hasattr(constraint, 'lt'):
                constraints['lt'] = constraint.lt
            if hasattr(constraint, 'min_length'):
                constraints['min_length'] = constraint.min_length
            if hasattr(constraint, 'max_length'):
                constraints['max_length'] = constraint.max_length
            if hasattr(constraint, 'pattern'):
                constraints['pattern'] = constraint.pattern

    # Also check metadata attribute for Annotated types
    if hasattr(field_info, 'metadata'):
        for metadata in field_info.metadata:
            if hasattr(metadata, 'ge'):
                constraints['ge'] = metadata.ge
            if hasattr(metadata, 'le'):
                constraints['le'] = metadata.le
            if hasattr(metadata, 'gt'):
                constraints['gt'] = metadata.gt
            if hasattr(metadata, 'lt'):
                constraints['lt'] = metadata.lt
            if hasattr(metadata, 'min_length'):
                constraints['min_length'] = metadata.min_length
            if hasattr(metadata, 'max_length'):
                constraints['max_length'] = metadata.max_length
            if hasattr(metadata, 'min_items'):
                constraints['min_items'] = metadata.min_items
            if hasattr(metadata, 'max_items'):
                constraints['max_items'] = metadata.max_items
            if hasattr(metadata, 'pattern'):
                constraints['pattern'] = metadata.pattern

    return constraints


# Define a function to provide example values based on type
def example_value_for_type(field_type: Type, constraints: dict = None, _recursion_depth: int = 0, _seen_types: set = None):
    """
    Generate example values for a given type, respecting Pydantic field constraints.

    Args:
        field_type: The type to generate an example for
        constraints: Optional dict of constraints (ge, le, gt, lt, min_length, max_length, etc.)
        _recursion_depth: Internal parameter to track recursion depth
        _seen_types: Internal parameter to track seen types for circular reference detection

    Returns:
        An example value that satisfies the type and constraints
    """
    if constraints is None:
        constraints = {}

    if _seen_types is None:
        _seen_types = set()

    # Prevent infinite recursion for self-referential models
    MAX_RECURSION_DEPTH = 10  # Allow deep nesting but prevent infinite loops
    if _recursion_depth > MAX_RECURSION_DEPTH:
        return None  # Stop recursion for deeply nested or circular models

    # Handle Any type specifically
    if field_type is Any:
        return "example_value"  # Return a simple string for Any type

    # Handle datetime types
    if field_type == datetime:
        return "2025-01-15T10:30:00"
    elif field_type == date:
        return "2025-01-15"
    elif field_type == time:
        return "10:30:00"

    # Handle UUID
    if field_type == UUID:
        return "12345678-1234-5678-1234-567812345678"

    origin = get_origin(field_type)

    # Handle Literal type
    if origin == Literal:
        literal_values = get_args(field_type)
        return literal_values[0] if literal_values else None

    if origin is None:  # Not a generic type
        # Check if it's an Enum (before BaseModel check)
        try:
            if issubclass(field_type, Enum):
                # Return the first enum value
                enum_values = list(field_type)
                return enum_values[0].value if enum_values else "enum_value"
        except TypeError:
            pass  # Not a class, continue

        if issubclass(field_type, BaseModel):  # Check if it's a custom Pydantic model
            # Check for circular reference
            type_id = id(field_type)
            if type_id in _seen_types:
                return None  # Circular reference detected, return None for Optional fields

            # Add to seen types
            _seen_types = _seen_types.copy()  # Create new set for this branch
            _seen_types.add(type_id)

            # Generate an example using all fields of the model
            example_data = {}
            if hasattr(field_type, 'model_fields'):
                for field_name, field_info in field_type.model_fields.items():
                    field_constraints = get_field_constraints(field_info)
                    field_type_hint = get_type_hints(field_type).get(field_name)
                    example_data[field_name] = example_value_for_type(
                        field_type_hint,
                        field_constraints,
                        _recursion_depth + 1,
                        _seen_types
                    )
            else:
                # Fallback for older Pydantic versions
                for field_name, field_type_hint in get_type_hints(field_type).items():
                    example_data[field_name] = example_value_for_type(
                        field_type_hint,
                        None,
                        _recursion_depth + 1,
                        _seen_types
                    )
            return field_type(**example_data)
        elif field_type == str:
            min_len = constraints.get('min_length', 0)
            max_len = constraints.get('max_length', None)
            # Generate string of appropriate length
            base_string = "example_string"

            if min_len > 0:
                # Need to meet minimum length requirement
                target_length = min(min_len, max_len) if max_len else min_len
            elif max_len:
                # Only max length constraint
                target_length = min(15, max_len)
            else:
                # No constraints
                target_length = 15

            # Generate string of target length
            if target_length <= len(base_string):
                return base_string[:target_length]
            else:
                # Need to pad or repeat to meet minimum length
                repeats = (target_length // len(base_string)) + 1
                repeated = (base_string + " ") * repeats
                return repeated[:target_length]
        elif field_type == int:
            # Respect numeric constraints
            if 'ge' in constraints:
                return constraints['ge']
            elif 'gt' in constraints:
                return constraints['gt'] + 1
            elif 'le' in constraints:
                return max(0, constraints['le'])
            elif 'lt' in constraints:
                return max(0, constraints['lt'] - 1)
            return 0
        elif field_type == float:
            # Respect numeric constraints
            # Always ensure float has decimal part to distinguish from int
            if 'ge' in constraints and 'le' in constraints:
                # Use middle value
                value = (constraints['ge'] + constraints['le']) / 2.0
                # Ensure it has a decimal part
                return value if value % 1 != 0 else value + 0.5
            elif 'ge' in constraints:
                value = float(constraints['ge'])
                # If ge is whole number, add 0.5 to clearly show it's a float
                return value if value % 1 != 0 else value + 0.5
            elif 'gt' in constraints:
                return float(constraints['gt']) + 0.5
            elif 'le' in constraints:
                value = max(0.0, float(constraints['le']))
                return value if value % 1 != 0 else value - 0.5 if value > 0.5 else 0.5
            elif 'lt' in constraints:
                return max(0.5, float(constraints['lt']) - 0.5)
            return 0.5  # Default to 0.5 instead of 0.0 to clearly show it's a float
        elif field_type == bool:
            return True
        else:
            return "example_value"  # More generic fallback value
    elif origin == list:  # It's a List
        args = get_args(field_type)
        if not args:
            return []  # No type specified for elements, return empty list
        element_type = args[0]
        # Respect min_length/max_length (Pydantic v2) and min_items/max_items (Pydantic v1)
        min_items = constraints.get('min_length') or constraints.get('min_items', 2)
        max_items = constraints.get('max_length') or constraints.get('max_items', None)

        # Calculate number of items to generate
        if max_items:
            num_items = min(min_items, max_items)
        else:
            num_items = min_items

        # Ensure at least 2 items for a good example (unless max is less)
        if max_items:
            num_items = max(min(num_items, max_items), min(2, max_items))
        else:
            num_items = max(num_items, 2)

        # Create a list with appropriate number of elements
        # Filter out None values (which occur from circular references or max recursion depth)
        items = []
        for _ in range(num_items):
            value = example_value_for_type(element_type, {}, _recursion_depth + 1, _seen_types)
            if value is not None:
                items.append(value)

        # If no valid items could be generated (all were None due to circular refs),
        # return None to signal this to parent Optional field
        if len(items) == 0 and num_items > 0:
            return None

        return items
    elif origin == dict:  # It's a Dict
        args = get_args(field_type)
        if not args or len(args) < 2:
            return {}  # No type specified for keys/values, return empty dict
        key_type, value_type = args[0], args[1]
        # Create a dict with example key-value pairs
        example_dict = {}
        for i in range(1, 3):  # Create 2 example key-value pairs
            example_dict[f"example_key_{i}"] = example_value_for_type(value_type, {}, _recursion_depth + 1, _seen_types)
        return example_dict
    elif origin == Union:  # Handle Optional (Union[Type, None])
        args = get_args(field_type)
        # If one of the args is NoneType, it's an Optional
        if type(None) in args:
            # Find the non-None type
            for arg in args:
                if arg is not type(None):
                    return example_value_for_type(arg, constraints, _recursion_depth, _seen_types)
        # For other Union types, use the first type
        return example_value_for_type(args[0], constraints, _recursion_depth, _seen_types) if args else None

def generate_json_example_from_pydantic(model_class: Type[BaseModel]) -> str:
    """
    Generate a valid JSON example from a Pydantic model class.

    This function first checks if the model has a json_schema_extra with example data.
    If not, it generates example values that respect field constraints (ge, le, min_length, etc.).

    Args:
        model_class: Pydantic BaseModel class to generate example from

    Returns:
        str: JSON string representation of a valid example
    """
    # First, check if the model has json_schema_extra with example data
    if hasattr(model_class, 'model_config'):
        model_config = model_class.model_config
        if isinstance(model_config, dict) and 'json_schema_extra' in model_config:
            schema_extra = model_config['json_schema_extra']
            if isinstance(schema_extra, dict) and 'example' in schema_extra:
                # Use the provided example directly (it's already validated)
                import json
                return json.dumps(schema_extra['example'])

    # Fallback: check old-style Config class for Pydantic v1 compatibility
    if hasattr(model_class, 'Config') and hasattr(model_class.Config, 'json_schema_extra'):
        schema_extra = model_class.Config.json_schema_extra
        if isinstance(schema_extra, dict) and 'example' in schema_extra:
            import json
            return json.dumps(schema_extra['example'])

    # Generate example data using field constraints
    example_data = {}
    if hasattr(model_class, 'model_fields'):
        # Pydantic v2
        for field_name, field_info in model_class.model_fields.items():
            field_constraints = get_field_constraints(field_info)
            field_type = get_type_hints(model_class).get(field_name)
            if field_type:
                # Always use field_name for model construction (not alias)
                example_data[field_name] = example_value_for_type(field_type, field_constraints)
    else:
        # Pydantic v1 or fallback
        for field_name, field_type in get_type_hints(model_class).items():
            example_data[field_name] = example_value_for_type(field_type)

    model_instance = model_class(**example_data)
    # Use model_dump_json with by_alias=True to:
    # 1. Export field aliases instead of field names
    # 2. Convert datetime/UUID objects to JSON-serializable strings
    # model_dump_json handles serialization directly and respects aliases better
    return model_instance.model_dump_json(by_alias=True)
