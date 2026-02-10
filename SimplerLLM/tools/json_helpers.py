import re
import json
from decimal import Decimal
from typing import Type, get_type_hints, List, get_origin, get_args, Union, Dict, Any, Literal, Set, FrozenSet, Tuple
from enum import Enum
from datetime import datetime, date, time
from uuid import UUID

try:
    from pydantic import BaseModel, ValidationError, RootModel
    HAS_ROOT_MODEL = True
except ImportError:
    from pydantic import BaseModel, ValidationError
    HAS_ROOT_MODEL = False
    RootModel = None



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

    # Check if this is a RootModel
    is_root_model = False
    if HAS_ROOT_MODEL and RootModel is not None:
        try:
            is_root_model = issubclass(model_class, RootModel)
        except TypeError:
            pass

    if isinstance(json_data, list):
        for item in json_data:
            try:
                # RootModel takes value directly, not as kwargs
                if is_root_model:
                    model_instance = model_class(item)
                else:
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
        # Handle RootModel - it takes the value directly, not as kwargs
        if HAS_ROOT_MODEL and RootModel is not None:
            try:
                if issubclass(model_class, RootModel):
                    model_instance = model_class(json_data)
                    return model_instance
            except TypeError:
                pass  # Not a class that can be checked with issubclass

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


def generate_string_from_pattern(pattern: str) -> str:
    """
    Generate a string that matches a regex pattern.

    Handles common patterns directly; for complex patterns,
    returns a best-effort example.
    """
    # Common pattern mappings (pattern -> example)
    common_patterns = {
        r'^\d{3}-\d{3}-\d{4}$': '555-123-4567',
        r'^\d{5}$': '90210',
        r'^[A-Z]{2}-\d{4}$': 'AB-1234',
        r'^\d{4}-\d{2}-\d{2}$': '2025-01-15',
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$': 'user@example.com',
        r'^\d{3}-\d{2}-\d{4}$': '123-45-6789',
        r'^https?://': 'https://example.com',
    }

    # Check for exact matches first
    if pattern in common_patterns:
        return common_patterns[pattern]

    # Simple pattern parsing fallback
    result = []
    clean = pattern.strip('^$')
    i = 0
    while i < len(clean):
        if clean[i:i+2] == '\\d':
            result.append('5')
            i += 2
        elif clean[i] == '-':
            result.append('-')
            i += 1
        elif clean[i] == '{':
            end = clean.find('}', i)
            if end != -1 and result:
                count = int(clean[i+1:end].split(',')[0]) - 1
                result.extend([result[-1]] * count)
            i = end + 1 if end != -1 else i + 1
        elif clean[i] == '[':
            end = clean.find(']', i)
            if end != -1:
                cls = clean[i+1:end]
                if 'A-Z' in cls:
                    result.append('A')
                elif 'a-z' in cls:
                    result.append('a')
                elif '0-9' in cls:
                    result.append('5')
                else:
                    result.append(cls[0] if cls else 'x')
            i = end + 1 if end != -1 else i + 1
        elif clean[i] not in '().*+?|':
            result.append(clean[i])
            i += 1
        else:
            i += 1

    return ''.join(result) if result else 'example_value'


def get_smart_example_for_field(field_name: str, field_type: Type, constraints: dict) -> Any:
    """
    Generate context-aware examples based on field name.

    This helps generate values that are more likely to pass custom validators.
    """
    name = field_name.lower()

    if field_type == str:
        # Password fields - include uppercase, lowercase, digit
        if 'password' in name or 'pwd' in name:
            min_len = constraints.get('min_length', 8)
            return f"Secure1{'x' * max(0, min_len - 7)}!"

        # Username fields - alphanumeric with underscores
        if 'username' in name or 'user_name' in name:
            return 'john_doe123'

        # Email fields
        if 'email' in name:
            return 'user@example.com'

        # URL/website fields
        if 'url' in name or 'website' in name:
            return 'https://example.com'

        # Phone fields
        if 'phone' in name or 'tel' in name:
            return '555-123-4567'

        # Name fields
        if 'name' in name:
            if 'first' in name:
                return 'John'
            elif 'last' in name:
                return 'Doe'
            else:
                return 'Example Name'

    return None  # No smart example, use default


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

    # Handle Decimal type
    if field_type == Decimal:
        if 'ge' in constraints:
            return str(constraints['ge'])
        elif 'gt' in constraints:
            return str(float(constraints['gt']) + 0.01)
        elif 'le' in constraints:
            return str(constraints['le'])
        elif 'lt' in constraints:
            return str(float(constraints['lt']) - 0.01)
        return "123.45"

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
            # Check for pattern constraint first
            if 'pattern' in constraints and constraints['pattern']:
                return generate_string_from_pattern(constraints['pattern'])

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
        elif field_type == dict:
            return {}  # Plain dict type returns empty dict
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
        # return empty list instead of None to maintain list type consistency
        if len(items) == 0 and num_items > 0:
            return []

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
    elif origin == set or origin == frozenset:  # Handle Set and FrozenSet
        args = get_args(field_type)
        if not args:
            return []  # No type specified, return empty list
        element_type = args[0]
        min_items = constraints.get('min_length') or constraints.get('min_items', 2)

        items = []
        for i in range(max(min_items, 2)):
            value = example_value_for_type(element_type, {}, _recursion_depth + 1, _seen_types)
            if value is not None:
                # For strings, append index to ensure uniqueness
                if element_type == str and value in items:
                    value = f"{value}_{i}"
                items.append(value)
        return items  # Return as list since JSON doesn't support sets
    elif origin == tuple:  # Handle Tuple types
        args = get_args(field_type)
        if not args:
            return []  # No type args, return empty list

        # Handle Tuple[T, ...] (variable length homogeneous tuple)
        if len(args) == 2 and args[1] is Ellipsis:
            element_type = args[0]
            return [example_value_for_type(element_type, {}, _recursion_depth + 1, _seen_types) for _ in range(2)]

        # Handle Tuple[T1, T2, T3] (fixed length heterogeneous tuple)
        return [example_value_for_type(arg, {}, _recursion_depth + 1, _seen_types) for arg in args]
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
    # Handle RootModel specially
    if HAS_ROOT_MODEL and RootModel is not None:
        try:
            if issubclass(model_class, RootModel):
                root_type = get_type_hints(model_class).get('root')
                if root_type:
                    example_value = example_value_for_type(root_type, {})
                    return json.dumps(example_value, default=str)
        except TypeError:
            pass  # Not a class that can be checked with issubclass

    # First, check if the model has json_schema_extra with example data
    if hasattr(model_class, 'model_config'):
        model_config = model_class.model_config
        if isinstance(model_config, dict) and 'json_schema_extra' in model_config:
            schema_extra = model_config['json_schema_extra']
            if isinstance(schema_extra, dict) and 'example' in schema_extra:
                # Use the provided example directly (it's already validated)
                return json.dumps(schema_extra['example'])

    # Fallback: check old-style Config class for Pydantic v1 compatibility
    if hasattr(model_class, 'Config') and hasattr(model_class.Config, 'json_schema_extra'):
        schema_extra = model_class.Config.json_schema_extra
        if isinstance(schema_extra, dict) and 'example' in schema_extra:
            return json.dumps(schema_extra['example'])

    # Generate example data using field constraints
    example_data = {}
    if hasattr(model_class, 'model_fields'):
        # Pydantic v2
        for field_name, field_info in model_class.model_fields.items():
            field_constraints = get_field_constraints(field_info)
            field_type = get_type_hints(model_class).get(field_name)
            if field_type:
                # Try smart example first (handles field validators)
                smart = get_smart_example_for_field(field_name, field_type, field_constraints)
                if smart is not None:
                    example_data[field_name] = smart
                else:
                    example_data[field_name] = example_value_for_type(field_type, field_constraints)
    else:
        # Pydantic v1 or fallback
        for field_name, field_type in get_type_hints(model_class).items():
            smart = get_smart_example_for_field(field_name, field_type, {})
            if smart is not None:
                example_data[field_name] = smart
            else:
                example_data[field_name] = example_value_for_type(field_type)

    model_instance = model_class(**example_data)
    # Use model_dump_json with by_alias=True to:
    # 1. Export field aliases instead of field names
    # 2. Convert datetime/UUID objects to JSON-serializable strings
    # model_dump_json handles serialization directly and respects aliases better
    return model_instance.model_dump_json(by_alias=True)


def extract_schema_constraints(model_class: Type[BaseModel], prefix: str = "", _seen_types: set = None) -> list:
    """
    Recursively extract Literal/Enum constraints from a Pydantic model.

    Args:
        model_class: The Pydantic model class to extract constraints from.
        prefix: Prefix for nested field paths (e.g., "shipping." for nested fields).
        _seen_types: Internal set to track visited types and prevent infinite recursion.

    Returns:
        list: A list of constraint strings describing allowed values for Literal/Enum fields.

    Example:
        >>> class Task(BaseModel):
        ...     status: Literal["todo", "in_progress", "done"]
        >>> constraints = extract_schema_constraints(Task)
        >>> print(constraints)
        ['- "status": MUST be exactly one of: [\'todo\', \'in_progress\', \'done\']']
    """
    if _seen_types is None:
        _seen_types = set()

    # Prevent infinite recursion for self-referencing models
    type_id = id(model_class)
    if type_id in _seen_types:
        return []
    _seen_types = _seen_types.copy()
    _seen_types.add(type_id)

    constraints = []

    if not hasattr(model_class, 'model_fields'):
        return constraints

    for field_name, field_info in model_class.model_fields.items():
        field_type = get_type_hints(model_class).get(field_name)
        if field_type is None:
            continue

        full_path = f"{prefix}{field_name}" if prefix else field_name
        origin = get_origin(field_type)

        # Extract field constraints for length/numeric limits
        field_constraints = get_field_constraints(field_info)

        # Add string length constraints
        if field_constraints.get('min_length') or field_constraints.get('max_length'):
            min_len = field_constraints.get('min_length')
            max_len = field_constraints.get('max_length')
            if min_len and max_len:
                constraints.append(f'- "{full_path}": MUST be between {min_len} and {max_len} characters')
            elif max_len:
                constraints.append(f'- "{full_path}": MUST be at most {max_len} characters')
            elif min_len:
                constraints.append(f'- "{full_path}": MUST be at least {min_len} characters')

        # Add numeric constraints
        if field_constraints.get('ge') is not None or field_constraints.get('le') is not None:
            ge = field_constraints.get('ge')
            le = field_constraints.get('le')
            if ge is not None and le is not None:
                constraints.append(f'- "{full_path}": MUST be between {ge} and {le}')
            elif le is not None:
                constraints.append(f'- "{full_path}": MUST be at most {le}')
            elif ge is not None:
                constraints.append(f'- "{full_path}": MUST be at least {ge}')

        # Handle Literal types directly
        if origin == Literal:
            values = get_args(field_type)
            constraints.append(f'- "{full_path}": MUST be exactly one of: {list(values)}')

        # Handle Enum types and nested BaseModel
        elif origin is None and isinstance(field_type, type):
            try:
                if issubclass(field_type, Enum):
                    values = [e.value for e in field_type]
                    constraints.append(f'- "{full_path}": MUST be exactly one of: {values}')
                elif issubclass(field_type, BaseModel):
                    # Recurse into nested models
                    nested = extract_schema_constraints(field_type, f"{full_path}.", _seen_types)
                    constraints.extend(nested)
            except TypeError:
                pass

        # Handle Optional[T] which is Union[T, None]
        elif origin == Union:
            args = get_args(field_type)
            for arg in args:
                if arg is type(None):
                    continue
                arg_origin = get_origin(arg)
                if arg_origin == Literal:
                    values = get_args(arg)
                    constraints.append(f'- "{full_path}": MUST be exactly one of: {list(values)}')
                elif isinstance(arg, type):
                    try:
                        if issubclass(arg, Enum):
                            values = [e.value for e in arg]
                            constraints.append(f'- "{full_path}": MUST be exactly one of: {values}')
                        elif issubclass(arg, BaseModel):
                            nested = extract_schema_constraints(arg, f"{full_path}.", _seen_types)
                            constraints.extend(nested)
                    except TypeError:
                        pass

        # Handle List[BaseModel] to extract constraints from list element types
        elif origin == list:
            args = get_args(field_type)
            if args:
                element_type = args[0]
                if isinstance(element_type, type):
                    try:
                        if issubclass(element_type, BaseModel):
                            nested = extract_schema_constraints(element_type, f"{full_path}[].", _seen_types)
                            constraints.extend(nested)
                    except TypeError:
                        pass

        # Handle Dict types to clarify they must be objects, not arrays
        elif origin == dict:
            constraints.append(f'- "{full_path}": MUST be an object/dictionary {{}}, NOT an array []')

    return constraints
