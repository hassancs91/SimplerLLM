"""
Web Search + JSON Service - Combines web search with Pydantic structured output
"""
import time
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel, Field, create_model
from services.llm_service import LLMService

# Import SimplerLLM
try:
    from SimplerLLM.language.llm_addons import generate_pydantic_json_model
    from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse
except ImportError:
    from simplerllm.language.llm_addons import generate_pydantic_json_model
    from simplerllm.language.llm_providers.llm_response_models import LLMFullResponse


class WebSearchJsonService:
    """Service for web search with structured JSON output generation."""

    # Supported field types mapping
    TYPE_MAP = {
        'string': str,
        'number': float,
        'integer': int,
        'boolean': bool,
    }

    def __init__(self):
        self.llm_service = LLMService()

    def build_dynamic_model(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model from user-defined schema.

        Args:
            schema: Dict with 'fields' list containing field definitions
                   Each field has: name, type, description, item_type (for lists)

        Returns:
            Dynamically created Pydantic model class
        """
        fields = {}

        for field_def in schema.get('fields', []):
            name = field_def.get('name', '').strip()
            if not name:
                continue

            field_type = field_def.get('type', 'string')
            description = field_def.get('description', '')

            if field_type == 'list':
                item_type_str = field_def.get('item_type', 'string')
                item_type = self.TYPE_MAP.get(item_type_str, str)
                python_type = List[item_type]
            else:
                python_type = self.TYPE_MAP.get(field_type, str)

            # Create field with description for better LLM prompting
            fields[name] = (python_type, Field(description=description))

        if not fields:
            raise ValueError("Schema must have at least one valid field")

        # Dynamically create the model
        DynamicModel = create_model('DynamicSchema', **fields)
        return DynamicModel

    def parse_pydantic_code(self, code: str) -> Type[BaseModel]:
        """
        Parse pasted Pydantic model code and return the model class.
        Executes the code in a restricted namespace.

        Args:
            code: Python code defining Pydantic models

        Returns:
            The last defined BaseModel subclass in the code
        """
        # Clean the code - remove import statements as we provide them
        lines = code.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip import lines - we'll provide our own namespace
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                continue
            cleaned_lines.append(line)
        cleaned_code = '\n'.join(cleaned_lines)

        # Create a restricted namespace with necessary types
        namespace = {
            'BaseModel': BaseModel,
            'Field': Field,
            'List': List,
            'Optional': Optional,
            'Dict': Dict,
            'Any': Any,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
        }

        try:
            exec(cleaned_code, namespace)
        except Exception as e:
            raise ValueError(f"Failed to parse Pydantic code: {e}")

        # Find the last defined BaseModel subclass (that's the main output model)
        model_class = None
        for name, obj in namespace.items():
            if (isinstance(obj, type) and
                issubclass(obj, BaseModel) and
                obj is not BaseModel and
                not name.startswith('_')):
                model_class = obj

        if model_class is None:
            raise ValueError("No valid Pydantic BaseModel class found in the code")

        return model_class

    def get_web_search_providers(self) -> List[Dict[str, Any]]:
        """Get list of providers that support web search."""
        return [
            {
                'id': 'openai',
                'name': 'OpenAI (Web Search)',
                'models': ['gpt-4o', 'gpt-4o-mini'],
                'default_model': 'gpt-4o',
                'description': 'Uses OpenAI Responses API with web_search tool'
            },
            {
                'id': 'perplexity',
                'name': 'Perplexity (Native Search)',
                'models': ['sonar', 'sonar-pro', 'sonar-reasoning', 'sonar-reasoning-pro'],
                'default_model': 'sonar-pro',
                'description': 'Built-in web search with citations, always enabled'
            }
        ]

    def generate_with_web_search(
        self,
        prompt: str,
        provider: str,
        model: str,
        schema: Optional[Dict[str, Any]] = None,
        schema_code: Optional[str] = None,
        schema_mode: str = "form",
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute web search and generate structured JSON output.

        Args:
            prompt: The search query/prompt
            provider: LLM provider ('openai' or 'perplexity')
            model: Model name
            schema: Schema definition for the output structure (form mode)
            schema_code: Pydantic model code (code mode)
            schema_mode: "form" or "code"
            settings: Optional settings (temperature, max_tokens)

        Returns:
            Dict with 'data', 'sources', 'tokens', 'process_time'
        """
        settings = settings or {}
        start_time = time.time()

        # Build Pydantic model based on mode
        try:
            if schema_mode == "code" and schema_code:
                model_class = self.parse_pydantic_code(schema_code)
            elif schema:
                model_class = self.build_dynamic_model(schema)
            else:
                raise ValueError("Either schema or schema_code must be provided")
        except ValueError as e:
            raise ValueError(f"Invalid schema: {e}")

        # Get LLM instance
        llm = self.llm_service.get_llm(provider, model)
        if not llm:
            raise ValueError(f"Failed to create LLM instance for {provider}/{model}. Check API key configuration.")

        # Generate with web search
        # For Perplexity, web_search is always enabled
        # For OpenAI, we explicitly enable it
        result = generate_pydantic_json_model(
            model_class=model_class,
            prompt=prompt,
            llm_instance=llm,
            web_search=True,
            full_response=True,
            temperature=settings.get('temperature', 0.7),
            max_tokens=settings.get('max_tokens', 2000),
            max_retries=settings.get('max_retries', 3)
        )

        process_time = time.time() - start_time

        # Handle error responses (string)
        if isinstance(result, str):
            raise ValueError(f"Generation failed: {result}")

        # Extract data from LLMFullResponse
        if isinstance(result, LLMFullResponse):
            data = {}
            if result.model_object:
                data = result.model_object.model_dump()

            sources = result.web_sources or []

            return {
                'data': data,
                'sources': sources,
                'tokens': {
                    'input': result.input_token_count or 0,
                    'output': result.output_token_count or 0
                },
                'process_time': round(process_time, 2),
                'provider_used': provider,
                'model_used': model
            }
        else:
            # Fallback for BaseModel result (shouldn't happen with full_response=True)
            return {
                'data': result.model_dump() if hasattr(result, 'model_dump') else {},
                'sources': [],
                'tokens': {'input': 0, 'output': 0},
                'process_time': round(process_time, 2),
                'provider_used': provider,
                'model_used': model
            }

    def generate_code_example(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider: str,
        model: str
    ) -> str:
        """
        Generate Python code example for the current configuration.

        Returns Python code that replicates this web search + JSON generation.
        """
        # Build field definitions for the code
        field_lines = []
        for field in schema.get('fields', []):
            name = field.get('name', '')
            ftype = field.get('type', 'string')
            desc = field.get('description', '')

            if ftype == 'list':
                item_type = field.get('item_type', 'string')
                type_str = f"List[{item_type}]"
            else:
                type_map = {'string': 'str', 'number': 'float', 'integer': 'int', 'boolean': 'bool'}
                type_str = type_map.get(ftype, 'str')

            if desc:
                field_lines.append(f'    {name}: {type_str} = Field(description="{desc}")')
            else:
                field_lines.append(f'    {name}: {type_str}')

        fields_code = '\n'.join(field_lines)

        # Map provider to enum
        provider_map = {
            'openai': 'LLMProvider.OPENAI',
            'perplexity': 'LLMProvider.PERPLEXITY'
        }
        provider_enum = provider_map.get(provider, 'LLMProvider.OPENAI')

        code = f'''from pydantic import BaseModel, Field
from typing import List
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

# Define your output schema
class OutputSchema(BaseModel):
{fields_code}

# Create LLM instance
llm = LLM.create(provider={provider_enum}, model_name="{model}")

# Generate structured output with web search
result = generate_pydantic_json_model(
    model_class=OutputSchema,
    prompt="""{prompt}""",
    llm_instance=llm,
    web_search=True,
    full_response=True,
    max_tokens=2000
)

# Access the structured data
print("Data:", result.model_object.model_dump())
print("Sources:", result.web_sources)
'''
        return code


# Singleton instance
websearch_json_service = WebSearchJsonService()
