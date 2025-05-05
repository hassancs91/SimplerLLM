import os
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError, HttpUrl
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# --- Constants ---
API_BASE_URL = "https://simplerllm.com/api/v1/prompts/"
API_KEY_ENV_VAR = "SIMPLERLLM_API_KEY"

# --- Custom Exceptions ---
class PromptManagerError(Exception):
    """Base exception for Prompt Manager errors."""
    pass

class AuthenticationError(PromptManagerError):
    """Raised when API authentication fails."""
    pass

class PromptNotFoundError(PromptManagerError):
    """Raised when a prompt ID is not found."""
    pass

class NetworkError(PromptManagerError):
    """Raised for network-related issues during API calls."""
    pass

class MissingAPIKeyError(PromptManagerError):
    """Raised when the API key is missing."""
    pass

class VariableError(PromptManagerError):
    """Raised for issues related to prompt variables."""
    pass


# --- Pydantic Models for API Response ---
class PromptVariable(BaseModel):
    """Represents a variable within a prompt template."""
    name: str
    description: Optional[str] = ""
    _id: str # Keep the original ID field if needed

class PromptHubData(BaseModel):
    """Represents the data structure returned by the SimplerLLM Prompt Manager API."""
    id: str
    name: str
    description: Optional[str] = ""
    template: str
    variables: List[PromptVariable] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: int
    createdAt: str # Could use datetime if needed, but string is simpler
    updatedAt: Optional[str] = None # Make optional as it might be missing
    cached: Optional[bool] = False # Assuming it might be optional

class PromptSummaryData(BaseModel):
    """Represents summary data for a prompt as returned by the list endpoint."""
    id: str
    name: str
    description: Optional[str] = ""
    tags: List[str] = Field(default_factory=list)
    version: int
    createdAt: str
    updatedAt: str
    isPublic: Optional[bool] = None # Field might not always be present
    cacheEnabled: Optional[bool] = None # Field might not always be present
    ownership: Optional[str] = None # e.g., "owned", "shared"
    cached: Optional[bool] = False


# --- Managed Prompt Class ---
class ManagedPrompt:
    """
    Represents a prompt fetched from the SimplerLLM Prompt Manager.

    Provides methods to access prompt details, set variables, and format the final prompt string.
    Instantiate this class using the `fetch_prompt_from_hub` function.
    """
    def __init__(self, data: PromptHubData):
        """
        Initializes the ManagedPrompt instance.

        Args:
            data: A PromptHubData object containing the fetched prompt details.
        """
        self._data = data
        self._variable_values: Dict[str, Any] = {}
        self._variable_names = {var.name for var in self._data.variables}

    @property
    def id(self) -> str:
        """The unique ID of the prompt."""
        return self._data.id

    @property
    def name(self) -> str:
        """The name of the prompt."""
        return self._data.name

    @property
    def description(self) -> Optional[str]:
        """The description of the prompt."""
        return self._data.description

    @property
    def template(self) -> str:
        """The raw template string of the prompt."""
        return self._data.template

    @property
    def variables(self) -> List[PromptVariable]:
        """A list of variables defined in the prompt template."""
        return self._data.variables

    @property
    def tags(self) -> List[str]:
        """A list of tags associated with the prompt."""
        return self._data.tags

    @property
    def version(self) -> int:
        """The version number of the prompt."""
        return self._data.version

    def set_variable(self, name: str, value: Any):
        """
        Sets the value for a specific variable in the template.

        Args:
            name: The name of the variable (e.g., 'TARGET_AUDIENCE').
            value: The value to substitute for the variable.

        Raises:
            VariableError: If the variable name is not defined in the prompt template.
        """
        if name not in self._variable_names:
            raise VariableError(f"Variable '{name}' is not defined in the prompt template. Available variables: {list(self._variable_names)}")
        self._variable_values[name] = str(value) # Ensure value is string for replacement

    def set_variables(self, **kwargs: Any):
        """
        Sets multiple variable values using keyword arguments.

        Args:
            **kwargs: Keyword arguments where the key is the variable name
                      and the value is the value to substitute.

        Raises:
            VariableError: If any variable name is not defined in the prompt template.
        """
        for name, value in kwargs.items():
            self.set_variable(name, value) # Reuse single set_variable for validation

    def get_formatted_prompt(self) -> str:
        """
        Formats the prompt template by substituting the set variable values.

        Returns:
            The final prompt string with all variables replaced.

        Raises:
            VariableError: If any required variable has not been set using
                           `set_variable` or `set_variables`.
        """
        missing_vars = self._variable_names - set(self._variable_values.keys())
        if missing_vars:
            raise VariableError(f"Missing values for required variables: {list(missing_vars)}")

        formatted_template = self.template
        # Use regex to replace {{VARIABLE_NAME}} patterns
        for name, value in self._variable_values.items():
            # Ensure correct escaping for regex patterns if needed, but simple names should be fine
            placeholder = f"{{{{{name}}}}}" # Matches {{VAR_NAME}}
            formatted_template = formatted_template.replace(placeholder, value)

        return formatted_template

    def __repr__(self) -> str:
        return f"ManagedPrompt(id='{self.id}', name='{self.name}', version={self.version})"


# --- Fetch Function ---
def fetch_prompt_from_hub(prompt_id: str, api_key: Optional[str] = None) -> ManagedPrompt:
    """
    Fetches a prompt template from the SimplerLLM Prompt Manager API.

    Args:
        prompt_id: The unique ID of the prompt to fetch.
        api_key: Your SimplerLLM API key. If None, it attempts to read the key
                 from the 'SIMPLERLLM_API_KEY' environment variable.

    Returns:
        A ManagedPrompt object containing the fetched prompt data and methods
        to interact with it.

    Raises:
        MissingAPIKeyError: If the API key is not provided and not found in env vars.
        NetworkError: If there's a problem connecting to the API.
        AuthenticationError: If the provided API key is invalid (401 status).
        PromptNotFoundError: If the prompt ID does not exist (404 status).
        PromptManagerError: For other API-related errors (non-200 status codes).
        ValidationError: If the API response does not match the expected format.
    """
    resolved_api_key = api_key or os.getenv(API_KEY_ENV_VAR)
    if not resolved_api_key:
        raise MissingAPIKeyError(f"API key not provided and environment variable '{API_KEY_ENV_VAR}' not set.")

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{API_BASE_URL}{prompt_id}"

    try:
        response = requests.get(url, headers=headers, timeout=30) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

    except requests.exceptions.Timeout as e:
        raise NetworkError(f"API request timed out: {e}") from e
    except requests.exceptions.ConnectionError as e:
        raise NetworkError(f"Could not connect to API: {e}") from e
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            raise AuthenticationError("Authentication failed. Check your API key.") from e
        elif status_code == 404:
            raise PromptNotFoundError(f"Prompt with ID '{prompt_id}' not found.") from e
        else:
            raise PromptManagerError(f"API request failed with status {status_code}: {e.response.text}") from e
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"An unexpected network error occurred: {e}") from e

    try:
        data = response.json()
        prompt_data = PromptHubData.model_validate(data)
        return ManagedPrompt(prompt_data)
    except ValueError as e: # Catches JSONDecodeError
        raise PromptManagerError(f"Failed to decode API response as JSON: {e}") from e
    except ValidationError as e:
        raise PromptManagerError(f"API response validation failed: {e}") from e


# --- List Function ---
def list_prompts_from_hub(api_key: Optional[str] = None, include_shared: bool = True) -> List[PromptSummaryData]:
    """
    Lists prompts available to the user from the SimplerLLM Prompt Manager API.

    Args:
        api_key: Your SimplerLLM API key. If None, it attempts to read the key
                 from the 'SIMPLERLLM_API_KEY' environment variable.
        include_shared: Whether to include prompts shared with the user (default True).

    Returns:
        A list of PromptSummaryData objects representing the available prompts.

    Raises:
        MissingAPIKeyError: If the API key is not provided and not found in env vars.
        NetworkError: If there's a problem connecting to the API.
        AuthenticationError: If the provided API key is invalid (401 status).
        PromptManagerError: For other API-related errors (non-200 status codes).
        ValidationError: If the API response does not match the expected format.
    """
    resolved_api_key = api_key or os.getenv(API_KEY_ENV_VAR)
    if not resolved_api_key:
        raise MissingAPIKeyError(f"API key not provided and environment variable '{API_KEY_ENV_VAR}' not set.")

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    # Base URL for listing prompts doesn't include a specific ID
    list_url = API_BASE_URL.rstrip('/') # Ensure no trailing slash before adding params
    params = {"includeShared": str(include_shared).lower()} # API expects 'true' or 'false'

    try:
        response = requests.get(list_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        raise NetworkError(f"API request timed out: {e}") from e
    except requests.exceptions.ConnectionError as e:
        raise NetworkError(f"Could not connect to API: {e}") from e
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            raise AuthenticationError("Authentication failed. Check your API key.") from e
        # Add other specific status code checks if needed (e.g., 404 if the base endpoint itself could be missing)
        else:
            raise PromptManagerError(f"API request failed with status {status_code}: {e.response.text}") from e
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"An unexpected network error occurred: {e}") from e

    try:
        data = response.json()
        if not isinstance(data, list):
             raise PromptManagerError(f"API response was not a list as expected. Received: {type(data)}")
        # Validate each item in the list
        prompt_list = [PromptSummaryData.model_validate(item) for item in data]
        return prompt_list
    except ValueError as e: # Catches JSONDecodeError
        raise PromptManagerError(f"Failed to decode API response as JSON: {e}") from e
    except ValidationError as e:
        raise PromptManagerError(f"API response validation failed for one or more items: {e}") from e


# --- Fetch Specific Version Function ---
def fetch_prompt_version_from_hub(prompt_id: str, version: int, api_key: Optional[str] = None) -> ManagedPrompt:
    """
    Fetches a specific version of a prompt template from the SimplerLLM Prompt Manager API.

    Args:
        prompt_id: The unique ID of the prompt to fetch.
        version: The specific version number of the prompt to fetch.
        api_key: Your SimplerLLM API key. If None, it attempts to read the key
                 from the 'SIMPLERLLM_API_KEY' environment variable.

    Returns:
        A ManagedPrompt object containing the fetched prompt data for the specific version.

    Raises:
        MissingAPIKeyError: If the API key is not provided and not found in env vars.
        NetworkError: If there's a problem connecting to the API.
        AuthenticationError: If the provided API key is invalid (401 status).
        PromptNotFoundError: If the prompt ID or version number does not exist (404 status).
        PromptManagerError: For other API-related errors (non-200 status codes).
        ValidationError: If the API response does not match the expected format.
    """
    resolved_api_key = api_key or os.getenv(API_KEY_ENV_VAR)
    if not resolved_api_key:
        raise MissingAPIKeyError(f"API key not provided and environment variable '{API_KEY_ENV_VAR}' not set.")

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    # Construct URL for specific version
    url = f"{API_BASE_URL}{prompt_id}/versions/{version}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        raise NetworkError(f"API request timed out: {e}") from e
    except requests.exceptions.ConnectionError as e:
        raise NetworkError(f"Could not connect to API: {e}") from e
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            raise AuthenticationError("Authentication failed. Check your API key.") from e
        elif status_code == 404:
            raise PromptNotFoundError(f"Prompt with ID '{prompt_id}' and version '{version}' not found.") from e
        else:
            raise PromptManagerError(f"API request failed with status {status_code}: {e.response.text}") from e
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"An unexpected network error occurred: {e}") from e

    try:
        data = response.json()
        # Reuse the same detailed data model as fetching the latest version
        prompt_data = PromptHubData.model_validate(data)
        return ManagedPrompt(prompt_data)
    except ValueError as e: # Catches JSONDecodeError
        raise PromptManagerError(f"Failed to decode API response as JSON: {e}") from e
    except ValidationError as e:
        raise PromptManagerError(f"API response validation failed: {e}") from e
