"""
Custom exceptions for the Guardrails system.
"""


class GuardrailException(Exception):
    """Base exception for all guardrail-related errors."""

    def __init__(self, message: str, guardrail_name: str = "", metadata: dict = None):
        """
        Initialize guardrail exception.

        Args:
            message: Error message
            guardrail_name: Name of the guardrail that raised the exception
            metadata: Additional metadata about the error
        """
        super().__init__(message)
        self.guardrail_name = guardrail_name
        self.metadata = metadata or {}


class GuardrailBlockedException(GuardrailException):
    """
    Exception raised when a guardrail blocks a request or response.

    This exception is raised when a guardrail determines that content
    should not be processed (input) or returned (output).
    """

    def __init__(self, message: str, guardrail_name: str = "", metadata: dict = None):
        """
        Initialize blocked exception.

        Args:
            message: Reason for blocking
            guardrail_name: Name of the guardrail that blocked
            metadata: Additional metadata (e.g., detected patterns, violation details)
        """
        super().__init__(message, guardrail_name, metadata)


class GuardrailValidationException(GuardrailException):
    """
    Exception raised when guardrail validation fails unexpectedly.

    This is different from GuardrailBlockedException - this indicates
    an error in the guardrail itself, not blocked content.
    """

    def __init__(self, message: str, guardrail_name: str = "", original_exception: Exception = None):
        """
        Initialize validation exception.

        Args:
            message: Error message
            guardrail_name: Name of the guardrail that failed
            original_exception: The original exception that caused the failure
        """
        super().__init__(message, guardrail_name)
        self.original_exception = original_exception


class GuardrailConfigurationException(GuardrailException):
    """
    Exception raised when a guardrail is misconfigured.

    Examples: missing required config parameters, invalid parameter values, etc.
    """

    def __init__(self, message: str, guardrail_name: str = "", config: dict = None):
        """
        Initialize configuration exception.

        Args:
            message: Error message
            guardrail_name: Name of the misconfigured guardrail
            config: The problematic configuration
        """
        super().__init__(message, guardrail_name, config or {})
        self.config = config or {}


class GuardrailTimeoutException(GuardrailException):
    """
    Exception raised when a guardrail execution times out.

    Useful for guardrails that make external API calls or perform
    expensive computations.
    """

    def __init__(self, message: str, guardrail_name: str = "", timeout_seconds: float = None):
        """
        Initialize timeout exception.

        Args:
            message: Error message
            guardrail_name: Name of the guardrail that timed out
            timeout_seconds: The timeout value that was exceeded
        """
        metadata = {"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        super().__init__(message, guardrail_name, metadata)
