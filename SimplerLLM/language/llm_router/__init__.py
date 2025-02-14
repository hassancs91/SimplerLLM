"""
LLM Router package initialization.

This package provides a system for intelligent content routing using LLMs.
"""

from .models import RouterResponse, Choice, PromptTemplate
from .router import LLMRouter

__all__ = ['RouterResponse', 'Choice', 'PromptTemplate', 'LLMRouter']
