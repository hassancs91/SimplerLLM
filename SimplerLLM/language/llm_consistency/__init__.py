"""
Self-Consistency module for SimplerLLM.

This module provides self-consistency voting for improving LLM reliability
by generating multiple responses and returning the most consistent answer.
"""

from .consistency import SelfConsistency
from .models import (
    ConsistencyResult,
    SampleResponse,
    AnswerGroup,
    AnswerType,
)

__all__ = [
    "SelfConsistency",
    "ConsistencyResult",
    "SampleResponse",
    "AnswerGroup",
    "AnswerType",
]
