"""
LLM Backends Module
Provides abstraction layer for different LLM providers
"""

from .base import LLMBackend
from .factory import BackendFactory

__all__ = ['LLMBackend', 'BackendFactory']
