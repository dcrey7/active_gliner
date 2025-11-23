"""
LLM Module Exceptions

Backend-level exceptions only.
Validation errors are returned in result dicts, not raised as exceptions.
"""


class LLMConfigError(Exception):
    """Configuration errors (API key missing, invalid model)"""
    pass


class LLMRuntimeError(Exception):
    """Runtime errors (connection, timeout, empty response)"""
    pass


class HardQuotaError(LLMRuntimeError):
    """Hard quota exceeded - STOP generation"""
    pass
