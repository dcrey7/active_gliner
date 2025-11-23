"""
Label Studio connection configuration.

Provides connection settings from environment variables.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


DEFAULT_BASE_URL = "http://slimmer-labelstudio-frontend:8080"


def get_connection_config() -> dict:
    """
    Get Label Studio connection configuration from environment.

    Returns:
        dict: {'base_url': str, 'api_key': str}

    Raises:
        ValueError: If LABEL_STUDIO_API_KEY not set

    Example:
        >>> config = get_connection_config()
        >>> client = LabelStudio(**config)
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY")

    if not api_key:
        raise ValueError(
            "LABEL_STUDIO_API_KEY environment variable not set. "
            "Please set it in .env file or environment."
        )

    return {
        'base_url': DEFAULT_BASE_URL,
        'api_key': api_key
    }
