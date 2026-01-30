"""
utils for text
"""

import re
from typing import Dict, Any


def count_words(text: str) -> int:
    """
    Count words in text for length-based threshold.

    Args:
        text: Input text.

    Returns:
        Number of words.
    """
    if not text:
        return 0
    return len(text.split())
