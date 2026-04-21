"""
tools/search.py — Simulated search tool for teaching purposes.

In a real system, this would call a search API (Brave, Serper, etc.).
For this learning pipeline, we return deterministic mock results so
training and evaluation are reproducible without internet access.

The model generates: <tool>search(query)</tool>
"""

import re

# Mock knowledge base: query keywords → fact snippets
MOCK_KNOWLEDGE = {
    "population": {
        "world": "World population is approximately 8.1 billion people (2024).",
        "switzerland": "Switzerland has a population of about 8.9 million people.",
        "china": "China has a population of approximately 1.4 billion people.",
        "usa": "The United States has a population of approximately 335 million people.",
    },
    "distance": {
        "moon": "The average distance from Earth to the Moon is 384,400 kilometers.",
        "sun": "The average distance from Earth to the Sun is 149.6 million kilometers (1 AU).",
        "mars": "The average distance from Earth to Mars is 225 million kilometers.",
    },
    "speed": {
        "light": "The speed of light is 299,792,458 meters per second.",
        "sound": "The speed of sound in air is approximately 343 meters per second at 20°C.",
    },
    "science": {
        "gravity": "The gravitational acceleration on Earth's surface is 9.81 m/s².",
        "avogadro": "Avogadro's number is 6.022 × 10²³ mol⁻¹.",
        "pi": "Pi (π) is approximately 3.14159265358979.",
    },
    "history": {
        "ww2": "World War II lasted from 1939 to 1945.",
        "moon landing": "The first Moon landing was on July 20, 1969 (Apollo 11).",
        "internet": "The World Wide Web was invented by Tim Berners-Lee in 1989.",
    },
}


def _find_relevant_fact(query: str) -> str:
    """Search mock knowledge base for relevant facts."""
    query_lower = query.lower()
    results = []

    for category, entries in MOCK_KNOWLEDGE.items():
        for key, fact in entries.items():
            if any(word in query_lower for word in key.split()):
                results.append(fact)

    if results:
        return " | ".join(results[:3])  # Return up to 3 matching facts
    return f"No specific data found for '{query}'. Please rely on general knowledge."


def run(query: str) -> str:
    """Tool entry point called by the ToolDispatcher.

    Args:
        query: Search query string from model's tool call.

    Returns:
        Search result string.
    """
    result = _find_relevant_fact(query)
    return f"Search result for '{query}': {result}"
