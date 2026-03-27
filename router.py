"""
Query intent classifier using Gemini Flash.
Returns a QueryIntent enum value + optional structured data.
"""
from __future__ import annotations

import json
import re

from gemini import generate
from models import QueryIntent

_SYSTEM = """You are an electronics parts query classifier.
Classify the user query into exactly ONE of these intents:
- PART_LOOKUP   : user wants a specific part by name/number (e.g. "LM2596", "ESP32-WROOM-32D")
- SEMANTIC      : user describes a function/spec (e.g. "convert 12V to 5V at 1A", "NE555 equivalent")
- COMPARE       : user wants to compare 2+ parts (e.g. "LM7805 vs AMS1117")
- CATEGORY      : user wants to browse a category (e.g. "types of motor drivers", "all BJT transistors")
- CIRCUIT_HELP  : user needs circuit calculation/advice (e.g. "what resistor for 5mm LED at 5V")
- BOM           : user pasted a multi-item parts list (messy or structured)

Respond ONLY with valid JSON, no markdown:
{"intent": "<INTENT>", "parts": ["part1", "part2"]}

"parts" should list the key part names/numbers extracted from the query (empty list if none).
"""


async def classify(query: str) -> tuple[QueryIntent, list[str]]:
    """Returns (intent, extracted_parts)."""
    raw = await generate(query, system=_SYSTEM)
    # Strip markdown fences if model adds them
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(raw)
        intent = QueryIntent(data.get("intent", "SEMANTIC"))
        parts = data.get("parts", [])
    except (json.JSONDecodeError, ValueError):
        intent = QueryIntent.SEMANTIC
        parts = []
    return intent, parts
