"""
State schema for the Weather Advisory Bot.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class BotState(TypedDict):
    # ── Session memory (accumulates across turns) ──
    messages: Annotated[list, add_messages]

    # ── Per-turn working fields (overwritten each invocation) ──
    # Extracted from user query by the LLM
    user_query: str
    city: str
    activities: list[str]
    time_context: str
    vulnerable_groups: list[str]

    # Resolved location
    latitude: float
    longitude: float
    location_name: str

    # Weather data from Open-Meteo
    weather_data: dict

    # SOP matching results
    matched_sops: list[dict]

    # Error handling
    error: str
    error_type: str  # "location_error" | "weather_error" | ""
