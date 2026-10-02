"""
LangGraph nodes for the Weather Advisory Bot.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
import state
import weather
import sop_engine
from prompts import EXTRACT_INTENT_PROMPT, COMPOSE_RESPONSE_PROMPT, NO_SOP_RESPONSE_PROMPT


def get_llm():
    """Initialize the LLM. Using Groq API."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0
    )


def extract_intent(state_dict: state.BotState) -> dict:
    """
    Extract structured intent from the user's query and conversation history.
    """
    # Get the latest message
    messages = state_dict.get("messages", [])
    if not messages:
        return {"error": "No messages in state", "error_type": "internal"}
        
    latest_msg = messages[-1].content
    
    # Format history for the prompt (excluding the latest message)
    history_str = ""
    if len(messages) > 1:
        for m in messages[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "Bot"
            history_str += f"{role}: {m.content}\n"
    if not history_str:
        history_str = "No previous history."

    # Build prompt
    prompt = EXTRACT_INTENT_PROMPT.format(
        history=history_str,
        latest_message=latest_msg
    )
    
    llm = get_llm()
    # Force JSON output if the model supports it, otherwise prompt engineering handles it
    response = llm.invoke(prompt)
    
    # Parse JSON from response
    try:
        # Strip markdown code blocks if present
        raw_content = response.content
        if isinstance(raw_content, list):
            content = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)])
        else:
            content = str(raw_content)
        content = content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        extracted = json.loads(content.strip())
        
        # Ensure default values
        city = extracted.get("city")
        activities = extracted.get("activities", [])
        time_context = extracted.get("time_context", "today")
        vulnerable = extracted.get("vulnerable_groups", [])
        
        # Handle case where LLM returns a string instead of list
        if isinstance(activities, str): activities = [activities]
        if isinstance(vulnerable, str): vulnerable = [vulnerable]
        
        return {
            "user_query": latest_msg,
            "city": city,
            "activities": activities,
            "time_context": time_context,
            "vulnerable_groups": vulnerable,
            # Clear any previous errors
            "error": "",
            "error_type": ""
        }
    except Exception as e:
        # Fallback if parsing fails
        return {
            "error": f"Failed to parse intent: {str(e)}",
            "error_type": "intent_error"
        }


def resolve_location(state_dict: state.BotState) -> dict:
    """
    Resolve the city name to coordinates.
    """
    city = state_dict.get("city")
    
    if not city:
        # If city couldn't be extracted and isn't in history
        return {
            "error": "I couldn't identify a location in your request. Could you specify which city you're asking about?",
            "error_type": "location_error"
        }
        
    loc = weather.geocode_city(city)
    
    if not loc:
        return {
            "error": f"I couldn't find a location matching '{city}'. Please check the spelling or try a nearby larger city.",
            "error_type": "location_error"
        }
        
    # Format a nice location name (e.g., "Bhopal, Madhya Pradesh")
    name_parts = [loc["name"]]
    if loc.get("admin1"):
        name_parts.append(loc["admin1"])
    elif loc.get("country"):
        name_parts.append(loc["country"])
        
    loc_name = ", ".join(name_parts)
        
    return {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "location_name": loc_name
    }


def fetch_weather(state_dict: state.BotState) -> dict:
    """
    Fetch weather data for the resolved coordinates.
    """
    lat = state_dict.get("latitude")
    lon = state_dict.get("longitude")
    
    if lat is None or lon is None:
        return {"error": "Missing coordinates", "error_type": "internal"}
        
    data = weather.fetch_weather(lat, lon)
    
    if not data:
        return {
            "error": "I couldn't retrieve weather data for that location right now. The weather service might be down.",
            "error_type": "weather_error"
        }
        
    return {"weather_data": data}


def match_sops(state_dict: state.BotState) -> dict:
    """
    Match SOPs against the fetched weather data.
    """
    weather_data = state_dict.get("weather_data")
    activities = state_dict.get("activities", [])
    vulnerable_groups = state_dict.get("vulnerable_groups", [])
    
    if not weather_data:
        return {"error": "Missing weather data", "error_type": "internal"}
        
    matched = sop_engine.match_sops(weather_data, activities, vulnerable_groups)
    
    return {"matched_sops": matched}


def compose_response(state_dict: state.BotState) -> dict:
    """
    Compose the final response based on matched SOPs.
    """
    query = state_dict.get("user_query", "")
    location = state_dict.get("location_name", "your location")
    weather_data = state_dict.get("weather_data", {})
    matched_sops = state_dict.get("matched_sops", [])
    
    weather_summary = weather.format_weather_summary(weather_data)
    llm = get_llm()
    
    if not matched_sops:
        # NO MATCH PATH
        prompt = NO_SOP_RESPONSE_PROMPT.format(
            query=query,
            location=location,
            weather_summary=weather_summary
        )
        response = llm.invoke(prompt)
        raw_content = response.content
        if isinstance(raw_content, list):
            final_text = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)])
        else:
            final_text = str(raw_content)
        msg = AIMessage(content=final_text)
        return {"messages": [msg]}
        
    # MATCH PATH
    # Format SOPs for the prompt
    sops_str = ""
    for i, sop in enumerate(matched_sops):
        sops_str += f"--- POLICY {i+1} ---\n"
        sops_str += f"ID: [{sop['sop_id']}]\n"
        sops_str += f"Name: {sop['sop_name']}\n"
        sops_str += f"Severity: {sop['severity'].upper()}\n"
        sops_str += f"Advice: {sop['advice']}\n\n"
        
    prompt = COMPOSE_RESPONSE_PROMPT.format(
        query=query,
        location=location,
        weather_data=weather_summary,
        matched_sops=sops_str
    )
    
    response = llm.invoke(prompt)
    raw_content = response.content
    if isinstance(raw_content, list):
        final_text = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)])
    else:
        final_text = str(raw_content)
    msg = AIMessage(content=final_text)
    
    return {"messages": [msg]}


def handle_failure(state_dict: state.BotState) -> dict:
    """
    Handle location or weather API failures gracefully.
    """
    error = state_dict.get("error", "An unknown error occurred.")
    
    # Just return the error as the bot's response
    msg = AIMessage(content=error)
    return {"messages": [msg]}
