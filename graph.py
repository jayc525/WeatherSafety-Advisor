"""
Builds the LangGraph for the Weather Advisory Bot.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import BotState
import nodes
from langchain_groq import ChatGroq
import os


def get_llm():
    """Initialize the LLM. Using Groq API."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0
    )

llm=get_llm()

def route_initial(state_dict: BotState) -> str:
    messages = state_dict.get("messages", [])
    if not messages:
        return "extract_intent"
        
    qwery = messages[-1].content

    responses = llm.invoke(
        f"""
        Determine whether the following user message is a simple greeting (like hi, hello, hey, good morning)
        OR if it's asking a question or requesting weather/safety info.

        user message: "{qwery}"

        return ONLY the word "greeting" if it is just a greeting.
        return ONLY the word "intent" if they are asking something.
        """
    )
    result = responses.content.strip().lower()

    if "greeting" in result:
        return "handle_greeting"

    return "extract_intent"

def route_after_intent(state_dict: BotState) -> str:
    """Route after intent extraction."""
    if state_dict.get("error_type"):
        return "handle_failure"
    return "resolve_location"


def route_after_location(state_dict: BotState) -> str:
    """Route after location resolution."""
    if state_dict.get("error_type") == "location_error":
        return "handle_failure"
    return "fetch_weather"


def route_after_weather(state_dict: BotState) -> str:
    """Route after weather fetch."""
    if state_dict.get("error_type") == "weather_error":
        return "handle_failure"
    return "match_sops"


def build_graph():
    """Build and compile the LangGraph."""
    
    # Initialize the graph with our state schema
    workflow = StateGraph(BotState)
    
    # Add nodes
    workflow.add_node("handle_greeting", nodes.handle_greeting)
    workflow.add_node("extract_intent", nodes.extract_intent)
    workflow.add_node("resolve_location", nodes.resolve_location)
    workflow.add_node("fetch_weather", nodes.fetch_weather)
    workflow.add_node("match_sops", nodes.match_sops)
    workflow.add_node("compose_response", nodes.compose_response)
    workflow.add_node("handle_failure", nodes.handle_failure)
    
    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_initial,
        {
            "handle_greeting": "handle_greeting",
            "extract_intent": "extract_intent"
        }
    )
    # Add conditional edges
    workflow.add_conditional_edges(
        "extract_intent",
        route_after_intent,
        {
            "resolve_location": "resolve_location",
            "handle_failure": "handle_failure"
        }
    )
    
    workflow.add_conditional_edges(
        "resolve_location",
        route_after_location,
        {
            "fetch_weather": "fetch_weather",
            "handle_failure": "handle_failure"
        }
    )
    
    workflow.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "match_sops": "match_sops",
            "handle_failure": "handle_failure"
        }
    )
    
    # Linear edges for the rest of the success path
    workflow.add_edge("match_sops", "compose_response")
    
    # End points
    workflow.add_edge("compose_response", END)
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_failure", END)
    
    # Compile with memory (for session management)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# Create a global instance
agent = build_graph()
