"""
Streamlit Chat Frontend for Weather Advisory Bot.
"""

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Must import graph after loading dotenv so Gemini API key is available
from graph import agent


def init_session():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Generate a unique thread ID for this browser session
    # This allows LangGraph to maintain memory for this specific user
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


st.set_page_config(
    page_title="Weather Safety Advisor",
    page_icon="🌤️",
    layout="centered"
)

st.title("🌤️ Weather Safety Advisor")
st.markdown("""
Ask about outdoor activities and I'll check live weather conditions against our safety policies.
*Examples:*
- "Is it safe to cycle in Bhopal today?"
- "Can I take my toddler to the park in Chennai this afternoon?"
- "Is today a good day for a picnic in London?"
""")

init_session()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to Streamlit history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process with LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Checking policies and live weather..."):
            # The config object passes the thread_id to LangGraph's checkpointer
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Create a HumanMessage object
            input_message = HumanMessage(content=prompt)
            
            try:
                # Run the graph
                # We pass the message in the "messages" key because our state schema
                # uses add_messages to append it.
                result = agent.invoke(
                    {"messages": [input_message]},
                    config=config
                )
                
                # Extract the bot's reply from the final state
                final_messages = result.get("messages", [])
                if final_messages:
                    bot_reply = final_messages[-1].content
                else:
                    bot_reply = "Sorry, I couldn't generate a response."
                    
                st.markdown(bot_reply)
                
                # Optional: Expandable section to show graph state for debugging/evaluation
                with st.expander("Show internal state (Debug)"):
                    st.write("**Extracted Intent:**")
                    st.json({
                        "city": result.get("city"),
                        "activities": result.get("activities"),
                        "time_context": result.get("time_context"),
                        "vulnerable_groups": result.get("vulnerable_groups")
                    })
                    
                    st.write("**Resolved Location:**")
                    st.write(result.get("location_name", "N/A"))
                    
                    st.write("**Matched SOPs:**")
                    sops = result.get("matched_sops", [])
                    if sops:
                        for s in sops:
                            st.write(f"- [{s['sop_id']}] {s['sop_name']} ({s['severity']})")
                    else:
                        st.write("None")
                
                # Add to Streamlit history
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
