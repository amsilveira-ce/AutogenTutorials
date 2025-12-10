import asyncio
import streamlit as st
from src.config import UIConfig
from src.team import ResearchTeam


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "agent_team" not in st.session_state:
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        st.session_state["event_loop"] = loop
        asyncio.set_event_loop(loop)
        
        # Initialize the research team
        st.session_state["agent_team"] = ResearchTeam()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def display_sidebar():
    """Display the sidebar with instructions and information."""
    with st.sidebar:
        st.header("How to Use")
        st.markdown("""
        1. **Enter your research query** in the chat
        2. **Specify the number of papers** you want (e.g., "find 3 papers...")
        3. **Wait for the AI team** to collaborate
        4. **Review the literature review** generated
        
        ### Example Queries:
        - "Find 3 papers on transformer architectures"
        - "Search for 5 recent papers on multi-agent reinforcement learning"
        - "Look up papers about explainable AI in healthcare, 4 results"
        """)
        
        st.divider()
        
        st.header("🤖 Agent Team")
        st.markdown("""
        - **Researcher**: Finds relevant papers
        - **Reviewer**: Validates selection
        - **Writer**: Creates literature review
        """)
        
        st.divider()
        
        st.header("⚙️ Configuration")
        st.markdown(f"""
        - **Researcher Model**: granite3.3:2b
        - **Writer Model**: granite3.3:8b
        - **Max Turns**: {UIConfig}
        """)


def display_chat_history():
    """Display all previous messages in the chat."""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(prompt: str):
    """
    Process user input and get response from the agent team.
    
    Args:
        prompt (str): User's research query
    """
    # Add user message to history
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤖 AI team is collaborating... This may take a moment."):
            # Run the async task
            loop = st.session_state["event_loop"]
            response = loop.run_until_complete(
                st.session_state["agent_team"].run_chat(prompt)
            )
            
            # Display the response
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response
    })


def main():
    """Main application entry point."""
    # Configure the page
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout="wide"
    )
    
    # Display header
    st.title(f"{UIConfig.PAGE_ICON} {UIConfig.PAGE_TITLE}")
    st.markdown("""
    Enter a research topic, and our AI team will find relevant papers and generate 
    a comprehensive literature review. Powered by AutoGen and Ollama.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display chat history
    display_chat_history()
    
    # Handle user input
    if prompt := st.chat_input(UIConfig.DEFAULT_PLACEHOLDER):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()