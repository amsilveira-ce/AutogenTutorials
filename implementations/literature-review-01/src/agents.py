"""
Agent creation and configuration.
Each agent has a specific role in the literature review pipeline.
"""

from autogen_agentchat.agents import AssistantAgent
from tools import Search_arXiv
from prompts import RESEARCHER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT
from config import ModelConfig, AgentConfig
from autogen_ext.models.ollama import OllamaChatCompletionClient



ollama_client_lamma = OllamaChatCompletionClient(
    model=ModelConfig.LLAMA_3_1,
    
    follow_redirects=False,
)

ollama_client_granite = OllamaChatCompletionClient(
            model="granite3.3:8b",
            model_info= ModelConfig.granite_capabilities
        )


def create_researcher_agent() -> AssistantAgent:
    """
    Create the Researcher agent.
    
    This agent is responsible for:
    - Understanding the research topic
    - Formulating effective search queries
    - Using the arxiv_search tool to find papers
    - Returning structured paper information
    
    Returns:
        AssistantAgent configured as a researcher
    """
    return AssistantAgent(
        name=AgentConfig.RESEARCHER_NAME,
        description=AgentConfig.RESEARCHER_DESCRIPTION,
        model_client=ollama_client_lamma,
        tools=[Search_arXiv],  # Give it access to arXiv search
        reflect_on_tool_use=False,
        model_client_stream=True,
        system_message=RESEARCHER_PROMPT
    )

def create_reviewer_agent() -> AssistantAgent:
    """
    Create the Reviewer agent.
    
    This agent is responsible for:
    - Evaluating paper relevance to the user's query
    - Checking coverage of key topics
    - Identifying gaps or off-topic papers
    - Validating the paper selection
    
    Returns:
        AssistantAgent configured as a reviewer
    """
    return AssistantAgent(
        name=AgentConfig.REVIEWER_NAME,
        description= AgentConfig.REVIEWER_DESCRIPTION,
        model_client=ollama_client_granite,
        model_client_stream=True,
        system_message=REVIEWER_PROMPT
    )

def create_writer_agent() -> AssistantAgent:
    """
    Create the Writer agent.
    
    This agent is responsible for:
    - Synthesizing information from multiple papers
    - Creating a structured literature review
    - Comparing and contrasting different approaches
    - Identifying trends and future directions
    
    Returns:
        AssistantAgent configured as an academic writer
    """
    return AssistantAgent(
        name=AgentConfig.WRITER_NAME,
        description= AgentConfig.WRITER_DESCRIPTION,
        model_client=ollama_client_granite,
        model_client_stream=True,
        system_message=WRITER_PROMPT
    )


