"""
Configuration settings for the AutoGen Literature Review system.
"""
from dataclasses import dataclass
from autogen_ext.models.ollama._model_info import ModelInfo, ModelFamily


@dataclass
class ModelConfig:
    """Configuration for AI models"""

    OLLAMA_HOST = "http://localhost:11434"

    LLAMA_3_1= "llama3.1:8b" 
    GRANITE_3_3 = "granite3.3:8b"      

    default_options_hyperparameters = {
        "temperature": 0.7,
        "top_k": 50,
    }

    default_options_hyperparameters_researcher = {
        "temperature": 0.2,
        "top_k": 10,
    }

    granite_capabilities = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "multiple_system_messages": False,
        "family": ModelFamily.UNKNOWN,
    }


class AgentConfig:
    """Configuration for agent behavior"""
    MAX_TURNS_SEQUENTIAL = 3

    RESEARCHER_NAME = "Researcher"
    RESEARCHER_DESCRIPTION = "A agent that can search papers"


    REVIEWER_NAME = "Reviewer"
    REVIEWER_DESCRIPTION = "A agent that can review the papers retrieved"

    WRITER_NAME = "Writer"
    WRITER_DESCRIPTION = "A agent that can write the final report"


class UIConfig:
    """Configuration for the user interface"""
    PAGE_TITLE = "AI Literature Review Assistant"
    PAGE_ICON = "📚"
    DEFAULT_PLACEHOLDER = "e.g., 'Find 3 papers on multi-agent systems for customer service'"
