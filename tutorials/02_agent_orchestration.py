from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo, ModelFamily
import subprocess
import sys 


# === Memory configuration ===

# Configuration for Mem0Memory for local deployment 
#   - Vector Store: Qdrant (local file-based)
#   - Embedder: Ollama with nomic-embed-text
#   - LLM: Ollama with llama3.1:8b for memory extraction

mem0_configuration = {
 "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "./database/workflow_mem0_db"  # Local Qdrant database
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest"  # Embeddings model
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:8b",  # For memory extraction
                "temperature": 0
            }
        }
}


# === Tool definition ===
def search_arXiv(query: str, max_results: int)-> str: 
    """
    Get arxiv papers metadata    
    :param query: the topic related to the papers retrieved 
    :param max_results: The number of required papers
    :returns: The found papers related to the query in strig format
    """
    # Ensure that the arxiv library is imported 
    try:
        import arxiv
    except ModuleNotFoundError: 
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "arxiv"])
            
        except Exception as install_error:
            print(f"Details: {install_error}")

    # Handles in case the query is not a string or max_results a integer
    if not query or not isinstance(query, str): 
        raise ValueError(f"Query must be a nom-empty string. Found value: {query}")
    
    if not isinstance(max_results, int) and max_results>1000 or max_results<=1:
            raise ValueError(f"must be a integer between 1 and 1000. Found value: {max_results}")
    

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance 
        )
        
        results = list(search.results())

        if not results:
            raise ValueError("No papers found related to the search query")

        paper_details = []

        for i, result in enumerate(results):
            summary = result.summary.replace('\n', ' ')
            authors = ", ".join(author.name for author in result.authors)

            paper_details.append(
                f"Paper {i+1}:\n"
                f"  Title: {result.title}\n"
                f"  Authors: {authors}\n"
                f"  URL: {result.entry_id}\n"
                f"  Abstract: {summary}...\n"
            )
            
        return "\n".join(paper_details)
    
    except arxiv.ArxivError as e: 
        raise arxiv.ArxivError(f"arXiv API error: {str(e)}") from e 

# === Model configuration ===
small_model = OllamaChatCompletionClient(
    model="granite3.3:2b",
    native_tool_calls=True,
    hide_tools="if_any_run",
    model_info=ModelInfo(
        vision=False,
        function_calling=True,
        json_output=False,
        family=ModelFamily.UNKNOWN,
        structured_output=False
    )
)


larger_model = OllamaChatCompletionClient(
    model = "llama3.1:8b",
    follow_redirects=False,
)