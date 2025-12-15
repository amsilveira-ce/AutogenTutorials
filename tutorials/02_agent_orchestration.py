from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo, ModelFamily
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

import subprocess
import asyncio
import sys 


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
ollama_client_llama = OllamaChatCompletionClient(
    model="llama3.1:8b",
    follow_redirects=False,
    options={
        "temperature": 0.7,
        "top_k": 50,
    }
)

ollama_client_granite = OllamaChatCompletionClient(
            model="granite3.3:8b",
            model_info=  {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "multiple_system_messages": False,
                "family": ModelFamily.UNKNOWN,
            },
            options={
                "temperature": 0.7,
                "top_k": 50,
            }
        )


# === Memory === 
# user_memory = ListMemory()


PLANNER_PROMPT = """
You are a research planning expert for literature review systems.

Your task is to create a SIMPLE, NUMBERED, STEP-BY-STEP PLAN for conducting a literature review.

REQUIREMENTS:
- Include at least 3 SPECIFIC search queries based on the research topic
- If you receive feedback, carefully adjust the plan based on that feedback
- Keep the plan focused and executable

When you receive a research topic, produce a plan in EXACTLY this format:

PLAN:
1. [First specific action with search query]
2. [Second specific action with search query]
3. [Third specific action with search query]
4. [Optional fourth action]

Rules:
- Generate 3-4 steps only
- Step 1 should ALWAYS be: "Search arXiv for [N] papers on [specific query 1]"
- Include at least 2-3 different search queries across steps (vary keywords, aspects, or related topics)
- Each step must be specific and actionable
- Steps must be ordered logically (search → review → synthesize)
- Use clear, concise language
- Do NOT add extra commentary before or after the plan
- Start directly with "PLAN:"

Example for topic "Human-AI interaction and cognitive flow":

PLAN:
1. Search arXiv for 9 recent papers on "human-AI interaction cognitive flow"
2. Search arXiv for 6 recent papers on "AI assistants user experience"
3. Review and select the 3 most relevant papers based on recency, relevance, and coverage
4. Write a comprehensive literature review synthesizing the selected papers

Example for topic "Quantum computing applications":

PLAN:
1. Search arXiv for 12 papers on "quantum computing optimization algorithms"
2. Search arXiv for 6 papers on "quantum machine learning applications"
3. Review and select the 5 most impactful papers across both searches
4. Write a literature review highlighting practical applications and future directions

FEEDBACK HANDLING:
- If human requests more papers: Increase the numbers in search steps
- If human wants different focus: Adjust search query keywords
- If human wants additional topics: Add a new search step
- If human wants fewer steps: Consolidate searches or remove optional steps

"""

# -- Input function
def human_input(prompt: str) -> str:
    return input(f"\nUSER INPUT REQUIRED:\n{prompt}\n> ")

# -- Cancelation condition
def is_approved(message: TextMessage) -> bool:
    approval_words = ["approved", "looks good", "yes", "ok", "proceed"]
    return any(word in message.content.lower() for word in approval_words)


user = UserProxyAgent(name="User", description="Human in loop", input_func=human_input)
planner = AssistantAgent(name="PlannerAgent", description="Agent resposible for planning the roadmap of the literature review", system_message=PLANNER_PROMPT,model_client=ollama_client_granite)


async def main():
    cancellation_token = CancellationToken()

    termination_condition = (
    TextMentionTermination("approved")
    | TextMentionTermination("looks good")
    | TextMentionTermination("yes")
    | TextMentionTermination("ok")
    | TextMentionTermination("proceed")
    )

    team = SelectorGroupChat(
        participants=[user, planner],
        model_client=ollama_client_granite,
        termination_condition=termination_condition,
    )

    task = "Create a research plan for Human-AI interaction and cognitive flow."

    result = team.run_stream(
        task=task,
        cancellation_token=cancellation_token,
    )
    await Console(result)
    


if __name__ == "__main__":
    asyncio.run(main())