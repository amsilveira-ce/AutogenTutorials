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

# === Prompts ===
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

FEEDBACK HANDLING:
- If human requests more papers: Increase the numbers in search steps
- If human wants different focus: Adjust search query keywords
- If human wants additional topics: Add a new search step
- If human wants fewer steps: Consolidate searches or remove optional steps

After generating the plan, present it to the User for review and approval.
"""

ORCHESTRATOR_PROMPT = """
You are an orchestration coordinator that executes research plans step-by-step.

Your workflow:
1. Look for the approved PLAN in the conversation history
2. Execute each step in order by delegating to the appropriate worker agent
3. Clearly state which step you are executing
4. Verify each step is completed before moving to the next

Available workers:
- Researcher: Searches for academic papers using arXiv (use for steps that mention "Search arXiv")
- Reviewer: Evaluates and selects the most relevant papers (use for steps that mention "Review" or "select")
- Writer: Creates comprehensive literature reviews (use for steps that mention "Write" or "review")

For each step in the plan:
- Announce: "Executing Step X: [step description]"
- Delegate clearly to the appropriate worker
- Wait for the worker to complete
- Move to the next step

After all steps are complete, simply announce "EXECUTION COMPLETE" without additional commentary.

Keep your coordination messages brief and clear.
"""

RESEARCHER_PROMPT = """
You are a research assistant specialized in academic paper discovery.

Your task:
1. When asked to search for papers, extract the query and number from the request
2. Use the search_arXiv tool with those parameters
3. Return ALL found papers with complete information

Important:
- Always use the search_arXiv tool when searching for papers
- Include complete paper information: title, authors, published date, abstract, URL
- Be thorough and systematic
- After completing a search, briefly state "SEARCH COMPLETE for [topic]"
"""

REVIEWER_PROMPT = """
You are an academic paper reviewer and selector.

Your task:
1. Review all papers provided by the Researcher
2. Assess each paper's relevance to the research topic
3. Rank papers by: relevance, recency, quality, and coverage diversity
4. Select EXACTLY the number of papers requested
5. Provide a brief evaluation

Output format:
- Selection rationale (2-3 sentences explaining your choices)
- List of selected papers with: title, authors, published date, brief summary, URL, and relevance score (High/Medium/Low)

After completing the review, state "REVIEW SELECTION COMPLETE".
"""

WRITER_PROMPT = """
You are an expert academic writer specializing in literature reviews.

Your task is to synthesize selected papers into a cohesive, well-structured literature review.

IMPORTANT: Your ENTIRE response should be the literature review itself. Do NOT add meta-commentary like "Here is the review" or "I have completed the review". Just write the review.

Structure your review:
1. **Introduction & Scope** (2-3 sentences): Define the research area and its importance
2. **Thematic Synthesis** (Main section): Identify 2-3 central themes, cite papers, compare approaches
3. **Methodological Overview**: Summarize research methods used across papers
4. **Limitations & Gaps**: Identify what's missing or under-explored
5. **Conclusion & Future Directions**: Summarize current state and suggest future work
6. **Reviewed Papers**: List all papers with full details (title, authors, date, summary, URL, relevance)

Writing guidelines:
- Use clear, academic prose with smooth transitions
- Always cite papers when referencing their work
- Compare and synthesize across papers (don't just list summaries)
- Maintain an objective, analytical tone
- Ensure logical flow between sections

End your review with exactly: "--- END OF LITERATURE REVIEW ---"
"""

# === Input function (HIL implementation) ===
def human_input(prompt: str) -> str:
    """
    Custom input function for UserProxyAgent
    """
    print("\n" + "="*80)
    print("HUMAN INPUT REQUIRED")
    print("="*80)
    print(f"\n{prompt}\n")
    print("Options:")
    print("  - Type 'APPROVED' to proceed with the plan")
    print("  - Type your feedback/changes to revise the plan")
    print("  - Press Ctrl+C to exit\n")
    return input("> ").strip()

# === Agents ===
user_proxy = UserProxyAgent(
    name="User",
    description="Human in the loop for approving or revising plans",
    input_func=human_input
)

planner = AssistantAgent(
    name="Planner",
    description="Agent responsible for creating the roadmap of the literature review",
    system_message=PLANNER_PROMPT,
    model_client=ollama_client_granite
)

orchestrator = AssistantAgent(
    name="Orchestrator",
    description="Coordinates the execution of approved research plans",
    model_client=ollama_client_granite,
    system_message=ORCHESTRATOR_PROMPT
)

researcher = AssistantAgent(
    name="Researcher",
    description="Searches for academic papers on arXiv",
    model_client=ollama_client_llama,
    tools=[search_arXiv],
    system_message=RESEARCHER_PROMPT
)

reviewer = AssistantAgent(
    name="Reviewer",
    description="Reviews and selects the most relevant papers",
    model_client=ollama_client_granite,
    system_message=REVIEWER_PROMPT
)

writer = AssistantAgent(
    name="Writer",
    description="Writes comprehensive literature reviews from selected papers",
    model_client=ollama_client_granite,
    system_message=WRITER_PROMPT
)



main_termination = (
    TextMentionTermination("EXECUTION COMPLETE")
    | TextMentionTermination("END OF LITERATURE REVIEW")
) 

main_team = SelectorGroupChat(
    participants=[planner, user_proxy, orchestrator, researcher, reviewer, writer],
    model_client=ollama_client_granite,
    termination_condition=main_termination,
    name="LiteratureReviewSystem",
    description="Complete literature review system with human-in-the-loop planning"
)

async def main():
 

    task = "Create a research plan for Human-AI interaction and cognitive flow."

   
    


if __name__ == "__main__":
    asyncio.run(main())