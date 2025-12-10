
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama._model_info import ModelInfo, ModelFamily

import subprocess
import sys 


# === Model configuration ===
ollama_client_lamma = OllamaChatCompletionClient(
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


# === Tool definition ===
def Search_arXiv(query: str, max_results: int)-> str: 
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


# === Prompts ===
RESEARCHER_PROMPT = """
You are a research assistant specialized in academic paper discovery.

Your task:
1. Analyze the user's research topic
2. Formulate the most effective arXiv search query
3. Use the arxiv_search tool to find relevant papers
4. Select exactly the number of papers requested by the user multiplied by 3
5. Pass ALL found papers with complete information to the Reviewer for selection

Important:
- Cast a wide net - find more papers than requested so the Reviewer has options
- Always include complete paper information: title, authors, published date, summary, PDF URL
- Return structured data that the Reviewer can easily evaluate
- Don't filter or select papers yourself - that's the Reviewer's job
- If not explict number is given to you, the number is 3
"""

REVIEWER_PROMPT = """
You are an academic paper reviewer and selector with expertise in evaluating research relevance.

Your PRIMARY task is to select EXACTLY the number of papers the user requested, choosing the MOST RELEVANT ones.

Your workflow:
1. Review all papers provided by the Researcher
2. Assess each paper's relevance to the user's original query
3. Rank papers by relevance (most relevant first)
4. Select EXACTLY the number of papers the user requested (e.g., if they asked for 3, select the top 3)
5. Provide a brief evaluation of your selection

Evaluation criteria (in order of importance):
1. **Direct Relevance**: Does the paper directly address the user's topic?
2. **Recency**: Prefer more recent papers when relevance is similar
3. **Quality**: Consider paper citations and venue reputation
4. **Coverage**: Together, do selected papers cover different aspects of the topic?

Your output must include:

1. **Selection Rationale** (3-5 sentences):
   - Why you chose these specific papers
   - What aspects of the query they cover
   - Any notable papers you excluded and why

2. **Selected Papers** (EXACTLY the number requested):
   For each paper, provide:
   - Title
   - Authors
   - Published date
   - Summary
   - PDF URL
   - **Relevance score** (High/Medium/Low with 1 sentence justification)

CRITICAL: You must return EXACTLY the number of papers requested by the user. If the user asked for 3 papers, return 3. If they asked for 5, return 5.
"""

WRITER_PROMPT =  """
You are an expert academic writer specializing in literature reviews.

You will receive a curated selection of papers that have been chosen for their relevance to the user's query. Your task is to synthesize these papers into a cohesive literature review.

Structure your review as follows:

1. **Introduction & Scope** (2-3 sentences):
   - Define the research area covered by the papers
   - Provide context for why this topic matters

2. **Thematic Synthesis** (The Core - this is the most important section):
   - Identify 2-3 central themes that emerge from the papers
   - For each theme, create a detailed paragraph that:
     * Clearly states the theme
     * Explains how each relevant paper contributes to it (cite paper titles)
     * Compares and contrasts approaches: "While Paper A focuses on..., Paper B offers..."
     * Shows connections and divergences between papers
     * Synthesizes insights across papers rather than summarizing each individually

3. **Methodological Overview**:
   - Summarize the research methods used across papers
   - Note common approaches and unique methodologies
   - Cite specific papers when mentioning methods
   - Identify methodological strengths and limitations

4. **Limitations & Gaps**:
   - Identify limitations visible from the abstracts
   - Point out areas that appear under-explored in this collection
   - Be specific about what's missing or could be improved

5. **Conclusion & Future Directions**:
   - Summarize the current state of research (1-2 sentences)
   - Suggest 2-3 promising directions for future work
   - Base suggestions on logical extensions of the reviewed work

6. **Reviewed Papers** (Place this section LAST):
   List each paper using EXACTLY this format:
   
   **Paper**: [Full Paper Title]
   - **Authors**: [Author names separated by commas]
   - **Published Date**: [YYYY-MM-DD format]
   - **Summary**: [Brief summary from the abstract]
   - **PDF URL**: [URL as clickable link]
   - **Relevance Score**: [High/Medium/Low with brief justification]
   
   Example:
   **Paper**: Navigating the State of Cognitive Flow: Context-Aware AI Interventions
   - **Authors**: Dinithi Dissanayake, Suranga Nanayakkara
   - **Published Date**: 2025-04-16
   - **Summary**: This paper proposes a context-aware cognitive augmentation framework that adapts interventions based on type, timing, and scale to maintain or restore flow.
   - **PDF URL**: [http://arxiv.org/abs/2504.16021v1](http://arxiv.org/abs/2504.16021v1)
   - **Relevance Score**: High - Directly addresses context-aware AI interventions

Writing guidelines:
- Write in clear, academic prose with smooth transitions
- Always cite papers when referencing their work (use paper titles in the synthesis sections)
- Compare and synthesize across papers - avoid just listing summaries
- Maintain an objective, analytical tone
- Use transitional phrases to connect ideas between papers
- Ensure the review flows logically from one section to the next
- The paper list at the end should be properly formatted with the exact structure shown above
"""


# === Agents ===
researcher = AssistantAgent(
        name="Researcher",
        description="A agent that can search stuff",
        model_client=ollama_client_lamma,
        tools=[Search_arXiv],
        reflect_on_tool_use=False,
        model_client_stream=True,
        system_message= RESEARCHER_PROMPT
    )

reviewer = AssistantAgent(
    name="Reviwer",
    description="A agent that can review the papers retrieved",
    model_client=ollama_client_granite,
    model_client_stream=True,
    system_message=REVIEWER_PROMPT
)

writer = AssistantAgent(
    name="Writer",
    description="A agent that can write the final report",
    model_client=ollama_client_granite,
    model_client_stream=True, 
    system_message=WRITER_PROMPT
)

async def main():

    task = "Find the 3 most recent papers on 'Human-AI interaction' "

    team = RoundRobinGroupChat([researcher,reviewer, writer], max_turns=3)

    result =  await Console(team.run_stream(task=task),output_stats=True,  # Enable stats printing.
    )

    print(result.messages)


if __name__ == "__main__":
    #
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted.")