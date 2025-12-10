"""
Team coordination and workflow management.
Defines how agents work together to complete the literature review task.
"""

from autogen_agentchat.teams import RoundRobinGroupChat
from agents import create_researcher_agent, create_reviewer_agent, create_writer_agent
from config import AgentConfig


class ResearchTeam:
    """
    Manages a team of AI agents that collaborate on literature reviews.
    
    The workflow is:
    1. Researcher finds relevant papers
    2. Reviewer validates the selection
    3. Writer creates the literature review
    """
    
    def __init__(self):
        """Initialize the research team with all necessary agents."""
        # Create specialized agents
        self.researcher = create_researcher_agent()
        self.reviewer = create_reviewer_agent()
        self.writer = create_writer_agent()
        
        # Create the team with round-robin coordination
        # Each agent speaks once in sequence
        self.team = RoundRobinGroupChat(
            participants=[self.researcher, self.reviewer, self.writer],
            max_turns=AgentConfig.MAX_TURNS_SEQUENTIAL
        )
    
    async def run_chat(self, task: str) -> str:
        """
        Execute a literature review task and return formatted results.
        
        Args:
            task (str): The user's research query
            
        Returns:
            str: Formatted conversation log showing the team's work
        """
        stream = self.team.run_stream(task=task)
        
        # Build a structured conversation log
        conversation_flow = [{
            "source": "User",
            "content": task,
            "type": "text"
        }]
        
        # Process the event stream
        async for event in stream:
            event_type = getattr(event, 'type', '')
            source = getattr(event, 'source', 'system')
            
            # Handle tool call requests
            if event_type == 'ToolCallRequestEvent':
                tool_calls = getattr(event, 'content', [])
                for call in tool_calls:
                    conversation_flow.append({
                        "source": source,
                        "type": "tool_request",
                        "tool_name": call.name,
                        "arguments": call.arguments
                    })
            
            # Handle tool execution results
            elif event_type == 'ToolCallExecutionEvent':
                conversation_flow.append({
                    "source": source,
                    "type": "tool_execution"
                })
            
            # Handle streaming text chunks
            elif event_type == 'ModelClientStreamingChunkEvent':
                content_chunk = getattr(event, 'content', '')
                
                # Find existing text entry for this source or create new one
                found = False
                for i in range(len(conversation_flow) - 1, -1, -1):
                    if (conversation_flow[i].get("source") == source and 
                        conversation_flow[i].get("type") == "text"):
                        conversation_flow[i]["content"] += content_chunk
                        found = True
                        break
                
                if not found:
                    conversation_flow.append({
                        "source": source,
                        "content": content_chunk,
                        "type": "text"
                    })
        
        # Format the conversation for display
        return self._format_conversation(conversation_flow)
    
    def _format_conversation(self, conversation_flow: list) -> str:
        """
        Format the conversation flow into readable markdown.
        
        Args:
            conversation_flow (list): List of conversation events
            
        Returns:
            str: Formatted markdown string
        """
        formatted_log = []
        
        for item in conversation_flow:
            item_type = item["type"]
            source = item["source"]
            
            if item_type == "text":
                content = item.get("content", "").strip()
                if content:
                    formatted_log.append(f"**{source.title()}**:\n\n{content}")
            
            elif item_type == "tool_request":
                tool_name = item["tool_name"]
                arguments = item["arguments"]
                formatted_log.append(
                    f"🛠️ **Tool Call**: `{source}` is calling `{tool_name}` "
                    f"with arguments:\n```json\n{arguments}\n```"
                )
            
            elif item_type == "tool_execution":
                formatted_log.append(" **Tool Result**: Data received successfully.")
        
        return "\n\n".join(formatted_log)