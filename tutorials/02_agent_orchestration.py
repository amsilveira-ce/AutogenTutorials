"""
=================================================================
TUTORIAL: Hierarchical Agent Orchestration with Memory in AutoGen
=================================================================

This tutorial demonstrates a complete hierarchical agent system where:
1. A PLANNER creates a structured plan
2. An ORCHESTRATOR manages execution with memory
3. WORKER AGENTS execute individual steps
4. Memory tracks progress and validates completion

Educational Goals:
- Learn hierarchical agent design patterns
- Understand memory management in multi-agent systems
- See validation and feedback loops in action
- Master orchestration patterns for complex workflows

Workflow: User → Planner → Orchestrator → Workers → User
"""

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
