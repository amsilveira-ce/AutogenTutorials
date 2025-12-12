import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent
from autogen_ext.memory.mem0 import Mem0Memory
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main() -> None:
    
    # --- 1. CONFIGURE LOCAL MEMORY (With Nomic Embeddings) ---
    local_mem0_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "./local_mem0_db" 
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest"  # <--- UPDATED HERE
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:8b",
                "temperature": 0
            }
        }
    }

    print("Initializing Local Memory with Nomic Embeddings...")
    mem0_memory = Mem0Memory(
        is_cloud=False,
        config=local_mem0_config,
        limit=5,
    )

    # --- 2. ADD DATA ---
    await mem0_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type="text/plain", 
            metadata={"category": "preferences", "type": "units"},
        )
    )

    await mem0_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type="text/plain",
            metadata={"category": "preferences", "type": "dietary"},
        )
    )

    # --- 3. CONFIGURE AGENT CLIENT ---
    ollama_client_llama = OllamaChatCompletionClient(
        model="llama3.1:8b", 
        options={
            "temperature": 0.7,
            "top_k": 50,
        }
    )

    # --- 4. CREATE AGENT ---
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=ollama_client_llama,
        memory=[mem0_memory],
    )

    # --- 5. RUN ---
    print("\n--- Running Task ---")
    stream = assistant_agent.run_stream(task="What are my dietary preferences?")
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())