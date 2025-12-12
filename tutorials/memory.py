import asyncio 
from autogen_core.memory import ListMemory, MemoryContent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage

async def main() -> None:
    
    memory = ListMemory(name="chat_history_testing")

    content_test1 = MemoryContent(content="User prefers formal language", mime_type="text/plain")
    await memory.add(content_test1)
    model_context = BufferedChatCompletionContext(buffer_size=3)

   

    small_message = UserMessage(content="Hello im from canada", source="user")
    
    await model_context.add_message(small_message)

    content_test2 = MemoryContent(content="User said it is alergic to garlic bread", mime_type="text/plain")
    content_test3 = MemoryContent(content="the user asked about help to cook a banana", mime_type="text/plain")

   
  
    await memory.add(content_test2)
    await memory.add(content_test3)

    await memory.update_context(model_context)



    last = await model_context.get_messages()

    print(last[0].content)




asyncio.run(main())