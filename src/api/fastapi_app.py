# fastapi_app.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uuid

from src.core.chatbot import chatbot, run_async, checkpointer
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessageChunk


app = FastAPI(title="LangGraph Chatbot API")

# Request schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

# Response schema
class ChatResponse(BaseModel):
    thread_id: str
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Use provided thread_id or generate new one
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Wrap user message
    initial_message = {"messages": [HumanMessage(content=req.message)]}

    # Call chatbot graph
    result = await chatbot.ainvoke(initial_message, config=config)

    reply = result["messages"][-1].content
    return ChatResponse(thread_id=thread_id, reply=reply)


@app.get("/threads")
async def list_threads():
    """List all thread IDs stored in checkpoint DB."""
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return {"threads": list(all_threads)}


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a specific thread from checkpoint DB."""
    await checkpointer.adelete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}

@app.get("/conversation/{thread_id}")
async def load_conversation(thread_id: str):
    """Fetch all messages for a given thread_id from checkpoint DB."""
    state = await chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    formatted = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted.append({"role": role, "content": msg.content})
    return {"thread_id": thread_id, "messages": formatted}



