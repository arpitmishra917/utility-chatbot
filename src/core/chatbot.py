from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool,BaseTool
import requests
import aiosqlite
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import uuid
import os 
from src.core.ingest import ingest

import threading

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)



class chatstate(TypedDict):

    messages: Annotated[list[BaseMessage],add_messages]

llm=ChatOpenAI(model_name="gpt-4.1")

# search tool
search_tool=DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=GQD3UUFLBNAOLWNU"
    r = requests.get(url)
    return r.json()



@tool
def pdf_rag_search(query: str, k: int = 5) -> dict:
    """
    Perform a Retrieval-Augmented Generation (RAG) search over a local FAISS index
    that was built from uploaded PDF files.

    Workflow:
    1. Check if a FAISS index already exists at the given `index_path`.
       - Requires both `index.faiss` and `index.pkl` files to be present.
       - If the index exists, it is loaded into memory using the provided embeddings.
       - If the index does not exist, the ingestion pipeline is triggered:
         * All PDF files in `pdf_dir` are loaded.
         * Each PDF is split into overlapping text chunks using a recursive character splitter.
         * Embeddings are generated for each chunk using OpenAI's `text-embedding-3-small` model.
         * A FAISS vectorstore is built and saved locally at `index_path`.

    2. Once the vectorstore is available, run a similarity search:
       - The `query` string is embedded and compared against the stored chunks.
       - The top `k` most relevant chunks are retrieved.

    3. Return a dictionary containing:
       - `"query"`: the original search query string
       - `"results"`: a list of text chunks (strings) that are most relevant to the query
       - If any error occurs (e.g., missing index, ingestion failure), return an `"error"` key.

    Parameters
    ----------
    query : str
        The natural language search query to run against the PDF index.
    k : int, optional
        The number of top matching chunks to return (default = 5).
    pdf_dir : str, optional
        Directory containing uploaded PDF files (default = "data/pdfs").
    index_path : str, optional
        Directory where the FAISS index is stored (default = "data/faiss_idx").

    Returns
    -------
    dict
        A dictionary with either:
        - {"query": query, "results": [list of text chunks]}
        - {"error": "error message"} if something goes wrong.
    """
    # Path to your FAISS index (built separately with an ingestion script)
    INDEX_PATH = "data/faiss_idx"

    # Load embeddings + FAISS index once
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if (
        INDEX_PATH is not None
        and os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))
        and os.path.exists(os.path.join(INDEX_PATH, "index.pkl"))
    ):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore=ingest(pdf_dir="data/pdfs", index_path="data/faiss_idx")

    if vectorstore is None:
        return {"error": "FAISS index not loaded"}

    try:
        docs = vectorstore.similarity_search(query, k=k)
        results = [doc.page_content for doc in docs]
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}


client = MultiServerMCPClient({
  "expense-tracker": {
    # 1. Change transport to 'http'
    "transport": "streamable_http",
    
    # 2. Specify the URL where the server is running
    "url": "https://developed-pink-sailfish.fastmcp.app/mcp", 
    
    # Remove 'command', 'args', and 'cwd' as the client doesn't need to launch the server
    # it just connects to the existing running process.
  }
})



import traceback

def load_mcp_tools() -> list[BaseTool]:
    try:
        tools = run_async(client.get_tools())
        print("✅ Loaded MCP tools:", [t.name for t in tools])
        return tools
    except Exception as e:
        print("❌ Error loading MCP tools:")
        traceback.print_exc()
        return []
    
    
mcp_tools=load_mcp_tools()

tools=[get_stock_price,calculator,search_tool,pdf_rag_search, *mcp_tools]

llm_with_tools=llm.bind_tools(tools)


async def chat_node(state: chatstate) -> chatstate:
    """LLM node that may answer or request a tool call."""
    messages=state["messages"]
    response=await llm_with_tools.ainvoke(messages)
    return {"messages":[response]}


tool_node=ToolNode(tools)

async def _init_checkpointer():
    conn= await aiosqlite.connect(database="chatbot.db",check_same_thread=False)
    # return  AsyncSqliteSaver.from_conn_string("chatbot.db")

    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())



graph=StateGraph(chatstate)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)

graph.add_edge("tools","chat_node")

chatbot=graph.compile(checkpointer=checkpointer)



# async def _alist_threads():
#     all_threads = set()
#     async for checkpoint in checkpointer.alist(None):
#         all_threads.add(checkpoint.config["configurable"]["thread_id"])
#     return list(all_threads)


# def retrieve_all_threads():
#     return run_async(_alist_threads())


# async def delete_thread(thread_id):
#     await checkpointer.adelete_thread(thread_id)


# async def run_chat():


#     # 1. Generate a unique thread ID for this test run
#     test_thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": test_thread_id}}

#     initial_message = {"messages": [HumanMessage(content="use pdf rag tool and tell me whats into the pdf data i ahve provided to you ")]}
    
#     # FIX: Use the asynchronous invocation method: .ainvoke()
    
#     result = await chatbot.ainvoke(initial_message,config=config)

#     print(result["messages"][-1].content)


# if __name__ == "__main__":
#     run_async(run_chat())