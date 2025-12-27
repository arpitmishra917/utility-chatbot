import streamlit as st
from langchain_core.messages import HumanMessage
import requests
import uuid
API_URL = "http://127.0.0.1:8000"

import os
import time



# **********************************************  utility function  *********************************************

def generate_thread_id():
    thread_id=uuid.uuid4()
    return str(thread_id)

def add_thread(thread_id, title="new chat"):
    """Adds a thread and title, ensuring it only appears once."""
    
    # Ensure dictionaries are initialized (Safeguard)
    if "chat_titles" not in st.session_state:
        st.session_state["chat_titles"] = {}
    if "chat_thread" not in st.session_state:
        st.session_state["chat_thread"] = []
    
    # Add to list and dictionary only if not already present
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].insert(0, thread_id) # Insert at front for new chats
    
    # Always set the title (to update if "new chat" is replaced later)
    if thread_id not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][thread_id] = title

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state["thread_id"]=thread_id
    add_thread(st.session_state["thread_id"],"new chat")
    st.session_state["message_history"]=[]


def load_conversation(thread_id):
    return requests.get(f"{API_URL}/conversation/{thread_id}").json()



# ******************************************************  session setup   *******************************************

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=generate_thread_id()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}


# 1. Load existing thread IDs from API
threads_from_api = requests.get(f"{API_URL}/threads").json()["threads"]
st.session_state["chat_thread"] = threads_from_api

for threadid in st.session_state["chat_thread"]:
    if threadid not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][threadid] = f"Conversation {threadid[:8]}"

add_thread(st.session_state["thread_id"],"new chat")



# **********************************************  sidebar ui  *****************************************************

st.sidebar.title("Multi utility chatbot")

if st.sidebar.button("New chat"):
    reset_chat()


# --- Configuration ---
PDF_DIR = "data/pdfs"
index_path = "data/faiss_idx"
UPLOADER_KEY = "pdf_uploader" # Unique key for the widget

# --- State Initialization (Top of script) ---
if 'delete_msg' not in st.session_state:
    st.session_state.delete_msg = None


def handle_pdf_deletion(fname_to_delete: str):
    """Performs all disk operations and Streamlit state cleanup."""
    
    try:
        # 1. CRITICAL: Clear Streamlit UI and Resource State
        # The correct way to clear a file_uploader widget is to DELETE its key from session_state.
        if UPLOADER_KEY in st.session_state:
            del st.session_state[UPLOADER_KEY]
            
        st.cache_resource.clear()
        st.cache_data.clear()
        
        # 2. Delete the PDF
        pdf_path = os.path.join(PDF_DIR, fname_to_delete)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            time.sleep(0.05) # Small delay for OS sync
        
        # 3. Delete FAISS index files
        for f in ["index.faiss", "index.pkl"]:
            fpath = os.path.join(index_path, f)
            if os.path.exists(fpath):
                os.remove(fpath)
                
        # 4. Set success message and trigger final rerun
        st.session_state.delete_msg = f"🗑️ Successfully deleted **{fname_to_delete}** and cleared index."
        st.rerun()

    except Exception as e:
        st.session_state.delete_msg = f"❌ Error deleting {fname_to_delete}: {e}"
        st.rerun()


# --- Main Application Logic ---

# Check if a deletion message exists from a previous run and display it
if st.session_state.delete_msg:
    st.sidebar.markdown(st.session_state.delete_msg)
    st.session_state.delete_msg = None # Clear message after display


# 1. File Upload Code (MUST be updated with the key)
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], key=UPLOADER_KEY)

if uploaded_file is not None:
    os.makedirs(PDF_DIR, exist_ok=True)
    
    target_path = os.path.join(PDF_DIR, uploaded_file.name)
    if not os.path.exists(target_path):
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("✅ PDF uploaded ")


# 2. Sidebar Deletion Logic
st.sidebar.header("Uploaded Pdf's")
os.makedirs(PDF_DIR, exist_ok=True)
pdf_files = os.listdir(PDF_DIR)

if not pdf_files:
    st.sidebar.info("No PDFs uploaded yet.")
else:
    for fname in pdf_files:
        col1, col2 = st.sidebar.columns([0.7, 0.3])
        col1.write(fname)
        
        if col2.button("Delete", key=f"del_{fname}"):
            # When button is clicked, call the handler function directly
            handle_pdf_deletion(fname)

st.sidebar.header("My conversations")

for thread_id in st.session_state["chat_thread"]:
    title=st.session_state["chat_titles"].get(thread_id, str(thread_id))
    col1,col2=st.sidebar.columns([0.75,0.25])
    if col1.button(title, key=f"load_{thread_id}"):
        st.session_state["thread_id"]=thread_id
        st.session_state["message_history"]=load_conversation(thread_id)["messages"]

    if col2.button("del",key=f"delete_{thread_id}"):
        requests.delete(f"{API_URL}/threads/{thread_id}")
        if thread_id in st.session_state["chat_thread"]:
            st.session_state["chat_thread"].remove(thread_id)
        if st.session_state["thread_id"] == thread_id:
            reset_chat()
        st.rerun()





# ******************************************************  main ui **************************************************

# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input=st.chat_input("type here")

if user_input:
    current_thread = st.session_state["thread_id"]
    if current_thread not in st.session_state["chat_titles"]:
        add_thread(current_thread, title="new chat") 
    
    if st.session_state["chat_titles"][current_thread]=="new chat":
        st.session_state["chat_titles"][current_thread]=user_input[:30]
    # first add the message to the message_history
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text(user_input)
    

    response=requests.post(f"{API_URL}/chat",json={"message":user_input,"thread_id":str(st.session_state["thread_id"])}).json()
    ai_message=response["reply"]
    st.session_state["thread_id"] = response["thread_id"]

    st.session_state["message_history"].append({"role":"assistant","content":ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)

