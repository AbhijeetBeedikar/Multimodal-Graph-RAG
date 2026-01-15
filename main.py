import importlib
import rag_system.rag.rag_pipeline as rag
importlib.reload(rag)
import rag_system.retrieval.orchestrator as orch
importlib.reload(orch)
import rag_system.graph.graph_builder as graph
importlib.reload(graph)
import rag_system.graph.graph_search as search
importlib.reload(search)
import rag_system.vector_db.indexing as indexing
importlib.reload(indexing)
import pickle
from pathlib import Path
#TODO: Store Knowledge graph as a st.session_state rather than an external file. Initialize it to an empty graph initially and then grow out the graph.
def get_all_text_from_collection(client, collection_name):
    collected = []
    scroll_filter = None
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=200,    # Qdrant default max
            offset=next_page,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            break

        for p in points:
            # assuming you stored text under payload["text"]
            if "text" in p.payload:
                collected.append(p.payload["text"])

        if next_page is None:
            break
    
    return collected
def process(mediapath):
    # Filling the Vector Database
    client = None
    for file in os.listdir(mediapath):

        (client, collection_name) = rag.ingest_and_index(mediapath + file)
        # Building the Graph
       # corpus_text = get_all_text_from_collection(client, collection_name)
       # print("corpus text obtained")
       # corp = " ".join(corpus_text)
       # graph.process_document_for_graph(corp)
        #ORIGINAL:
            #for i in corpus_text:
             #   graph.process_document_for_graph(i)
        print("ingested and indexed", file)
    return client
def user_input(query,client):
  # loading updated graph for hybrid retrieval
  with open("knowledge_graph.gpickle", "rb") as f:
    G = pickle.load(f)
  return orch.orchestrate(query, client, G) # returns response in the form of {"response":,"context": }

import streamlit as st
import os
import shutil
import time

st.title("Multimodal RAG 🤖")
if 'cli' not in st.session_state:
    st.session_state.cli = None
uploaded_files = st.file_uploader(
    "Upload documents for the RAG pipeline",
    type=['png', 'jpg', 'jpeg', 'mp3', 'pdf', 'txt'],
    accept_multiple_files=True
)
cols1, cols2 = st.columns(2)
with cols1:
    upload = st.button("Upload")
with cols2:
    delete = st.button("Delete")
if uploaded_files:
    if upload:
        # Create the directory if it doesn't exist
        Path("local_data/").mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:

            file_path = os.path.join("local_data/", uploaded_file.name)

            # Write the bytes to local memory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.cli = process("local_data/")
        st.success(f"Saved all files")

if delete:
    shutil.rmtree("local_data/")
    os.mkdir("local_data/")
    st.session_state.cli = None
# 1. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# 2. Display previous chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Generate & Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Simulate a "typing" effect for a professional feel
        if not st.session_state.cli:
            assistant_response = f"**Normal_LLM_Output**"
        else:
            assistant_response = user_input(prompt,st.session_state.cli)["response"]  # Replace this with your RAG or MCTS logic!

        #response_place
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.025)
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
