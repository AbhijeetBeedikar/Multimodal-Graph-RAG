import streamlit as st
import os
import shutil
import time
st.title("Multimodal RAG 🤖")

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
        for uploaded_file in uploaded_files:
            # Construct the local path
            file_path = os.path.join("C:/Users/a6hij/PycharmProjects/RAG_Pipeline/local_data/", uploaded_file.name)

            # Write the bytes to local memory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Saved {uploaded_file.name}")
if delete:
    shutil.rmtree("C:/Users/a6hij/PycharmProjects/RAG_Pipeline/local_data/")
    os.mkdir("C:/Users/a6hij/PycharmProjects/RAG_Pipeline/local_data/")
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
        assistant_response = f"Echo: {prompt}"  # Replace this with your RAG or MCTS logic!

        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})