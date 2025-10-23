import streamlit as st
import tempfile
import os

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- NEW IMPORT: Using HuggingFaceEmbeddings for the MPNet model ---
from langchain_huggingface import HuggingFaceEmbeddings
# We keep the Google LLM for the chat functionality
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# --- Configuration Constants ---
CHROMA_PERSIST_DIR = "vector_store"
CHROMA_COLLECTION_NAME = "mpnet_rag_collection"
RETRIEVER_KEY = "retriever"
CHAT_CHAIN_KEY = "chat_chain"
# --- MPNet Model Identifier (Change 1/4) ---
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
# Note: For production use, you might set model_kwargs={'device': 'cuda'} 
# if you have a powerful GPU. We omit it here for maximum compatibility.

# --- Utility Functions ---

# We remove @st.cache_resource here because the HuggingFace model load can sometimes
# cause issues with Streamlit's caching mechanism on repeated runs, 
# although in a production environment, you might try to cache the entire setup.
def get_mpnet_embeddings():
    """Initializes and returns the MPNet embedding function."""
    # This will download the model locally if not already present
    return HuggingFaceEmbeddings(model_name=MPNET_MODEL_NAME)

@st.cache_resource
def get_chroma_retriever():
    """Initializes and returns the ChromaDB retriever, ensuring persistence."""
    try:
        # 1. Initialize MPNet Embeddings (Change 2/4)
        embeddings = get_mpnet_embeddings()
        
        # 2. Load the persistent Chroma client (creates dir/collection if not exist)
        db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # 3. Create a retriever instance
        return db.as_retriever(search_kwargs={"k": 5})

    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        st.warning("Did you install the required packages? Run: `pip install sentence-transformers langchain-huggingface`")
        return None

def process_and_store_documents(uploaded_files):
    """Loads documents, splits them, and adds them to ChromaDB."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    all_docs = []
    
    # Process each uploaded file
    for file in uploaded_files:
        # Use tempfile to save uploaded file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            temp_path = tmp.name
        
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load_and_split(text_splitter)
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        finally:
            os.remove(temp_path) # Clean up temp file
            
    # Get the embedding instance (Change 3/4)
    embeddings = get_mpnet_embeddings()
    
    # Get the existing ChromaDB instance
    db = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    # Add documents to the database
    if all_docs:
        # Add a placeholder message for embedding progress
        with st.spinner(f"Creating {len(all_docs)} embeddings using MPNet... this may take a moment."):
            db.add_documents(all_docs)
        db.persist()
        
        # Update the retriever and chain in session state
        st.session_state[RETRIEVER_KEY] = db.as_retriever(search_kwargs={"k": 5})
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
        
        return f"Successfully added **{len(all_docs)}** document chunks to the database using **MPNet** embeddings."
    else:
        return "No documents were successfully processed."

@st.cache_resource
def get_conversational_chain(retriever):
    """
    Defines the RAG chain using the retriever and adds conversational memory.
    (Keeps ChatGoogleGenerativeAI for the LLM)
    """
    # Note: We keep the Google model for the generation step (Change 4/4)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    custom_prompt_template = """
    You are a helpful assistant. Please analyze the provided context and the conversation history carefully.
    Your answer MUST be based ONLY on the given context from the retrieved documents.
    If the context does not contain the answer, politely state, "I could not find the answer in the provided documents."
    **IMPORTANT:** Please respond in the same language as the user's latest question.

    Chat History:
    {chat_history}

    Context from Documents:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # Create the memory object to store history
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        input_key='question',
        output_key='answer'
    )
    
    # Create a PromptTemplate
    CUSTOM_PROMPT = PromptTemplate.from_template(custom_prompt_template)
    
    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=False,
        return_generated_question=False,
    )
    return chain

def user_input(user_question):
    """Processes the user question, retrieves, and generates a RAG response."""
    
    # Ensure chain is initialized
    if CHAT_CHAIN_KEY not in st.session_state or not st.session_state[CHAT_CHAIN_KEY]:
        st.error("Please process documents first to initialize the chat chain.")
        return

    chain = st.session_state[CHAT_CHAIN_KEY]
    
    # Append user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response using the conversational chain
    with st.spinner("Gemini is thinking..."):
        # The chain handles the retrieval, memory, and generation
        result = chain({"question": user_question})
        response_text = result["answer"]

    # Append assistant message to history and display it
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)


# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="MPNet RAG Chat (ChromaDB)", layout="wide")
    st.title("MPNet RAG: Chat with your Documents! 💬 (Local Embeddings)")
    st.markdown("This chatbot uses **MPNet** for document vectorization (local, open-source embeddings) and **Gemini** for chat response generation. It maintains conversational memory.")
    
    # Set the GOOGLE_API_KEY from Streamlit secrets (still needed for the Gemini LLM)
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            # We assume st.secrets is available in the Streamlit environment
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except:
            # Display a warning if the key is missing, as the LLM won't work
            st.warning("Google API Key not found. Please set it in your environment or secrets for the LLM to function.")
            # We allow the app to run so the user can still try to upload documents (which use MPNet)
            # but the chat won't work without the LLM key.

    # --- Initialize Session State (History and Components) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if RETRIEVER_KEY not in st.session_state:
        st.session_state[RETRIEVER_KEY] = get_chroma_retriever()
        
    if CHAT_CHAIN_KEY not in st.session_state and st.session_state[RETRIEVER_KEY] is not None:
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
    
    
    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input Section ---
    if user_question := st.chat_input("Ask a question about your documents..."):
        user_input(user_question)

    # --- Sidebar for Document Management ---
    with st.sidebar:
        st.subheader("Document Upload (PDFs)")
        
        uploaded_files = st.file_uploader(
            "Upload your PDF files here", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process & Add to Knowledge Base"):
            if uploaded_files:
                with st.spinner("Processing... Creating embeddings and storing in ChromaDB."):
                    result_message = process_and_store_documents(uploaded_files)
                    st.success(result_message)
                    
                    # Clear history after processing new documents
                    st.session_state.messages = []
                    st.info("Chat history cleared. You can now start querying the new documents!")
            else:
                st.warning("Please upload at least one PDF file before processing.")

        st.markdown("---")
        # Display status of the Knowledge Base
        retriever_status = "Ready to Query" if st.session_state.get(RETRIEVER_KEY) else "No documents loaded/processed."
        st.info(f"Knowledge Base Status: **{retriever_status}**")
        st.caption(f"Embedding Model: `{MPNET_MODEL_NAME}`")
        st.caption(f"Vector Store Location: `{CHROMA_PERSIST_DIR}/`")


if __name__ == "__main__":
    main()
