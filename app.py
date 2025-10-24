import streamlit as st
import tempfile
import os
import datetime
from typing import List, Dict
import json

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# --- Configuration Constants ---
CHROMA_PERSIST_DIR = "vector_store"
CHROMA_COLLECTION_NAME = "mpnet_rag_collection"
RETRIEVER_KEY = "retriever"
CHAT_CHAIN_KEY = "chat_chain"
EXTRACTION_LLM_KEY = "extraction_llm"
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
GEMINI_MODEL = "gemini-2.5-flash"  # Using Gemini 2.5 Flash for both chat and extraction

# --- Requirements Documentation ---
def show_requirements():
    """Display system requirements and capabilities"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 System Requirements")
    
    with st.sidebar.expander("View Requirements & Capabilities"):
        st.markdown("""
        ### 🎯 **System Requirements**

        **Core Components:**
        - ✅ MPNet (all-mpnet-base-v2) for document embeddings
        - ✅ Gemini 2.5 Flash for chat responses
        - ✅ Gemini 2.5 Flash for conversation extraction
        - ✅ ChromaDB for vector storage
        - ✅ Streamlit for web interface

        **Key Features:**
        - 📄 PDF document processing and chunking
        - 💬 Conversational RAG with memory
        - 🧠 Local MPNet embeddings (no API calls)
        - 🤖 Dual Gemini 2.5 Flash usage (chat + extraction)
        - 📊 Conversation statistics and analytics
        - 💾 Persistent vector storage
        - 📤 Text file extraction with AI formatting

        **Extraction Capabilities:**
        - 🪄 AI-powered conversation analysis
        - 📑 Multiple output formats
        - 🔍 Theme and topic identification
        - 📝 Action item extraction
        - 🎨 Professional markdown formatting
        - ⬇️ Downloadable text files

        **Model Specifications:**
        - **Embedding Model**: MPNet (sentence-transformers/all-mpnet-base-v2)
        - **Chat Model**: Gemini 2.5 Flash
        - **Extraction Model**: Gemini 2.5 Flash
        - **Vector DB**: ChromaDB with persistence
        """)

# --- Conversation Extraction Functions with Gemini 2.5 Flash ---

@st.cache_resource
def get_extraction_llm():
    """Initialize Gemini 2.5 Flash specifically for extraction"""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,  # Lower temperature for consistent formatting
        max_output_tokens=4096,
        timeout=60
    )

def generate_gemini_extraction(messages: List[Dict], extraction_style: str = "Comprehensive Analysis") -> str:
    """
    Use Gemini 2.5 Flash to generate a comprehensive, well-formatted extraction of the conversation
    """
    if not messages:
        return "No conversation to extract."
    
    # Prepare conversation text for Gemini
    conversation_text = "CONVERSATION HISTORY:\n\n"
    for i, message in enumerate(messages, 1):
        role = "USER" if message["role"] == "user" else "ASSISTANT"
        conversation_text += f"{i}. {role}: {message['content']}\n\n"
    
    # Style-specific prompts
    style_prompts = {
        "Comprehensive Analysis": """
        Create a comprehensive analysis with:
        - Executive summary
        - Detailed conversation breakdown
        - Key insights and findings
        - Topic analysis
        - Action items and recommendations
        - Conclusion
        """,
        "Summary Report": """
        Create a concise summary with:
        - Brief overview
        - Main points discussed
        - Key conclusions
        - Quick takeaways
        """,
        "Q&A Format": """
        Format as Q&A with:
        - Clear question-answer pairs
        - Grouped by topic
        - Direct quotes where relevant
        - Follow-up questions identified
        """,
        "Technical Documentation": """
        Format as technical documentation:
        - Structured sections
        - Code blocks if technical content
        - Technical specifications
        - Implementation notes
        - API references if applicable
        """
    }
    
    # Create a detailed prompt for Gemini 2.5 Flash
    extraction_prompt = f"""
    You are an expert analyst. Please analyze the following conversation and create a professional, well-structured extraction document.
    
    **EXTRACTION STYLE: {extraction_style}**
    {style_prompts.get(extraction_style, style_prompts["Comprehensive Analysis"])}
    
    **FORMATTING REQUIREMENTS:**
    1. Use proper markdown formatting with headers, subheaders, and sections
    2. Include metadata section with date, participants, and statistics
    3. Use bullet points and numbered lists for structured content
    4. Apply **bold** and *italic* text for emphasis where appropriate
    5. Include tables for statistics if relevant
    6. Ensure professional tone and clear organization
    7. Highlight key insights and action items
    
    **ADDITIONAL ANALYSIS:**
    - Identify main topics and themes
    - Extract key insights and findings
    - Note any decisions made or conclusions reached
    - List action items or follow-up tasks
    - Analyze the quality and depth of the conversation
    
    **CONVERSATION TO ANALYZE:**
    {conversation_text}
    
    Please generate the formatted extraction:
    """
    
    try:
        # Use the dedicated extraction LLM
        if EXTRACTION_LLM_KEY not in st.session_state:
            st.session_state[EXTRACTION_LLM_KEY] = get_extraction_llm()
        
        extraction_llm = st.session_state[EXTRACTION_LLM_KEY]
        
        # Generate the extraction
        response = extraction_llm.invoke(extraction_prompt)
        return response.content
        
    except Exception as e:
        st.error(f"Gemini 2.5 Flash extraction failed: {e}")
        # Fallback to basic formatting
        return create_basic_extraction(messages)

def create_basic_extraction(messages: List[Dict]) -> str:
    """
    Fallback basic extraction if Gemini fails
    """
    extraction_content = []
    
    # Header with metadata
    extraction_content.append("# CHAT CONVERSATION EXTRACTION")
    extraction_content.append("")
    extraction_content.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    extraction_content.append(f"**Model:** {GEMINI_MODEL}")
    extraction_content.append(f"**Total Messages:** {len(messages)}")
    extraction_content.append("")
    extraction_content.append("---")
    extraction_content.append("")
    
    # Conversation content
    extraction_content.append("## CONVERSATION CONTENT")
    extraction_content.append("")
    
    for i, message in enumerate(messages, 1):
        role = message["role"].upper()
        content = message["content"]
        
        extraction_content.append(f"### Message {i}: {role}")
        extraction_content.append("")
        
        if role == "USER":
            extraction_content.append(f"**Question:** {content}")
        else:
            # Format assistant responses with better structure
            lines = content.split('\n')
            if any(line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')) for line in lines):
                extraction_content.append("**Response:**")
                extraction_content.append("")
                for line in lines:
                    if line.strip().startswith(('-', '*', '•')):
                        extraction_content.append(f"- {line.strip().lstrip('-*• ')}")
                    elif line.strip().startswith(tuple(f"{n}." for n in range(1, 10))):
                        extraction_content.append(line)
                    else:
                        extraction_content.append(line)
            else:
                extraction_content.append(f"**Response:** {content}")
        
        extraction_content.append("")
        extraction_content.append("---")
        extraction_content.append("")
    
    # Summary section
    extraction_content.append("## CONVERSATION SUMMARY")
    extraction_content.append("")
    
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    
    extraction_content.append(f"**Total Questions:** {len(user_messages)}")
    extraction_content.append(f"**Total Responses:** {len(assistant_messages)}")
    extraction_content.append("")
    
    if user_messages:
        extraction_content.append("**Questions Asked:**")
        for i, question in enumerate(user_messages, 1):
            extraction_content.append(f"{i}. {question['content']}")
    
    return "\n".join(extraction_content)

def save_conversation_to_file(messages: List[Dict], filename: str = None, use_gemini: bool = True, extraction_style: str = "Comprehensive Analysis") -> str:
    """
    Save conversation to a text file using Gemini 2.5 Flash for formatting
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_extraction_{timestamp}.txt"
    
    # Generate extraction content
    if use_gemini:
        formatted_content = generate_gemini_extraction(messages, extraction_style)
    else:
        formatted_content = create_basic_extraction(messages)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        return filename
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def show_extraction_output(messages: List[Dict]):
    """
    Display the formatted extraction output in the Streamlit app
    """
    if not messages:
        st.warning("No conversation to extract.")
        return
    
    # Extraction options
    st.markdown("## 📄 AI-Powered Conversation Extraction")
    st.markdown(f"*Using {GEMINI_MODEL} for intelligent formatting*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_gemini = st.checkbox("Use Gemini 2.5 Flash AI", value=True, 
                                 help="Use AI to create a well-structured, insightful summary")
    
    with col2:
        extraction_style = st.selectbox(
            "Extraction Style",
            ["Comprehensive Analysis", "Summary Report", "Q&A Format", "Technical Documentation"],
            help="Choose the format style for the extraction"
        )
    
    # Generate and display preview
    if st.button("🪄 Generate AI Extraction", type="primary"):
        with st.spinner(f"{GEMINI_MODEL} is analyzing and formatting the conversation..."):
            formatted_content = generate_gemini_extraction(messages, extraction_style)
            
            # Display preview
            st.markdown("### 📋 Extraction Preview")
            st.markdown("---")
            
            # Show preview in a scrollable area
            preview = formatted_content[:2500] + "..." if len(formatted_content) > 2500 else formatted_content
            st.text_area("Preview", preview, height=400, key="extraction_preview")
            
            # Store the full content in session state for downloading
            st.session_state.current_extraction = formatted_content
            st.session_state.use_gemini = use_gemini
            st.session_state.extraction_style = extraction_style
    
    # Save file and provide download
    if "current_extraction" in st.session_state:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to TXT File"):
                filename = save_conversation_to_file(
                    messages, 
                    use_gemini=st.session_state.get('use_gemini', True),
                    extraction_style=st.session_state.get('extraction_style', 'Comprehensive Analysis')
                )
                if filename:
                    st.success(f"✅ Conversation saved to: `{filename}`")
        
        with col2:
            # Provide download button for the current extraction
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"conversation_extraction_{timestamp}.txt"
            
            st.download_button(
                label="⬇️ Download Extraction",
                data=st.session_state.current_extraction,
                file_name=download_filename,
                mime="text/plain",
                type="primary"
            )

def get_conversation_stats(messages: List[Dict]) -> Dict:
    """
    Generate comprehensive statistics about the conversation
    """
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    
    total_chars = sum(len(msg["content"]) for msg in messages)
    total_words = sum(len(msg["content"].split()) for msg in messages)
    
    avg_user_chars = sum(len(msg["content"]) for msg in user_messages) / len(user_messages) if user_messages else 0
    avg_assistant_chars = sum(len(msg["content"]) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
    
    return {
        "total_messages": len(messages),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_user_message_length": round(avg_user_chars, 1),
        "avg_assistant_message_length": round(avg_assistant_chars, 1),
        "conversation_duration_minutes": len(messages) * 0.5  # Estimate
    }

def display_conversation_analytics(messages: List[Dict]):
    """Display advanced conversation analytics"""
    if not messages:
        return
    
    stats = get_conversation_stats(messages)
    
    st.markdown("### 📊 Conversation Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", stats["total_messages"])
    with col2:
        st.metric("Questions", stats["user_messages"])
    with col3:
        st.metric("Responses", stats["assistant_messages"])
    with col4:
        st.metric("Total Words", stats["total_words"])
    
    # Additional stats in expander
    with st.expander("Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Message Lengths:**")
            st.write(f"Avg Question: {stats['avg_user_message_length']} chars")
            st.write(f"Avg Response: {stats['avg_assistant_message_length']} chars")
            st.write(f"Total Characters: {stats['total_characters']}")
        
        with col2:
            st.write("**Conversation Metrics:**")
            st.write(f"Estimated Duration: {stats['conversation_duration_minutes']:.1f} min")
            st.write(f"Words per Message: {stats['total_words']/stats['total_messages']:.1f}")

# --- Original Utility Functions ---

def get_mpnet_embeddings():
    """Initializes and returns the MPNet embedding function."""
    return HuggingFaceEmbeddings(model_name=MPNET_MODEL_NAME)

@st.cache_resource
def get_chroma_retriever():
    """Initializes and returns the ChromaDB retriever, ensuring persistence."""
    try:
        embeddings = get_mpnet_embeddings()
        
        db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
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
    
    for file in uploaded_files:
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
            os.remove(temp_path)
            
    embeddings = get_mpnet_embeddings()
    
    db = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    if all_docs:
        with st.spinner(f"Creating {len(all_docs)} embeddings using MPNet... this may take a moment."):
            db.add_documents(all_docs)
        db.persist()
        
        st.session_state[RETRIEVER_KEY] = db.as_retriever(search_kwargs={"k": 5})
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
        
        return f"Successfully added **{len(all_docs)}** document chunks to the database using **MPNet** embeddings."
    else:
        return "No documents were successfully processed."

@st.cache_resource
def get_conversational_chain(retriever):
    """
    Defines the RAG chain using the retriever and adds conversational memory.
    Uses Gemini 2.5 Flash for chat responses.
    """
    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        max_output_tokens=2048
    )
    
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
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        input_key='question',
        output_key='answer'
    )
    
    CUSTOM_PROMPT = PromptTemplate.from_template(custom_prompt_template)
    
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
    
    if CHAT_CHAIN_KEY not in st.session_state or not st.session_state[CHAT_CHAIN_KEY]:
        st.error("Please process documents first to initialize the chat chain.")
        return

    chain = st.session_state[CHAT_CHAIN_KEY]
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner(f"{GEMINI_MODEL} is thinking..."):
        result = chain({"question": user_question})
        response_text = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

# --- Streamlit App UI ---
def main():
    st.set_page_config(
        page_title="MPNet RAG with Gemini 2.5 Flash", 
        layout="wide",
        page_icon="🤖"
    )
    
    st.title("🤖 MPNet RAG: Chat with AI-Powered Extraction")
    st.markdown(f"""
    This advanced chatbot uses:
    - **🧠 MPNet** for local document embeddings
    - **⚡ Gemini 2.5 Flash** for intelligent chat responses  
    - **⚡ Gemini 2.5 Flash** for AI-powered conversation extraction
    - **💾 ChromaDB** for persistent vector storage
    """)
    
    # Set the GOOGLE_API_KEY from Streamlit secrets
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except:
            st.warning("Google API Key not found. Please set it in your environment or secrets for the LLM to function.")

    # Show requirements documentation
    show_requirements()

    # --- Initialize Session State (History and Components) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if RETRIEVER_KEY not in st.session_state:
        st.session_state[RETRIEVER_KEY] = get_chroma_retriever()
        
    if CHAT_CHAIN_KEY not in st.session_state and st.session_state[RETRIEVER_KEY] is not None:
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
    
    # --- Main Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # --- Display Chat History ---
        st.markdown("### 💬 Conversation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --- Chat Input Section ---
        if user_question := st.chat_input(f"Ask a question about your documents... (Powered by {GEMINI_MODEL})"):
            user_input(user_question)

    with col2:
        # --- Analytics and Extraction Panel ---
        st.markdown("### 📈 Analytics & Extraction")
        
        if st.session_state.messages:
            display_conversation_analytics(st.session_state.messages)
            show_extraction_output(st.session_state.messages)
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation History", use_container_width=True):
                st.session_state.messages = []
                if 'current_extraction' in st.session_state:
                    del st.session_state.current_extraction
                st.rerun()
        else:
            st.info("💡 Start a conversation to see analytics and extraction options!")
            st.image("https://via.placeholder.com/300x200/4CAF50/white?text=Start+Chatting", use_column_width=True)

    # --- Sidebar for Document Management ---
    with st.sidebar:
        st.header("📁 Document Management")
        
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            accept_multiple_files=True,
            type=["pdf"],
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Process & Add to Knowledge Base", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing... Creating embeddings and storing in ChromaDB."):
                    result_message = process_and_store_documents(uploaded_files)
                    st.success(result_message)
                    
                    # Clear history after processing new documents
                    st.session_state.messages = []
                    if 'current_extraction' in st.session_state:
                        del st.session_state.current_extraction
                    st.info("Chat history cleared. You can now start querying the new documents!")
            else:
                st.warning("Please upload at least one PDF file before processing.")

        st.markdown("---")
        
        # System status
        st.subheader("🔧 System Status")
        retriever_status = "✅ Ready" if st.session_state.get(RETRIEVER_KEY) else "❌ No documents"
        st.info(f"**Knowledge Base:** {retriever_status}")
        
        # Model information
        st.markdown("**🤖 Active Models:**")
        st.caption(f"• Embeddings: `{MPNET_MODEL_NAME}`")
        st.caption(f"• Chat: `{GEMINI_MODEL}`")
        st.caption(f"• Extraction: `{GEMINI_MODEL}`")
        st.caption(f"• Vector DB: `ChromaDB`")
        
        st.markdown("---")
        st.caption(f"Vector Store: `{CHROMA_PERSIST_DIR}/`")


if __name__ == "__main__":
    main()
