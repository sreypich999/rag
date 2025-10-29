import streamlit as st
import tempfile
import os
import datetime
from typing import List, Dict
import google.generativeai as genai

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
GEMINI_MODEL = "gemini-2.5-flash"

# Set Google API Key directly
GEMINI_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# --- Gemini PDF Extraction ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using the Gemini 2.5 Flash model"""
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"❌ Error configuring the API. Details: {e}")
        return None

    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        with st.spinner(f"⏳ Uploading and extracting text from PDF..."):
            pdf_file = genai.upload_file(path=pdf_path)
        
        with st.spinner("🔍 Gemini is extracting text from the PDF..."):
            response = model.generate_content([
                """You are an expert at data extraction. Extract ALL text from this PDF document. 
                
                Important instructions:
                1. Preserve the original layout, paragraphs, headings, and line breaks
                2. Extract text exactly as they appear
                3. Include all visible text elements
                
                Extract:
                - Headers and footers
                - Section headings and subheadings  
                - Paragraph text
                - Lists and bullet points
                - Table content (as structured text)
                - Captions and labels
                - Any visible text elements
                
                Maintain the original formatting and structure.""",
                pdf_file
            ])

        extracted_text = response.text
        
        # Clean up the uploaded file after use
        try:
            genai.delete_file(pdf_file.name)
        except:
            pass

        return extracted_text

    except Exception as e:
        st.error(f"❌ An error occurred during PDF extraction: {e}")
        try:
            if 'pdf_file' in locals():
                genai.delete_file(pdf_file.name)
        except:
            pass
        return None

def save_text_to_file(text_content, filename):
    """Save extracted text to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_content)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def process_pdfs_with_gemini(uploaded_files):
    """Process multiple PDFs using Gemini for text extraction"""
    all_docs = []
    saved_files = []
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            temp_path = tmp.name
        
        try:
            extracted_text = extract_text_from_pdf(temp_path)
            
            if extracted_text:
                # Save to text file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_filename = f"extracted_{file.name}_{timestamp}.txt"
                
                if save_text_to_file(extracted_text, txt_filename):
                    saved_files.append(txt_filename)
                    st.success(f"✅ Gemini extracted and saved: {txt_filename}")
                
                # Also prepare for vector store
                from langchain.schema import Document
                doc = Document(
                    page_content=extracted_text,
                    metadata={
                        "source": file.name,
                        "page": 1,
                        "total_pages": 1,
                        "extraction_method": "gemini"
                    }
                )
                all_docs.append(doc)
            else:
                st.error(f"❌ Failed to extract text from {file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
    
    return all_docs, saved_files

# --- Core RAG Functions ---
def get_mpnet_embeddings():
    """Initialize MPNet embedding function"""
    return HuggingFaceEmbeddings(model_name=MPNET_MODEL_NAME)

@st.cache_resource
def get_chroma_retriever():
    """Initialize ChromaDB retriever"""
    try:
        embeddings = get_mpnet_embeddings()
        
        # Check if vector store exists, if not return None
        if not os.path.exists(CHROMA_PERSIST_DIR):
            return None
            
        db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        return db.as_retriever(search_kwargs={"k": 10})

    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None

def process_and_store_documents(uploaded_files):
    """Process PDFs and store in ChromaDB"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Process PDFs using Gemini extraction
    all_docs, saved_files = process_pdfs_with_gemini(uploaded_files)
    
    if not all_docs:
        return "No documents were successfully processed.", saved_files
    
    # Split documents into chunks
    split_docs = []
    for doc in all_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            from langchain.schema import Document
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "chunk": i + 1,
                    "extraction_method": "gemini"
                }
            )
            split_docs.append(chunk_doc)
    
    embeddings = get_mpnet_embeddings()
    
    try:
        # Create or load ChromaDB
        db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        
        # ChromaDB automatically persists
        st.session_state[RETRIEVER_KEY] = db.as_retriever(search_kwargs={"k": 10})
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
        
        return f"Added **{len(split_docs)}** document chunks to knowledge base.", saved_files
            
    except Exception as e:
        st.error(f"Error storing documents in ChromaDB: {e}")
        return f"Error: {e}", saved_files

@st.cache_resource
def get_conversational_chain(_retriever):
    """Define RAG chain with Gemini 2.5 Flash"""
    try:
        model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=2048,
            google_api_key=GEMINI_API_KEY
        )
        
        # Simple prompt template
        prompt_template = """
        You are a helpful assistant. Answer the question based on the given context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True, 
            input_key='question',
            output_key='answer'
        )
        
        CUSTOM_PROMPT = PromptTemplate.from_template(prompt_template)
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=_retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            return_source_documents=False,
            return_generated_question=False,
        )
        return chain
        
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

def user_input(user_question):
    """Process user question and generate RAG response"""
    if CHAT_CHAIN_KEY not in st.session_state or not st.session_state[CHAT_CHAIN_KEY]:
        st.error("Please process documents first to initialize the chat chain.")
        return

    chain = st.session_state[CHAT_CHAIN_KEY]
    
    if chain is None:
        st.error("Chat chain is not properly initialized.")
        return
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner(f"{GEMINI_MODEL} is thinking..."):
        try:
            result = chain({"question": user_question})
            response_text = result["answer"]

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
        except Exception as e:
            st.error(f"Error generating response: {e}")

# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        layout="wide",
        page_icon="📄"
    )
    
    st.title("📄 PDF Chat Assistant")
    st.markdown(f"""
    Smart PDF processing and chat:
    - **📤 Upload PDFs** - Extract text using Gemini AI and save as .txt files
    - **💬 Chat** - Ask questions about your documents
    - **🧠 Gemini {GEMINI_MODEL}** - Powered by Google's latest AI for extraction and chat
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if RETRIEVER_KEY not in st.session_state:
        st.session_state[RETRIEVER_KEY] = get_chroma_retriever()
        
    if CHAT_CHAIN_KEY not in st.session_state and st.session_state[RETRIEVER_KEY] is not None:
        st.session_state[CHAT_CHAIN_KEY] = get_conversational_chain(st.session_state[RETRIEVER_KEY])
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Your Documents")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question := st.chat_input(f"Ask about your documents... (Powered by {GEMINI_MODEL})"):
            user_input(user_question)

    with col2:
        st.markdown("### 📊 Status")
        
        if st.session_state.messages:
            st.metric("Conversation Messages", len(st.session_state.messages))
            
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("💡 Upload PDFs and start chatting!")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF Files", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("🚀 Process with Gemini & Extract Text", use_container_width=True):
            if uploaded_files:
                with st.spinner("Gemini is processing documents..."):
                    result_message, saved_files = process_and_store_documents(uploaded_files)
                    st.success(result_message)
                    
                    if saved_files:
                        st.markdown("**📄 Text Files Saved:**")
                        for file in saved_files:
                            st.code(file, language="text")
                    
                    st.session_state.messages = []
                    st.info("Chat history cleared. Start querying your documents!")
            else:
                st.warning("Please upload PDF files first.")

        st.markdown("---")
        st.subheader("🔧 System Status")
        retriever_status = "✅ Ready" if st.session_state.get(RETRIEVER_KEY) else "❌ No documents"
        chat_chain_status = "✅ Ready" if st.session_state.get(CHAT_CHAIN_KEY) else "❌ Not initialized"
        
        st.info(f"**Knowledge Base:** {retriever_status}")
        st.info(f"**Chat Chain:** {chat_chain_status}")
        
        st.markdown("**🤖 Active Models:**")
        st.caption(f"• Embeddings: MPNet")
        st.caption(f"• Chat: {GEMINI_MODEL}")
        st.caption(f"• PDF Extraction: {GEMINI_MODEL}")

if __name__ == "__main__":
    main()
