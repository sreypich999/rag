# RAG Chroma Gemini App

A powerful PDF Chat Assistant that combines Retrieval-Augmented Generation (RAG) with Google's Gemini AI for intelligent document processing and conversational querying.

## 🚀 Features

- **📄 PDF Text Extraction**: Advanced text extraction from PDF documents using Google's Gemini 2.5 Flash model
- **🧠 Intelligent Chat**: Conversational AI powered by Gemini 2.5 Flash for answering questions about your documents
- **💾 Vector Storage**: Persistent document storage using ChromaDB with MPNet embeddings
- **🔍 Semantic Search**: Efficient retrieval of relevant document chunks using vector similarity
- **💬 Streamlit Interface**: Modern, responsive web interface for easy interaction
- **📁 Multi-Document Support**: Process and chat with multiple PDF files simultaneously
- **💾 Text File Export**: Automatically saves extracted text as .txt files for future reference

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Models**:
  - [Google Gemini 2.5 Flash](https://ai.google.dev/models/gemini) (for both text extraction and chat)
  - [Sentence Transformers MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (for embeddings)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- **RAG Framework**: [LangChain](https://www.langchain.com/)
- **PDF Processing**: [PyPDF2](https://pypdf2.readthedocs.io/) + Gemini AI

## 📋 Prerequisites

- Python 3.10 or higher
- Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🔧 Installation


1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   - Copy `.env.example` to `.env` (if exists) or create a `.env` file
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## 🚀 Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the app**:
   - Open your browser and go to `http://localhost:8501`

3. **Upload PDFs**:
   - Use the sidebar to upload one or more PDF files
   - Click "🚀 Process with Gemini & Extract Text" to process documents

4. **Start chatting**:
   - Ask questions about your uploaded documents in the chat interface
   - The AI will provide answers based on the content of your PDFs

## 📁 Project Structure

```
rag-chroma-gemini-app/
├── app.py                 # Main Streamlit application
├── main.py               # Alternative entry point
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project configuration
├── README.md            # This file
├── .env                 # Environment variables (API keys)
├── .streamlit/          # Streamlit configuration
├── vector_store/        # ChromaDB persistent storage
└── extracted_*.txt      # Auto-generated text files from PDFs
```

## 🔑 Configuration

### API Keys
The application requires a Google Gemini API key. You can obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey).

### Models Used
- **Text Extraction**: `gemini-2.5-flash`
- **Chat Model**: `gemini-2.5-flash`
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`

### Vector Store Settings
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top-K Retrieval**: 10 most similar chunks

