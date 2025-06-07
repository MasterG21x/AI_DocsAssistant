AI Document Assistant
======================

A Streamlit-based application that uses RAG (Retrieval-Augmented Generation)
and summarization to help users ask questions and get summaries based on uploaded documents.
Whole project is running local so without GPU some taskts might take long.

Features:
---------
- Upload documents (.txt, .md, .pdf, .docx)
- Handle big documents
- Build a knowledge base with Chroma vector store
- Ask questions about the uploaded documents
- Generate summaries in different styles and lengths

Requirements:
-------------
- Python 3.9+
- Ollama (locally installed and running)
- CUDA (optional, for GPU acceleration)

How to run:
-----------
1. Install Python dependencies:
   pip install -r requirements.txt

2. Make sure Ollama is installed and running:
   ollama serve
   (Or run a specific model: ollama run llama3)

3. Run the Streamlit app:
   streamlit run app.py

Project Structure:
------------------
app.py                # Streamlit UI
RagQA.py              # RAG question-answering module
document_manager.py   # File handling and chunking
summarizer.py         # Summarizer module
requirements.txt      # Dependencies
files/                # Uploaded documents
chroma/               # Vector database

Notes:
------
- Ollama must be running locally (localhost:11434).
- This is a prototype app — error handling and file security are minimal.

