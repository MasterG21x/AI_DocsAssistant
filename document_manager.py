from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from pathlib import Path
import os

class DocumentManager():
    """
    A class to manage document loading, splitting, and file handling
    for the RAG (Retrieval-Augmented Generation) system.
    """
    def __init__(self, model_name, chunk_size):
        """
        Initialize the DocumentManager with a specified model and text splitter.
        """
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0, top_p=0.9)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=20
        )

    def split_text(self, text):
        """
        Split a single text string into document chunks.
        """
        return self.text_splitter.create_documents([text])

    def load_documents(self, directory_path):
        """
        Load all documents from the specified directory.
        Supports .txt, .md, and .pdf files.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            return []
        docs = []
        for file in Path(directory_path).iterdir():
            if file.suffix in ('.txt', '.md'):
                loader = TextLoader(str(file), encoding='utf-8')
            elif file.suffix == '.pdf':
                loader = PyMuPDFLoader(str(file))
            elif file.suffix == '.docx':
                loader = Docx2txtLoader(str(file)) 
            else:
                print(f"Unsupported file format: {file.name}")
                continue
            docs.extend(loader.load())
        return docs

    def save_uploaded_files(self, uploaded_files, target_path='files/'):
        """
        Save uploaded files to the specified directory.
        """
        os.makedirs(target_path, exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join(target_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        print(f"Saved  {len(uploaded_files)} files to {target_path}")
        
    def show_available_files(self, target_path = 'files/'):
        """
        List available files in the specified directory.
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
            return []
        file_names = [f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
        return file_names 
    
    def load_single_file_as_text(self, chosen_file): 
        """
        Load the content of a single file as a string.
        Supports .txt, .md, and .pdf files.
        """
        filepath = os.path.join('files', chosen_file)
        file = Path(filepath)
        if file.suffix in ('.txt', '.md'):
            loader = TextLoader(str(file), encoding='utf-8')
        elif file.suffix == '.pdf':
            loader = PyMuPDFLoader(str(file))
        elif file.suffix == '.docx':
            loader = Docx2txtLoader(str(file)) 
        else:
            raise ValueError(f"Unsupported file format: {file.name}")  
        
        try:
            docs = loader.load()             
        except RuntimeError as e:               
            raise RuntimeError(f"Skipping file {file.name}: {e}")               
        if not docs:
            raise RuntimeError(f"Plik {file.name} is empty.")

        return "\n\n".join(doc.page_content for doc in docs)