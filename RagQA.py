from document_manager import DocumentManager
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from chromadb import Settings
import os
DATA_PATH = 'files/'
CHROMA_PATH = 'chroma'

class RagQA():
    """
    A class for Retrieval-Augmented Generation Question Answering (RAG QA).
    Handles document chunking, embedding storage, and querying.
    """
    def __init__(self, model_name):
        """
        Initialize the RAG QA system with a specified LLM and embedding model.
        """
        self.document_manager = DocumentManager(model_name=model_name, chunk_size=1000)
        self.llm = self.document_manager.llm
        self.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.base_map_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the following question using ONLY the provided context:
            {context}
            ---------
            Answer the question based on the context above: {question}
            """
        )
        
    def split_text(self, text):
        """
        Split a single text string into document chunks using the DocumentManager.
        """
        return self.document_manager.split_text(text)
        
    def save_to_chroma(self, docs):
        """
        Save documents to a Chroma vector store.
        Resets the database before storing new embeddings.
        """
        try:
            chunks_nested = [self.split_text(doc.page_content) for doc in docs]
            chunks = [chunk for sublist in chunks_nested for chunk in sublist]

            db = Chroma.from_documents(chunks, self.embeddings_model, persist_directory=CHROMA_PATH)
            db.persist()
            
            print(f"Saved {len(chunks)} documents to {CHROMA_PATH}")

        except Exception as e:
            print(f"Error saving to Chroma: {e}")
            raise RuntimeError(f"Error saving to the database: {e}")
        
    def load_from_chroma(self):
        """
        Load the Chroma vector store from disk.
        """
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError("Chroma database does not exist.")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings_model)
        return db

        
    def query_to_db(self, question, threshold=0.2):
        """
        Perform a similarity search on the vector store using the given question.
        """
        try:
            db = self.load_from_chroma()
            results = db.similarity_search_with_relevance_scores(question, k=3)

            if not results or results[0][1] <  threshold:
                return None

            context = "\n\n".join(doc.page_content for doc, _ in results)
            sources = [doc.metadata.get("source") for doc, _ in results]
            return context, sources

        except FileNotFoundError as fnf:
            raise fnf
        except Exception as e:
            raise RuntimeError(f"Error querying the database: {e}")
        
    def inspect_chroma(self, preview_count=3, preview_length=100):
        """
        Preview the stored documents in the Chroma vector store.
        """
        try:
            db = self.load_from_chroma()
            docs = db.get(include=["documents"])['documents']
            previews = [f"---Chunk {i+1}---\n{doc[:preview_length]}..." for i, doc in enumerate(docs[:preview_count])]
            return docs, previews
        except Exception as e:
            raise RuntimeError(f" Error previewing the database: {e}")
    
    def query_answer(self, question):
        """
        Generate an answer to the user's question using the retrieved context.
        """
        context_data = self.query_to_db(question)
        if not context_data:
            return None
        prompt = self.base_map_prompt.format(context=context_data, question=question)
        try:
            return self.llm(prompt)
        except Exception as e:
            raise RuntimeError(f"Error generating the answer: {e}")


