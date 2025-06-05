from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from langchain.embeddings import OllamaEmbeddings
import numpy as np

class Summarizer():
    """
    A class for summarizing text documents using an LLM.
    Handles chunking, embedding, and summarization tasks.
    """
    def __init__(self, model_name):
        """
        Initialize the Summarizer with the specified model.
        """
        self.llm = Ollama(model=model_name, temperature=0, top_p=0.9)
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=5000, chunk_overlap=20)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.length_options = {'short': 100, 'medium': 300, 'long': 600}
        self.style_options = ['informative', 'technical', 'simple language']
        
        self.base_map_prompt = PromptTemplate(
            input_variables=["text", "style", "length"],
            template="""
            Summarize the following text in the '{style}' style with a length of '{length}':

            Text: {text}
            """
        )
        self.base_combine_prompt = PromptTemplate(
            input_variables=["text", "style", "length"],
            template="""
            Combine the following summaries into a single one, keeping the '{style}' style and length '{length}':

            Summaries:
            {text}
            """
        )
        
    def _validate_input(self, text, length, style):
        """
        Validate that the input text, style, and length are correct.
        """
        if len(text.strip()) < 150:
            raise ValueError("The text is too short for summarization.")
        if style not in self.style_options:
            raise ValueError(f"Unknown style: {style}")
        if length not in self.length_options:
            raise ValueError(f"Unknown length: {length}")
        
    def split_text(self, text):
        """
        Split the input text into chunks.
        """
        chunks = self.text_splitter.create_documents([text])
        return chunks
    
    def paragraphs_selector(self, documents):
        """
        Use KMeans clustering to select representative paragraphs from the documents.
        """
        texts = [doc.page_content for doc in documents]
        embeddings_vectors = self.embeddings.embed_documents(texts)
        kmeans = KMeans(n_clusters=10, random_state=42).fit(embeddings_vectors)
        
        closest = []
        for i in range(10):
            distances = np.linalg.norm(embeddings_vectors - kmeans.cluster_centers_[i], axis=1)         
            closest_index = np.argmin(distances)
            closest.append(closest_index)
        selected_indices = sorted(set(closest))
        return [documents[i] for i in selected_indices]

    def create_summary(self, text, length, style):
        """
        Generate a summary for the given text with the specified length and style.
        """
        self._validate_input(text, length, style)
        chunks = self.split_text(text)

        if len(chunks) < 2:
            return self._summarize_single_chunk(text, length, style)
        elif len(chunks) <= 20:
            return self._summarize_small_doc(chunks, length, style)
        elif len(chunks) <= 300:
            return self._summarize_large_doc(chunks, length, style)
        else:
            raise ValueError("The document is too large to process.")
        
    def _summarize_single_chunk(self, text, length, style):
        """
        Summarize a single text chunk.
        """
        if self.length_options[length] > len(text):
            raise ValueError("The text is too short for the chosen summary length.")
        prompt = self.base_map_prompt.format(text=text, style=style, length=length)
        return self.llm(prompt)

    def _summarize_small_doc(self, chunks, length, style):
        """
        Summarize a small document using map-reduce summarization.
        """
        if all(len(chunk.page_content) < self.length_options[length] for chunk in chunks):
            raise ValueError("All chunks are too short.")
        return self._run_map_reduce(chunks, length, style)

    def _summarize_large_doc(self, chunks, length, style):
        """
        Summarize a large document by selecting key paragraphs and applying map-reduce.
        """
        selected_docs = self.paragraphs_selector(chunks)
        return self._run_map_reduce(selected_docs, length, style)

    def _run_map_reduce(self, docs, length, style):
        """
        Run a map-reduce summarization chain on the provided documents.
        """
        map_prompt = self.base_map_prompt.partial(
            style=style,
            length=self.length_options['short']
        )
        combine_prompt = self.base_combine_prompt.partial(
            style=style,
            length=length
        )
        chain = load_summarize_chain(
            self.llm,
            chain_type='map_reduce',
            map_prompt=map_prompt,
            combine_prompt=combine_prompt
        )
        return chain.run(docs)