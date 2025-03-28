import os
import uuid
import yaml
from typing import List, Optional

import faiss
from langchain.docstore.document import Document
# Remove this line
#from langchain_huggingface import HuggingFaceInferenceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
# Remove this line
#from langchain_community.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class DocumentEmbedder:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Document Embedder with enhanced error handling and configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration with error handling
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        
        # Validate required configuration keys
        required_keys = ['chunking', 'vector_store', 'huggingface']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key: {key}")
        
        # Initialize embedding model with error handling
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError("Hugging Face API key not found. Set HUGGINGFACEHUB_API_TOKEN environment variable.")
        
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=api_key
        )
        
        # Initialize text splitter using configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap']
        )
        
        # Ensure vector store directory exists
        self.vector_store_path = self.config['vector_store']['path']
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Optional[FAISS]:
        """
        Initialize or load FAISS vector store
        
        Returns:
            Initialized FAISS vector store or None
        """
        try:
            if os.path.exists(self.vector_store_path):
                return FAISS.load_local(
                    self.vector_store_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document with expanded file type support and error handling
        
        Args:
            file_path (str): Path to document file
        
        Returns:
            List of Document objects
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        if file_extension not in loaders:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        try:
            loader = loaders[file_extension](file_path)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = file_path
            
            return documents
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting into chunks
        
        Args:
            documents (List[Document]): List of documents
        
        Returns:
            List of processed document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store with improved handling
        
        Args:
            documents (List[Document]): Documents to add
        
        Returns:
            List of document IDs
        """
        chunks = self.process_documents(documents)
        
        try:
            if self.vector_store:
                doc_ids = self.vector_store.add_documents(chunks)
            else:
                self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
                doc_ids = [str(uuid.uuid4()) for _ in chunks]
            
            self.vector_store.save_local(self.vector_store_path)
            return doc_ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            return []
    
    def add_document_from_file(self, file_path: str) -> List[str]:
        """
        Add document from file to vector store
        
        Args:
            file_path (str): Path to document file
        
        Returns:
            List of document IDs
        """
        documents = self.load_document(file_path)
        return self.add_documents(documents)
    
    def get_document_count(self) -> int:
        """
        Get total number of document chunks in vector store
        
        Returns:
            Number of document chunks
        """
        return self.vector_store.index.ntotal if self.vector_store else 0

    def search_documents(self, query: str, top_k: Optional[int] = None):
        """
        Search documents in vector store
        
        Args:
            query (str): Search query
            top_k (Optional[int]): Number of top results to return
        
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            return []
        
        top_k = top_k or self.config['retrieval']['top_k']
        return self.vector_store.similarity_search(query, k=top_k)