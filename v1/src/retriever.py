from typing import List, Dict, Any, Optional
import os
from langchain.docstore.document import Document

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import yaml

class RAGRetriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize RAG Retriever with robust error handling
        
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
        required_keys = ['embedding', 'vector_store', 'llm', 'retrieval']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key: {key}")
        
        # Initialize embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config['embedding']['model_name']
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
        
        # Initialize vector store
        self.vector_store_path = self.config['vector_store']['path']
        self.vector_store = self._initialize_vector_store()
        
        # Initialize LLM with API key handling
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            self.llm = OpenAI(
                temperature=self.config['llm']['temperature'],
                model_name=self.config['llm']['model_name'],
                max_tokens=self.config['llm']['max_tokens'],
                openai_api_key=api_key
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}")
        
        # Initialize QA chain
        self.qa_chain = self._initialize_qa_chain()
    
    def _initialize_vector_store(self):
        """
        Initialize vector store with error handling
        
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
    
    def _initialize_qa_chain(self):
        """
        Initialize QA chain if vector store exists
        
        Returns:
            Initialized RetrievalQA chain or None
        """
        if not self.vector_store:
            print("No vector store available. QA chain cannot be initialized.")
            return None
        
        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": self.config['retrieval']['top_k']}
                )
            )
        except Exception as e:
            print(f"Error initializing QA chain: {e}")
            return None
    
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): Query string
            top_k (int, optional): Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.vector_store:
            print("No vector store available for document retrieval.")
            return []
        
        try:
            top_k = top_k or self.config['retrieval']['top_k']
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using the RAG pipeline
        
        Args:
            query (str): Query string
            
        Returns:
            Dictionary containing answer and retrieved documents
        """
        if not self.qa_chain:
            return {
                "answer": "No documents have been indexed or QA chain could not be initialized.", 
                "documents": []
            }
        
        try:
            # Get answer
            result = self.qa_chain({"query": query})
            
            # Get supporting documents
            documents = self.retrieve_documents(query)
            
            return {
                "answer": result["result"],
                "documents": documents
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {e}",
                "documents": []
            }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if not self.vector_store:
            print("No vector store available.")
            return None
        
        try:
            results = self.vector_store.get([doc_id])
            if results and len(results['documents']) > 0:
                return Document(
                    page_content=results['documents'][0],
                    metadata=results['metadatas'][0] if results['metadatas'] else {}
                )
            return None
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None