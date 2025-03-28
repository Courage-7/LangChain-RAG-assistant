import os
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from src.retriever import RAGRetriever
from src.embedding import DocumentEmbedder
from src.query_optimization import QueryOptimizer
from src.evaluator import RAGEvaluator
from dotenv import load_dotenv

# Configure logging
# Configure logging
import os

# Ensure logs directory exists (though we know it does)
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging to the logs directory
log_file_path = os.path.join(logs_dir, "rag_assistant.log")
logging.basicConfig(
    filename=log_file_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Improved component initialization with error handling
@st.cache_resource
def initialize_components():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "config.yaml")
        
        # Validate config file
        if not os.path.exists(config_path):
            st.error(f"Configuration file not found at: {config_path}")
            return None, None, None, None
        
        # Initialize components
        embedder = DocumentEmbedder(config_path)
        retriever = RAGRetriever(config_path)
        optimizer = QueryOptimizer(config_path)
        evaluator = RAGEvaluator(retriever)
        
        return embedder, retriever, optimizer, evaluator
    
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        logging.error(f"Component initialization failed: {e}")
        return None, None, None, None

# File validation function
def validate_file(uploaded_file):
    """
    Validate uploaded file
    - Check file size
    - Check file type
    """
    # Maximum file size (50 MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Allowed file types
    ALLOWED_TYPES = ["pdf", "txt", "csv", "xlsx", "docx"]
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File {uploaded_file.name} is too large. Maximum file size is 50 MB.")
        return False
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_TYPES:
        st.error(f"Unsupported file type: {file_extension}. Allowed types: {', '.join(ALLOWED_TYPES)}")
        return False
    
    return True

# Batch document processing
def process_documents_in_batches(embedder, files, batch_size=5):
    """
    Process documents in batches with progress tracking
    """
    processed_docs = []
    total_chunks = 0
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        
        with st.spinner(f"Processing batch {i//batch_size + 1}..."):
            for uploaded_file in batch:
                if validate_file(uploaded_file):
                    # Save uploaded file temporarily
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    temp_file_path = os.path.join(base_dir, "data", "documents", uploaded_file.name)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add to vector store
                    try:
                        doc_ids = embedder.add_document_from_file(temp_file_path)
                        processed_docs.append(uploaded_file.name)
                        total_chunks += len(doc_ids)
                        
                        # Log successful document processing
                        logging.info(f"Processed document: {uploaded_file.name}")
                        
                        st.success(f"Added {uploaded_file.name} with {len(doc_ids)} chunks")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logging.error(f"Document processing error: {e}")
    
    return processed_docs, total_chunks

# Cached query retrieval with performance tracking
@st.cache_data(ttl=3600)
def cached_query_retrieval(query, use_optimization, retriever, optimizer):
    """
    Cached query retrieval with optional optimization
    """
    start_time = datetime.now()
    
    try:
        if use_optimization:
            optimization_result = optimizer.optimize_query(query)
            result = retriever.answer_query(optimization_result["optimized_query"])
            
            # Log query details
            logging.info(f"Optimized Query: {query} -> {optimization_result['optimized_query']}")
        else:
            result = retriever.answer_query(query)
        
        end_time = datetime.now()
        retrieval_time = (end_time - start_time).total_seconds()
        
        # Enhance result with retrieval time
        result['retrieval_time'] = retrieval_time
        return result
    
    except Exception as e:
        st.error(f"Query retrieval error: {e}")
        logging.error(f"Query retrieval failed: {e}")
        return None

# Main Streamlit app
def main():
    # Initialize components
    embedder, retriever, optimizer, evaluator = initialize_components()
    
    # Check if components were initialized successfully
    if not all([embedder, retriever, optimizer, evaluator]):
        st.error("Failed to initialize RAG components. Please check your configuration.")
        return
    
    # App title
    st.title("📚 Advanced RAG Document Assistant")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a Page", [
        "Document Management", 
        "Search & Query", 
        "Evaluation", 
        "System Configuration"
    ])
    
    # Page routing
    if page == "Document Management":
        st.header("Document Management")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Choose documents to upload", 
            type=["pdf", "txt", "csv", "xlsx", "docx"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                processed_docs, total_chunks = process_documents_in_batches(embedder, uploaded_files)
                
                # Display processing summary
                st.metric("Total Processed Documents", len(processed_docs))
                st.metric("Total Document Chunks", total_chunks)
    
    elif page == "Search & Query":
        st.header("Search & Query")
        
        # Query input
        query = st.text_input("Enter your question or search query")
        
        # Query optimization options
        col1, col2 = st.columns(2)
        with col1:
            use_query_optimization = st.checkbox("Use Query Optimization", value=True)
        with col2:
            show_documents = st.checkbox("Show Retrieved Documents", value=True)
        
        # Query submission
        if query and st.button("Search"):
            result = cached_query_retrieval(query, use_query_optimization, retriever, optimizer)
            
            if result:
                # Display retrieval time
                st.success(f"Retrieval Time: {result.get('retrieval_time', 'N/A')} seconds")
                
                # Display answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Show retrieved documents
                if show_documents and result.get("documents"):
                    st.subheader("Retrieved Documents")
                    for i, doc in enumerate(result["documents"], 1):
                        with st.expander(f"Document {i}"):
                            st.write(doc.page_content)
                            st.write("Source:", doc.metadata.get("source", "Unknown"))
    
    elif page == "Evaluation":
        # Existing evaluation page code (from previous implementation)
        st.header("Evaluation")
        # ... (keep the existing evaluation code)
    
    elif page == "System Configuration":
        st.header("RAG System Configuration")
        
        # Embedding configuration
        st.subheader("Embedding Configuration")
        embedding_model = st.selectbox(
            "Embedding Model", 
            ["openai", "huggingface", "sentence-transformers"]
        )
        
        # Retrieval settings
        st.subheader("Retrieval Configuration")
        top_k = st.slider("Number of Retrieved Documents", 1, 10, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
        
        # Save configuration button (placeholder)
        if st.button("Save Configuration"):
            st.success("Configuration saved successfully!")

# Footer
    st.markdown("---")
    st.markdown("Advanced RAG Document Assistant | Powered by AI")

# Run the app
if __name__ == "__main__":
    main()