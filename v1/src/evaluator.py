from typing import List, Dict, Any
import numpy as np
from langchain.docstore.document import Document
from src.retriever import RAGRetriever
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RAGEvaluator:
    def __init__(self, retriever: RAGRetriever, config_path: str = "config/config.yaml"):
        """
        Initialize RAG Evaluator with enhanced configuration and error handling
        
        Args:
            retriever (RAGRetriever): RAG retriever instance
            config_path (str): Path to configuration file
        """
        self.retriever = retriever
        
        # Load configuration
        try:
            import yaml
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file not found at {config_path}. Using default settings.")
            self.config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            self.config = {}
        
        # Initialize evaluation tools with error handling
        try:
            self.rouge = Rouge()
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error initializing evaluation tools: {e}")
            raise
    
    def calculate_precision(
        self, 
        retrieved_docs: List[Document], 
        ground_truth: List[Document]
    ) -> float:
        """
        Calculate precision
        
        Args:
            retrieved_docs (List[Document]): Retrieved documents
            ground_truth (List[Document]): Relevant documents
        
        Returns:
            Precision score
        """
        if not retrieved_docs or not ground_truth:
            return 0.0
        
        # Convert to sets of document contents for comparison
        retrieved_contents = set(doc.page_content for doc in retrieved_docs)
        ground_truth_contents = set(doc.page_content for doc in ground_truth)
        
        # Calculate precision
        correct_retrievals = len(retrieved_contents.intersection(ground_truth_contents))
        return correct_retrievals / len(retrieved_docs) if retrieved_docs else 0.0
    
    def calculate_recall(
        self, 
        retrieved_docs: List[Document], 
        ground_truth: List[Document]
    ) -> float:
        """
        Calculate recall
        
        Args:
            retrieved_docs (List[Document]): Retrieved documents
            ground_truth (List[Document]): Relevant documents
        
        Returns:
            Recall score
        """
        if not retrieved_docs or not ground_truth:
            return 0.0
        
        # Convert to sets of document contents for comparison
        retrieved_contents = set(doc.page_content for doc in retrieved_docs)
        ground_truth_contents = set(doc.page_content for doc in ground_truth)
        
        # Calculate recall
        correct_retrievals = len(retrieved_contents.intersection(ground_truth_contents))
        return correct_retrievals / len(ground_truth) if ground_truth else 0.0
    
    def calculate_f1_score(
        self, 
        retrieved_docs: List[Document], 
        ground_truth: List[Document]
    ) -> float:
        """
        Calculate F1 score
        
        Args:
            retrieved_docs (List[Document]): Retrieved documents
            ground_truth (List[Document]): Relevant documents
        
        Returns:
            F1 score
        """
        precision = self.calculate_precision(retrieved_docs, ground_truth)
        recall = self.calculate_recall(retrieved_docs, ground_truth)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(
        self, 
        retrieved_docs: List[Document], 
        ground_truth: List[Document]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            retrieved_docs (List[Document]): Retrieved documents
            ground_truth (List[Document]): Relevant documents
        
        Returns:
            Mean Reciprocal Rank
        """
        ground_truth_contents = set(doc.page_content for doc in ground_truth)
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc.page_content in ground_truth_contents:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_rouge_score(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE score for answer evaluation
        
        Args:
            generated_answer (str): Generated answer
            reference_answer (str): Reference/ground truth answer
        
        Returns:
            Dictionary of ROUGE scores
        """
        if not generated_answer or not reference_answer:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        
        try:
            scores = self.rouge.get_scores(generated_answer, reference_answer)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    def calculate_semantic_similarity(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings
        
        Args:
            generated_answer (str): Generated answer
            reference_answer (str): Reference/ground truth answer
        
        Returns:
            Semantic similarity score
        """
        if not generated_answer or not reference_answer:
            return 0.0
        
        try:
            gen_embedding = self.sentence_model.encode([generated_answer])[0]
            ref_embedding = self.sentence_model.encode([reference_answer])[0]
            
            # Reshape for cosine_similarity function
            gen_embedding = gen_embedding.reshape(1, -1)
            ref_embedding = ref_embedding.reshape(1, -1)
            
            return float(cosine_similarity(gen_embedding, ref_embedding)[0][0])
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def evaluate_retrieval(
        self, 
        query: str, 
        ground_truth: List[Document]
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation
        
        Args:
            query (str): Search query
            ground_truth (List[Document]): Relevant documents
        
        Returns:
            Evaluation metrics dictionary
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve_documents(query)
        
        # Compute metrics
        metrics = {
            "precision": self.calculate_precision(retrieved_docs, ground_truth),
            "recall": self.calculate_recall(retrieved_docs, ground_truth),
            "f1_score": self.calculate_f1_score(retrieved_docs, ground_truth),
            "mrr": self.mean_reciprocal_rank(retrieved_docs, ground_truth)
        }
        
        return metrics
    
    def evaluate_answer(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate generated answer against reference answer
        
        Args:
            generated_answer (str): Generated answer
            reference_answer (str): Reference/ground truth answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        rouge_scores = self.calculate_rouge_score(generated_answer, reference_answer)
        semantic_similarity = self.calculate_semantic_similarity(generated_answer, reference_answer)
        
        metrics = {
            "rouge_scores": rouge_scores,
            "semantic_similarity": semantic_similarity
        }
        
        return metrics