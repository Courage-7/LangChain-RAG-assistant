from typing import List, Dict, Any, Optional
import re
import os
from langchain_community.llms import OpenAI
import yaml

class QueryOptimizer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Query Optimizer with robust configuration handling
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Load configuration with error handling
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file not found at {config_path}. Using default settings.")
            self.config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            self.config = {}
        
        # Retrieve LLM configuration with sensible defaults
        llm_config = self.config.get('llm', {})
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM with error handling
        try:
            self.llm = OpenAI(
                temperature=llm_config.get('temperature', 0.2),
                model_name=llm_config.get('model_name', 'gpt-3.5-turbo'),
                max_tokens=llm_config.get('max_tokens', 100),
                openai_api_key=api_key
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with additional context and keywords
        
        Args:
            query (str): Original query
            
        Returns:
            Expanded query
        """
        prompt = f"""
        Please expand the following search query to improve document retrieval.
        Add relevant keywords and context while preserving the original intent.
        Original query: {query}
        Expanded query:
        """
        
        try:
            expanded_query = self.llm(prompt).strip()
            return expanded_query
        except Exception as e:
            print(f"Error expanding query: {e}")
            return query
    
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of the query for better retrieval
        
        Args:
            query (str): Original query
            num_variations (int): Number of variations to generate
            
        Returns:
            List of query variations
        """
        prompt = f"""
        Generate {num_variations} different variations of the following query.
        Each variation should preserve the original intent but use different wording.
        Separate each variation with a newline.
        
        Original query: {query}
        
        Variations:
        """
        
        try:
            result = self.llm(prompt).strip()
            variations = [v.strip() for v in result.split('\n') if v.strip()]
            return variations[:num_variations]
        except Exception as e:
            print(f"Error generating query variations: {e}")
            return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query (str): Query string
            
        Returns:
            List of keywords
        """
        # First try LLM-based keyword extraction
        prompt = f"""
        Extract the 5 most important keywords from the following query.
        Return only the keywords separated by commas, with no additional text.
        
        Query: {query}
        
        Keywords:
        """
        
        try:
            result = self.llm(prompt).strip()
            keywords = [k.strip() for k in result.split(',') if k.strip()]
            
            # If LLM fails or returns insufficient keywords, fallback to regex
            if not keywords:
                raise ValueError("No keywords extracted")
            
            return keywords
        except Exception:
            # Fallback to simple word extraction
            words = re.findall(r'\b\w+\b', query.lower())
            return [w for w in words if len(w) > 3]
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive query optimization
        
        Args:
            query (str): Original query
            
        Returns:
            Dictionary with optimized query and metadata
        """
        expanded_query = self.expand_query(query)
        variations = self.generate_query_variations(query)
        keywords = self.extract_keywords(query)
        
        return {
            "original_query": query,
            "optimized_query": expanded_query,
            "variations": variations,
            "keywords": keywords
        }