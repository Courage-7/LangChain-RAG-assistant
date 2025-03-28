import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import yaml

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_optimization import QueryOptimizer

class TestQueryOptimizer(unittest.TestCase):
    def setUp(self):
        # Create a mock config file
        self.mock_config = {
            'llm': {
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.3,
                'max_tokens': 500
            }
        }
        
        # Mock the yaml.safe_load to return our mock config
        with patch('yaml.safe_load', return_value=self.mock_config):
            # Mock the OpenAI class
            with patch('src.query_optimization.OpenAI'):
                self.optimizer = QueryOptimizer()
    
    def test_expand_query(self):
        # Mock the LLM response
        self.optimizer.llm = MagicMock()
        self.optimizer.llm.return_value = "expanded query with additional context"
        
        # Test the expand_query method
        original_query = "test query"
        expanded_query = self.optimizer.expand_query(original_query)
        
        # Assert that the LLM was called with the correct prompt
        self.optimizer.llm.assert_called_once()
        call_args = self.optimizer.llm.call_args[0][0]
        self.assertIn(original_query, call_args)
        
        # Assert that the expanded query is returned
        self.assertEqual(expanded_query, "expanded query with additional context")
    
    def test_expand_query_exception(self):
        # Mock the LLM to raise an exception
        self.optimizer.llm = MagicMock(side_effect=Exception("Test exception"))
        
        # Test the expand_query method with an exception
        original_query = "test query"
        expanded_query = self.optimizer.expand_query(original_query)
        
        # Assert that the original query is returned on exception
        self.assertEqual(expanded_query, original_query)
    
    def test_generate_query_variations(self):
        # Mock the LLM response
        self.optimizer.llm = MagicMock()
        self.optimizer.llm.return_value = "variation 1\nvariation 2\nvariation 3"
        
        # Test the generate_query_variations method
        original_query = "test query"
        variations = self.optimizer.generate_query_variations(original_query)
        
        # Assert that the LLM was called with the correct prompt
        self.optimizer.llm.assert_called_once()
        call_args = self.optimizer.llm.call_args[0][0]
        self.assertIn(original_query, call_args)
        
        # Assert that the variations are returned
        self.assertEqual(variations, ["variation 1", "variation 2", "variation 3"])
    
    def test_generate_query_variations_exception(self):
        # Mock the LLM to raise an exception
        self.optimizer.llm = MagicMock(side_effect=Exception("Test exception"))
        
        # Test the generate_query_variations method with an exception
        original_query = "test query"
        variations = self.optimizer.generate_query_variations(original_query)
        
        # Assert that a list with the original query is returned on exception
        self.assertEqual(variations, [original_query])
    
    def test_extract_keywords(self):
        # Mock the LLM response
        self.optimizer.llm = MagicMock()
        self.optimizer.llm.return_value = "keyword1, keyword2, keyword3, keyword4, keyword5"
        
        # Test the extract_keywords method
        original_query = "test query with multiple keywords"
        keywords = self.optimizer.extract_keywords(original_query)
        
        # Assert that the LLM was called with the correct prompt
        self.optimizer.llm.assert_called_once()
        call_args = self.optimizer.llm.call_args[0][0]
        self.assertIn(original_query, call_args)
        
        # Assert that the keywords are returned
        self.assertEqual(keywords, ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"])
    
    def test_extract_keywords_exception(self):
        # Mock the LLM to raise an exception
        self.optimizer.llm = MagicMock(side_effect=Exception("Test exception"))
        
        # Test the extract_keywords method with an exception
        original_query = "test query with multiple keywords"
        keywords = self.optimizer.extract_keywords(original_query)
        
        # Assert that fallback keyword extraction is used
        self.assertEqual(keywords, ["test", "query", "with", "multiple", "keywords"])
    
    def test_optimize_query(self):
        # Mock the individual optimization methods
        self.optimizer.expand_query = MagicMock(return_value="expanded query")
        self.optimizer.generate_query_variations = MagicMock(return_value=["var1", "var2", "var3"])
        self.optimizer.extract_keywords = MagicMock(return_value=["key1", "key2", "key3"])
        
        # Test the optimize_query method
        original_query = "test query"
        result = self.optimizer.optimize_query(original_query)
        
        # Assert that the individual methods were called
        self.optimizer.expand_query.assert_called_once_with(original_query)
        self.optimizer.generate_query_variations.assert_called_once_with(original_query)
        self.optimizer.extract_keywords.assert_called_once_with(original_query)
        
        # Assert that the result contains the expected data
        self.assertEqual(result["original_query"], original_query)
        self.assertEqual(result["optimized_query"], "expanded query")
        self.assertEqual(result["variations"], ["var1", "var2", "var3"])
        self.assertEqual(result["keywords"], ["key1", "key2", "key3"])

if __name__ == "__main__":
    unittest.main()