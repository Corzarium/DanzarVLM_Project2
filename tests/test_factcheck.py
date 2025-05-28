#!/usr/bin/env python3
# tests/test_factcheck.py

import unittest
from unittest.mock import Mock, patch
from services.danzar_factcheck import FactCheckService

class TestFactCheckService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.rag_service = Mock()
        self.model_client = Mock()
        self.fact_check = FactCheckService(
            rag_service=self.rag_service,
            model_client=self.model_client
        )
        
    def test_known_supported_query(self):
        """Test that a known-supported query returns a grounded answer."""
        # Mock RAG service to return relevant docs
        self.rag_service.query.return_value = [
            "EverQuest is a fantasy MMORPG released in 1999.",
            "The game features multiple playable races and classes."
        ]
        
        # Mock model to return a grounded response
        self.model_client.generate.return_value = "EverQuest is a fantasy MMORPG from 1999 with various races and classes."
        
        # Test the query
        response = self.fact_check.fact_checked_generate("What is EverQuest?")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was called with grounded prompt
        self.model_client.generate.assert_called_once()
        prompt = self.model_client.generate.call_args[1]['prompt']
        self.assertIn("EverQuest is a fantasy MMORPG", prompt)
        self.assertEqual(self.model_client.generate.call_args[1]['temperature'], 0.0)
        
        # Verify response
        self.assertEqual(response, "EverQuest is a fantasy MMORPG from 1999 with various races and classes.")
        
    def test_unsupported_query(self):
        """Test that an unsupported query triggers the fallback."""
        # Mock RAG service to return no docs
        self.rag_service.query.return_value = []
        
        # Test the query
        response = self.fact_check.fact_checked_generate("What is the meaning of life?")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was not called
        self.model_client.generate.assert_not_called()
        
        # Verify fallback response
        self.assertEqual(
            response,
            "I'm not sure; I couldn't find any info on that—would you like me to search more?"
        )
        
    def test_empty_query(self):
        """Test that an empty query returns the fallback."""
        response = self.fact_check.fact_checked_generate("")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was not called
        self.model_client.generate.assert_not_called()
        
        # Verify fallback response
        self.assertEqual(
            response,
            "I'm not sure; I couldn't find any info on that—would you like me to search more?"
        )
        
    def test_rag_error(self):
        """Test that RAG errors trigger the fallback."""
        # Mock RAG service to raise an exception
        self.rag_service.query.side_effect = Exception("RAG error")
        
        # Test the query
        response = self.fact_check.fact_checked_generate("What is EverQuest?")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was not called
        self.model_client.generate.assert_not_called()
        
        # Verify fallback response
        self.assertEqual(
            response,
            "I'm not sure; I couldn't find any info on that—would you like me to search more?"
        )
        
    def test_llm_error(self):
        """Test that LLM errors trigger the fallback."""
        # Mock RAG service to return docs
        self.rag_service.query.return_value = ["Some context"]
        
        # Mock model to raise an exception
        self.model_client.generate.side_effect = Exception("LLM error")
        
        # Test the query
        response = self.fact_check.fact_checked_generate("What is EverQuest?")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was called
        self.model_client.generate.assert_called_once()
        
        # Verify fallback response
        self.assertEqual(
            response,
            "I'm not sure; I couldn't find any info on that—would you like me to search more?"
        )
        
    def test_empty_llm_response(self):
        """Test that empty LLM responses trigger the fallback."""
        # Mock RAG service to return docs
        self.rag_service.query.return_value = ["Some context"]
        
        # Mock model to return empty response
        self.model_client.generate.return_value = ""
        
        # Test the query
        response = self.fact_check.fact_checked_generate("What is EverQuest?")
        
        # Verify RAG was queried
        self.rag_service.query.assert_called_once()
        
        # Verify model was called
        self.model_client.generate.assert_called_once()
        
        # Verify fallback response
        self.assertEqual(
            response,
            "I'm not sure; I couldn't find any info on that—would you like me to search more?"
        )

if __name__ == '__main__':
    unittest.main() 