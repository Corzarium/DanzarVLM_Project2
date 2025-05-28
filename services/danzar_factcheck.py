#!/usr/bin/env python3
# services/danzar_factcheck.py

import logging
import time
import requests
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("DanzarVLM.FactCheck")

class FactCheckService:
    """Service for fact-checking LLM responses against RAG store."""
    
    def __init__(self, rag_service, model_client, default_collection: str = "multimodal_rag_default"):
        self.rag_service = rag_service
        self.model_client = model_client
        self.default_collection = default_collection
        self.logger = logger  # Use the module-level logger
        
        # Setup fallback logging
        self.fallback_log_path = Path("logs/fallbacks.log")
        self.fallback_log_path.parent.mkdir(exist_ok=True)
        
    def _log_fallback(self, query: str, collection: str, reason: str):
        """Log a fallback event with timestamp and context."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Query: '{query}' | Collection: {collection} | Reason: {reason}\n"
        
        try:
            with open(self.fallback_log_path, "a") as f:
                f.write(log_entry)
        except Exception as e:
            self.logger.error(f"Failed to log fallback: {e}")

    def _search_web(self, query: str) -> Optional[str]:
        """
        Search the web using DuckDuckGo API for relevant information.
        
        Args:
            query: The search query
            
        Returns:
            Optional[str]: Relevant text from search results or None if no results
        """
        try:
            # Use DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            if data.get("Abstract"):
                return data["Abstract"]
            elif data.get("RelatedTopics"):
                # Get the first related topic's text
                for topic in data["RelatedTopics"]:
                    if "Text" in topic:
                        return topic["Text"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}", exc_info=True)
            return None

    def fact_checked_generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """
        Generate a response with fact checking against RAG store.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # First try to get relevant context from RAG
            try:
                rag_results = self.rag_service.query_rag(
                    query_text=prompt,
                    top_k=5,
                    collection_name="multimodal_rag_default"
                )
                
                if rag_results and len(rag_results) > 0:
                    # Build grounded prompt with context
                    context = "\n".join(rag_results)  # rag_results is already a list of strings
                    grounded_prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {prompt}

Answer:"""
                    
                    # Generate response with zero temperature for deterministic output
                    response = self.model_client.generate(
                        prompt=grounded_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response
                    
            except Exception as e:
                self.logger.error(f"RAG query failed: {e}", exc_info=True)
            
            # If RAG fails or returns no results, try web search
            self.logger.info("Attempting web search for additional context...")
            web_result = self._search_web(prompt)
            
            if web_result:
                # Build prompt with web search results
                web_prompt = f"""Based on the following information found online, please answer the question:

Information:
{web_result}

Question: {prompt}

Answer:"""
                
                try:
                    response = self.model_client.generate(
                        prompt=web_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response
                except Exception as e:
                    self.logger.error(f"LLM generation failed: {e}", exc_info=True)
                    return f"While I don't have that in my knowledge base, I found this online: {web_result}"
            
            # If both RAG and web search fail, return a clear message
            return "I apologize, but I'm unable to find any relevant information about that topic. Would you like to try a different question?"
            
        except Exception as e:
            self.logger.error(f"Fact check generation failed: {e}", exc_info=True)
            return "I apologize, but I encountered an error while trying to find information. Please try again." 