#!/usr/bin/env python3
# services/danzar_factcheck.py

import logging
import time
import requests
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger("DanzarVLM.FactCheck")

class FactCheckService:
    """Service for fact-checking LLM responses against RAG store."""
    
    def __init__(self, rag_service, model_client, app_context=None, default_collection: str = "multimodal_rag_default"):
        self.rag_service = rag_service
        self.model_client = model_client
        self.ctx = app_context  # Store app context for settings access
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

    def _search_web(self, query: str, fact_check: bool = True) -> Optional[str]:
        """
        Search the web using SearxNG with rate limiting to prevent errors.
        
        Args:
            query: The search query
            fact_check: Whether to perform fact-checking across sources
            
        Returns:
            Optional[str]: Relevant text from search results or None if no results
        """
        try:
            import requests
            import time
            
            self.logger.info(f"[FactCheck] Searching web for: '{query}' (fact_check={fact_check})")
            
            # Get SearxNG configuration from app context
            searxng_config = {}
            if self.ctx and hasattr(self.ctx, 'global_settings'):
                searxng_config = self.ctx.global_settings.get("SEARXNG_SETTINGS", {})
            
            # Default SearxNG configuration
            endpoint = searxng_config.get("endpoint", "http://localhost:8080")
            timeout = searxng_config.get("timeout", 30)
            max_results = searxng_config.get("max_results", 6 if fact_check else 3)
            categories = searxng_config.get("categories", "general")
            engines = searxng_config.get("engines", "google,bing,duckduckgo")
            format_type = searxng_config.get("format", "json")
            safesearch = searxng_config.get("safesearch", 1)
            
            # Check if SearxNG is enabled
            if not searxng_config.get("enabled", True):
                self.logger.warning("[FactCheck] SearxNG is disabled in configuration")
                return None
            
            # Add rate limiting to prevent overloading SearxNG
            max_retries = 2
            retry_delay = 2.0  # seconds (shorter than DuckDuckGo since SearxNG is self-hosted)
            
            for attempt in range(max_retries):
                try:
                    # Prepare SearxNG search parameters
                    search_url = f"{endpoint}/search"
                    params = {
                        'q': query,
                        'format': format_type,
                        'categories': categories,
                        'engines': engines,
                        'safesearch': safesearch,
                        'pageno': 1
                    }
                    
                    # Make request to SearxNG
                    response = requests.get(
                        search_url,
                        params=params,
                        timeout=timeout,
                        headers={
                            'User-Agent': 'DanzarAI-FactCheck/1.0',
                            'Accept': 'application/json'
                        }
                    )
                    
                    response.raise_for_status()
                    search_data = response.json()
                    
                    # Extract results from SearxNG response
                    results = search_data.get('results', [])
                    
                    if results:
                        # Limit results based on configuration
                        results = results[:max_results]
                        
                        if fact_check and len(results) >= 3:
                            # Perform fact-checking across sources
                            # Convert SearxNG results to DuckDuckGo-like format for compatibility
                            formatted_results = []
                            for result in results:
                                formatted_result = {
                                    'title': result.get('title', 'No title'),
                                    'body': result.get('content', result.get('snippet', 'No description')),
                                    'href': result.get('url', 'Unknown source')
                                }
                                formatted_results.append(formatted_result)
                            
                            verified_info = self._fact_check_sources(query, formatted_results)
                            if verified_info:
                                # Store verified information in RAG
                                self._store_verified_info_in_rag(query, verified_info, formatted_results)
                                return verified_info
                        
                        # Fallback to regular search summary
                        search_summary = ""
                        for i, result in enumerate(results[:3], 1):
                            title = result.get('title', 'No title')
                            content = result.get('content', result.get('snippet', 'No description'))
                            source = result.get('url', 'Unknown source')
                            # Limit content length to avoid token overflow
                            content = content[:300] + "..." if len(content) > 300 else content
                            search_summary += f"Result {i}: {title}\nSource: {source}\n{content}\n\n"
                        
                        self.logger.info(f"[FactCheck] SearxNG search returned {len(results)} results")
                        return search_summary.strip()
                    
                    self.logger.warning(f"[FactCheck] No SearxNG search results for: '{query}'")
                    return None
                
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"[FactCheck] SearxNG request failed, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        raise e
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"[FactCheck] SearxNG error, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        raise e
            
            # If we get here, all retries failed
            self.logger.error(f"[FactCheck] SearxNG search failed after {max_retries} attempts")
            return None
            
        except Exception as e:
            self.logger.error(f"SearxNG web search failed: {e}")
            return None

    def _fact_check_sources(self, query: str, search_results: List[Dict]) -> Optional[str]:
        """
        Fact-check information across multiple sources - requires at least 2 sources.
        
        Args:
            query: The original search query
            search_results: List of search results from SearxNG
            
        Returns:
            Optional[str]: Verified information if consistent across sources
        """
        try:
            # Require at least 2 sources minimum for any verification
            min_sources = 2
            if self.ctx and hasattr(self.ctx, 'global_settings'):
                min_sources = max(2, self.ctx.global_settings.get("FACT_CHECK_SETTINGS", {}).get("min_sources_for_verification", 2))
            
            if len(search_results) < min_sources:
                self.logger.warning(f"[FactCheck] Not enough sources for fact-checking ({len(search_results)} < {min_sources})")
                return None
            
            self.logger.info(f"[FactCheck] Cross-referencing across {len(search_results)} sources (minimum {min_sources} required)")
            
            # Extract key information from each source
            source_summaries = []
            source_domains = []
            
            for i, result in enumerate(search_results[:6]):  # Use up to 6 sources for better verification
                title = result.get('title', '')
                body = result.get('body', '')
                source_url = result.get('href', '')
                
                # Extract domain for diversity check
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(source_url).netloc
                    source_domains.append(domain)
                except:
                    domain = "unknown"
                    source_domains.append(domain)
                
                # Create a summary for this source
                summary = f"Source {i+1} ({domain}):\nTitle: {title}\nContent: {body[:350]}"
                source_summaries.append(summary)
            
            # Check for source diversity (avoid echo chambers)
            unique_domains = len(set(source_domains))
            self.logger.info(f"[FactCheck] Using {unique_domains} unique domains from {len(source_summaries)} sources")
            
            # Use LLM to analyze consistency across sources
            fact_check_prompt = f"""
            Cross-reference the following {len(source_summaries)} sources about "{query}" to verify factual accuracy.

            Sources from {unique_domains} different domains:
            {chr(10).join(source_summaries)}

            Task: Analyze these sources and determine if they contain consistent, reliable information.

            Requirements:
            1. Identify facts that appear in at least {min(2, len(source_summaries))} sources
            2. Note any significant conflicts between sources
            3. Focus on verifiable facts (dates, names, definitions, established information)
            4. Ignore opinions, speculation, or unverifiable claims

            Response format:
            - Start with "VERIFIED:" if facts are consistent across multiple sources
            - Start with "CONFLICTED:" if sources significantly disagree
            - Start with "INSUFFICIENT:" if not enough verifiable information

            Provide only the verified facts, not opinions or speculation.
            """
            
            messages = [
                {"role": "system", "content": "You are a fact-checking assistant specializing in cross-referencing multiple sources for accuracy verification."},
                {"role": "user", "content": fact_check_prompt}
            ]
            
            # Use the default model from the profile instead of hardcoded "mistral"
            model_to_use = 'qwen2.5:3b'  # Default fallback
            if self.ctx and hasattr(self.ctx, 'active_profile') and self.ctx.active_profile:
                model_to_use = getattr(self.ctx.active_profile, 'conversational_llm_model', 'qwen2.5:3b')
            
            fact_check_response = self.model_client.generate(
                messages=messages,
                temperature=0.1,  # Very low temperature for consistency
                max_tokens=600,   # Increased for better verification responses
                model=model_to_use
            )
            
            if fact_check_response:
                if fact_check_response.startswith("VERIFIED:"):
                    verified_facts = fact_check_response.replace("VERIFIED:", "").strip()
                    self.logger.info(f"[FactCheck] Information verified across {len(source_summaries)} sources ({unique_domains} domains)")
                    return f"[CROSS-REFERENCED from {len(source_summaries)} sources]: {verified_facts}"
                elif fact_check_response.startswith("CONFLICTED:"):
                    self.logger.warning(f"[FactCheck] Sources conflict for query: '{query}'")
                    return None
                elif fact_check_response.startswith("INSUFFICIENT:"):
                    self.logger.warning(f"[FactCheck] Insufficient verifiable information for: '{query}'")
                    return None
                else:
                    self.logger.warning(f"[FactCheck] Unexpected response format from fact-checker")
                    return None
            else:
                self.logger.error(f"[FactCheck] No response from fact-checking model")
                return None
                
        except Exception as e:
            self.logger.error(f"[FactCheck] Fact-checking failed: {e}", exc_info=True)
            return None

    def _store_verified_info_in_rag(self, query: str, verified_info: str, sources: List[Dict]):
        """
        Store verified information in the RAG database.
        
        Args:
            query: The original search query
            verified_info: The verified information to store
            sources: List of sources used for verification
        """
        try:
            self.logger.info(f"[FactCheck] Storing verified information in RAG for query: '{query}'")
            
            # Create metadata about the verification
            source_urls = [result.get('href', 'Unknown') for result in sources[:3]]
            metadata = {
                "query": query,
                "verification_date": time.strftime("%Y-%m-%d"),
                "verification_time": time.time(),
                "sources": source_urls,
                "source_count": len(sources),
                "type": "fact_checked_web_search",
                "confidence": "high"  # Since it's verified across multiple sources
            }
            
            # Format the information for storage
            formatted_text = f"Query: {query}\n\nVerified Information:\n{verified_info}\n\nSources: {', '.join(source_urls[:3])}"
            
            # Try to store using RAG service
            success = False
            try:
                # Try the standard add_document method if it exists
                if hasattr(self.rag_service, 'add_document'):
                    success = self.rag_service.add_document(
                        text=formatted_text,
                        metadata=metadata,
                        collection=self.default_collection
                    )
                elif hasattr(self.rag_service, 'store_text'):
                    success = self.rag_service.store_text(
                        text=formatted_text,
                        metadata=metadata,
                        collection=self.default_collection
                    )
                else:
                    # Fallback: try direct Qdrant storage
                    self.logger.info("[FactCheck] Using direct Qdrant storage fallback")
                    success = self._store_directly_in_qdrant(formatted_text, metadata)
                    
            except Exception as storage_error:
                self.logger.warning(f"[FactCheck] RAG service storage failed, trying direct Qdrant: {storage_error}")
                success = self._store_directly_in_qdrant(formatted_text, metadata)
            
            if success:
                self.logger.info(f"[FactCheck] ✅ Verified information stored in RAG successfully")
                # Log to fallback file for tracking
                self._log_fallback(query, self.default_collection, f"STORED_VERIFIED_INFO: {len(source_urls)} sources")
            else:
                self.logger.error(f"[FactCheck] ❌ Failed to store verified information in RAG")
                
        except Exception as e:
            self.logger.error(f"[FactCheck] Failed to store verified info in RAG: {e}", exc_info=True)

    def _store_directly_in_qdrant(self, text: str, metadata: Dict) -> bool:
        """
        Direct storage to Qdrant as a fallback method.
        
        Args:
            text: The text to store
            metadata: Metadata about the text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to generate embedding using sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(text).tolist()
            except ImportError:
                self.logger.warning("[FactCheck] sentence-transformers not available, using simple hash-based embedding")
                # Simple fallback: create a basic embedding from text hash
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                # Convert hash to a 384-dimensional vector (matching typical embedding size)
                embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 64), 2)]
                embedding.extend([0.0] * (384 - len(embedding)))  # Pad to 384 dimensions
            
            # Store in Qdrant directly if we have access to it
            if hasattr(self.rag_service, 'qdrant_service') and self.rag_service.qdrant_service:
                qdrant_service = self.rag_service.qdrant_service
                import uuid
                
                point_id = str(uuid.uuid4())
                success = qdrant_service.add_texts(
                    collection_name=self.default_collection,
                    texts=[text],
                    vectors=[embedding],
                    metadatas=[metadata]
                )
                return success
            elif hasattr(self.rag_service, 'ingest_text'):
                # Use the RAG service's ingest method instead
                success = self.rag_service.ingest_text(
                    text=text,
                    metadata=metadata,
                    collection=self.default_collection
                )
                return success
            else:
                self.logger.warning("[FactCheck] No direct Qdrant access available")
                return False
                
        except Exception as e:
            self.logger.error(f"[FactCheck] Direct Qdrant storage failed: {e}")
            return False

    def fact_checked_generate(self, prompt, temperature=0.7, max_tokens=1024, model=None, llm_provider="quadrant"):
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
            # Query RAG service
            docs = self.rag_service.query(
                collection=self.default_collection,
                query_text=prompt,
                n_results=5
            )
            
            if docs:
                # Build context-aware message
                messages = [
                    {"role": "system", "content": "You are a knowledgeable assistant. Use the provided context to answer questions accurately."},
                    {"role": "user", "content": f"Context:\n{docs}\n\nQuestion: {prompt}"}
                ]
                
                # Generate response using context
                response = self.model_client.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model
                )
                
                if response:
                    return response
            
            # Fallback response if no context or generation fails
            return "I apologize, but I'm unable to find any relevant information about that topic. Would you like to try a different question?"
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}", exc_info=True)
            return "I encountered an error while processing your question. Please try again."