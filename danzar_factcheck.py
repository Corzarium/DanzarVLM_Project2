# danzar_factcheck.py

import time
import json
import re
import logging
from urllib.parse import quote # Explicitly import quote for URL encoding
from typing import List, Dict, Any, Optional
import requests # Ensure requests is imported

# Assuming default_api is available for tool calls within this module's methods
# If not, the tool calls would need to be orchestrated by the calling code (LLMService).
# Given the instruction to "wrap the core response generator to invoke search(query_text)",
# it's more likely the LLMService will call methods in FactChecker, and FactChecker
# methods will make the tool calls (like web_search).

# Set up logging for unsupported facts
logging.basicConfig(
    filename='logs/unsupported_facts.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
unsupported_facts_logger = logging.getLogger('unsupported_facts_logger')
unsupported_facts_logger.setLevel(logging.INFO)
# Prevent log messages from propagating to the root logger
unsupported_facts_logger.propagate = False
# Add a file handler if not already added
if not unsupported_facts_logger.handlers:
    file_handler = logging.FileHandler('logs/unsupported_facts.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    unsupported_facts_logger.addHandler(file_handler)


__version__ = "1.1.0" # Added for debugging module loading issues

class FactCheckService:
    def __init__(self, rag_service, model_client):
        self.rag_service = rag_service
        self.model_client = model_client
        self.logger = logging.getLogger("DanzarVLM.FactCheck")
        self.logger.info(f"[FactCheckService] Initializing FactCheckService version {__version__}")

    def _is_question(self, text: str) -> bool:
        """Determines if the given text is a question."""
        text = text.strip().lower()
        # Simple heuristic: ends with '?' or starts with common question words
        if text.endswith('?'):
            return True
        question_starters = ["what", "where", "when", "who", "whom", "whose", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should", "has", "have", "had"]
        return any(text.startswith(f"{qs} ") for qs in question_starters)

    def fact_checked_generate(self, prompt: str, temperature: float, max_tokens: int, model: str = "mistral") -> str:
        # Ensure the 'model' parameter is recognized.
        """
        Generates an LLM response, fact-checking it against RAG and web search.
        """
        self.logger.info(f"Attempting fact-checked generation for prompt: '{prompt[:100]}...'")

        # 1. Initial LLM Response (always generate, even if it's a question)
        # This is the base response that will be fact-checked.
        # 1. Initial LLM Response (always generate, even if it's a question)
        # This is the base response that will be fact-checked.
        # Add a system message to inform the LLM about its web search capability
        messages = [
            {"role": "system", "content": "You are an AI assistant that can perform web searches to find current information. If a user asks for information that might be outdated or requires external knowledge, you should indicate that you will search for it and then provide the information you find."},
            {"role": "user", "content": prompt}
        ]
        
        raw_llm_response = self.model_client.generate(
            prompt=messages, # Pass messages directly
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            endpoint="chat/completions" # Ensure correct endpoint for chat models
        )
        
        if not raw_llm_response:
            self.logger.error("LLM generation failed, returning generic error.")
            return "I'm sorry, I couldn't generate a response at this time."

        # If it's not a question, return the raw LLM response directly
        if not self._is_question(prompt):
            self.logger.info("Prompt is not a question, returning direct LLM response.")
            return raw_llm_response

        self.logger.info("Prompt is a question, proceeding with fact-checking.")

        # 2. RAG Lookup
        rag_sources = self.rag_service.query_rag(
            collection_name=self.rag_service.server_side_default_collection, # Use the default collection
            query_text=prompt,
            top_k=5 # Get more results for better context
        )
        
        rag_context = ""
        if rag_sources:
            rag_context = "\n\n".join(rag_sources)
            self.logger.info(f"Retrieved {len(rag_sources)} RAG sources.")
        else:
            self.logger.info("No RAG sources found for the query.")

        # 3. Web Search (Always perform if it's a question)
        self.logger.info("Performing web search for additional context...")
        web_search_results = self._perform_web_search(prompt) # Use the new web search function
        if web_search_results:
            self.logger.info(f"Retrieved {len(web_search_results)} web search results.")
        else:
            self.logger.info("No web search results found.")

        # Combine all supporting documents
        supporting_documents = rag_context
        if web_search_results:
            # Add a separator if RAG context exists
            if supporting_documents:
                supporting_documents += "\n\n"
            supporting_documents += "Web Search Results:\n"
            for i, result in enumerate(web_search_results):
                supporting_documents += f"[Doc {i+1}] Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\nSnippet: {result.get('snippet', 'No snippet')}\n\n"

        if not supporting_documents.strip():
            self.logger.warning("No supporting documents (RAG or web) found. Cannot fact-check.")
            return "I'm sorry, I couldn't find enough information to answer that question reliably."

        # 4. Reflection, Scoring, and Filtering
        sentences = self._tokenize_sentences(raw_llm_response)
        checked_sentences = self._reflect_and_score(sentences, supporting_documents)

        # 5. Construct Final Response and Apply Fallback Logic
        supported_sentences = [s for s in checked_sentences if s['supported']]
        unsupported_sentences = [s for s in checked_sentences if not s['supported']]

        self.logger.debug(f"Supported sentences: {len(supported_sentences)}, Unsupported: {len(unsupported_sentences)}")

        final_response_parts = []
        unsupported_count = len(unsupported_sentences)
        total_sentences = len(sentences)
        unsupported_percentage = (unsupported_count / total_sentences) * 100 if total_sentences > 0 else 0

        # Log unsupported claims
        for s in unsupported_sentences:
            unsupported_facts_logger.info(
                f"Query: {prompt}\n"
                f"Unsupported Claim: {s['sentence']}\n"
                f"Reason/Confidence: {s.get('reason', 'Not supported by search/RAG')}\n"
                f"---"
            )

        # Apply fallback based on percentage or lack of supported sentences
        if unsupported_percentage > 30 or len(supported_sentences) < 3: # Require at least 3 supported sources
            self.logger.info("High percentage of unsupported claims or insufficient supported claims. Applying fallback.")
            if len(supported_sentences) > 0:
                 # If some sentences are supported, but not enough for full confidence
                 final_response_parts.append("I found some information, but I need to verify more details. Here's what I'm confident about: ")
                 final_response_parts.extend([s['sentence'] for s in supported_sentences])
            else:
                 # If no sentences supported at all
                 return "I'm sorry, I couldn't find enough reliable information to answer that question."
        else:
            self.logger.info("Sufficiently supported claims. Constructing response from supported sentences.")
            final_response_parts.extend([s['sentence'] for s in supported_sentences])

        final_response = " ".join(final_response_parts).strip()

        if not final_response:
             return "I'm sorry, I couldn't find enough reliable information to answer that question."

        return final_response

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Simple sentence tokenization."""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a web search using DuckDuckGo Instant Answer API.
        Returns a list of dictionaries, each with 'title', 'url', 'snippet'.
        """
        self.logger.info(f"Performing web search for: '{query}'")
        search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&t=DanzarVLM"
        try:
            response = requests.get(search_url, timeout=5)
            response.raise_for_status()
            data = response.json()

            results = []
            # Check for Abstract (Wikipedia, etc.)
            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append({
                    "title": data.get("AbstractSource", "Abstract"),
                    "url": data["AbstractURL"],
                    "snippet": data["AbstractText"]
                })
            
            # Check for Related Topics (more general search results)
            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"]:
                    if "Result" in topic: # This is a direct link result
                        # Extract title, URL, and snippet from the HTML-like string in "Result"
                        # This is a bit fragile, but common for DDG API
                        match = re.search(r'<a href="([^"]+)">([^<]+)</a>(.+)', topic["Result"])
                        if match:
                            url = match.group(1)
                            title = match.group(2)
                            snippet = re.sub(r'<[^>]+>', '', match.group(3)).strip() # Strip HTML tags from snippet
                            results.append({"title": title, "url": url, "snippet": snippet})
                    elif "Text" in topic and "FirstURL" in topic: # This is a text-only topic
                        results.append({
                            "title": topic.get("Text", "Related Topic"),
                            "url": topic["FirstURL"],
                            "snippet": topic.get("Text", "No snippet available.")
                        })
            
            # Check for Results (actual search results)
            if data.get("Results"):
                for res in data["Results"]:
                    results.append({
                        "title": res.get("Title", "Search Result"),
                        "url": res.get("FirstURL", "N/A"),
                        "snippet": res.get("Text", "No snippet available.")
                    })

            self.logger.info(f"DuckDuckGo search for '{query}' returned {len(results)} results.")
            return results
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Web search failed for '{query}': {e}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from DuckDuckGo API: {e}", exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during web search: {e}", exc_info=True)
            return []

    def _reflect_and_score(self, sentences: List[str], supporting_documents: str) -> List[Dict[str, Any]]:
        """
        Refined reflection and scoring process.
        This version uses a more robust keyword matching and aims for multi-source verification.
        """
        self.logger.debug("[FactCheckService] Reflecting and scoring sentences against supporting documents.")
        checked_sentences = []
        doc_lower = supporting_documents.lower()

        for sentence in sentences:
            is_supported = False
            reason = "No supporting documents provided."
            sentence_lower = sentence.lower()

            # Extract key phrases/entities from the sentence
            # This is a simple regex-based approach; for better results, consider NLP libraries
            key_phrases = re.findall(r'\b[a-z0-9]+\b', sentence_lower) # Simple word extraction
            key_phrases = [p for p in key_phrases if len(p) > 2 and p not in self._get_common_words()]

            if supporting_documents and key_phrases:
                # Count how many unique supporting documents contain at least one key phrase
                # For multi-source verification, we need to know which document each snippet came from.
                # The current `supporting_documents` string combines all, so we'll simulate.
                # A better approach would be to pass `List[Dict[str, Any]]` for documents.
                
                # For now, a simple check: if a significant portion of key phrases are found
                # and we can infer "multiple sources" by checking for distinct URL patterns
                # or just by the sheer volume of supporting_documents.
                
                found_in_docs = [phrase for phrase in key_phrases if phrase in doc_lower]
                
                # Heuristic for multi-source verification:
                # If the combined document string contains "Doc 1", "Doc 2", "Doc 3" etc.
                # This is a very weak heuristic and needs improvement if actual source tracking is required.
                num_simulated_sources = len(re.findall(r'\[Doc \d+\]', supporting_documents))
                
                if len(found_in_docs) / len(key_phrases) > 0.6: # Higher threshold for confidence
                    if num_simulated_sources >= 3: # Check for 3 or more simulated sources
                        is_supported = True
                        reason = f"Supported by multiple sources ({num_simulated_sources} simulated) with high keyword match."
                    else:
                        is_supported = True # Still supported, but not multi-source verified
                        reason = f"Supported by documents ({num_simulated_sources} simulated sources), but not multi-source verified."
                else:
                    reason = f"Low keyword match ({len(found_in_docs)}/{len(key_phrases)}) in documents."
            else:
                reason = "No relevant supporting documents or key phrases for verification."

            checked_sentences.append({
                'sentence': sentence,
                'supported': is_supported,
                'reason': reason
            })

        return checked_sentences

    def _get_common_words(self) -> set:
        """Returns a set of common English stop words."""
        return set(["a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "of", "at", "by", "for", "with", "and", "or", "in", "on", "it", "its", "that", "this", "those", "these", "to", "from", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

# Mock classes for testing
class MockAppContext:
    def __init__(self):
        self.logger = logging.getLogger("MockLogger")
        if not self.logger.handlers:
             self.logger.addHandler(logging.NullHandler())
        self.global_settings = {}

class MockRAGService:
    def query_rag(self, collection_name: str, query_text: str, top_k: int) -> List[str]:
        # Simulate RAG results
        if "everquest" in query_text.lower():
            return ["EverQuest is a 3D fantasy massively multiplayer online role-playing game (MMORPG) that was released in 1999.",
                    "It is known for its immersive world, challenging gameplay, and strong community.",
                    "EverQuest has had numerous expansions over the years, adding new content and features."]
        return []

class MockModelClient:
    def generate(self, prompt: str, temperature: float, max_tokens: int, model: str, endpoint: str) -> str:
        # Simulate LLM response
        if "everquest" in prompt.lower():
            return "EverQuest is a very old MMORPG. It came out in 1999 and has many expansions. It is famous for its community."
        return "This is a generic LLM response."

if __name__ == "__main__":
    print("Running illustrative tests for FactCheckService...")
    
    mock_app_context = MockAppContext()
    mock_rag_service = MockRAGService()
    mock_model_client = MockModelClient()
    
    fact_checker = FactCheckService(mock_rag_service, mock_model_client)

    # Test 1: Question with sufficient RAG support
    print("\n--- Test 1: Question with sufficient RAG support ---")
    query1 = "What can you tell me about EverQuest?"
    response1 = fact_checker.fact_checked_generate(query1, 0.7, 256)
    print(f"Query: {query1}\nResponse: {response1}")
    assert "EverQuest is a 3D fantasy massively multiplayer online role-playing game (MMORPG) that was released in 1999." in response1
    assert "It is known for its immersive world, challenging gameplay, and strong community." in response1
    assert "EverQuest has had numerous expansions over the years, adding new content and features." in response1
    assert not response1.startswith("I'm sorry, I couldn't find enough reliable information")
    print("Test 1 Passed.")

    # Test 2: Non-question
    print("\n--- Test 2: Non-question ---")
    query2 = "Tell me a story."
    response2 = fact_checker.fact_checked_generate(query2, 0.7, 256)
    print(f"Query: {query2}\nResponse: {response2}")
    assert response2 == "This is a generic LLM response."
    print("Test 2 Passed.")

    # Test 3: Question with insufficient RAG, but web search provides
    print("\n--- Test 3: Question with insufficient RAG, but web search provides ---")
    original_rag_query = mock_rag_service.query_rag
    mock_rag_service.query_rag = lambda collection_name, query_text, top_k: ["EverQuest is an MMORPG from 1999."]
    
    original_web_search = fact_checker._perform_web_search
    fact_checker._perform_web_search = lambda query: [
        {"title": "EQ Wiki", "url": "wiki.com", "snippet": "EverQuest has a huge world."},
        {"title": "EQ Forum", "url": "forum.com", "snippet": "The community is very active."},
        {"title": "EQ History", "url": "history.com", "snippet": "Many expansions have been released."}
    ]
    
    query3 = "Tell me more about EverQuest."
    response3 = fact_checker.fact_checked_generate(query3, 0.7, 256)
    print(f"Query: {query3}\nResponse: {response3}")
    assert "I found some information, but I need to verify more details." in response3
    assert "EverQuest is an MMORPG from 1999." in response3
    assert "EverQuest has a huge world." in response3
    assert "The community is very active." in response3
    assert "Many expansions have been released." in response3
    
    mock_rag_service.query_rag = original_rag_query
    fact_checker._perform_web_search = original_web_search
    print("Test 3 Passed.")

    # Test 4: Question with no reliable information
    print("\n--- Test 4: Question with no reliable information ---")
    mock_rag_service.query_rag = lambda collection_name, query_text, top_k: []
    fact_checker._perform_web_search = lambda query: []
    
    query4 = "What is the capital of Norrath in EverQuest?"
    response4 = fact_checker.fact_checked_generate(query4, 0.7, 256)
    print(f"Query: {query4}\nResponse: {response4}")
    assert response4 == "I'm sorry, I couldn't find enough reliable information to answer that question."
    print("Test 4 Passed.")

    print("\nAll illustrative tests finished.")