#!/usr/bin/env python3
"""
Web Search Utility for DanzarAI
"""

import requests
import logging
from urllib.parse import quote_plus
import json

logger = logging.getLogger(__name__)

def web_search(query: str) -> str:
    """
    Perform a web search and return formatted results
    """
    try:
        logger.info(f"🔍 Web search for: {query}")
        
        # Use DuckDuckGo Instant Answer API
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        
        headers = {
            'User-Agent': 'DanzarAI/1.0 (Gaming Assistant Bot)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Get instant answer if available
        if data.get('Abstract'):
            results.append(f"**Summary**: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"**Source**: {data['AbstractURL']}")
        
        # Get definition if available
        if data.get('Definition'):
            results.append(f"**Definition**: {data['Definition']}")
            if data.get('DefinitionURL'):
                results.append(f"**Source**: {data['DefinitionURL']}")
        
        # Get answer if available
        if data.get('Answer'):
            results.append(f"**Answer**: {data['Answer']}")
            if data.get('AnswerType'):
                results.append(f"**Type**: {data['AnswerType']}")
        
        # Get related topics
        if data.get('RelatedTopics'):
            related = data['RelatedTopics'][:3]  # Limit to 3 related topics
            for i, topic in enumerate(related):
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"**Related {i+1}**: {topic['Text']}")
                    if topic.get('FirstURL'):
                        results.append(f"**Link**: {topic['FirstURL']}")
        
        if results:
            formatted_results = "\n".join(results)
            logger.info(f"✅ Web search successful, found {len(results)} result items")
            return formatted_results
        else:
            logger.warning(f"🔍 No results found for query: '{query}'")
            return f"No specific results found for '{query}'. You may want to try a more specific search term."
            
    except requests.exceptions.Timeout:
        logger.error(f"⏰ Web search timeout for query: '{query}'")
        return f"Web search timed out for '{query}'. Please try again."
        
    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Web search request error: {e}")
        return f"Web search failed due to network error: {str(e)}"
        
    except Exception as e:
        logger.error(f"❌ Web search error: {e}")
        return f"Web search encountered an error: {str(e)}"