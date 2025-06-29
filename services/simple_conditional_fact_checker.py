#!/usr/bin/env python3
"""
Simple Conditional Fact Checker
Provides conditional fact-checking without external dependencies
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.web_search import web_search
from services.memory_service import MemoryService


@dataclass
class FactCheckResult:
    """Result of a fact-checking operation"""
    needs_fact_check: bool
    confidence: float
    reasoning: str
    search_queries: List[str]
    fact_check_results: Optional[str] = None
    corrected_response: Optional[str] = None


class SimpleConditionalFactChecker:
    """Simple conditional fact-checking service without external dependencies"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.memory_service = getattr(app_context, 'memory_service', None)
        self.rag_service = getattr(app_context, 'rag_service_instance', None)
        
        # Challenge detection patterns
        self.challenge_patterns = [
            # Direct challenges
            r'\b(that\'?s\s+not\s+right|that\'?s\s+wrong|you\'?re\s+wrong|you\'?re\s+incorrect)\b',
            r'\b(are\s+you\s+sure|are\s+you\s+certain|is\s+that\s+correct|is\s+that\s+right)\b',
            r'\b(I\s+think\s+you\'?re\s+wrong|I\s+don\'?t\s+think\s+so|that\s+doesn\'?t\s+sound\s+right)\b',
            r'\b(actually|in\s+fact|as\s+a\s+matter\s+of\s+fact|well\s+actually)\b',
            r'\b(no\s+that\'?s\s+not|no\s+you\'?re\s+wrong|no\s+that\'?s\s+incorrect)\b',
            
            # Doubt expressions
            r'\b(I\s+doubt|I\'?m\s+not\s+sure|I\'?m\s+uncertain|I\'?m\s+not\s+convinced)\b',
            r'\b(that\s+seems\s+unlikely|that\s+doesn\'?t\s+make\s+sense|that\s+can\'?t\s+be\s+right)\b',
            
            # Request for verification
            r'\b(can\s+you\s+verify|can\s+you\s+check|can\s+you\s+confirm|can\s+you\s+look\s+up)\b',
            r'\b(please\s+verify|please\s+check|please\s+confirm|please\s+look\s+up)\b',
            
            # Internet search requests
            r'\b(can\s+you\s+search|can\s+you\s+look\s+it\s+up|can\s+you\s+find\s+out)\b',
            r'\b(do\s+an\s+internet\s+search|search\s+the\s+web|look\s+it\s+up\s+online)\b',
            
            # Specific fact challenges
            r'\b(that\'?s\s+not\s+what\s+I\s+heard|that\'?s\s+not\s+what\s+I\s+read|that\'?s\s+not\s+what\s+I\s+know)\b',
            r'\b(I\s+heard\s+differently|I\s+read\s+differently|I\s+know\s+differently)\b',
        ]
        
        # Uncertainty patterns for LLM responses
        self.uncertainty_patterns = [
            # LLM uncertainty indicators
            r'\b(I\s+think|I\s+believe|I\'?m\s+not\s+sure|I\'?m\s+uncertain)\b',
            r'\b(as\s+far\s+as\s+I\s+know|to\s+my\s+knowledge|from\s+what\s+I\s+recall)\b',
            r'\b(I\'?m\s+not\s+completely\s+sure|I\'?m\s+not\s+100%\s+sure|I\'?m\s+not\s+certain)\b',
            r'\b(this\s+might\s+be|this\s+could\s+be|this\s+may\s+be|this\s+possibly\s+is)\b',
            r'\b(if\s+I\s+remember\s+correctly|if\s+my\s+memory\s+serves|if\s+I\'?m\s+not\s+mistaken)\b',
        ]
        
        self.logger.info("[SimpleConditionalFactChecker] Initialized successfully")
    
    async def should_fact_check(self, user_text: str, llm_response: str = None) -> FactCheckResult:
        """
        Determine if fact-checking should be triggered
        """
        try:
            return await self._detect_fact_check_needed(user_text, llm_response)
        except Exception as e:
            self.logger.error(f"[SimpleConditionalFactChecker] Error in should_fact_check: {e}")
            return FactCheckResult(
                needs_fact_check=False,
                confidence=0.0,
                reasoning="Error occurred during detection",
                search_queries=[]
            )
    
    async def _detect_fact_check_needed(self, user_text: str, llm_response: str = None) -> FactCheckResult:
        """Detect if fact-checking is needed using pattern matching"""
        text_lower = user_text.lower()
        
        # Check for challenge patterns in user text
        challenge_score = 0
        for pattern in self.challenge_patterns:
            if re.search(pattern, text_lower):
                challenge_score += 1
        
        # Check for uncertainty patterns in LLM response
        uncertainty_score = 0
        if llm_response:
            llm_lower = llm_response.lower()
            for pattern in self.uncertainty_patterns:
                if re.search(pattern, llm_lower):
                    uncertainty_score += 1
        
        # Determine if fact-checking is needed
        needs_fact_check = challenge_score > 0 or uncertainty_score >= 2
        confidence = min(1.0, (challenge_score + uncertainty_score) / 3.0)
        
        # Generate search queries
        search_queries = self._generate_search_queries(user_text, llm_response)
        
        # Determine reasoning
        if challenge_score > 0:
            reasoning = "User challenged or questioned the information"
        elif uncertainty_score >= 2:
            reasoning = "LLM expressed uncertainty about the information"
        else:
            reasoning = "No specific trigger detected"
        
        return FactCheckResult(
            needs_fact_check=needs_fact_check,
            confidence=confidence,
            reasoning=reasoning,
            search_queries=search_queries
        )
    
    def _generate_search_queries(self, user_text: str, llm_response: str = None) -> List[str]:
        """Generate search queries for fact-checking"""
        queries = []
        
        # Extract key terms from user text
        key_terms = self._extract_key_terms(user_text)
        if key_terms:
            queries.append(f"{' '.join(key_terms)} fact check")
            queries.append(f"{' '.join(key_terms)} verification")
        
        # If LLM response provided, extract disputed facts
        if llm_response:
            disputed_facts = self._extract_disputed_facts(user_text, llm_response)
            for fact in disputed_facts:
                queries.append(f"{fact} correct information")
        
        # Add general verification query
        if key_terms:
            queries.append(f"verify {' '.join(key_terms)}")
        
        return queries[:5]  # Limit to 5 queries
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms for search queries"""
        # Remove common words and extract nouns/important terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _extract_disputed_facts(self, user_text: str, llm_response: str) -> List[str]:
        """Extract facts that are being disputed"""
        facts = []
        
        # Look for specific claims in the user text
        claim_patterns = [
            r'\b(that|this|it)\s+(is|was|will\s+be|should\s+be)\s+([^.!?]+)',
            r'\b(you\s+said|you\s+claimed|you\s+mentioned)\s+([^.!?]+)',
        ]
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, user_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    fact = ' '.join(match).strip()
                else:
                    fact = match.strip()
                if fact and len(fact) > 10:
                    facts.append(fact)
        
        return facts[:3]  # Return top 3 disputed facts
    
    async def perform_fact_check(self, result: FactCheckResult) -> str:
        """Perform the actual fact-checking"""
        if not result.search_queries:
            return "No search queries generated for fact-checking."
        
        self.logger.info(f"[SimpleConditionalFactChecker] Performing fact-check with queries: {result.search_queries}")
        
        fact_check_results = []
        
        for query in result.search_queries[:3]:  # Limit to 3 queries
            try:
                search_result = await asyncio.get_event_loop().run_in_executor(
                    None, web_search, query
                )
                
                if search_result and "No specific results found" not in search_result:
                    fact_check_results.append(f"**Query**: {query}\n**Result**: {search_result}\n")
                
                # Small delay between searches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"[SimpleConditionalFactChecker] Error searching for '{query}': {e}")
                fact_check_results.append(f"**Query**: {query}\n**Error**: Search failed - {str(e)}\n")
        
        if fact_check_results:
            combined_results = "\n".join(fact_check_results)
            
            # Store in memory if available
            if self.memory_service:
                try:
                    memory_entry = {
                        "content": f"Fact-check results for: {result.search_queries[0]}",
                        "metadata": {
                            "type": "fact_check",
                            "queries": result.search_queries,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": result.confidence
                        }
                    }
                    await self.memory_service.add_memory(memory_entry)
                except Exception as e:
                    self.logger.error(f"[SimpleConditionalFactChecker] Error storing fact-check in memory: {e}")
            
            return f"🔍 **Fact-Check Results**\n\n{combined_results}"
        else:
            return "🔍 **Fact-Check Results**\n\nNo definitive information found to verify or dispute the claims."
    
    async def integrate_with_llm_response(self, user_text: str, original_response: str, fact_check_results: str) -> str:
        """Integrate fact-check results into LLM response"""
        if not fact_check_results or "No definitive information found" in fact_check_results:
            return original_response
        
        # Create enhanced response with fact-check results
        enhanced_response = f"{original_response}\n\n🔍 **Fact-Check Summary**:\n{fact_check_results}"
        
        return enhanced_response 