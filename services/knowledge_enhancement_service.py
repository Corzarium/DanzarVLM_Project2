#!/usr/bin/env python3
"""
Knowledge Enhancement Service - Automatically improves RAG database with verified information
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class SourceResult:
    """Result from a single information source"""
    source_name: str
    content: str
    url: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = 0.0

@dataclass
class VerifiedInformation:
    """Information verified across multiple sources"""
    query: str
    verified_content: str
    sources: List[SourceResult]
    confidence_score: float
    verification_method: str
    metadata: Dict[str, Any]

class KnowledgeEnhancementService:
    """
    Service that enhances the RAG knowledge base by:
    1. Searching multiple sources when RAG has no answer
    2. Cross-verifying information across sources
    3. Adding verified information to the RAG database
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.settings = app_context.global_settings
        
        # Services
        self.rag_service = None
        self.fact_check_service = None
        
        # Configuration
        self.min_sources_for_verification = 2
        self.min_confidence_threshold = 0.7
        self.similarity_threshold = 0.6  # For content similarity comparison
        self.max_search_sources = 5
        
        # Knowledge enhancement settings
        self.auto_enhance_enabled = self.settings.get('AUTO_ENHANCE_KNOWLEDGE', True)
        self.enhancement_collection = self.settings.get('ENHANCEMENT_COLLECTION', 'enhanced_knowledge')
        
        # Statistics
        self.enhancement_stats = {
            "total_queries": 0,
            "enhanced_queries": 0,
            "successful_enhancements": 0,
            "failed_enhancements": 0
        }
        
        # Store original queries to prevent self-referential searches
        self.original_queries = {}  # query_id -> original_query
        
        self.logger.info("[KnowledgeEnhancement] Service initialized")
    
    def set_services(self, rag_service=None, fact_check_service=None):
        """Set required services"""
        if rag_service:
            self.rag_service = rag_service
            self.logger.info("[KnowledgeEnhancement] RAG service connected")
        
        if fact_check_service:
            self.fact_check_service = fact_check_service
            self.logger.info("[KnowledgeEnhancement] Fact check service connected")
    
    async def enhance_knowledge_if_needed(self, original_query: str, rag_results: List[Dict], user_name: str = "User") -> Optional[VerifiedInformation]:
        """
        Enhanced method that stores original query and uses LLM to generate better search queries
        """
        if not self.auto_enhance_enabled:
            return None
        
        self.enhancement_stats["total_queries"] += 1
        
        # Store the original query to prevent self-referential searches
        query_id = f"{original_query}_{time.time()}"
        self.original_queries[query_id] = original_query
        
        try:
            # Check if RAG results are sufficient
            if self._has_sufficient_rag_results(rag_results):
                self.logger.info(f"[KnowledgeEnhancement] RAG results sufficient for: '{original_query}'")
                return None
            
            self.logger.info(f"[KnowledgeEnhancement] RAG results insufficient, searching external sources for: '{original_query}'")
            self.enhancement_stats["enhanced_queries"] += 1
            
            # Generate intelligent search queries using LLM
            search_queries = await self._generate_llm_search_queries(original_query, rag_results, user_name)
            
            # Search multiple sources with LLM-generated queries
            source_results = await self._search_multiple_sources_with_llm_queries(search_queries)
            
            if len(source_results) < self.min_sources_for_verification:
                self.logger.info(f"[KnowledgeEnhancement] Insufficient sources ({len(source_results)}) for verification")
                self.enhancement_stats["failed_enhancements"] += 1
                return None
            
            # Verify information across sources
            verified_info = await self._verify_information_across_sources(original_query, source_results)
            
            if verified_info:
                self.logger.info(f"[KnowledgeEnhancement] ✅ Successfully enhanced knowledge for: '{original_query}'")
                self.enhancement_stats["successful_enhancements"] += 1
                
                # Optionally add to RAG database for future queries
                await self._add_to_rag_database(verified_info)
                
                return verified_info
            else:
                self.logger.info(f"[KnowledgeEnhancement] Could not verify information for: '{original_query}'")
                self.enhancement_stats["failed_enhancements"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error enhancing knowledge: {e}")
            self.enhancement_stats["failed_enhancements"] += 1
            return None
        finally:
            # Clean up stored query
            if query_id in self.original_queries:
                del self.original_queries[query_id]
    
    async def _generate_llm_search_queries(self, original_query: str, rag_results: List[Dict], user_name: str) -> List[str]:
        """
        Use LLM to generate intelligent search queries when it doesn't know the answer
        """
        try:
            # Check if we have a model client available
            if not hasattr(self.app_context, 'model_client') or not self.app_context.model_client:
                self.logger.warning("[KnowledgeEnhancement] No model client available, using fallback search generation")
                return self._generate_fallback_search_queries(original_query)
            
            # Create a prompt for the LLM to generate search queries
            rag_context = ""
            if rag_results:
                rag_context = f"\n\nCurrent knowledge base contains limited information:\n"
                for i, result in enumerate(rag_results[:2]):
                    text = result.get('text', '')[:100]
                    rag_context += f"- {text}...\n"
            
            prompt = f"""You are a research assistant helping to find information. A user asked: "{original_query}"

{rag_context}

I need to search the web to find comprehensive information to answer this question. Generate 3 specific, targeted search queries that would help find authoritative information.

IMPORTANT RULES:
1. Focus on the user's original question, not on any partial answers
2. Use simple, direct search terms that would find official sources
3. Avoid complex phrases or descriptions
4. For game questions, focus on official wikis, guides, or documentation

Format your response as a simple numbered list:
1. [search query 1]
2. [search query 2] 
3. [search query 3]

Search queries:"""

            # Generate search queries using the LLM
            response = self.app_context.model_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            if response and response.strip():
                # Parse the LLM response to extract search queries
                search_queries = self._parse_llm_search_response(response, original_query)
                
                if search_queries:
                    self.logger.info(f"[KnowledgeEnhancement] LLM generated {len(search_queries)} search queries")
                    for i, query in enumerate(search_queries):
                        self.logger.info(f"[KnowledgeEnhancement] LLM Query {i+1}: '{query}'")
                    return search_queries
            
            # Fallback if LLM generation fails
            self.logger.warning("[KnowledgeEnhancement] LLM search generation failed, using fallback")
            return self._generate_fallback_search_queries(original_query)
            
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error generating LLM search queries: {e}")
            return self._generate_fallback_search_queries(original_query)
    
    def _parse_llm_search_response(self, response: str, original_query: str) -> List[str]:
        """Parse LLM response to extract search queries"""
        try:
            queries = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for numbered list items
                if re.match(r'^\d+\.?\s*', line):
                    query = re.sub(r'^\d+\.?\s*', '', line).strip()
                    if query and len(query) > 5:  # Minimum query length
                        # Remove any quotes or brackets
                        query = query.strip('"\'[]')
                        
                        # Check if it's not self-referential
                        if not self._is_self_referential(query, original_query):
                            queries.append(query)
                
                # Also look for lines that might be queries without numbers
                elif line and not line.startswith(('search', 'queries', 'format', 'important')):
                    if len(line) > 5 and not self._is_self_referential(line, original_query):
                        queries.append(line.strip('"\'[]'))
            
            # If we didn't get good queries from LLM, use fallback
            if not queries:
                self.logger.warning("[KnowledgeEnhancement] Could not parse LLM search queries, using fallback")
                return self._generate_fallback_search_queries(original_query)
            
            return queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error parsing LLM search response: {e}")
            return self._generate_fallback_search_queries(original_query)
    
    def _is_self_referential(self, search_query: str, original_query: str) -> bool:
        """Check if a search query is self-referential (searching for bot responses)"""
        search_lower = search_query.lower()
        original_lower = original_query.lower()
        
        # Check for bot response patterns that indicate we're searching for our own output
        bot_patterns = [
            '[everquest]', '[eq]', 'a utility support class', 'excel at crowd control',
            'hybrid tank life tap', 'melee dps class', 'martial artists who fight',
            'skilled in archery and nature magic', 'query:', 'answer:', 'everquest shadow knight',
            'everquest ranger', 'everquest monk', 'everquest bard', 'everquest warrior',
            'a hybrid melee caster class', 'a hybrid tank', 'life tap pet class'
        ]
        
        for pattern in bot_patterns:
            if pattern in search_lower:
                self.logger.warning(f"[KnowledgeEnhancement] Blocked self-referential search: '{search_query}' contains bot pattern: '{pattern}'")
                return True
        
        # Check if it's too similar to a known bot response format
        if search_lower.startswith(('everquest ', 'eq ')) and any(term in search_lower for term in ['class', 'hybrid', 'utility', 'support']):
            # But allow if it's the original user query
            if search_lower != original_lower:
                self.logger.warning(f"[KnowledgeEnhancement] Blocked self-referential search: '{search_query}' looks like bot response format")
                return True
        
        # Check for exact matches with stored original queries to prevent circular searches
        for stored_query in self.original_queries.values():
            if search_lower == stored_query.lower():
                continue  # Allow exact original query
            
            # Check if this looks like a bot response to the stored query
            if any(bot_word in search_lower for bot_word in ['hybrid', 'utility', 'support', 'excel at', 'skilled in']):
                if stored_query.lower() in search_lower or any(word in search_lower for word in stored_query.lower().split()):
                    self.logger.warning(f"[KnowledgeEnhancement] Blocked circular search: '{search_query}' appears to be bot response to previous query")
                    return True
        
        return False
    
    def _generate_fallback_search_queries(self, original_query: str) -> List[str]:
        """Generate search queries without LLM as fallback - focus on original user intent"""
        queries = []
        query_lower = original_query.lower()
        
        # Always start with the original query
        queries.append(original_query)
        
        # Game-specific enhancements - but keep them simple and user-focused
        if any(game in query_lower for game in ['everquest', 'eq']):
            if 'class' in query_lower:
                # Focus on comprehensive class information
                queries.extend([
                    "EverQuest character classes complete list",
                    "EverQuest all classes guide",
                    "EverQuest class overview"
                ])
            elif 'quest' in query_lower:
                queries.extend([
                    f"{original_query} walkthrough",
                    f"{original_query} guide"
                ])
            elif 'zone' in query_lower or 'area' in query_lower:
                queries.extend([
                    f"{original_query} guide",
                    f"{original_query} information"
                ])
            else:
                # General EverQuest queries
                queries.extend([
                    f"{original_query} guide",
                    f"{original_query} wiki"
                ])
        else:
            # General query enhancements
            queries.extend([
                f"{original_query} guide",
                f"{original_query} explanation"
            ])
        
        # Remove duplicates and self-referential queries
        unique_queries = []
        for query in queries:
            if query not in unique_queries and not self._is_self_referential(query, original_query):
                unique_queries.append(query)
        
        # Limit to 3 queries to avoid overwhelming the search
        return unique_queries[:3]
    
    def _has_sufficient_rag_results(self, rag_results: List[Dict]) -> bool:
        """Check if RAG results are sufficient to answer the query"""
        if not rag_results:
            return False
        
        # Check confidence scores
        high_confidence_results = [r for r in rag_results if r.get('score', 0) >= 0.7]
        
        if len(high_confidence_results) >= 2:
            return True
        
        # Check content quality
        total_content_length = sum(len(r.get('text', '')) for r in rag_results)
        if total_content_length >= 200:  # Minimum content threshold
            return True
        
        return False
    
    async def _search_multiple_sources_with_llm_queries(self, search_queries: List[str]) -> List[SourceResult]:
        """Search multiple sources using LLM-generated queries"""
        source_results = []
        
        try:
            # Use fact check service for web search if available
            if self.fact_check_service and hasattr(self.fact_check_service, '_search_web'):
                for i, search_query in enumerate(search_queries[:3]):  # Limit to 3 queries
                    try:
                        self.logger.info(f"[KnowledgeEnhancement] Searching web for LLM query {i+1}: '{search_query}'")
                        
                        # Use fact check service for web search
                        web_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.fact_check_service._search_web(search_query, fact_check=True)
                        )
                        
                        if web_result and len(web_result.strip()) > 50:  # Minimum content length
                            source_result = SourceResult(
                                source_name=f"llm_web_search_{i+1}",
                                content=web_result,
                                confidence=0.8,  # High confidence for fact-checked results
                                timestamp=time.time()
                            )
                            source_results.append(source_result)
                            
                            # Add delay between searches to avoid rate limiting
                            await asyncio.sleep(1.0)
                        
                    except Exception as e:
                        self.logger.warning(f"[KnowledgeEnhancement] LLM web search {i+1} failed: {e}")
                        continue
            else:
                self.logger.warning("[KnowledgeEnhancement] Fact check service not available for web search")
            
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error in LLM-guided web search: {e}")
        
        return source_results
    
    async def _verify_information_across_sources(self, query: str, source_results: List[SourceResult]) -> Optional[VerifiedInformation]:
        """Verify information by comparing content across multiple sources"""
        if len(source_results) < self.min_sources_for_verification:
            return None
        
        try:
            # Extract key information from each source
            extracted_info = []
            for source in source_results:
                info = self._extract_key_information(source.content, query)
                if info:
                    extracted_info.append((source, info))
            
            if len(extracted_info) < 2:
                self.logger.info("[KnowledgeEnhancement] Insufficient extracted information for verification")
                return None
            
            # Find common/consistent information
            verified_content = self._find_consistent_information(extracted_info)
            
            if not verified_content:
                self.logger.info("[KnowledgeEnhancement] No consistent information found across sources")
                return None
            
            # Calculate confidence based on source agreement
            confidence_score = self._calculate_verification_confidence(extracted_info, verified_content)
            
            # Create verified information object
            verified_info = VerifiedInformation(
                query=query,
                verified_content=verified_content,
                sources=source_results,
                confidence_score=confidence_score,
                verification_method="cross_source_verification",
                metadata={
                    "num_sources": len(source_results),
                    "extraction_method": "keyword_and_similarity",
                    "timestamp": time.time(),
                    "query_type": self._classify_query_type(query)
                }
            )
            
            self.logger.info(f"[KnowledgeEnhancement] Information verified with confidence: {confidence_score:.2f}")
            return verified_info
            
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error during verification: {e}")
            return None
    
    def _extract_key_information(self, content: str, query: str) -> Optional[str]:
        """Extract key information relevant to the query from source content"""
        try:
            # Clean the content
            content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
            content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
            
            # For EverQuest class queries, look for class lists
            if 'class' in query.lower() and 'everquest' in query.lower():
                return self._extract_everquest_classes(content)
            
            # For general queries, extract relevant sentences
            return self._extract_relevant_sentences(content, query)
            
        except Exception as e:
            self.logger.warning(f"[KnowledgeEnhancement] Error extracting information: {e}")
            return None
    
    def _extract_everquest_classes(self, content: str) -> Optional[str]:
        """Extract EverQuest class information from content"""
        # Known EverQuest classes
        known_classes = [
            'warrior', 'paladin', 'shadowknight', 'ranger', 'bard', 'rogue',
            'monk', 'berserker', 'wizard', 'magician', 'necromancer', 
            'enchanter', 'cleric', 'druid', 'shaman', 'beastlord'
        ]
        
        content_lower = content.lower()
        found_classes = []
        
        for class_name in known_classes:
            if class_name in content_lower:
                found_classes.append(class_name.title())
        
        if len(found_classes) >= 10:  # If we found most classes, it's likely a complete list
            return f"EverQuest has {len(found_classes)} character classes: {', '.join(sorted(found_classes))}."
        
        return None
    
    def _extract_relevant_sentences(self, content: str, query: str) -> Optional[str]:
        """Extract sentences most relevant to the query"""
        sentences = re.split(r'[.!?]+', content)
        query_words = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            if overlap >= 2:  # At least 2 words in common
                relevant_sentences.append((sentence, overlap))
        
        if relevant_sentences:
            # Sort by relevance and take top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:3]]
            return ' '.join(top_sentences)
        
        return None
    
    def _find_consistent_information(self, extracted_info: List[Tuple[SourceResult, str]]) -> Optional[str]:
        """Find information that's consistent across multiple sources"""
        if len(extracted_info) < 2:
            return None
        
        # For now, use a simple approach: find the most similar content
        contents = [info[1] for info in extracted_info]
        
        # Calculate similarity between all pairs
        max_similarity = 0.0
        best_content = None
        
        for i, content1 in enumerate(contents):
            similar_count = 0
            for j, content2 in enumerate(contents):
                if i != j:
                    similarity = SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
                    if similarity > self.similarity_threshold:
                        similar_count += 1
            
            # If this content is similar to at least one other source
            if similar_count > 0:
                avg_similarity = similar_count / (len(contents) - 1)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    best_content = content1
        
        return best_content if max_similarity > 0.3 else None
    
    def _calculate_verification_confidence(self, extracted_info: List[Tuple[SourceResult, str]], verified_content: str) -> float:
        """Calculate confidence score based on source agreement"""
        if not extracted_info or not verified_content:
            return 0.0
        
        # Base confidence from number of sources
        source_confidence = min(len(extracted_info) / 3.0, 1.0)  # Max at 3 sources
        
        # Content similarity confidence
        similarities = []
        for source, content in extracted_info:
            similarity = SequenceMatcher(None, verified_content.lower(), content.lower()).ratio()
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Source quality confidence (based on source confidence scores)
        source_quality = sum(source.confidence for source, _ in extracted_info) / len(extracted_info)
        
        # Combined confidence
        final_confidence = (source_confidence * 0.3 + avg_similarity * 0.4 + source_quality * 0.3)
        
        return min(final_confidence, 1.0)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for metadata"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['class', 'character', 'build']):
            return 'character_classes'
        elif any(word in query_lower for word in ['quest', 'mission', 'task']):
            return 'quests'
        elif any(word in query_lower for word in ['item', 'equipment', 'gear']):
            return 'items'
        elif any(word in query_lower for word in ['guide', 'how to', 'tutorial']):
            return 'guides'
        else:
            return 'general'
    
    async def _add_to_rag_database(self, verified_info: VerifiedInformation) -> bool:
        """Add verified information to the RAG database"""
        if not self.rag_service:
            self.logger.error("[KnowledgeEnhancement] RAG service not available")
            return False
        
        try:
            # Prepare document for RAG database
            document_text = f"Query: {verified_info.query}\n\nAnswer: {verified_info.verified_content}"
            
            # Prepare metadata
            metadata = {
                "source": "knowledge_enhancement",
                "query": verified_info.query,
                "confidence": verified_info.confidence_score,
                "verification_method": verified_info.verification_method,
                "num_sources": len(verified_info.sources),
                "query_type": verified_info.metadata.get("query_type", "general"),
                "timestamp": verified_info.metadata.get("timestamp", time.time()),
                "auto_enhanced": True
            }
            
            # Add to RAG database
            success = self.rag_service.add_documents(
                texts=[document_text],
                collection_name=self.enhancement_collection,
                metadatas=[metadata]
            )
            
            if success:
                self.logger.info(f"[KnowledgeEnhancement] ✅ Added verified information to collection '{self.enhancement_collection}'")
                
                # Also try to add to the main game collection if it's game-specific
                if verified_info.metadata.get("query_type") in ["character_classes", "quests", "items"]:
                    try:
                        # Determine target collection based on query
                        target_collection = self._determine_target_collection(verified_info.query)
                        if target_collection and target_collection != self.enhancement_collection:
                            self.rag_service.add_documents(
                                texts=[document_text],
                                collection_name=target_collection,
                                metadatas=[metadata]
                            )
                            self.logger.info(f"[KnowledgeEnhancement] ✅ Also added to game-specific collection '{target_collection}'")
                    except Exception as e:
                        self.logger.warning(f"[KnowledgeEnhancement] Could not add to game-specific collection: {e}")
                
                return True
            else:
                self.logger.error("[KnowledgeEnhancement] Failed to add document to RAG database")
                return False
                
        except Exception as e:
            self.logger.error(f"[KnowledgeEnhancement] Error adding to RAG database: {e}")
            return False
    
    def _determine_target_collection(self, query: str) -> Optional[str]:
        """Determine the target collection based on query content"""
        query_lower = query.lower()
        
        if 'everquest' in query_lower:
            return 'Everquest'
        elif any(game in query_lower for game in ['wow', 'world of warcraft']):
            return 'WorldOfWarcraft'
        elif any(game in query_lower for game in ['minecraft']):
            return 'Minecraft'
        
        return None
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge enhancement"""
        return {
            "auto_enhance_enabled": self.auto_enhance_enabled,
            "enhancement_collection": self.enhancement_collection,
            "min_sources_required": self.min_sources_for_verification,
            "confidence_threshold": self.min_confidence_threshold,
            "similarity_threshold": self.similarity_threshold
        } 