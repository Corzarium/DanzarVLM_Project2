#!/usr/bin/env python3
"""
Agentic RAG Service - Advanced retrieval with intelligent routing and multi-agent coordination
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.knowledge_enhancement_service import KnowledgeEnhancementService

class QueryType(Enum):
    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_MULTI_HOP = "complex_multi_hop"
    GAME_SPECIFIC = "game_specific"
    CONVERSATIONAL = "conversational"
    RECENT_CONTEXT = "recent_context"
    REQUIRES_WEB = "requires_web"

class RetrievalSource(Enum):
    VECTOR_DB = "vector_db"
    WEB_SEARCH = "web_search"
    MEMORY = "memory"
    HYBRID = "hybrid"

@dataclass
class QueryPlan:
    """Query execution plan created by the planner agent"""
    query_type: QueryType
    sub_queries: List[str]
    sources: List[RetrievalSource]
    confidence: float
    requires_iteration: bool
    max_iterations: int = 2

@dataclass
class RetrievalResult:
    """Result from a retrieval operation"""
    source: RetrievalSource
    query: str
    content: List[str]
    scores: List[float]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: float

class AgenticRAGService:
    """
    Agentic RAG service with intelligent routing and multi-agent coordination
    
    Architecture:
    1. Router Agent: Analyzes queries and plans retrieval strategy
    2. Query Planner: Breaks complex queries into sub-queries
    3. Retrieval Agents: Execute parallel retrieval from multiple sources
    4. Filter Agent: Re-ranks and validates results
    5. Generator Agent: Synthesizes final answer with context
    """
    
    def __init__(self, app_context):
        """Initialize the Agentic RAG Service with multi-agent coordination"""
        self.ctx = app_context
        self.logger = app_context.logger
        self.max_iterations = 2
        self.web_search_enabled = True
        
        # Services for different retrieval methods
        self.rag_service = None  # Will be set via set_services()
        self.fact_check_service = None  # Will be set via set_services()
        self.memory_service = None  # Will be set via set_services()
        
        # Add knowledge enhancement service
        self.knowledge_enhancement_service = KnowledgeEnhancementService(app_context)
        
        # Conversational memory (in-memory, separate from RAG)
        from services.conversation_buffer import ConversationBufferService
        self.conversation_buffer = ConversationBufferService(
            app_context, 
            max_turns=15,  # Keep last 15 conversation turns
            max_age_minutes=120  # Keep conversations for 2 hours
        )
        
        # Legacy conversation tracking (keeping for compatibility)
        self.conversation_history = []
        self.conversation_context = {}
        
        # Configuration attributes
        self.max_conversation_history = 5
        self.max_retrieval_time = 3.0
        
        self.logger.info("[AgenticRAG] Initialized with conversation buffer and multi-agent coordination")
    
    def set_services(self, rag_service=None, fact_check_service=None, memory_service=None):
        """Set the underlying services"""
        self.rag_service = rag_service
        self.fact_check_service = fact_check_service
        self.memory_service = memory_service
        
        # Set services for knowledge enhancement
        self.knowledge_enhancement_service.set_services(
            rag_service=rag_service,
            fact_check_service=fact_check_service
        )
        
        # Log available services
        services = []
        if rag_service: services.append("RAG")
        if fact_check_service: services.append("FactCheck")
        if memory_service: services.append("Memory")
        services.append("KnowledgeEnhancement")
        
        self.logger.info(f"[AgenticRAG] Services available: {', '.join(services)}")
    
    def _store_conversation_turn(self, query: str, response: str, user_name: str):
        """Store a conversation turn for context retention"""
        conversation_turn = {
            "user": user_name,
            "query": query,
            "response": response,
            "timestamp": time.time()
        }
        
        self.conversation_history.append(conversation_turn)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_conversation_history:
            self.conversation_history = self.conversation_history[-self.max_conversation_history:]
        
        # Extract and update conversation context
        self._extract_conversation_context(query, response)
        
        self.logger.info(f"[AgenticRAG:Memory] Stored conversation turn, history length: {len(self.conversation_history)}")
    
    def _extract_conversation_context(self, query: str, response: str):
        """Extract important context from the conversation"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract game context
        game_keywords = {
            'everquest': ['everquest', 'eq', 'norrath', 'plane of', 'steamfont', 'qeynos', 'freeport', 'classes'],
            'wow': ['world of warcraft', 'wow', 'azeroth'],
            'ff14': ['final fantasy', 'ff14', 'ffxiv', 'eorzea']
        }
        
        for game, keywords in game_keywords.items():
            if any(keyword in query_lower or keyword in response_lower for keyword in keywords):
                self.conversation_context['current_game'] = game
                self.conversation_context['game_confidence'] = 0.9
                self.conversation_context['last_topic'] = query_lower
                break
        
        # Extract topic context
        topics = {
            'classes': ['class', 'classes', 'warrior', 'cleric', 'wizard'],
            'zones': ['zone', 'area', 'location', 'steamfont', 'plane'],
            'leveling': ['level', 'leveling', 'experience', 'xp'],
            'items': ['item', 'equipment', 'weapon', 'armor']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                self.conversation_context['current_topic'] = topic
                break
        
        self.logger.info(f"[AgenticRAG:Memory] Updated context: {self.conversation_context}")
    
    def _get_contextual_query(self, query: str) -> str:
        """Enhance query with conversational context"""
        enhanced_query = query
        
        # Add game context if missing and we have conversation context
        current_game = self.conversation_context.get('current_game')
        if current_game and current_game not in query.lower():
            if current_game == 'everquest':
                enhanced_query = f"EverQuest {query}"
                self.logger.info(f"[AgenticRAG:Memory] Enhanced query with game context: '{enhanced_query}'")
        
        return enhanced_query
    
    def _get_recent_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context"""
        if not self.conversation_history:
            return "We haven't had any previous conversation yet. This is the start of our discussion!"
        
        # Get the most recent conversation entries
        recent_turns = self.conversation_history[-3:]  # Last 3 turns for better context
        
        # Check if user is asking about the last specific question
        summary_parts = []
        
        if self.conversation_context:
            if 'current_game' in self.conversation_context:
                game = self.conversation_context['current_game']
                summary_parts.append(f"We were discussing {game}")
            
            if 'current_topic' in self.conversation_context:
                topic = self.conversation_context['current_topic']
                summary_parts.append(f"specifically about {topic}")
        
        # Add recent conversation details
        if len(recent_turns) >= 2:
            last_query = recent_turns[-2]['query']  # Previous question (not the current one)
            summary_parts.append(f"Your last question was: '{last_query}'")
        
        if len(recent_turns) >= 1:
            current_query = recent_turns[-1]['query']
            if current_query not in ['What have we been talking about?', 'What was the last question I asked?']:
                summary_parts.append(f"We just discussed: '{current_query}'")
        
        if summary_parts:
            return '. '.join(summary_parts) + '.'
        
        # Fallback to basic summary
        last_turn = recent_turns[-1]
        return f"We were just discussing: '{last_turn['query']}'."
    
    def _get_last_question_specifically(self) -> str:
        """Get the last question the user asked specifically"""
        if not self.conversation_history or len(self.conversation_history) < 2:
            return "I don't have record of a previous question in our conversation."
        
        # Get the second-to-last entry (since the last entry is likely the current question)
        for i in range(len(self.conversation_history) - 2, -1, -1):
            turn = self.conversation_history[i]
            query = turn['query']
            
            # Skip meta-conversational questions
            if not any(phrase in query.lower() for phrase in [
                'what have we been', 'what was the last', 'what did i ask', 
                'what were we discussing', 'conversation about'
            ]):
                return f"Your last question was: '{query}'"
        
        return "I can see our conversation history but can't identify a specific previous question."
    
    def _router_agent(self, query: str, user_context: Dict[str, Any]) -> QueryPlan:
        """
        Router Agent: Analyzes query and determines optimal retrieval strategy with conversational context
        """
        self.logger.info(f"[AgenticRAG:Router] Analyzing query: '{query}'")
        
        # Enhanced conversational query detection - FIXED to include simple greetings
        query_lower = query.lower().strip()
        
        # Simple greeting patterns (should NOT trigger RAG)
        simple_greeting_patterns = [
            r'^(hi|hello|hey|sup|yo)(\s|$)',
            r'^(thanks?|thank you|thx)(\s|$)',
            r'^(bye|goodbye|cya|see ya)(\s|$)',
            r'^(how are you|what\'s up|wassup)(\?|\s|$)',
            r'^(ok|okay|cool|nice|awesome)(\s|$)',
            r'^(yes|yeah|yep|no|nope|nah)(\s|$)',
            r'^(lol|haha|lmao)(\s|$)'
        ]
        
        # Conversation context patterns
        conversational_patterns = [
            'what were we', 'what are we', 'were we talking', 'are we talking', 
            'what was that', 'our conversation', 'we discussed', 'just said',
            'what have we been', 'what was the last', 'last question',
            'what did i ask', 'what did we talk', 'previous question',
            'what were we discussing', 'what was our last', 'conversation about',
            'what have we talked', 'what was i asking', 'recent conversation'
        ]
        
        import re
        is_simple_greeting = any(re.match(pattern, query_lower) for pattern in simple_greeting_patterns)
        is_conversational_query = any(phrase in query_lower for phrase in conversational_patterns)
        
        # Handle simple greetings first (NO RAG needed)
        if is_simple_greeting:
            self.logger.info(f"[AgenticRAG:Router] Detected simple greeting - bypassing RAG")
            plan = QueryPlan(
                query_type=QueryType.CONVERSATIONAL,
                sub_queries=[query],
                sources=[],  # NO sources needed for simple greetings
                confidence=0.95,
                requires_iteration=False
            )
            self.logger.info(f"[AgenticRAG:Router] Plan: simple greeting, sources: [], confidence: 0.95")
            return plan
        
        # Handle conversation context queries
        if is_conversational_query:
            self.logger.info(f"[AgenticRAG:Router] Detected conversational context query")
            plan = QueryPlan(
                query_type=QueryType.CONVERSATIONAL,
                sub_queries=[query],
                sources=[RetrievalSource.MEMORY],
                confidence=0.95,
                requires_iteration=False
            )
            self.logger.info(f"[AgenticRAG:Router] Plan: conversational, sources: ['memory'], confidence: 0.95")
            return plan

        # Enhance query with conversational context for other queries
        contextual_query = self._get_contextual_query(query)
        query_lower = contextual_query.lower()
        
        # Log context usage
        if contextual_query != query:
            self.logger.info(f"[AgenticRAG:Memory] Enhanced query with game context: '{contextual_query}'")
            self.logger.info(f"[AgenticRAG:Router] Enhanced with context: '{contextual_query}'")
        
        # Detect query type using multiple signals
        if any(keyword in query_lower for keyword in ['what is', 'define', 'explain']):
            if any(game_term in query_lower for game_term in ['class', 'race', 'skill', 'quest', 'everquest']):
                query_type = QueryType.GAME_SPECIFIC
                sources = [RetrievalSource.VECTOR_DB, RetrievalSource.WEB_SEARCH]
            else:
                query_type = QueryType.SIMPLE_FACTUAL
                sources = [RetrievalSource.VECTOR_DB]
        
        elif any(keyword in query_lower for keyword in ['how to', 'strategy', 'best way', 'guide']):
            query_type = QueryType.COMPLEX_MULTI_HOP
            sources = [RetrievalSource.VECTOR_DB, RetrievalSource.WEB_SEARCH]
        
        elif any(keyword in query_lower for keyword in ['recent', 'latest', 'current', 'now']):
            query_type = QueryType.RECENT_CONTEXT
            sources = [RetrievalSource.MEMORY, RetrievalSource.WEB_SEARCH]
        
        else:
            # Default to game-specific with comprehensive search
            query_type = QueryType.GAME_SPECIFIC
            sources = [RetrievalSource.VECTOR_DB, RetrievalSource.WEB_SEARCH]
        
        # Calculate confidence based on keyword matches and context
        confidence = 0.7
        if any(game_term in query_lower for game_term in ['everquest', 'eq', 'class', 'race']):
            confidence += 0.2
        if user_context.get('recent_queries'):
            confidence += 0.1
        if self.conversation_context:
            confidence += 0.1  # Boost confidence when we have conversation context
        
        plan = QueryPlan(
            query_type=query_type,
            sub_queries=[contextual_query],  # Use contextual query
            sources=sources,
            confidence=min(confidence, 1.0),
            requires_iteration=query_type in [QueryType.COMPLEX_MULTI_HOP, QueryType.REQUIRES_WEB]
        )
        
        self.logger.info(f"[AgenticRAG:Router] Plan: {query_type.value}, sources: {[s.value for s in sources]}, confidence: {confidence:.2f}")
        return plan
    
    def _query_planner_agent(self, query: str, plan: QueryPlan) -> QueryPlan:
        """
        Query Planner Agent: Breaks complex queries into sub-queries for parallel processing
        """
        if plan.query_type == QueryType.COMPLEX_MULTI_HOP:
            # Break down complex queries
            if "best" in query.lower() and "class" in query.lower():
                plan.sub_queries = [
                    "What classes are available in EverQuest?",
                    "Which classes are best for beginners?",
                    "Class recommendations and strategies"
                ]
            elif "how to" in query.lower():
                plan.sub_queries = [
                    query,
                    f"Guide for {query.replace('how to', '').strip()}",
                    f"Tips and strategies for {query.replace('how to', '').strip()}"
                ]
        
        self.logger.info(f"[AgenticRAG:Planner] Generated {len(plan.sub_queries)} sub-queries")
        return plan
    
    async def _retrieval_agent_vector(self, query: str) -> RetrievalResult:
        """Retrieval Agent: Vector database search"""
        start_time = time.time()
        
        try:
            if not self.rag_service:
                return RetrievalResult(
                    source=RetrievalSource.VECTOR_DB,
                    query=query,
                    content=[],
                    scores=[],
                    metadata={"error": "RAG service not available"},
                    confidence=0.0,
                    timestamp=time.time()
                )
            
            # Get collection name from active profile if available
            collection = "Everquest"  # Default fallback
            if hasattr(self.ctx, 'active_profile') and self.ctx.active_profile:
                if hasattr(self.ctx.active_profile, 'rag_collection_name'):
                    collection = self.ctx.active_profile.rag_collection_name or "Everquest"
            
            self.logger.info(f"[AgenticRAG:VectorAgent] Querying collection '{collection}' for: '{query}'")
            
            results = self.rag_service.query(
                collection=collection,
                query_text=query,
                n_results=3
            )
            
            # Extract scores if available - handle both string and dict formats
            scores = []
            clean_content = []
            for result in results:
                if isinstance(result, dict):
                    # Handle dictionary format from our RAG service
                    score = result.get("score", 0.5)
                    text = result.get("text", str(result))
                    collection_name = result.get("collection", "unknown")
                    
                    scores.append(float(score))
                    # Format the content with collection info
                    clean_content.append(f"[{collection_name}] {text}")
                    
                elif isinstance(result, str) and result.startswith("[Score:"):
                    # Handle legacy string format with embedded scores
                    score_end = result.find("]")
                    if score_end > 0:
                        score_str = result[7:score_end]
                        try:
                            scores.append(float(score_str))
                        except:
                            scores.append(0.5)
                        clean_content.append(result[score_end+1:].strip())
                    else:
                        scores.append(0.5)
                        clean_content.append(result)
                else:
                    # Handle plain string format
                    scores.append(0.5)
                    clean_content.append(str(result))
            
            confidence = max(scores) if scores else 0.0
            
            return RetrievalResult(
                source=RetrievalSource.VECTOR_DB,
                query=query,
                content=clean_content,
                scores=scores,
                metadata={"collection": collection, "processing_time": time.time() - start_time},
                confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"[AgenticRAG:VectorAgent] Error: {e}")
            return RetrievalResult(
                source=RetrievalSource.VECTOR_DB,
                query=query,
                content=[],
                scores=[],
                metadata={"error": str(e)},
                confidence=0.0,
                timestamp=time.time()
            )
    
    async def _retrieval_agent_web(self, query: str) -> RetrievalResult:
        """Retrieval Agent: Web search"""
        start_time = time.time()
        
        try:
            if not self.fact_check_service or not self.web_search_enabled:
                return RetrievalResult(
                    source=RetrievalSource.WEB_SEARCH,
                    query=query,
                    content=[],
                    scores=[],
                    metadata={"error": "Web search not available"},
                    confidence=0.0,
                    timestamp=time.time()
                )
            
            self.logger.info(f"[AgenticRAG:WebAgent] Searching web for: '{query}'")
            
            # Try fact-checked search first
            if hasattr(self.fact_check_service, '_search_web'):
                result = self.fact_check_service._search_web(query, fact_check=True)
                if result:
                    return RetrievalResult(
                        source=RetrievalSource.WEB_SEARCH,
                        query=query,
                        content=[result],
                        scores=[0.8],  # High confidence for fact-checked results
                        metadata={"verified": True, "processing_time": time.time() - start_time},
                        confidence=0.8,
                        timestamp=time.time()
                    )
            
            # Fallback to regular search
            result = self.fact_check_service._search_web(query, fact_check=False)
            if result:
                return RetrievalResult(
                    source=RetrievalSource.WEB_SEARCH,
                    query=query,
                    content=[f"[UNVERIFIED] {result}"],
                    scores=[0.6],
                    metadata={"verified": False, "processing_time": time.time() - start_time},
                    confidence=0.6,
                    timestamp=time.time()
                )
            
            return RetrievalResult(
                source=RetrievalSource.WEB_SEARCH,
                query=query,
                content=[],
                scores=[],
                metadata={"no_results": True},
                confidence=0.0,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"[AgenticRAG:WebAgent] Error: {e}")
            return RetrievalResult(
                source=RetrievalSource.WEB_SEARCH,
                query=query,
                content=[],
                scores=[],
                metadata={"error": str(e)},
                confidence=0.0,
                timestamp=time.time()
            )
    
    async def _parallel_retrieval(self, plan: QueryPlan) -> List[RetrievalResult]:
        """
        Optimized parallel retrieval with timeout controls
        """
        self.logger.info(f"[AgenticRAG:ParallelRetrieval] Starting optimized retrieval")
        start_time = time.time()
        results = []
        
        try:
            # Simplified retrieval - focus on vector DB first
            if RetrievalSource.VECTOR_DB in plan.sources:
                for query in plan.sub_queries[:1]:  # Limit to first query for performance
                    try:
                        vector_result = await asyncio.wait_for(
                            self._retrieval_agent_vector(query), 
                            timeout=self.max_retrieval_time
                        )
                        if vector_result and vector_result.content:
                            results.append(vector_result)
                            self.logger.info(f"[AgenticRAG:ParallelRetrieval] Vector retrieval successful: {len(vector_result.content)} results")
                            break  # Exit early if we get good results
                    except asyncio.TimeoutError:
                        self.logger.warning(f"[AgenticRAG:ParallelRetrieval] Vector retrieval timeout for query: {query}")
                    except Exception as e:
                        self.logger.error(f"[AgenticRAG:ParallelRetrieval] Vector retrieval error: {e}")
            
            # Only try web search if vector search failed and it's enabled
            if not results and self.web_search_enabled and RetrievalSource.WEB_SEARCH in plan.sources:
                try:
                    web_result = await asyncio.wait_for(
                        self._retrieval_agent_web(plan.sub_queries[0]), 
                        timeout=self.max_retrieval_time
                    )
                    if web_result and web_result.content:
                        results.append(web_result)
                        self.logger.info("[AgenticRAG:ParallelRetrieval] Web search fallback successful")
                except asyncio.TimeoutError:
                    self.logger.warning("[AgenticRAG:ParallelRetrieval] Web search timeout")
                except Exception as e:
                    self.logger.error(f"[AgenticRAG:ParallelRetrieval] Web search error: {e}")
            
            processing_time = time.time() - start_time
            self.logger.info(f"[AgenticRAG:ParallelRetrieval] Completed in {processing_time:.2f}s with {len(results)} results")
            
        except Exception as e:
            self.logger.error(f"[AgenticRAG:ParallelRetrieval] Critical error: {e}")
        
        return results
    
    def _filter_agent(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Filter Agent: Re-ranks and validates results based on relevance and quality
        """
        self.logger.info(f"[AgenticRAG:Filter] Processing {len(results)} results")
        
        # Filter out empty results
        valid_results = [r for r in results if r.content and any(c.strip() for c in r.content)]
        
        # Sort by confidence score (descending)
        valid_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Take top 3 results
        top_results = valid_results[:3]
        
        self.logger.info(f"[AgenticRAG:Filter] Filtered to {len(top_results)} high-quality results")
        
        for i, result in enumerate(top_results):
            self.logger.info(f"[AgenticRAG:Filter] Result {i+1}: {result.source.value}, confidence={result.confidence:.3f}")
        
        return top_results
    
    def _generator_agent(self, query: str, results: List[RetrievalResult]) -> str:
        """
        Generator Agent: Creates final response from filtered results using advanced RAG techniques
        """
        if not results:
            return self._generate_helpful_fallback_response(query)
        
        # Advanced RAG: Prompt compression and context optimization
        # Based on research: eliminate unnecessary details while keeping essence
        compressed_context = self._compress_context(results, query)
        
        # Generate natural language response using context-aware prompting
        response = self._generate_natural_response(query, compressed_context, results)
        
        if response and len(response.strip()) > 10:
            self.logger.info(f"[AgenticRAG:Generator] Generated response of {len(response)} characters")
            return response.strip()
        
        # Enhanced fallback with better context synthesis
        self.logger.warning("[AgenticRAG:Generator] Fallback to context synthesis")
        return self._synthesize_from_context(query, results)
    
    def _compress_context(self, results: List[RetrievalResult], query: str) -> str:
        """
        Advanced RAG: Prompt compression to eliminate noise and focus on relevant information
        Based on research recommendation to remove unnecessary details while keeping essence
        """
        query_keywords = set(query.lower().split())
        compressed_pieces = []
        
        for result in results[:3]:  # Limit to top 3 for efficiency
            if not result.content:
                continue
            
            # Extract most relevant sentences based on query keywords
            relevant_parts = []
            for content_piece in result.content:
                if not content_piece:
                    continue
                
                # Score relevance based on keyword overlap
                content_lower = content_piece.lower()
                keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
                
                if keyword_matches > 0:
                    # Trim to essential information (max 150 chars per piece)
                    trimmed = content_piece[:150].strip()
                    if trimmed:
                        relevant_parts.append(trimmed)
            
            if relevant_parts:
                # Take the best relevant part
                best_part = relevant_parts[0]
                compressed_pieces.append(best_part)
        
        return "\n\n".join(compressed_pieces) if compressed_pieces else "No specific information found."
    
    def _generate_natural_response(self, query: str, context: str, results: List[RetrievalResult]) -> str:
        """
        Generate natural language response using available LLM services
        """
        # Use profile-based system prompt instead of hardcoded one
        system_prompt = self.ctx.active_profile.system_prompt_commentary
        
        # Construct optimized prompt
        user_prompt = f"""Context: {context}

Question: {query}

Please provide a direct, helpful answer based on the context. Keep it concise and focused."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Try multiple LLM access methods
            model_client = None
            
            # Method 1: Direct model client access
            if hasattr(self, 'model_client') and getattr(self, 'model_client', None):
                model_client = getattr(self, 'model_client')
            
            # Method 2: Through LLM service
            elif hasattr(self.ctx, 'llm_service_instance') and self.ctx.llm_service_instance:
                llm_service = self.ctx.llm_service_instance
                if hasattr(llm_service, 'model_client') and llm_service.model_client:
                    model_client = llm_service.model_client
            
            # Method 3: Through app context
            elif hasattr(self.ctx, 'model_client') and self.ctx.model_client:
                model_client = self.ctx.model_client
            
            if model_client:
                response = model_client.generate(
                    messages=messages,
                    temperature=0.3,  # Low temperature for factual responses
                    max_tokens=200,   # Reasonable limit for Discord
                    model=getattr(self.ctx.active_profile, 'conversational_llm_model', 'llama3.2:latest')
                )
                
                if response and response.strip():
                    # Clean up the response
                    cleaned_response = self._clean_response(response)
                    return cleaned_response
            
        except Exception as e:
            self.logger.error(f"[AgenticRAG:Generator] LLM generation failed: {e}")
        
        # If LLM fails, synthesize from context
        return self._synthesize_from_context(query, results)
    
    def _synthesize_from_context(self, query: str, results: List[RetrievalResult]) -> str:
        """
        Synthesize a coherent response when LLM is not available
        Advanced RAG fallback with better context understanding
        """
        if not results:
            # Instead of generic "I don't have information" responses, provide helpful conversational responses
            return self._generate_helpful_fallback_response(query)
        
        # Analyze query intent for better synthesis
        query_lower = query.lower()
        
        # Handle specific query types
        if 'class' in query_lower and ('list' in query_lower or 'names' in query_lower):
            return self._extract_class_list(results, query)
        elif 'what' in query_lower and 'about' in query_lower:
            return self._extract_general_info(results, query)
        elif any(word in query_lower for word in ['how', 'guide', 'strategy']):
            return self._extract_how_to_info(results, query)
        
        # General synthesis
        best_result = results[0]
        if best_result.content:
            content = best_result.content[0] if isinstance(best_result.content, list) else str(best_result.content)
            
            # Extract first meaningful sentence or two
            sentences = content.split('.')
            meaningful_sentences = []
            
            for sentence in sentences[:3]:  # Max 3 sentences
                sentence = sentence.strip()
                if len(sentence) > 10 and not sentence.isupper():  # Avoid headers/noise
                    meaningful_sentences.append(sentence)
            
            if meaningful_sentences:
                response = '. '.join(meaningful_sentences[:2]) + '.'
                # Ensure reasonable length
                if len(response) > 300:
                    response = response[:300] + '...'
                return response
        
        return f"I found some information about '{query}', but it needs clarification. Could you be more specific?"
    
    def _generate_helpful_fallback_response(self, query: str) -> str:
        """Generate helpful, natural responses when no RAG data is available"""
        query_lower = query.lower().strip()
        
        # Handle greetings
        if any(word in query_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm here to help with gaming questions. What would you like to know?"
        
        # Handle thanks
        if any(word in query_lower for word in ['thank', 'thanks', 'thx']):
            return "You're welcome! Feel free to ask me anything else about gaming."
        
        # Handle goodbyes
        if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'cya']):
            return "Goodbye! Happy gaming, and feel free to come back anytime!"
        
        # Handle how are you / status questions
        if any(phrase in query_lower for phrase in ['how are you', 'how are you doing', 'what\'s up', 'how\'s it going']):
            return "I'm doing great and ready to help with any gaming questions you have! What's on your mind?"
        
        # Handle game-specific questions
        if any(game in query_lower for game in ['everquest', 'eq', 'wow', 'world of warcraft', 'final fantasy', 'ff14']):
            return f"I'd love to help with that! While I don't have my full knowledge base loaded right now, I can still try to assist. What specifically would you like to know?"
        
        # Handle class/character questions
        if any(word in query_lower for word in ['class', 'classes', 'character', 'build', 'spec']):
            return "Character builds and classes are always interesting to discuss! What game are you playing, and what kind of character are you thinking about?"
        
        # Handle general gaming questions
        if any(word in query_lower for word in ['game', 'gaming', 'play', 'strategy', 'guide', 'help']):
            return "I'm here to help with gaming! What specific game or topic are you interested in? I'll do my best to assist."
        
        # Handle questions about items, spells, etc.
        if any(word in query_lower for word in ['item', 'spell', 'weapon', 'armor', 'quest']):
            return "That sounds like something I'd love to help with! Can you tell me which game you're asking about? That way I can give you the most relevant information."
        
        # Handle apologetic responses (like "I'm sorry")
        if any(word in query_lower for word in ['sorry', 'apologize', 'my bad']):
            return "No worries at all! Is there anything I can help you with?"
        
        # Default helpful response for other queries
        return f"That's an interesting question! While I don't have my full knowledge database available right now, I'm still here to help. Could you tell me a bit more about what you're looking for? I'll do my best to assist you."
    
    def _extract_class_list(self, results: List[RetrievalResult], query: str) -> str:
        """Extract class names from results when user asks for a list"""
        classes = set()
        common_classes = [
            'bard', 'berserker', 'cleric', 'druid', 'enchanter', 'magician', 
            'monk', 'necromancer', 'paladin', 'ranger', 'rogue', 'shadow knight',
            'shaman', 'warrior', 'wizard', 'beastlord'
        ]
        
        for result in results:
            if result.content:
                content_text = ' '.join(result.content) if isinstance(result.content, list) else str(result.content)
                content_lower = content_text.lower()
                
                for class_name in common_classes:
                    if class_name in content_lower:
                        classes.add(class_name.title())
        
        if classes:
            class_list = sorted(list(classes))
            return f"EverQuest classes: {', '.join(class_list)}."
        else:
            return "I found information about classes but couldn't extract a clear list. The data might contain technical details instead of class names."
    
    def _extract_general_info(self, results: List[RetrievalResult], query: str) -> str:
        """Extract general information when user asks 'what about X'"""
        best_content = []
        
        for result in results[:2]:  # Top 2 results only
            if result.content:
                content = result.content[0] if isinstance(result.content, list) else str(result.content)
                # Get first substantial sentence
                sentences = content.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if 20 < len(sentence) < 200 and not sentence.isupper():
                        best_content.append(sentence)
                        break
        
        if best_content:
            return '. '.join(best_content) + '.'
        else:
            return f"I found some technical information about '{query}' but couldn't extract a clear summary."
    
    def _extract_how_to_info(self, results: List[RetrievalResult], query: str) -> str:
        """Extract how-to or guide information"""
        steps = []
        
        for result in results:
            if result.content:
                content_text = ' '.join(result.content) if isinstance(result.content, list) else str(result.content)
                
                # Look for step-like patterns
                if any(word in content_text.lower() for word in ['step', 'first', 'then', 'next', 'guide']):
                    trimmed = content_text[:200].strip()
                    if trimmed:
                        steps.append(trimmed)
        
        if steps:
            return steps[0] + ('...' if len(steps[0]) >= 200 else '')
        else:
            return f"I found information about '{query}' but couldn't extract clear guidance."
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response for better presentation"""
        if not response:
            return response
        
        # Remove common LLM artifacts
        response = response.strip()
        
        # Remove redundant prefixes
        prefixes_to_remove = [
            "Based on the context provided,",
            "According to the information,",
            "From the context,",
            "The context indicates that"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure it doesn't exceed Discord limits (we'll handle this in the Discord service too)
        if len(response) > 1900:  # Leave room for other text
            response = response[:1900] + "..."
        
        return response
    
    def _reflection_agent(self, query: str, results: List[RetrievalResult], response: str) -> Dict[str, Any]:
        """
        Reflection Agent: Evaluates response quality and determines if refinement is needed
        """
        self.logger.info(f"[AgenticRAG:Reflection] Evaluating response quality for query: '{query}'")
        
        # Calculate overall confidence
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            max_confidence = max(r.confidence for r in results)
        else:
            avg_confidence = 0.0
            max_confidence = 0.0
        
        # Evaluate response quality
        response_length = len(response.split())
        has_specific_info = any(keyword in response.lower() for keyword in [
            'class', 'race', 'spell', 'item', 'zone', 'level', 'skill'
        ])
        
        # Quality score calculation
        quality_score = 0.0
        quality_factors = []
        
        # Factor 1: Confidence scores
        if max_confidence > 0.7:
            quality_score += 0.3
            quality_factors.append("high_confidence_results")
        elif max_confidence > 0.5:
            quality_score += 0.2
            quality_factors.append("medium_confidence_results")
        
        # Factor 2: Response specificity
        if has_specific_info and response_length > 10:
            quality_score += 0.3
            quality_factors.append("specific_information")
        elif response_length > 5:
            quality_score += 0.1
            quality_factors.append("basic_information")
        
        # Factor 3: Multiple sources
        unique_sources = set(r.source for r in results)
        if len(unique_sources) > 1:
            quality_score += 0.2
            quality_factors.append("multiple_sources")
        elif len(unique_sources) == 1:
            quality_score += 0.1
            quality_factors.append("single_source")
        
        # Factor 4: Result count
        if len(results) >= 2:
            quality_score += 0.2
            quality_factors.append("sufficient_results")
        
        needs_refinement = quality_score < 0.6
        
        reflection_metadata = {
            "quality_score": quality_score,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "needs_refinement": needs_refinement,
            "quality_factors": quality_factors,
            "response_length": response_length,
            "unique_sources": len(unique_sources),
            "result_count": len(results)
        }
        
        self.logger.info(f"[AgenticRAG:Reflection] Quality score: {quality_score:.2f}, needs refinement: {needs_refinement}")
        return reflection_metadata
    
    def _query_refinement_agent(self, original_query: str, previous_results: List[RetrievalResult]) -> List[str]:
        """
        Query Refinement Agent: Generates alternative queries when initial results are poor
        """
        self.logger.info(f"[AgenticRAG:Refinement] Generating refined queries for: '{original_query}'")
        
        refined_queries = []
        query_lower = original_query.lower()
        
        # Strategy 1: Add context keywords
        if 'class' in query_lower and 'everquest' not in query_lower:
            refined_queries.append(f"{original_query} in EverQuest")
        
        # Strategy 2: Simplify complex queries
        if len(original_query.split()) > 5:
            # Extract key terms
            key_terms = []
            for word in ['class', 'race', 'level', 'skill', 'quest', 'zone', 'item']:
                if word in query_lower:
                    key_terms.append(word)
            if key_terms:
                refined_queries.append(f"EverQuest {' '.join(key_terms)}")
        
        # Strategy 3: Alternative phrasings
        replacements = {
            'what classes are': 'list of classes',
            'tell me about': 'information about',
            'how to': 'guide for',
            'best way': 'strategy for'
        }
        
        for old_phrase, new_phrase in replacements.items():
            if old_phrase in query_lower:
                refined_query = original_query.lower().replace(old_phrase, new_phrase)
                refined_queries.append(refined_query.title())
        
        # Strategy 4: Add specific game terms
        if not any(term in query_lower for term in ['eq', 'everquest', 'norrath']):
            refined_queries.append(f"EverQuest {original_query}")
        
        # Limit to 2 refined queries to avoid over-processing
        refined_queries = refined_queries[:2]
        
        self.logger.info(f"[AgenticRAG:Refinement] Generated {len(refined_queries)} refined queries")
        for i, query in enumerate(refined_queries):
            self.logger.info(f"[AgenticRAG:Refinement] Refined query {i+1}: '{query}'")
        
        return refined_queries
    
    async def _iterative_retrieval(self, query: str, initial_plan: QueryPlan) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Optimized iterative retrieval with limited iterations for performance
        """
        self.logger.info(f"[AgenticRAG:Iterative] Starting fast iterative retrieval for: '{query}'")
        
        iteration_metadata = {
            "iterations": 0,
            "total_results": 0,
            "refinement_applied": False,
            "final_quality_score": 0.0
        }
        
        # Single iteration approach for performance
        results = await self._parallel_retrieval(initial_plan)
        iteration_metadata["iterations"] = 1
        iteration_metadata["total_results"] = len(results)
        
        # Only refine if we got zero results and it's a simple refinement
        if not results and self.max_iterations > 1:
            self.logger.info("[AgenticRAG:Iterative] No results, trying one simple refinement")
            
            # Simple refinement: add "EverQuest" if not present
            if 'everquest' not in query.lower() and 'eq' not in query.lower():
                refined_query = f"EverQuest {query}"
                self.logger.info(f"[AgenticRAG:Iterative] Trying refined query: '{refined_query}'")
                
                # Create simple refined plan
                refined_plan = QueryPlan(
                    query_type=initial_plan.query_type,
                    sub_queries=[refined_query],
                    sources=[RetrievalSource.VECTOR_DB],  # Only vector DB for speed
                    confidence=initial_plan.confidence,
                    requires_iteration=False
                )
                
                refined_results = await self._parallel_retrieval(refined_plan)
                
                if refined_results:
                    self.logger.info(f"[AgenticRAG:Iterative] Refinement successful: {len(refined_results)} results")
                    results.extend(refined_results)
                    iteration_metadata["iterations"] = 2
                    iteration_metadata["total_results"] += len(refined_results)
                    iteration_metadata["refinement_applied"] = True
        
        # Quick quality evaluation
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            iteration_metadata["final_quality_score"] = min(avg_confidence + 0.2, 1.0)  # Boost for having results
        
        self.logger.info(f"[AgenticRAG:Iterative] Completed {iteration_metadata['iterations']} iterations with {iteration_metadata['total_results']} results")
        
        return results, iteration_metadata
    
    async def smart_retrieve(self, query: str, user_name: str = "User") -> Tuple[str, Dict[str, Any]]:
        """
        Main agentic retrieval method with conversation buffer for short-term memory
        and automatic knowledge enhancement
        """
        start_time = time.time()
        self.logger.info(f"[AgenticRAG:Main] Starting retrieval for: '{query}' from {user_name}")
        
        # Step 1: Check if this is a conversational query that should use memory buffer
        if self.conversation_buffer.is_conversational_query(query):
            self.logger.info("[AgenticRAG:Main] Detected conversational query - using conversation buffer")
            
            response_text, metadata = self.conversation_buffer.handle_conversational_query(query, user_name)
            
            # Add this turn to conversation history (but don't include the response yet to avoid recursion)
            self.conversation_buffer.add_conversation_turn(user_name, query, response_text)
            
            # Update legacy conversation tracking for compatibility
            self._store_conversation_turn(query, response_text, user_name)
            
            return response_text, metadata
        
        # Step 2: For non-conversational queries, use the full agentic RAG pipeline
        query_plan = self._router_agent(query, {"user_name": user_name, "recent_queries": [], "timestamp": time.time()})
        self.logger.info(f"[AgenticRAG:Main] Query plan: {query_plan.query_type.value} | Strategies: {[s.value for s in query_plan.sources]}")
        
        # Step 3: Iterative Retrieval with Refinement
        try:
            all_results, iteration_metadata = await self._iterative_retrieval(query, query_plan)
        except Exception as e:
            self.logger.error(f"[AgenticRAG:Main] Iterative retrieval failed: {e}")
            all_results = []
            iteration_metadata = {"iterations": 0, "total_results": 0, "refinement_applied": False}
        
        # Step 4: Filter and Re-rank Results
        filtered_results = self._filter_agent(all_results, query)
        self.logger.info(f"[AgenticRAG:Main] Filtered to {len(filtered_results)} high-quality results")
        
        # Step 5: Knowledge Enhancement - Try to improve knowledge base if results are insufficient
        enhanced_info = None
        if len(filtered_results) == 0 or (filtered_results and max(r.confidence for r in filtered_results) < 0.5):
            self.logger.info("[AgenticRAG:Main] Results insufficient - attempting knowledge enhancement")
            try:
                # Convert filtered_results to the format expected by knowledge enhancement
                rag_results = []
                for result in filtered_results:
                    for i, content in enumerate(result.content):
                        score = result.scores[i] if i < len(result.scores) else result.confidence
                        rag_results.append({
                            'text': content,
                            'score': score,
                            'collection': result.metadata.get('collection', 'unknown')
                        })
                
                enhanced_info = await self.knowledge_enhancement_service.enhance_knowledge_if_needed(query, rag_results, user_name)
                
                if enhanced_info:
                    self.logger.info(f"[AgenticRAG:Main] ✅ Knowledge enhanced with confidence: {enhanced_info.confidence_score:.2f}")
                    
                    # Create a new result from the enhanced information
                    enhanced_result = RetrievalResult(
                        source=RetrievalSource.WEB_SEARCH,
                        query=query,
                        content=[enhanced_info.verified_content],
                        scores=[enhanced_info.confidence_score],
                        metadata={
                            "source": "knowledge_enhancement",
                            "num_sources": len(enhanced_info.sources),
                            "verification_method": enhanced_info.verification_method,
                            "auto_enhanced": True
                        },
                        confidence=enhanced_info.confidence_score,
                        timestamp=time.time()
                    )
                    
                    # Add the enhanced result to our filtered results
                    filtered_results.append(enhanced_result)
                    self.logger.info("[AgenticRAG:Main] Added enhanced information to results")
                
            except Exception as e:
                self.logger.error(f"[AgenticRAG:Main] Knowledge enhancement failed: {e}")
        
        # Step 6: Generate Response
        if filtered_results:
            response_text = self._generator_agent(query, filtered_results)
        else:
            response_text = self._generate_helpful_fallback_response(query)
            self.logger.warning(f"[AgenticRAG:Main] No results after filtering and enhancement - using fallback response")
        
        # Step 7: Final Reflection on Generated Response
        reflection_metadata = self._reflection_agent(query, filtered_results, response_text)
        
        # Step 8: Store conversation turn in both buffer and legacy system
        self.conversation_buffer.add_conversation_turn(user_name, query, response_text)
        self._store_conversation_turn(query, response_text, user_name)
        
        # Compile comprehensive metadata
        processing_time = time.time() - start_time
        metadata = {
            "method": "agentic_rag_with_conversation_buffer",
            "processing_time": processing_time,
            "query_plan": {
                "intent": query_plan.query_type.value,
                "strategies": [s.value for s in query_plan.sources],
                "sub_queries": len(query_plan.sub_queries)
            },
            "retrieval": {
                "iterations": iteration_metadata["iterations"],
                "total_results": iteration_metadata["total_results"],
                "filtered_results": len(filtered_results),
                "refinement_applied": iteration_metadata["refinement_applied"],
                "final_quality_score": iteration_metadata.get("final_quality_score", 0.0)
            },
            "knowledge_enhancement": {
                "attempted": enhanced_info is not None or len(filtered_results) == 0,
                "successful": enhanced_info is not None,
                "confidence": enhanced_info.confidence_score if enhanced_info else 0.0,
                "sources_used": len(enhanced_info.sources) if enhanced_info else 0
            },
            "reflection": reflection_metadata,
            "conversation_buffer_stats": self.conversation_buffer.buffer.get_stats(),
            "agents_used": ["conversation_buffer", "router", "planner", "retrieval", "filter", "knowledge_enhancement", "generator", "reflection"],
            "success": len(filtered_results) > 0
        }
        
        # Enhanced logging
        self.logger.info(f"[AgenticRAG:Main] Retrieval completed in {processing_time:.2f}s")
        self.logger.info(f"[AgenticRAG:Main] Final quality score: {reflection_metadata['quality_score']:.2f}")
        self.logger.info(f"[AgenticRAG:Main] Used {iteration_metadata['iterations']} iterations, {len(filtered_results)} final results")
        if enhanced_info:
            self.logger.info(f"[AgenticRAG:Main] Knowledge enhancement successful with {len(enhanced_info.sources)} sources")
        
        return response_text, metadata 