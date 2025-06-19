#!/usr/bin/env python3
"""
Smart RAG Service - Optimized for fast decision-making and retrieval
Key optimizations:
1. Intelligent query classification (intent detection)
2. Parallel retrieval from multiple sources
3. Embedding caching to avoid recomputation
4. Streaming responses for immediate feedback
5. Contextual decision-making about when to use RAG vs direct LLM
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import re
import hashlib

# Import the FactCheckService for web search fallback
from .danzar_factcheck import FactCheckService

@dataclass
class QueryIntent:
    """Represents the detected intent of a user query"""
    intent_type: str  # 'factual', 'conversational', 'game_specific', 'recent_context'
    confidence: float  # 0.0 to 1.0
    keywords: List[str]
    requires_rag: bool
    urgency: str  # 'immediate', 'normal', 'low'

class IntentType(Enum):
    CONVERSATIONAL = "conversational"  # Hi, thanks, etc - no RAG needed
    FACTUAL = "factual"  # Specific facts that need RAG
    GAME_SPECIFIC = "game_specific"  # Game rules, mechanics - needs game RAG
    RECENT_CONTEXT = "recent_context"  # About recent conversation - needs memory
    COMPLEX = "complex"  # Needs multiple sources

class SmartRAGService:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.SmartRAGService")
        
        # Core services
        self.rag_service = getattr(app_context, 'rag_service_instance', None)
        self.memory_service = getattr(app_context, 'memory_service_instance', None)
        self.model_client = getattr(app_context, 'model_client_instance', None)
        
        # Initialize conversation memory service
        try:
            from .conversation_memory import ConversationMemoryService
            self.conversation_memory = ConversationMemoryService(app_context)
            self.logger.info("[SmartRAGService] Conversation memory service initialized")
        except Exception as e:
            self.logger.warning(f"[SmartRAGService] Failed to initialize conversation memory: {e}")
            self.conversation_memory = None
        
        # Initialize web search fallback service - allow web search even without RAG service
        if self.model_client:
            self.fact_check_service = FactCheckService(
                rag_service=self.rag_service,  # Can be None
                model_client=self.model_client,
                app_context=app_context
            )
            self.web_search_enabled = app_context.global_settings.get("FACT_CHECK_SETTINGS", {}).get("enable_web_fact_check", False)
        else:
            self.fact_check_service = None
            self.web_search_enabled = False
        
        # Performance optimization settings
        self.cache_embeddings = app_context.global_settings.get("SMART_RAG_SETTINGS", {}).get("cache_embeddings", True)
        self.max_context_pieces = app_context.global_settings.get("SMART_RAG_SETTINGS", {}).get("max_context_pieces", 3)
        
        # Threading for parallel retrieval
        max_workers = app_context.global_settings.get("SMART_RAG_SETTINGS", {}).get("parallel_threads", 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Embedding cache for performance
        if self.cache_embeddings:
            self.embedding_cache = {}
            self.cache_lock = threading.Lock()
        
        # Intent detection patterns
        self.intent_patterns = {
            IntentType.CONVERSATIONAL: [
                r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
                r'\b(how are you|good morning|good night)\b'
            ],
            IntentType.GAME_SPECIFIC: [
                r'\b(rule|mechanic|stat|damage|skill|ability|quest|item|weapon|armor)\b',
                r'\b(how to|what does|what is)\b.*\b(spell|character|class|level)\b',
                r'\b(class|classes|spells?)\b.*\b(everquest|eq)\b',
                r'\b(everquest|eq)\b.*\b(class|classes|spells?|warrior|wizard|cleric)\b',
                r'\b(warrior|wizard|cleric|rogue|paladin|shadow knight|monk|bard|magician|necromancer|enchanter|druid|shaman|ranger)\b',
                r'\bwhat.*classes.*in\b'
            ],
            IntentType.RECENT_CONTEXT: [
                r'\b(what did|remember|earlier|before|previous|last time)\b',
                r'\b(we were talking about|you said|you mentioned)\b'
            ],
            IntentType.FACTUAL: [
                r'\b(what|who|when|where|why|how)\b',
                r'\b(explain|define|tell me about)\b'
            ]
        }
        
        self.logger.info(f"[SmartRAGService] Initialized with intelligent intent detection")
        if self.web_search_enabled:
            self.logger.info(f"[SmartRAGService] Web search fallback enabled")
        else:
            self.logger.info(f"[SmartRAGService] Web search fallback disabled")

    def classify_query_intent(self, query: str) -> QueryIntent:
        """
        Quickly classify user query intent using optimized pattern matching
        Returns classification in <1ms for maximum speed
        """
        query_lower = query.lower().strip()
        
        # Enhanced conversational patterns (bypass RAG completely)
        conversational_patterns = [
            r'\b(hi|hello|hey|sup|yo)\b',
            r'\b(thanks?|thank you|thx)\b', 
            r'\b(bye|goodbye|cya|see ya)\b',
            r'\b(how are you|what\'s up|wassup)\b',
            r'\b(ok|okay|cool|nice|awesome)\b',
            r'^(yes|yeah|yep|no|nope|nah)$',
            r'\b(lol|haha|lmao)\b'
        ]
        
        # Enhanced game-specific patterns (prioritize RAG)
        game_specific_patterns = [
            r'\bwhat.*classes.*in\b',
            r'\bclasses.*in.*everquest\b',
            r'\b(warrior|cleric|paladin|ranger|shadow knight|monk|bard|rogue|wizard|magician|necromancer|enchanter|druid|shaman)\b',
            r'\b(everquest|eq)\b.*\b(class|race|spell|zone|guild|raid)\b',
            r'\b(what|how|where|when).*\b(everquest|eq)\b',
            r'\b(game|rules|mechanics|stats|combat|equipment)\b',
            r'\b(level|experience|skill|ability|talent)\b'
        ]
        
        # Quick recent context patterns
        recent_context_patterns = [
            r'\b(what did|what was|earlier|before|just said|previously)\b',
            r'\b(remind me|tell me again|repeat)\b',
            r'\b(last time|recent|just now)\b'
        ]
        
        # Count pattern matches for fast scoring
        conversational_score = sum(1 for pattern in conversational_patterns 
                                 if re.search(pattern, query_lower))
        game_specific_score = sum(1 for pattern in game_specific_patterns 
                                if re.search(pattern, query_lower))
        recent_context_score = sum(1 for pattern in recent_context_patterns 
                                 if re.search(pattern, query_lower))
        
        # FIXED: Prioritize game-specific content over conversational greetings
        # If query contains both greeting AND game content, treat as game-specific
        if game_specific_score >= 1:
            return QueryIntent(
                intent_type='game_specific',
                confidence=min(0.9, 0.6 + game_specific_score * 0.15),
                keywords=[],
                requires_rag=True,
                urgency='normal'
            )
        elif conversational_score >= 1 and game_specific_score == 0:
            # Only treat as conversational if NO game-specific content
            return QueryIntent(
                intent_type='conversational',
                confidence=min(0.9, 0.5 + conversational_score * 0.2),
                keywords=[],
                requires_rag=False,
                urgency='immediate'
            )
        elif recent_context_score >= 1:
            return QueryIntent(
                intent_type='recent_context',
                confidence=min(0.8, 0.5 + recent_context_score * 0.2),
                keywords=[],
                requires_rag=True,
                urgency='normal'
            )
        else:
            # Default to factual with lower confidence
            return QueryIntent(
                intent_type='factual',
                confidence=0.4,  # Lower confidence for generic queries
                keywords=[],
                requires_rag=True,
                urgency='low'
            )

    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text to avoid recomputation"""
        if not self.cache_embeddings:
            return None
            
        with self.cache_lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        # Generate new embedding if not cached
        try:
            if not self.rag_service or not hasattr(self.rag_service, 'embedding_model'):
                return None
                
            embedding = self.rag_service.embedding_model.encode(text).tolist()
            
            # Cache the result
            with self.cache_lock:
                self.embedding_cache[text_hash] = embedding
                
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None

    def parallel_retrieval(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """
        Ultra-fast parallel retrieval with aggressive timeouts for speed
        """
        start_time = time.time()
        results = {}
        
        # Define retrieval tasks based on intent - much more selective
        tasks = []
        
        if intent.intent_type == 'game_specific':
            # Only get game knowledge for game-specific queries
            tasks.append(('game_knowledge', self._retrieve_game_knowledge, query))
        elif intent.intent_type == 'recent_context':
            # Only get recent memories for context queries
            tasks.append(('recent_memories', self._retrieve_recent_memories, query))
        else:
            # For other queries, get minimal context
            tasks.append(('game_knowledge', self._retrieve_game_knowledge, query))
        
        # Execute tasks in parallel with very aggressive timeout
        future_to_task = {}
        for task_name, task_func, task_query in tasks:
            future = self.executor.submit(task_func, task_query)
            future_to_task[future] = task_name
        
        # Collect results with 2 second timeout (was 5 seconds)
        try:
            for future in as_completed(future_to_task, timeout=2.0):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result if result else []
                except Exception as e:
                    self.logger.warning(f"Retrieval task {task_name} failed: {e}")
                    results[task_name] = []
        except TimeoutError:
            self.logger.warning("[SmartRAG] Some retrieval tasks timed out after 2s")
            # Cancel unfinished futures and set their results to empty lists
            for future, task_name in future_to_task.items():
                if not future.done():
                    future.cancel()
                    results[task_name] = []
        
        results['retrieval_time'] = time.time() - start_time
        return results

    def _retrieve_game_knowledge(self, query: str) -> List[str]:
        """Retrieve game-specific knowledge - optimized for speed"""
        self.logger.info(f"[SmartRAG:DEBUG] Starting game knowledge retrieval for query: '{query}'")
        
        try:
            if not self.rag_service:
                self.logger.warning("[SmartRAG:DEBUG] No RAG service available")
                return []
            
            collection = self.ctx.active_profile.rag_collection_name
            if not collection:
                self.logger.warning("[SmartRAG:DEBUG] No RAG collection configured")
                return []
            
            self.logger.info(f"[SmartRAG:DEBUG] Using collection: '{collection}'")
            self.logger.info(f"[SmartRAG:DEBUG] RAG service type: {type(self.rag_service).__name__}")
            
            # Try to query the collection directly - don't rely on collection_exists check
            # since it might have bugs but the actual query could still work
            self.logger.info(f"[SmartRAG:DEBUG] Calling rag_service.query with collection='{collection}', query='{query[:50]}...', n_results=2")
            
            results = self.rag_service.query(
                collection=collection,
                query_text=query,
                n_results=2
            )
            
            self.logger.info(f"[SmartRAG:DEBUG] RAG service returned: {type(results)} with {len(results) if results else 0} items")
            
            if results:
                self.logger.info(f"[SmartRAG:DEBUG] Successfully retrieved {len(results)} knowledge pieces from '{collection}'")
                for i, result in enumerate(results):
                    self.logger.info(f"[SmartRAG:DEBUG] Result {i+1}: {result[:100]}...")
                return results
            else:
                self.logger.warning(f"[SmartRAG:DEBUG] No results found in collection '{collection}' for query '{query}'")
                return []
                
        except Exception as e:
            self.logger.error(f"[SmartRAG:DEBUG] Game knowledge retrieval failed: {e}")
            import traceback
            self.logger.error(f"[SmartRAG:DEBUG] Full traceback: {traceback.format_exc()}")
            return []

    def _retrieve_chat_history(self, query: str) -> List[str]:
        """Retrieve relevant chat history - minimal for speed"""
        try:
            if not self.rag_service:
                return []
            
            history_collection = self.ctx.active_profile.memory_rag_history_collection_name
            if not history_collection:
                return []
            
            # Reduced from 2 to 1 result for speed
            return self.rag_service.query(
                collection=history_collection,
                query_text=query,
                n_results=1
            )
        except Exception as e:
            self.logger.error(f"Chat history retrieval failed: {e}")
            return []

    def _retrieve_recent_memories(self, query: str) -> List[str]:
        """Retrieve recent memories - optimized for speed"""
        try:
            if not self.memory_service:
                return []
            
            # Get only 2 memories instead of 3, lower threshold
            memories = self.memory_service.get_relevant_memories(
                query=query,
                top_k=2,
                min_importance=0.2  # Lower threshold for speed
            )
            
            return [memory.content for memory in memories[:2]]
        except Exception as e:
            self.logger.error(f"Recent memories retrieval failed: {e}")
            return []

    def smart_generate_response(self, query: str, user_name: str = "User") -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using intelligent routing and parallel processing
        Returns: (response_text, metadata)
        """
        start_time = time.time()
        self.logger.info(f"[SmartRAG:DEBUG] === Starting smart_generate_response for '{query}' from {user_name} ===")
        
        # Step 1: Check for conversation context and follow-up questions
        is_followup = False
        if self.conversation_memory:
            is_followup = self.conversation_memory.is_followup_question(user_name, query)
            if is_followup:
                self.logger.info(f"[SmartRAG:DEBUG] Detected follow-up question from {user_name}")
        
        # Step 2: Classify intent (very fast)
        self.logger.info(f"[SmartRAG:DEBUG] Classifying query intent...")
        intent = self.classify_query_intent(query)
        self.logger.info(f"[SmartRAG:DEBUG] Query intent: {intent.intent_type}, confidence: {intent.confidence:.2f}, requires_rag: {intent.requires_rag}")
        
        # Step 3: Handle conversational queries immediately (but not follow-ups)
        if intent.intent_type == 'conversational' and intent.confidence > 0.6 and not is_followup:
            self.logger.info(f"[SmartRAG:DEBUG] Using conversational response (skipping RAG)")
            response = self._generate_conversational_response(query)
            metadata = {
                'method': 'conversational',
                'intent': intent.intent_type,
                'processing_time': time.time() - start_time,
                'used_rag': False
            }
            # Store in conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_conversation_turn(user_name, query, response)
            return response, metadata
        
        # Step 4: Enhanced retrieval with conversation context and aggressive web search
        self.logger.info(f"[SmartRAG:DEBUG] Starting retrieval phase (intent requires RAG: {intent.requires_rag})")
        context_pieces = []
        conversation_context = ""
        web_search_result = None
        cross_referenced_info = None
        
        # Get conversation context if available
        if self.conversation_memory and (is_followup or intent.intent_type == 'recent_context'):
            conversation_context = self.conversation_memory.get_contextual_prompt(user_name, query)
            if conversation_context:
                self.logger.info(f"[SmartRAG:DEBUG] Using conversation context for {user_name}")
        
        if intent.requires_rag or is_followup:
            self.logger.info(f"[SmartRAG:DEBUG] Performing parallel retrieval...")
            retrieval_results = self.parallel_retrieval(query, intent)
            self.logger.info(f"[SmartRAG:DEBUG] Parallel retrieval completed with keys: {list(retrieval_results.keys())}")
            
            # Simplified context building - take only the best result
            if intent.intent_type == 'game_specific':
                game_knowledge = retrieval_results.get('game_knowledge', [])
                self.logger.info(f"[SmartRAG:DEBUG] Game knowledge retrieval returned {len(game_knowledge)} results")
                context_pieces.extend(game_knowledge[:1])  # Only 1 piece
            elif intent.intent_type == 'recent_context':
                recent_memories = retrieval_results.get('recent_memories', [])
                self.logger.info(f"[SmartRAG:DEBUG] Recent memories retrieval returned {len(recent_memories)} results")
                context_pieces.extend(recent_memories[:1])  # Only 1 piece
            else:  # factual
                game_knowledge = retrieval_results.get('game_knowledge', [])
                self.logger.info(f"[SmartRAG:DEBUG] Factual query - game knowledge returned {len(game_knowledge)} results")
                context_pieces.extend(game_knowledge[:1])  # Only 1 piece
            
            self.logger.info(f"[SmartRAG:DEBUG] Final context pieces count: {len(context_pieces)}")
            for i, piece in enumerate(context_pieces):
                self.logger.info(f"[SmartRAG:DEBUG] Context piece {i+1}: {piece[:100]}...")
            
            # AGGRESSIVE WEB SEARCH: Try web search for all game-specific queries that have no context
            # This should catch EverQuest class questions and other game-specific info
            should_web_search = (
                self.web_search_enabled and 
                self.fact_check_service and 
                hasattr(self.fact_check_service, '_search_web') and
                (not context_pieces or intent.intent_type == 'game_specific') and
                intent.confidence > 0.3  # Lower threshold
            )
            
            self.logger.info(f"[SmartRAG:DEBUG] Web search evaluation - enabled: {self.web_search_enabled}, fact_check_service: {self.fact_check_service is not None}, should_search: {should_web_search}")
            
            if should_web_search and self.fact_check_service:
                self.logger.info(f"[SmartRAG:DEBUG] Triggering web search for query: '{query}' (intent: {intent.intent_type})")
                try:
                    # Try cross-referenced search first
                    cross_referenced_info = self.fact_check_service._search_web(query, fact_check=True)
                    if cross_referenced_info:
                        context_pieces = [cross_referenced_info]  # Replace any existing context with verified web info
                        self.logger.info(f"[SmartRAG:DEBUG] Cross-referenced web search provided verified context")
                    else:
                        self.logger.info(f"[SmartRAG:DEBUG] Cross-referencing failed, trying basic web search")
                        # Fallback to regular web search
                        fallback_search = self.fact_check_service._search_web(query, fact_check=False)
                        if fallback_search:
                            context_pieces = [f"[UNVERIFIED WEB SEARCH]: {fallback_search}"]
                            self.logger.info(f"[SmartRAG:DEBUG] Using unverified web search as fallback")
                        else:
                            self.logger.warning(f"[SmartRAG:DEBUG] No web search results found for query")
                except Exception as e:
                    self.logger.error(f"[SmartRAG:DEBUG] Web search failed: {e}")
            
            # Generate context-aware response
            if context_pieces or conversation_context:
                self.logger.info(f"[SmartRAG:DEBUG] Generating contextual response with {len(context_pieces)} context pieces")
                response = self._generate_contextual_response(
                    query, context_pieces, intent, 
                    cross_referenced_info is not None,
                    conversation_context=conversation_context,
                    user_name=user_name
                )
                if cross_referenced_info:
                    method = 'web_search_verified'
                elif context_pieces and "[UNVERIFIED WEB SEARCH]" in context_pieces[0]:
                    method = 'web_search_unverified'
                elif conversation_context:
                    method = 'conversation_contextual'
                else:
                    method = 'rag_contextual'
            else:
                response = self._generate_direct_response(query)
                method = 'direct_llm'
        else:
            response = self._generate_direct_response(query)
            method = 'direct_llm'
            retrieval_results = {'retrieval_time': 0}
        
        metadata = {
            'method': method,
            'intent': intent.intent_type,
            'intent_confidence': intent.confidence,
            'processing_time': time.time() - start_time,
            'retrieval_time': retrieval_results.get('retrieval_time', 0),
            'used_rag': intent.requires_rag and context_pieces and cross_referenced_info is None and "[UNVERIFIED WEB SEARCH]" not in str(context_pieces),
            'used_web_search': cross_referenced_info is not None or any("[UNVERIFIED WEB SEARCH]" in str(piece) for piece in context_pieces),
            'cross_referenced': cross_referenced_info is not None
        }
        
        # Store conversation turn in memory
        if self.conversation_memory and response:
            self.conversation_memory.add_conversation_turn(user_name, query, response)
        
        return response, metadata

    def _generate_conversational_response(self, query: str) -> str:
        """Generate quick conversational responses without any RAG context"""
        query_lower = query.lower().strip()
        
        # Ensure purely conversational responses with no game context
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            return f"Hello! How can I help you with gaming today?"
        elif any(word in query_lower for word in ['thanks', 'thank you', 'thx']):
            return "You're welcome! Anything else I can help with?"
        elif any(word in query_lower for word in ['bye', 'goodbye', 'cya']):
            return "Goodbye! Happy gaming!"
        elif any(word in query_lower for word in ['ok', 'okay', 'cool', 'nice']):
            return "Great! Let me know if you need anything else."
        else:
            return "I'm here to help! What would you like to know?"

    def _generate_direct_response(self, query: str) -> str:
        """Generate response without RAG context - optimized for speed"""
        try:
            if not self.model_client:
                return "I'm having trouble processing that. Could you try rephrasing?"
                
            messages = [
                {"role": "system", "content": "You are a helpful gaming assistant. Keep responses brief and helpful."},
                {"role": "user", "content": query}
            ]
            
            return self.model_client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=100,  # Reduced from 150 for speed
                model=self.ctx.active_profile.conversational_llm_model
            )
        except Exception as e:
            self.logger.error(f"Direct response generation failed: {e}")
            return "I'm having trouble processing that. Could you try rephrasing?"

    def _generate_contextual_response(self, query: str, context_pieces: List[str], intent: QueryIntent, is_verified: bool, conversation_context: str = "", user_name: str = "User") -> str:
        """Generate response using retrieved context - optimized for speed with verification awareness"""
        try:
            if not self.model_client:
                return "I'm having trouble processing that. Could you try rephrasing?"
                
            # Use only the first context piece for speed
            context_text = context_pieces[0] if context_pieces else ""
            
            # Check if this is unverified web search content
            is_unverified_web = "[UNVERIFIED WEB SEARCH]" in context_text
            if is_unverified_web:
                context_text = context_text.replace("[UNVERIFIED WEB SEARCH]: ", "")
            
            # Truncate context for speed
            if len(context_text) > 600:  # Slightly increased for web content
                context_text = context_text[:600] + "..."
            
            # Create appropriate system prompt based on verification status
            if is_verified:
                system_prompt = "Answer using the verified information from multiple sources. Be confident in your response but keep it brief."
                reliability_note = "This information has been cross-referenced across multiple sources."
            elif is_unverified_web:
                system_prompt = "Answer using the provided information, but note that this is from web search and may need verification. Be helpful but cautious."
                reliability_note = "This information comes from web search and hasn't been fully verified."
            else:
                system_prompt = "Answer using the context from our knowledge base. Keep it brief and helpful."
                reliability_note = ""
            
            user_content = f"Context: {context_text}\n\nQuestion: {query}"
            if reliability_note and (is_verified or is_unverified_web):
                user_content += f"\n\nNote: {reliability_note}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            return self.model_client.generate(
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=150,  # Slightly increased for web search responses
                model=self.ctx.active_profile.conversational_llm_model
            )
        except Exception as e:
            self.logger.error(f"Contextual response generation failed: {e}")
            return self._generate_direct_response(query)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True) 