import logging
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import yaml
import os
import asyncio
import threading
import time
import uuid

class MemoryManager:
    """
    Enhanced Memory-Augmented RAG Manager for DanzarAI combining:
    - Short-Term Memory (STM): In-RAM buffer for recent conversation turns
    - Long-Term Memory (LTM): Qdrant-based persistent storage with LLM summarization
    - Cross-session retrieval and consolidation
    
    Based on Memory-Augmented RAG architecture with proper session management.
    """
    
    def __init__(self, app_context=None, config_path: str = "config/memory_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.app_context = app_context
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.embedding_model = None
        self.qdrant_client = None
        self.stm_buffer = deque(maxlen=self.config.get('stm_max_turns', 100))
        self.ltm_buffer = deque(maxlen=50)  # In-memory LTM cache
        
        # Session management
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.session_turns = []
        
        # Threading and async support
        self.lock = threading.RLock()
        self._initialized = False
        self._cleanup_task = None
        self._consolidation_task = None
        
        # Initialize the memory system
        self._initialize_memory_system()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load memory configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"✅ Loaded memory config from {self.config_path}")
                return config
            else:
                # Default configuration
                default_config = {
                    'qdrant_url': 'http://localhost:6333',
                    'stm_collection': 'danzar_stm',
                    'ltm_collection': 'danzar_ltm',
                    'stm_max_turns': 100,
                    'stm_decay_minutes': 30,
                    'stm_decay_threshold': 0.05,
                    'stm_retrieve_k': 10,
                    'ltm_retrieve_k': 5,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'auto_cleanup_interval': 300,  # 5 minutes
                    'ltm_summary_threshold': 20,  # Create LTM summary after N turns
                    'consolidation_interval': 600,  # 10 minutes
                    'enable_llm_summarization': True,
                    'max_session_duration_hours': 24,
                }
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load memory config: {e}")
            return {}
    
    def _initialize_memory_system(self):
        """Initialize the complete memory system."""
        try:
            # Initialize embedding model
            self._initialize_embedding_model()
            
            # Initialize Qdrant client
            self._initialize_qdrant()
            
            # Initialize memory collections
            self._initialize_memory_collections()
            
            # Start background tasks
            self._start_cleanup_task()
            self._start_consolidation_task()
            
            self._initialized = True
            self.logger.info("✅ Enhanced Memory Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}", exc_info=True)
            self._initialized = False

    async def initialize(self) -> bool:
        """Async initialization method for compatibility."""
        return self._initialized

    async def store_interaction(self, username: str, user_message: str, bot_response: str, 
                               game_context: str = None) -> bool:
        """
        Store a complete interaction (user message + bot response) in memory.
        
        Args:
            username: Name of the user
            user_message: User's input message
            bot_response: Bot's response
            game_context: Optional game context
            
        Returns:
            True if stored successfully
        """
        try:
            # Store user message
            user_metadata = {
                'user': username,
                'user_id': username,
                'type': 'user_input',
                'session_id': self.current_session_id,
                'game_context': game_context,
                'timestamp': datetime.now().isoformat()
            }
            self.upsert_stm(user_message, user_metadata)
            
            # Store bot response
            bot_metadata = {
                'user': 'DanzarAI',
                'user_id': username,
                'type': 'bot_response',
                'session_id': self.current_session_id,
                'game_context': game_context,
                'timestamp': datetime.now().isoformat()
            }
            self.upsert_stm(bot_response, bot_metadata)
            
            # Add to session turns for consolidation
            self.session_turns.append({
                'user': username,
                'user_message': user_message,
                'bot_response': bot_response,
                'game_context': game_context,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.debug(f"Stored interaction for {username} in session {self.current_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store interaction: {e}")
            return False

    async def get_conversation_context(self, username: str = None, 
                                     recent_turns: int = 10) -> List[Dict]:
        """
        Get conversation context including both STM and LTM memories.
        
        Args:
            username: Optional username to filter by
            recent_turns: Number of recent turns to include
            
        Returns:
            List of conversation context dictionaries
        """
        if not self._initialized:
            return []
        
        try:
            context = []
            
            # Get recent STM context
            stm_context = self._get_stm_context(username, recent_turns)
            context.extend(stm_context)
            
            # Get relevant LTM context
            if username:
                ltm_context = await self._get_ltm_context(username, limit=5)
                context.extend(ltm_context)
            
            # Sort by timestamp
            context.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return context[:recent_turns * 2]  # Limit total context
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return []

    async def get_conversation_summary(self, username: str) -> str:
        """Get a summary of conversation history for a user."""
        try:
            # Get recent interactions
            context = await self.get_conversation_context(username, recent_turns=20)
            
            if not context:
                return "No recent conversation history found."
            
            # Create summary using LLM if available
            if self.app_context and hasattr(self.app_context, 'llm_service'):
                summary_prompt = self._create_summary_prompt(context)
                summary = await self.app_context.llm_service.generate_response(summary_prompt)
                return summary or "Conversation history available but unable to generate summary."
            else:
                # Fallback to simple summary
                return self._create_simple_summary(context)
                
        except Exception as e:
            self.logger.error(f"Failed to get conversation summary: {e}")
            return "Unable to retrieve conversation summary."

    async def clear_user_memory(self, username: str) -> bool:
        """Clear memory for a specific user."""
        try:
            with self.lock:
                # Clear from STM buffer
                self.stm_buffer = deque([
                    m for m in self.stm_buffer 
                    if m.get('metadata', {}).get('user_id') != username
                ], maxlen=self.config.get('stm_max_turns', 100))
                
                # Clear from Qdrant (this would require more complex filtering)
                # For now, we'll just clear the buffer
                
                self.logger.info(f"Cleared memory for user: {username}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear user memory: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            stats = self.get_memory_stats()
            
            # Add session info
            stats.update({
                'current_session_id': self.current_session_id,
                'session_start_time': self.session_start_time.isoformat(),
                'session_turns_count': len(self.session_turns),
                'session_duration_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

    async def cleanup(self):
        """Async cleanup method."""
        self.shutdown()

    def _get_stm_context(self, username: str = None, recent_turns: int = 10) -> List[Dict]:
        """Get recent STM context."""
        try:
            with self.lock:
                recent_memories = list(self.stm_buffer)[-recent_turns:]
                
                if username:
                    recent_memories = [
                        m for m in recent_memories 
                        if m.get('metadata', {}).get('user_id') == username
                    ]
                
                return recent_memories
                
        except Exception as e:
            self.logger.error(f"Failed to get STM context: {e}")
            return []

    async def _get_ltm_context(self, username: str, limit: int = 5) -> List[Dict]:
        """Get relevant LTM context for a user."""
        try:
            # Create a query to find relevant LTM memories
            query = f"user:{username} conversation history"
            
            ltm_memories = self.retrieve_memory(
                query=query,
                memory_type='ltm',
                limit=limit
            )
            
            return ltm_memories
            
        except Exception as e:
            self.logger.error(f"Failed to get LTM context: {e}")
            return []

    def _create_summary_prompt(self, context: List[Dict]) -> str:
        """Create a prompt for LLM summarization."""
        # Format context for summarization
        conversation_text = []
        for entry in context[-20:]:  # Last 20 entries
            user = entry.get('metadata', {}).get('user', 'Unknown')
            text = entry.get('text', '')
            conversation_text.append(f"{user}: {text}")
        
        conversation_block = "\n".join(conversation_text)
        
        # Use profile-based system prompt instead of hardcoded one
        system_prompt = self.app_context.active_profile.system_prompt_commentary if self.app_context and hasattr(self.app_context, 'active_profile') else "You are DANZAR, a vision-capable gaming assistant with a witty personality."
        
        prompt = f"""{system_prompt}

Create a brief, engaging summary of this conversation history. Focus on:
- Key topics discussed
- User preferences or patterns
- Important decisions or plans made
- Any recurring themes

Keep it concise (2-3 sentences) and maintain your signature snarky tone.

Conversation History:
{conversation_block}

Summary:"""
        
        return prompt

    def _create_simple_summary(self, context: List[Dict]) -> str:
        """Create a simple summary without LLM."""
        try:
            # Extract key information
            users = set()
            topics = set()
            
            for entry in context:
                user = entry.get('metadata', {}).get('user', 'Unknown')
                users.add(user)
                
                text = entry.get('text', '').lower()
                # Look for common topics
                if any(word in text for word in ['game', 'play', 'quest', 'level', 'item']):
                    topics.add('gaming')
                if any(word in text for word in ['help', 'question', 'how', 'what']):
                    topics.add('help requests')
            
            summary_parts = []
            if users:
                summary_parts.append(f"Conversation with: {', '.join(users)}")
            if topics:
                summary_parts.append(f"Topics: {', '.join(topics)}")
            summary_parts.append(f"Total interactions: {len(context)}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create simple summary: {e}")
            return "Conversation history available."

    def _start_consolidation_task(self):
        """Start background task for session consolidation."""
        def consolidation_worker():
            while self._initialized:
                try:
                    self._consolidate_session_if_needed()
                    time.sleep(self.config.get('consolidation_interval', 600))
                except Exception as e:
                    self.logger.error(f"Error in consolidation worker: {e}")
                    time.sleep(60)  # Wait before retrying
        
        consolidation_thread = threading.Thread(target=consolidation_worker, daemon=True)
        consolidation_thread.start()
        self.logger.info("✅ Memory consolidation task started")

    def _consolidate_session_if_needed(self):
        """Consolidate current session to LTM if conditions are met."""
        try:
            # Check if consolidation is needed
            threshold = self.config.get('ltm_summary_threshold', 20)
            max_duration = self.config.get('max_session_duration_hours', 24)
            
            current_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
            
            should_consolidate = (
                len(self.session_turns) >= threshold or 
                current_duration >= max_duration
            )
            
            if should_consolidate and self.session_turns:
                self._consolidate_session()
                
        except Exception as e:
            self.logger.error(f"Failed to check consolidation: {e}")

    def _consolidate_session(self):
        """Consolidate current session to LTM."""
        try:
            if not self.session_turns:
                return
            
            # Create session summary
            summary = self._create_session_summary_llm()
            
            if summary:
                # Store in LTM
                metadata = {
                    'type': 'session_summary',
                    'session_id': self.current_session_id,
                    'session_start': self.session_start_time.isoformat(),
                    'session_end': datetime.now().isoformat(),
                    'turn_count': len(self.session_turns),
                    'users': list(set(turn.get('user') for turn in self.session_turns))
                }
                
                self.upsert_ltm(summary, metadata, weight=2.0)  # Higher weight for summaries
                
                self.logger.info(f"Consolidated session {self.current_session_id} to LTM")
            
            # Start new session
            self._start_new_session()
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate session: {e}")

    def _create_session_summary_llm(self) -> str:
        """Create LLM-powered session summary."""
        try:
            if not self.session_turns:
                return ""
            
            # Format session data for LLM
            session_text = []
            for turn in self.session_turns:
                user = turn.get('user', 'Unknown')
                user_msg = turn.get('user_message', '')
                bot_msg = turn.get('bot_response', '')
                game_ctx = turn.get('game_context', '')
                
                session_text.append(f"{user}: {user_msg}")
                session_text.append(f"DanzarAI: {bot_msg}")
                if game_ctx:
                    session_text.append(f"[Game Context: {game_ctx}]")
            
            session_block = "\n".join(session_text)
            
            # Create summary prompt
            summary_prompt = self._create_summary_prompt([{'text': session_block}])
            
            # Use LLM if available
            if self.app_context and hasattr(self.app_context, 'llm_service'):
                # This would need to be async, but we're in a sync context
                # For now, return a simple summary
                return self._create_simple_session_summary()
            else:
                return self._create_simple_session_summary()
                
        except Exception as e:
            self.logger.error(f"Failed to create LLM session summary: {e}")
            return self._create_simple_session_summary()

    def _create_simple_session_summary(self) -> str:
        """Create a simple session summary without LLM."""
        try:
            if not self.session_turns:
                return ""
            
            # Extract key information
            users = set()
            topics = set()
            game_contexts = set()
            
            for turn in self.session_turns:
                user = turn.get('user', 'Unknown')
                users.add(user)
                
                user_msg = turn.get('user_message', '').lower()
                bot_msg = turn.get('bot_response', '').lower()
                game_ctx = turn.get('game_context', '')
                
                if game_ctx:
                    game_contexts.add(game_ctx)
                
                # Look for topics
                all_text = f"{user_msg} {bot_msg}"
                if any(word in all_text for word in ['game', 'play', 'quest', 'level', 'item']):
                    topics.add('gaming')
                if any(word in all_text for word in ['help', 'question', 'how', 'what']):
                    topics.add('help requests')
                if any(word in all_text for word in ['friend', 'social', 'people']):
                    topics.add('social topics')
            
            summary_parts = []
            summary_parts.append(f"Session with {', '.join(users)}")
            
            if game_contexts:
                summary_parts.append(f"Game contexts: {', '.join(game_contexts)}")
            
            if topics:
                summary_parts.append(f"Topics: {', '.join(topics)}")
            
            summary_parts.append(f"Total interactions: {len(self.session_turns)}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create simple session summary: {e}")
            return "Session summary available."

    def _start_new_session(self):
        """Start a new session."""
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.session_turns = []
        self.logger.info(f"Started new session: {self.current_session_id}")

    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        try:
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client connection."""
        try:
            qdrant_url = self.config.get('qdrant_url', 'http://localhost:6333')
            self.logger.info(f"Connecting to Qdrant at {qdrant_url}")
            self.qdrant_client = QdrantClient(qdrant_url)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"✅ Connected to Qdrant, found {len(collections.collections)} collections")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _initialize_memory_collections(self):
        """Initialize STM and LTM collections in Qdrant."""
        try:
            stm_collection = self.config.get('stm_collection', 'danzar_stm')
            ltm_collection = self.config.get('ltm_collection', 'danzar_ltm')
            
            # Create STM collection
            self._create_collection_if_not_exists(
                collection_name=stm_collection,
                vector_size=self.embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                on_disk_payload=True
            )
            
            # Create LTM collection
            self._create_collection_if_not_exists(
                collection_name=ltm_collection,
                vector_size=self.embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                on_disk_payload=True
            )
            
            self.stm_collection = stm_collection
            self.ltm_collection = ltm_collection
            
            self.logger.info(f"✅ Memory collections initialized: {stm_collection}, {ltm_collection}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory collections: {e}")
            raise
    
    def _create_collection_if_not_exists(self, collection_name: str, vector_size: int, 
                                       distance: models.Distance, on_disk_payload: bool = True):
        """Create a Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                        on_disk=on_disk_payload
                    )
                )
                self.logger.info(f"Created collection: {collection_name}")
            else:
                self.logger.debug(f"Collection {collection_name} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def _start_cleanup_task(self):
        """Start background task for memory cleanup and maintenance."""
        def cleanup_worker():
            while self._initialized:
                try:
                    self._cleanup_expired_stm()
                    time.sleep(self.config.get('auto_cleanup_interval', 300))
                except Exception as e:
                    self.logger.error(f"Error in cleanup worker: {e}")
                    time.sleep(60)  # Wait before retrying
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("✅ Memory cleanup task started")
    
    def upsert_stm(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a memory to Short-Term Memory (recent conversation turns).
        
        Args:
            text: The memory text to store
            metadata: Additional metadata (user, timestamp, etc.)
            
        Returns:
            The ID of the stored memory entry
        """
        if not self._initialized:
            self.logger.warning("Memory manager not initialized, skipping STM upsert")
            return None
        
        try:
            with self.lock:
                # Create memory entry
                timestamp = datetime.now()
                # Use integer ID for Qdrant compatibility
                entry_id = int(timestamp.timestamp() * 1000) + (hash(text) % 10000)
                
                # Generate embedding
                embedding = self.embedding_model.encode(text).tolist()
                
                # Create payload
                payload = {
                    'text': text,
                    'timestamp': timestamp.isoformat(),
                    'type': 'stm',
                    'weight': 1.0,  # Initial weight
                    'metadata': metadata or {}
                }
                
                # Add to in-memory buffer with consistent structure
                stm_entry = {
                    'id': entry_id,
                    'text': text,
                    'timestamp': timestamp,
                    'type': metadata.get('type', 'user_message') if metadata else 'user_message',
                    'embedding': embedding,
                    'weight': 1.0,
                    'metadata': metadata or {}
                }
                self.stm_buffer.append(stm_entry)
                
                # Store in Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.stm_collection,
                    points=[
                        models.PointStruct(
                            id=entry_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
                
                self.logger.debug(f"Added to STM: {text[:50]}... (ID: {entry_id})")
                return str(entry_id)
                
        except Exception as e:
            self.logger.error(f"Failed to upsert STM: {e}")
            return None
    
    def upsert_ltm(self, text: str, metadata: Dict[str, Any] = None, weight: float = 1.0) -> str:
        """
        Add a memory to Long-Term Memory (summaries, key facts, etc.).
        
        Args:
            text: The memory text to store
            metadata: Additional metadata (session_id, game, importance, etc.)
            weight: Importance weight (higher = more important)
            
        Returns:
            The ID of the stored memory entry
        """
        if not self._initialized:
            self.logger.warning("Memory manager not initialized, skipping LTM upsert")
            return None
        
        try:
            with self.lock:
                # Create memory entry
                timestamp = datetime.now()
                # Use integer ID for Qdrant compatibility
                entry_id = int(timestamp.timestamp() * 1000) + (hash(text) % 10000)
                
                # Generate embedding
                embedding = self.embedding_model.encode(text).tolist()
                
                # Create payload
                payload = {
                    'text': text,
                    'timestamp': timestamp.isoformat(),
                    'type': 'ltm',
                    'weight': weight,
                    'metadata': metadata or {}
                }
                
                # Add to in-memory cache
                ltm_entry = {
                    'id': entry_id,
                    'text': text,
                    'timestamp': timestamp,
                    'type': 'ltm',
                    'embedding': embedding,
                    'weight': weight,
                    'metadata': metadata or {}
                }
                self.ltm_buffer.append(ltm_entry)
                
                # Store in Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.ltm_collection,
                    points=[
                        models.PointStruct(
                            id=entry_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
                
                self.logger.info(f"Added to LTM: {text[:50]}... (ID: {entry_id}, weight: {weight})")
                return str(entry_id)
                
        except Exception as e:
            self.logger.error(f"Failed to upsert LTM: {e}")
            return None
    
    def retrieve_memory(self, query: str, memory_type: str = 'both', 
                       limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on semantic similarity.
        
        Args:
            query: The query text to search for
            memory_type: 'stm', 'ltm', or 'both'
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memory entries
        """
        if not self._initialized:
            self.logger.warning("Memory manager not initialized, returning empty results")
            return []
        
        try:
            with self.lock:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query).tolist()
                
                results = []
                
                # Search STM if requested
                if memory_type in ['stm', 'both']:
                    stm_limit = limit or self.config.get('stm_retrieve_k', 10)
                    stm_results = self.qdrant_client.search(
                        collection_name=self.stm_collection,
                        query_vector=query_embedding,
                        limit=stm_limit,
                        with_payload=True
                    )
                    
                    for result in stm_results:
                        results.append({
                            'id': result.id,
                            'text': result.payload.get('text', ''),
                            'timestamp': result.payload.get('timestamp', ''),
                            'type': 'stm',
                            'weight': result.payload.get('weight', 1.0),
                            'score': result.score,
                            'metadata': result.payload.get('metadata', {})
                        })
                
                # Search LTM if requested
                if memory_type in ['ltm', 'both']:
                    ltm_limit = limit or self.config.get('ltm_retrieve_k', 5)
                    ltm_results = self.qdrant_client.search(
                        collection_name=self.ltm_collection,
                        query_vector=query_embedding,
                        limit=ltm_limit,
                        with_payload=True
                    )
                    
                    for result in ltm_results:
                        results.append({
                            'id': result.id,
                            'text': result.payload.get('text', ''),
                            'timestamp': result.payload.get('timestamp', ''),
                            'type': 'ltm',
                            'weight': result.payload.get('weight', 1.0),
                            'score': result.score,
                            'metadata': result.payload.get('metadata', {})
                        })
                
                # Sort by relevance score
                results.sort(key=lambda x: x['score'], reverse=True)
                
                # Apply limit if specified
                if limit:
                    results = results[:limit]
                
                self.logger.debug(f"Retrieved {len(results)} memories for query: {query[:30]}...")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {e}")
            return []
    
    def get_conversation_context(self, user_id: str = None, 
                               recent_turns: int = 10) -> str:
        """
        Get recent conversation context for a user.
        
        Args:
            user_id: Optional user ID to filter by
            recent_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context string
        """
        if not self._initialized:
            return ""
        
        try:
            with self.lock:
                # Get recent STM entries
                recent_memories = list(self.stm_buffer)[-recent_turns:]
                
                if user_id:
                    # Filter by user if specified
                    recent_memories = [
                        m for m in recent_memories 
                        if m.get('metadata', {}).get('user_id') == user_id
                    ]
                
                # Format context
                context_parts = []
                for memory in recent_memories:
                    user = memory.get('metadata', {}).get('user', 'Unknown')
                    text = memory.get('text', '')
                    context_parts.append(f"{user}: {text}")
                
                return "\n".join(context_parts)
                
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return ""
    
    def create_session_summary(self, session_id: str, 
                             conversation_turns: List[str]) -> str:
        """
        Create a session summary for LTM storage.
        
        Args:
            session_id: Unique session identifier
            conversation_turns: List of conversation turns to summarize
            
        Returns:
            Summary text for LTM storage
        """
        try:
            if not conversation_turns:
                return ""
            
            # Simple summary for now - could be enhanced with LLM summarization
            summary = f"Session {session_id}: "
            
            # Extract key topics
            topics = set()
            for turn in conversation_turns:
                words = turn.lower().split()
                # Look for game names, key terms
                for word in words:
                    if any(game in word for game in ['everquest', 'rimworld', 'minecraft', 'game']):
                        topics.add(word)
            
            if topics:
                summary += f"Discussed: {', '.join(topics)}. "
            
            summary += f"Total turns: {len(conversation_turns)}"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create session summary: {e}")
            return ""
    
    def _cleanup_expired_stm(self):
        """Remove expired STM entries based on decay threshold."""
        try:
            decay_threshold = self.config.get('stm_decay_threshold', 0.05)
            decay_minutes = self.config.get('stm_decay_minutes', 30)
            
            # Calculate decay for STM entries
            current_time = datetime.now()
            expired_ids = []
            
            with self.lock:
                # Check in-memory buffer
                for entry in list(self.stm_buffer):
                    timestamp = entry.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    age_minutes = (current_time - timestamp).total_seconds() / 60
                    decay_factor = max(0, 1 - (age_minutes / decay_minutes))
                    
                    if decay_factor < decay_threshold:
                        expired_ids.append(entry['id'])
                        self.stm_buffer.remove(entry)
                
                # Remove from Qdrant
                if expired_ids:
                    self.qdrant_client.delete(
                        collection_name=self.stm_collection,
                        points_selector=models.PointIdsList(
                            points=expired_ids
                        )
                    )
                    self.logger.info(f"Cleaned up {len(expired_ids)} expired STM entries")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired STM: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            with self.lock:
                # Get collection counts
                stm_count = len(self.qdrant_client.scroll(
                    collection_name=self.stm_collection, 
                    limit=1_000_000
                )[0])
                
                ltm_count = len(self.qdrant_client.scroll(
                    collection_name=self.ltm_collection, 
                    limit=1_000_000
                )[0])
                
                return {
                    'stm_buffer_size': len(self.stm_buffer),
                    'ltm_buffer_size': len(self.ltm_buffer),
                    'stm_collection_count': stm_count,
                    'ltm_collection_count': ltm_count,
                    'initialized': self._initialized,
                    'config': {
                        'stm_max_turns': self.config.get('stm_max_turns'),
                        'stm_decay_minutes': self.config.get('stm_decay_minutes'),
                        'embedding_model': self.config.get('embedding_model')
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    def clear_memory(self, memory_type: str = 'both'):
        """Clear memory entries (use with caution)."""
        try:
            with self.lock:
                if memory_type in ['stm', 'both']:
                    self.stm_buffer.clear()
                    self.qdrant_client.delete_collection(self.stm_collection)
                    self._create_collection_if_not_exists(
                        collection_name=self.stm_collection,
                        vector_size=self.embedding_model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                    self.logger.info("Cleared STM")
                
                if memory_type in ['ltm', 'both']:
                    self.ltm_buffer.clear()
                    self.qdrant_client.delete_collection(self.ltm_collection)
                    self._create_collection_if_not_exists(
                        collection_name=self.ltm_collection,
                        vector_size=self.embedding_model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                    self.logger.info("Cleared LTM")
                    
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
    
    def shutdown(self):
        """Clean shutdown of the memory manager."""
        self.logger.info("Shutting down Memory Manager...")
        self._initialized = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        self.logger.info("Memory Manager shutdown complete") 