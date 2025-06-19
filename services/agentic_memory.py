#!/usr/bin/env python3
"""
Agentic Memory Service for DanzarVLM
Implements episodic, semantic, and procedural memory with dynamic linking
Based on A-MEM, Zep, and Mem0 architectures
"""

import time
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta

class MemoryType(Enum):
    EPISODIC = "episodic"      # Events, conversations, actions
    SEMANTIC = "semantic"      # Facts, knowledge, rules
    PROCEDURAL = "procedural"  # How-to, workflows, patterns

class MemoryPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MemoryNode:
    """A single memory node in the agentic memory graph"""
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    timestamp: float
    last_accessed: float
    access_count: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    # Dynamic linking (A-MEM style)
    linked_nodes: List[str] = None  # Node IDs this connects to
    link_strengths: Dict[str, float] = None  # Connection weights
    
    # Temporal context (Zep style)
    temporal_context: Dict[str, Any] = None
    
    # Decay factors
    importance_decay: float = 0.95
    recency_weight: float = 1.0
    
    def __post_init__(self):
        if self.linked_nodes is None:
            self.linked_nodes = []
        if self.link_strengths is None:
            self.link_strengths = {}
        if self.temporal_context is None:
            self.temporal_context = {}

@dataclass 
class MemoryQuery:
    """Query structure for agentic memory retrieval"""
    query_text: str
    memory_types: List[MemoryType]
    user_name: str
    context: Dict[str, Any]
    max_results: int = 10
    time_window_hours: Optional[int] = None
    min_relevance_score: float = 0.3

@dataclass
class AgenticAction:
    """Action that the agent can take based on memory analysis"""
    action_type: str  # "search_web", "recall_memory", "store_memory", "update_memory"
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

class MemoryGraphService:
    """Manages the dynamic memory graph with A-MEM style linking"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger("DanzarVLM.MemoryGraph")
        self._init_graph_db()
        
    def _init_graph_db(self):
        """Initialize graph database for memory connections"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # Memory links table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS memory_links (
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        link_strength REAL NOT NULL,
                        link_type TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_updated REAL NOT NULL,
                        PRIMARY KEY (source_id, target_id)
                    )
                ''')
                
                # Temporal context table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS temporal_context (
                        memory_id TEXT PRIMARY KEY,
                        context_data TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                ''')
                
                conn.commit()
                self.logger.info("[MemoryGraph] Graph database initialized")
        except Exception as e:
            self.logger.error(f"[MemoryGraph] Failed to initialize graph DB: {e}")
    
    def add_memory_link(self, source_id: str, target_id: str, strength: float, link_type: str):
        """Add or update a link between memory nodes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                now = time.time()
                
                c.execute('''
                    INSERT OR REPLACE INTO memory_links 
                    (source_id, target_id, link_strength, link_type, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (source_id, target_id, strength, link_type, now, now))
                
                conn.commit()
                
                # Also add reverse link with slightly lower strength
                c.execute('''
                    INSERT OR REPLACE INTO memory_links 
                    (source_id, target_id, link_strength, link_type, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (target_id, source_id, strength * 0.8, f"reverse_{link_type}", now, now))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[MemoryGraph] Failed to add memory link: {e}")
    
    def get_linked_memories(self, memory_id: str, min_strength: float = 0.3) -> List[Tuple[str, float, str]]:
        """Get memories linked to a given memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT target_id, link_strength, link_type
                    FROM memory_links
                    WHERE source_id = ? AND link_strength >= ?
                    ORDER BY link_strength DESC
                ''', (memory_id, min_strength))
                
                return c.fetchall()
        except Exception as e:
            self.logger.error(f"[MemoryGraph] Failed to get linked memories: {e}")
            return []

class AgenticMemoryService:
    """
    Main agentic memory service implementing episodic, semantic, and procedural memory
    with ReAct-style agent decision making
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = logging.getLogger("DanzarVLM.AgenticMemory")
        
        # Initialize database
        self.db_path = app_context.global_settings.get("AGENTIC_MEMORY", {}).get(
            "db_path", "data/agentic_memory.db"
        )
        self._init_agentic_db()
        
        # Initialize memory graph
        self.memory_graph = MemoryGraphService(self.db_path)
        
        # Memory storage by type
        self.episodic_memory: Dict[str, MemoryNode] = {}
        self.semantic_memory: Dict[str, MemoryNode] = {}
        self.procedural_memory: Dict[str, MemoryNode] = {}
        
        # Summarization buffer (every ~2k tokens)
        self.buffer_max_tokens = 2000
        self.current_buffer = deque(maxlen=50)  # Recent interactions
        self.buffer_token_count = 0
        
        # Agent state
        self.agent_lock = threading.Lock()
        self.pending_actions: deque = deque(maxlen=100)
        
        # Configuration
        config = app_context.global_settings.get("AGENTIC_MEMORY", {})
        self.max_memory_age_days = config.get("max_age_days", 30)
        self.summarization_enabled = config.get("enable_summarization", True)
        self.auto_linking_enabled = config.get("enable_auto_linking", True)
        
        # Start background threads
        self.start_background_processes()
        
        self.logger.info("[AgenticMemory] Initialized with summarization and auto-linking")
    
    def _init_agentic_db(self):
        """Initialize SQLite database for agentic memory storage"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # Main memory nodes table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS memory_nodes (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        timestamp REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        metadata TEXT NOT NULL,
                        embedding BLOB,
                        importance_decay REAL DEFAULT 0.95,
                        recency_weight REAL DEFAULT 1.0,
                        user_name TEXT,
                        game_context TEXT
                    )
                ''')
                
                # Summarization buffer table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS summarization_buffer (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        token_count INTEGER NOT NULL,
                        user_name TEXT,
                        summarized BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Agent actions log
                c.execute('''
                    CREATE TABLE IF NOT EXISTS agent_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_type TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reasoning TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        executed BOOLEAN DEFAULT FALSE,
                        result TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info(f"[AgenticMemory] Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Failed to initialize DB: {e}")
    
    def start_background_processes(self):
        """Start background threads for maintenance"""
        # Summarization thread
        if self.summarization_enabled:
            self.summarization_thread = threading.Thread(
                target=self._summarization_loop, daemon=True
            )
            self.summarization_thread.start()
        
        # Memory maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop, daemon=True
        )
        self.maintenance_thread.start()
        
        # Agent action processor
        self.agent_thread = threading.Thread(
            target=self._agent_loop, daemon=True
        )
        self.agent_thread.start()
    
    def add_to_buffer(self, content: str, user_name: str, estimated_tokens: int = None):
        """Add content to summarization buffer"""
        try:
            if estimated_tokens is None:
                # Rough token estimation (1 token ≈ 4 characters)
                estimated_tokens = len(content) // 4
            
            self.current_buffer.append({
                'content': content,
                'user_name': user_name,
                'timestamp': time.time(),
                'tokens': estimated_tokens
            })
            
            self.buffer_token_count += estimated_tokens
            
            # Store in DB
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO summarization_buffer 
                    (content, timestamp, token_count, user_name)
                    VALUES (?, ?, ?, ?)
                ''', (content, time.time(), estimated_tokens, user_name))
                conn.commit()
            
            self.logger.debug(f"[AgenticMemory] Added to buffer: {estimated_tokens} tokens")
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Failed to add to buffer: {e}")
    
    def store_memory(self, content: str, memory_type: MemoryType, user_name: str, 
                    priority: MemoryPriority = MemoryPriority.MEDIUM, 
                    metadata: Dict[str, Any] = None) -> str:
        """Store a new memory with automatic linking"""
        try:
            # Generate unique ID
            memory_id = hashlib.md5(
                f"{content}{time.time()}{user_name}".encode()
            ).hexdigest()
            
            # Create memory node
            memory_node = MemoryNode(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                priority=priority,
                timestamp=time.time(),
                last_accessed=time.time(),
                access_count=0,
                metadata=metadata or {}
            )
            
            # Store in appropriate memory type collection
            if memory_type == MemoryType.EPISODIC:
                self.episodic_memory[memory_id] = memory_node
            elif memory_type == MemoryType.SEMANTIC:
                self.semantic_memory[memory_id] = memory_node
            elif memory_type == MemoryType.PROCEDURAL:
                self.procedural_memory[memory_id] = memory_node
            
            # Store in database
            self._persist_memory_node(memory_node, user_name)
            
            # Auto-link to related memories if enabled
            if self.auto_linking_enabled:
                self._auto_link_memory(memory_id, content, memory_type)
            
            # Add to buffer for summarization
            self.add_to_buffer(content, user_name)
            
            self.logger.debug(f"[AgenticMemory] Stored {memory_type.value} memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Failed to store memory: {e}")
            return ""
    
    def _persist_memory_node(self, node: MemoryNode, user_name: str):
        """Persist memory node to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT OR REPLACE INTO memory_nodes 
                    (id, content, memory_type, priority, timestamp, last_accessed, 
                     access_count, metadata, importance_decay, recency_weight, 
                     user_name, game_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    node.id, node.content, node.memory_type.value, node.priority.value,
                    node.timestamp, node.last_accessed, node.access_count,
                    json.dumps(node.metadata), node.importance_decay, node.recency_weight,
                    user_name, self.app_context.active_profile.game_name
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Failed to persist memory node: {e}")
    
    def _auto_link_memory(self, memory_id: str, content: str, memory_type: MemoryType):
        """Automatically link new memory to related existing memories"""
        try:
            # Simple keyword-based linking for now
            # In production, would use semantic similarity
            content_words = set(content.lower().split())
            
            # Check each memory type for related content
            all_memories = []
            all_memories.extend(self.episodic_memory.values())
            all_memories.extend(self.semantic_memory.values())
            all_memories.extend(self.procedural_memory.values())
            
            for existing_memory in all_memories:
                if existing_memory.id == memory_id:
                    continue
                
                existing_words = set(existing_memory.content.lower().split())
                overlap = content_words.intersection(existing_words)
                
                # Calculate link strength based on word overlap
                if len(overlap) >= 2:  # At least 2 common words
                    strength = min(0.9, len(overlap) / len(content_words.union(existing_words)))
                    
                    if strength >= 0.3:  # Minimum threshold
                        link_type = f"{memory_type.value}_to_{existing_memory.memory_type.value}"
                        self.memory_graph.add_memory_link(
                            memory_id, existing_memory.id, strength, link_type
                        )
                        
                        self.logger.debug(f"[AgenticMemory] Auto-linked {memory_id} to {existing_memory.id} (strength: {strength:.2f})")
        
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Failed to auto-link memory: {e}")
    
    async def agentic_query(self, query: MemoryQuery) -> Tuple[List[MemoryNode], List[AgenticAction]]:
        """
        Perform agentic memory retrieval with ReAct-style reasoning
        Returns both retrieved memories and suggested actions
        """
        try:
            self.logger.info(f"[AgenticMemory] Processing agentic query: {query.query_text[:100]}...")
            
            # Step 1: Analyze query intent and determine strategy
            actions = await self._analyze_query_intent(query)
            
            # Step 2: Retrieve relevant memories
            memories = await self._retrieve_memories(query)
            
            # Step 3: Evaluate if additional actions are needed
            if len(memories) < query.max_results or not memories:
                # Suggest web search or memory expansion
                search_action = AgenticAction(
                    action_type="search_web",
                    parameters={"query": query.query_text, "user": query.user_name},
                    confidence=0.7,
                    reasoning="Insufficient local memory found, web search recommended"
                )
                actions.append(search_action)
            
            # Step 4: Check for memory consolidation opportunities
            if len(memories) > 5:
                consolidate_action = AgenticAction(
                    action_type="consolidate_memory",
                    parameters={"memory_ids": [m.id for m in memories[:5]]},
                    confidence=0.6,
                    reasoning="Multiple related memories found, consolidation may help"
                )
                actions.append(consolidate_action)
            
            return memories, actions
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Agentic query failed: {e}")
            return [], []
    
    async def _analyze_query_intent(self, query: MemoryQuery) -> List[AgenticAction]:
        """Analyze query to determine what actions the agent should take"""
        actions = []
        
        try:
            query_lower = query.query_text.lower()
            
            # Check for factual queries that might need web search
            factual_indicators = ['what is', 'how to', 'when did', 'where is', 'who is']
            if any(indicator in query_lower for indicator in factual_indicators):
                actions.append(AgenticAction(
                    action_type="prepare_web_search",
                    parameters={"query": query.query_text, "type": "factual"},
                    confidence=0.8,
                    reasoning="Factual query detected, web search may be beneficial"
                ))
            
            # Check for procedural queries
            procedural_indicators = ['how do i', 'steps to', 'process of', 'way to']
            if any(indicator in query_lower for indicator in procedural_indicators):
                actions.append(AgenticAction(
                    action_type="prioritize_procedural",
                    parameters={"memory_types": [MemoryType.PROCEDURAL.value]},
                    confidence=0.9,
                    reasoning="Procedural query detected, prioritizing how-to memories"
                ))
            
            # Check for conversational follow-ups
            followup_indicators = ['what about', 'also', 'and', 'too']
            if any(indicator in query_lower for indicator in followup_indicators):
                actions.append(AgenticAction(
                    action_type="expand_context",
                    parameters={"user": query.user_name, "time_window": 1},
                    confidence=0.7,
                    reasoning="Follow-up question detected, expanding conversational context"
                ))
            
            return actions
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Intent analysis failed: {e}")
            return []
    
    async def _retrieve_memories(self, query: MemoryQuery) -> List[MemoryNode]:
        """Retrieve relevant memories based on query"""
        try:
            all_relevant = []
            
            # Search each memory type
            for memory_type in query.memory_types:
                if memory_type == MemoryType.EPISODIC:
                    memories = self._search_episodic(query)
                elif memory_type == MemoryType.SEMANTIC:
                    memories = self._search_semantic(query)
                elif memory_type == MemoryType.PROCEDURAL:
                    memories = self._search_procedural(query)
                else:
                    memories = []
                
                all_relevant.extend(memories)
            
            # Remove duplicates and sort by relevance
            unique_memories = {m.id: m for m in all_relevant}.values()
            sorted_memories = sorted(
                unique_memories, 
                key=lambda m: self._calculate_relevance_score(m, query),
                reverse=True
            )
            
            # Apply time window filter if specified
            if query.time_window_hours:
                cutoff_time = time.time() - (query.time_window_hours * 3600)
                sorted_memories = [m for m in sorted_memories if m.timestamp >= cutoff_time]
            
            # Update access statistics
            for memory in sorted_memories[:query.max_results]:
                memory.last_accessed = time.time()
                memory.access_count += 1
            
            return sorted_memories[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Memory retrieval failed: {e}")
            return []
    
    def _search_episodic(self, query: MemoryQuery) -> List[MemoryNode]:
        """Search episodic memories"""
        return self._keyword_search(self.episodic_memory, query.query_text)
    
    def _search_semantic(self, query: MemoryQuery) -> List[MemoryNode]:
        """Search semantic memories"""
        return self._keyword_search(self.semantic_memory, query.query_text)
    
    def _search_procedural(self, query: MemoryQuery) -> List[MemoryNode]:
        """Search procedural memories"""
        return self._keyword_search(self.procedural_memory, query.query_text)
    
    def _keyword_search(self, memory_dict: Dict[str, MemoryNode], query: str) -> List[MemoryNode]:
        """Simple keyword-based search in memory collection"""
        query_words = set(query.lower().split())
        matches = []
        
        for memory in memory_dict.values():
            content_words = set(memory.content.lower().split())
            overlap = query_words.intersection(content_words)
            
            if overlap:
                matches.append(memory)
        
        return matches
    
    def _calculate_relevance_score(self, memory: MemoryNode, query: MemoryQuery) -> float:
        """Calculate relevance score for a memory given a query"""
        try:
            # Base keyword similarity
            query_words = set(query.query_text.lower().split())
            content_words = set(memory.content.lower().split())
            overlap = query_words.intersection(content_words)
            
            if not query_words:
                keyword_score = 0
            else:
                keyword_score = len(overlap) / len(query_words)
            
            # Recency factor
            age_hours = (time.time() - memory.timestamp) / 3600
            recency_score = max(0, 1 - (age_hours / (24 * 7)))  # Decay over a week
            
            # Priority factor  
            priority_score = memory.priority.value / 4.0
            
            # Access frequency factor
            frequency_score = min(1.0, memory.access_count / 10)
            
            # Combined score
            total_score = (
                keyword_score * 0.4 +
                recency_score * 0.2 +
                priority_score * 0.2 +
                frequency_score * 0.2
            )
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Relevance calculation failed: {e}")
            return 0.0
    
    def _summarization_loop(self):
        """Background thread for periodic summarization"""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                if self.buffer_token_count >= self.buffer_max_tokens:
                    self._perform_summarization()
                    
            except Exception as e:
                self.logger.error(f"[AgenticMemory] Summarization loop error: {e}")
    
    def _perform_summarization(self):
        """Perform summarization of buffer content"""
        try:
            self.logger.info("[AgenticMemory] Starting buffer summarization...")
            
            # Get buffer content
            buffer_content = list(self.current_buffer)
            if not buffer_content:
                return
            
            # Group by user and create summaries
            user_groups = defaultdict(list)
            for item in buffer_content:
                user_groups[item['user_name']].append(item)
            
            for user_name, user_items in user_groups.items():
                # Create conversation summary
                conversation_text = "\n".join([item['content'] for item in user_items])
                
                # Store as episodic memory
                summary_id = self.store_memory(
                    content=f"Conversation summary for {user_name}: {conversation_text[:500]}...",
                    memory_type=MemoryType.EPISODIC,
                    user_name=user_name,
                    priority=MemoryPriority.MEDIUM,
                    metadata={
                        "type": "conversation_summary",
                        "original_items": len(user_items),
                        "time_span": user_items[-1]['timestamp'] - user_items[0]['timestamp']
                    }
                )
                
                self.logger.debug(f"[AgenticMemory] Created summary {summary_id} for {user_name}")
            
            # Clear buffer
            self.current_buffer.clear()
            self.buffer_token_count = 0
            
            # Mark items as summarized in DB
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('UPDATE summarization_buffer SET summarized = TRUE WHERE summarized = FALSE')
                conn.commit()
            
            self.logger.info("[AgenticMemory] Summarization completed")
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Summarization failed: {e}")
    
    def _maintenance_loop(self):
        """Background thread for memory maintenance"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_memories()
                self._update_memory_weights()
                
            except Exception as e:
                self.logger.error(f"[AgenticMemory] Maintenance loop error: {e}")
    
    def _cleanup_old_memories(self):
        """Clean up old, low-importance memories"""
        try:
            cutoff_time = time.time() - (self.max_memory_age_days * 24 * 3600)
            removed_count = 0
            
            # Clean each memory type
            for memory_dict in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
                to_remove = []
                for memory_id, memory in memory_dict.items():
                    if (memory.timestamp < cutoff_time and 
                        memory.priority == MemoryPriority.LOW and 
                        memory.access_count < 2):
                        to_remove.append(memory_id)
                
                for memory_id in to_remove:
                    del memory_dict[memory_id]
                    removed_count += 1
            
            # Clean database
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    DELETE FROM memory_nodes 
                    WHERE timestamp < ? AND priority = 1 AND access_count < 2
                ''', (cutoff_time,))
                conn.commit()
            
            if removed_count > 0:
                self.logger.info(f"[AgenticMemory] Cleaned up {removed_count} old memories")
                
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Memory cleanup failed: {e}")
    
    def _update_memory_weights(self):
        """Update memory importance weights based on access patterns"""
        try:
            # Apply decay to all memories
            all_memories = []
            all_memories.extend(self.episodic_memory.values())
            all_memories.extend(self.semantic_memory.values())  
            all_memories.extend(self.procedural_memory.values())
            
            for memory in all_memories:
                # Time-based decay
                age_factor = (time.time() - memory.timestamp) / (24 * 3600)  # Days
                memory.recency_weight *= (memory.importance_decay ** age_factor)
                
                # Update in database
                self._persist_memory_node(memory, "system")
            
            self.logger.debug("[AgenticMemory] Updated memory weights")
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Weight update failed: {e}")
    
    def _agent_loop(self):
        """Background thread for processing agent actions"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                if self.pending_actions:
                    with self.agent_lock:
                        action = self.pending_actions.popleft()
                        self._execute_agent_action(action)
                        
            except Exception as e:
                self.logger.error(f"[AgenticMemory] Agent loop error: {e}")
    
    def _execute_agent_action(self, action: AgenticAction):
        """Execute a specific agent action"""
        try:
            self.logger.info(f"[AgenticMemory] Executing action: {action.action_type}")
            
            if action.action_type == "search_web":
                # Trigger web search through Smart RAG if available
                if hasattr(self.app_context, 'smart_rag_service'):
                    # This would trigger web search and store results
                    pass
                    
            elif action.action_type == "consolidate_memory":
                # Consolidate related memories
                memory_ids = action.parameters.get("memory_ids", [])
                self._consolidate_memories(memory_ids)
                
            elif action.action_type == "update_memory":
                # Update existing memory based on new information
                pass
                
            # Log action execution
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO agent_actions 
                    (action_type, parameters, confidence, reasoning, timestamp, executed)
                    VALUES (?, ?, ?, ?, ?, TRUE)
                ''', (
                    action.action_type,
                    json.dumps(action.parameters),
                    action.confidence,
                    action.reasoning,
                    time.time()
                ))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Action execution failed: {e}")
    
    def _consolidate_memories(self, memory_ids: List[str]):
        """Consolidate multiple related memories into one"""
        try:
            memories_to_consolidate = []
            
            # Collect memories from all types
            for memory_dict in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
                for memory_id in memory_ids:
                    if memory_id in memory_dict:
                        memories_to_consolidate.append(memory_dict[memory_id])
            
            if len(memories_to_consolidate) < 2:
                return
            
            # Create consolidated content
            consolidated_content = "Consolidated memory: " + " | ".join([
                m.content[:100] for m in memories_to_consolidate
            ])
            
            # Determine best memory type (most common)
            memory_types = [m.memory_type for m in memories_to_consolidate]
            best_type = max(set(memory_types), key=memory_types.count)
            
            # Store consolidated memory
            consolidated_id = self.store_memory(
                content=consolidated_content,
                memory_type=best_type,
                user_name="system",
                priority=MemoryPriority.HIGH,
                metadata={
                    "type": "consolidated",
                    "source_memories": memory_ids,
                    "consolidation_time": time.time()
                }
            )
            
            self.logger.info(f"[AgenticMemory] Consolidated {len(memory_ids)} memories into {consolidated_id}")
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Memory consolidation failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the agentic memory system"""
        try:
            stats = {
                "episodic_count": len(self.episodic_memory),
                "semantic_count": len(self.semantic_memory),
                "procedural_count": len(self.procedural_memory),
                "buffer_tokens": self.buffer_token_count,
                "buffer_items": len(self.current_buffer),
                "pending_actions": len(self.pending_actions)
            }
            
            # Add database stats
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT COUNT(*) FROM memory_nodes')
                stats["total_db_memories"] = c.fetchone()[0]
                
                c.execute('SELECT COUNT(*) FROM memory_links')
                stats["total_links"] = c.fetchone()[0]
                
                c.execute('SELECT COUNT(*) FROM agent_actions WHERE executed = TRUE')
                stats["executed_actions"] = c.fetchone()[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"[AgenticMemory] Stats calculation failed: {e}")
            return {}

# Integration helper functions for existing services
def upgrade_conversation_memory(conversation_memory_service, agentic_memory_service):
    """Helper to upgrade existing conversation memory to agentic memory"""
    try:
        # Migrate conversation turns to episodic memories
        for user_name, conversations in conversation_memory_service.conversations.items():
            for turn in conversations:
                content = f"User: {turn.user_message}\nAI: {turn.bot_response}"
                
                agentic_memory_service.store_memory(
                    content=content,
                    memory_type=MemoryType.EPISODIC,
                    user_name=user_name,
                    priority=MemoryPriority.MEDIUM if turn.importance > 0.7 else MemoryPriority.LOW,
                    metadata={
                        "type": "conversation_turn",
                        "topic": turn.topic,
                        "sentiment": turn.user_sentiment,
                        "original_timestamp": turn.timestamp
                    }
                )
        
        logging.getLogger("DanzarVLM.AgenticMemory").info(
            "[AgenticMemory] Successfully migrated conversation memory"
        )
        
    except Exception as e:
        logging.getLogger("DanzarVLM.AgenticMemory").error(
            f"[AgenticMemory] Migration failed: {e}"
        )