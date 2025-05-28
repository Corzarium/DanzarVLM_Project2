# services/memory_service.py

import os
import json
import time
from typing import Optional, Dict, List, Any
from collections import deque
from dataclasses import dataclass, asdict
import sqlite3
from datetime import datetime

@dataclass
class MemoryEntry:
    content: str
    source: str  # 'vlm_commentary', 'user_query', 'bot_response', etc.
    timestamp: float
    metadata: Dict[str, Any]
    importance_score: float = 1.0  # Higher = more important/memorable
    recall_count: int = 0  # How often this memory has been recalled
    last_recall_time: Optional[float] = None

class MemoryService:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = self.ctx.logger
        
        # Initialize memory storage
        self.memory_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "memory.db"
        )
        self._init_db()
        
        # Short-term memory (deque)
        self.recent_memories = deque(maxlen=self.ctx.active_profile.memory_deque_size)
        
        # Subscribe to profile changes
        if hasattr(self.ctx, 'active_profile_change_subscribers'):
            self.ctx.active_profile_change_subscribers.append(self._handle_profile_change)
        
        self.logger.info("[MemoryService] Initialized.")

    def _init_db(self):
        """Initialize SQLite database for persistent memory storage."""
        os.makedirs(os.path.dirname(self.memory_db_path), exist_ok=True)
        
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metadata TEXT NOT NULL,
                        importance_score REAL DEFAULT 1.0,
                        recall_count INTEGER DEFAULT 0,
                        last_recall_time REAL,
                        game_name TEXT NOT NULL
                    )
                ''')
                conn.commit()
            self.logger.info(f"[MemoryService] Memory database initialized at {self.memory_db_path}")
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to initialize memory database: {e}", exc_info=True)

    def _handle_profile_change(self, new_profile):
        """Update memory settings when game profile changes."""
        new_maxlen = new_profile.memory_deque_size
        if self.recent_memories.maxlen != new_maxlen:
            old_items = list(self.recent_memories)
            self.recent_memories = deque(old_items, maxlen=new_maxlen)
            self.logger.info(f"[MemoryService] Updated recent_memories maxlen to {new_maxlen}")

    def store_memory(self, entry: MemoryEntry):
        """Store a memory both in short-term and long-term storage."""
        try:
            # Add to short-term memory
            self.recent_memories.append(entry)
            
            # Store in SQLite
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO memories 
                    (content, source, timestamp, metadata, importance_score, recall_count, last_recall_time, game_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.content,
                    entry.source,
                    entry.timestamp,
                    json.dumps(entry.metadata),
                    entry.importance_score,
                    entry.recall_count,
                    entry.last_recall_time,
                    self.ctx.active_profile.game_name
                ))
                conn.commit()
            
            # Also store in RAG if available
            if self.ctx.rag_service_instance and self.ctx.active_profile.memory_rag_history_collection_name:
                self.ctx.rag_service_instance.ingest_text(
                    collection_name=self.ctx.active_profile.memory_rag_history_collection_name,
                    text_content=entry.content,
                    metadata={
                        **entry.metadata,
                        "source": entry.source,
                        "timestamp": entry.timestamp,
                        "importance_score": entry.importance_score
                    }
                )
            
            self.logger.debug(f"[MemoryService] Stored new memory: {entry.content[:50]}...")
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to store memory: {e}", exc_info=True)

    def get_relevant_memories(self, query: str, top_k: int = 5, min_importance: float = 0.5) -> List[MemoryEntry]:
        """Retrieve relevant memories using both RAG and SQL-based retrieval."""
        relevant_memories: List[MemoryEntry] = []
        retrieved_content_set = set() # Use a set to prevent duplicates

        try:
            # --- RAG-based retrieval for Game Knowledge ---
            game_knowledge_collection = self.ctx.active_profile.rag_collection_name
            if self.ctx.rag_service_instance and game_knowledge_collection:
                self.logger.debug(f"[MemoryService] Querying game knowledge RAG collection: {game_knowledge_collection}")
                game_knowledge_results = self.ctx.rag_service_instance.query_rag(
                    collection_name=game_knowledge_collection,
                    query_text=query,
                    top_k=top_k # Use profile's rag_top_k or the function's top_k
                )
                if game_knowledge_results:
                    self.logger.debug(f"[MemoryService] Retrieved {len(game_knowledge_results)} results from game knowledge RAG.")
                    for result in game_knowledge_results:
                         if result and result not in retrieved_content_set:
                            relevant_memories.append(MemoryEntry(
                                content=result,
                                source="rag_game_knowledge",
                                timestamp=time.time(), # Use current time as retrieval time
                                metadata={"retrieval_method": "rag_game_knowledge"},
                                importance_score=1.0 # Default importance for RAG results
                            ))
                            retrieved_content_set.add(result)


            # --- RAG-based retrieval for Chat History (if configured) ---
            history_collection = self.ctx.active_profile.memory_rag_history_collection_name
            if self.ctx.rag_service_instance and history_collection and history_collection != game_knowledge_collection: # Avoid double querying the same collection
                 self.logger.debug(f"[MemoryService] Querying chat history RAG collection: {history_collection}")
                 history_rag_results = self.ctx.rag_service_instance.query_rag(
                     collection_name=history_collection,
                     query_text=query,
                     top_k=getattr(self.ctx.active_profile, 'memory_rag_chat_lookback_k', 5) # Use specific lookback setting
                 )
                 if history_rag_results:
                     self.logger.debug(f"[MemoryService] Retrieved {len(history_rag_results)} results from history RAG.")
                     for result in history_rag_results:
                         if result and result not in retrieved_content_set:
                              relevant_memories.append(MemoryEntry(
                                 content=result,
                                 source="rag_chat_history",
                                 timestamp=time.time(), # Use current time
                                 metadata={"retrieval_method": "rag_chat_history"},
                                 importance_score=1.0
                             ))
                              retrieved_content_set.add(result)


            # --- SQL-based retrieval for Recent Memories ---
            # This gets recent memories (including user queries, bot responses, VLM commentary)
            # from the in-memory deque and potentially the SQLite DB for older but important ones.
            # We'll add these to relevant_memories, again checking for duplicates.
            self.logger.debug(f"[MemoryService] Retrieving recent memories from SQL/deque.")
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                now = time.time()
                # Get memories with high importance score and recent recall, or just recent ones
                # Prioritize recent and important memories
                c.execute('''
                    SELECT content, source, timestamp, metadata, importance_score, recall_count, last_recall_time\n\
                    FROM memories\n\
                    WHERE game_name = ? AND (importance_score >= ? OR timestamp > ?) -- Include recent memories regardless of importance\n\
                    ORDER BY (importance_score * (1 + recall_count)) DESC, timestamp DESC -- Order by combined score then recency\n\
                    LIMIT ?
                ''', (self.ctx.active_profile.game_name, min_importance, now - (3600 * 24 * 7), top_k * 2)) # Look back 7 days for recent, retrieve more initially to filter

                sql_results = c.fetchall()
                self.logger.debug(f"[MemoryService] Retrieved {len(sql_results)} potential memories from SQL.")

                for row in sql_results:
                    memory_content = row[0]
                    if memory_content and memory_content not in retrieved_content_set:
                        memory = MemoryEntry(
                            content=memory_content,
                            source=row[1],
                            timestamp=row[2],
                            metadata=json.loads(row[3]),
                            importance_score=row[4],
                            recall_count=row[5],
                            last_recall_time=row[6]
                        )
                        relevant_memories.append(memory)
                        retrieved_content_set.add(memory_content)
                        # Update recall stats in SQL for this memory if it was retrieved
                        self._update_memory_recall_stats(memory_content)


            # Sort memories (e.g., by a combination of recency and importance/relevance)
            # For simplicity, let's sort by timestamp descending (most recent first)
            # A more sophisticated approach might use embedding similarity scores if available from RAG
            relevant_memories.sort(key=lambda x: x.timestamp, reverse=True)


            self.logger.debug(f"[MemoryService] Final count of relevant memories after combining and filtering: {len(relevant_memories)}")

            # Return the top_k results after combining and sorting
            return relevant_memories[:top_k]

        except Exception as e:
            self.logger.error(f"[MemoryService] Error retrieving relevant memories: {e}", exc_info=True)
            return []

    def _update_memory_recall_stats(self, content: str):
        """Update recall statistics for a memory in the SQL database."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE memories 
                    SET recall_count = recall_count + 1,
                        last_recall_time = ?
                    WHERE content = ? AND game_name = ?
                ''', (time.time(), content, self.ctx.active_profile.game_name))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to update memory recall stats: {e}", exc_info=True)

    def update_memory_importance(self, content: str, importance_delta: float):
        """Update the importance score of a memory."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE memories 
                    SET importance_score = CASE
                        WHEN importance_score + ? BETWEEN 0 AND 10 THEN importance_score + ?
                        WHEN importance_score + ? > 10 THEN 10
                        ELSE 0
                    END
                    WHERE content = ? AND game_name = ?
                ''', (importance_delta, importance_delta, importance_delta, content, self.ctx.active_profile.game_name))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to update memory importance: {e}", exc_info=True)

    def cleanup_old_memories(self, max_age_days: int = 30, min_importance: float = 0.5):
        """Clean up old memories with low importance scores."""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    DELETE FROM memories 
                    WHERE timestamp < ? 
                    AND importance_score < ?
                    AND game_name = ?
                ''', (cutoff_time, min_importance, self.ctx.active_profile.game_name))
                conn.commit()
                
                deleted_count = c.rowcount
                self.logger.info(f"[MemoryService] Cleaned up {deleted_count} old memories with low importance.")
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to clean up old memories: {e}", exc_info=True)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT 
                        COUNT(*) as total_memories,
                        AVG(importance_score) as avg_importance,
                        AVG(recall_count) as avg_recalls,
                        MIN(timestamp) as oldest_memory,
                        MAX(timestamp) as newest_memory
                    FROM memories
                    WHERE game_name = ?
                ''', (self.ctx.active_profile.game_name,))
                
                row = c.fetchone()
                if row:
                    return {
                        "total_memories": row[0],
                        "avg_importance": round(row[1] or 0, 2),
                        "avg_recalls": round(row[2] or 0, 2),
                        "oldest_memory": datetime.fromtimestamp(row[3]).isoformat() if row[3] else None,
                        "newest_memory": datetime.fromtimestamp(row[4]).isoformat() if row[4] else None,
                        "short_term_memory_size": len(self.recent_memories),
                        "short_term_memory_capacity": self.recent_memories.maxlen
                    }
                return {}
        except Exception as e:
            self.logger.error(f"[MemoryService] Failed to get memory stats: {e}", exc_info=True)
            return {} 