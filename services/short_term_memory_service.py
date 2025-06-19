#!/usr/bin/env python3
"""
Short-Term Memory Service - Fast RAM-based conversation memory
Holds recent conversation context in memory for immediate access
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import logging

@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    content: str
    timestamp: float
    user_name: str
    entry_type: str  # 'user_input', 'bot_response', 'system_event'
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0.0 to 1.0, higher = more important

@dataclass
class ConversationContext:
    """Context for a specific conversation/user"""
    user_name: str
    entries: deque = field(default_factory=lambda: deque(maxlen=50))  # Last 50 entries
    last_activity: float = field(default_factory=time.time)
    topic_keywords: Dict[str, int] = field(default_factory=dict)  # keyword -> frequency
    current_game: Optional[str] = None
    conversation_summary: str = ""

class ShortTermMemoryService:
    """
    Fast RAM-based short-term memory service for conversations
    
    Features:
    - Per-user conversation contexts
    - Automatic cleanup of old conversations
    - Topic tracking and keyword extraction
    - Integration with long-term memory
    - Thread-safe operations
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.settings = app_context.global_settings
        
        # Memory storage
        self.conversations: Dict[str, ConversationContext] = {}
        self.global_context: deque = deque(maxlen=100)  # Global conversation flow
        
        # Configuration
        self.max_conversations = self.settings.get('SHORT_TERM_MEMORY', {}).get('max_conversations', 10)
        self.max_entries_per_user = self.settings.get('SHORT_TERM_MEMORY', {}).get('max_entries_per_user', 50)
        self.cleanup_interval = self.settings.get('SHORT_TERM_MEMORY', {}).get('cleanup_interval_minutes', 30)
        self.conversation_timeout = self.settings.get('SHORT_TERM_MEMORY', {}).get('conversation_timeout_minutes', 60)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Cleanup thread
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        
        self.logger.info(f"[ShortTermMemory] Initialized with max_conversations={self.max_conversations}, cleanup_interval={self.cleanup_interval}min")
    
    def start_cleanup_thread(self):
        """Start the automatic cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("[ShortTermMemory] Cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop the cleanup thread"""
        self.shutdown_event.set()
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
            self.logger.info("[ShortTermMemory] Cleanup thread stopped")
    
    def _cleanup_worker(self):
        """Background worker for cleaning up old conversations"""
        while not self.shutdown_event.wait(self.cleanup_interval * 60):  # Convert minutes to seconds
            try:
                self.cleanup_old_conversations()
            except Exception as e:
                self.logger.error(f"[ShortTermMemory] Cleanup error: {e}")
    
    def add_entry(self, user_name: str, content: str, entry_type: str = 'user_input', 
                  metadata: Optional[Dict[str, Any]] = None, importance: float = 1.0) -> bool:
        """
        Add a new memory entry for a user
        
        Args:
            user_name: Name of the user
            content: Content of the entry
            entry_type: Type of entry ('user_input', 'bot_response', 'system_event')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
        
        Returns:
            True if added successfully
        """
        try:
            with self.lock:
                # Get or create conversation context
                if user_name not in self.conversations:
                    if len(self.conversations) >= self.max_conversations:
                        # Remove oldest conversation
                        oldest_user = min(self.conversations.keys(), 
                                        key=lambda u: self.conversations[u].last_activity)
                        del self.conversations[oldest_user]
                        self.logger.info(f"[ShortTermMemory] Removed oldest conversation for {oldest_user}")
                    
                    self.conversations[user_name] = ConversationContext(user_name=user_name)
                    self.logger.info(f"[ShortTermMemory] Created new conversation context for {user_name}")
                
                context = self.conversations[user_name]
                
                # Create memory entry
                entry = MemoryEntry(
                    content=content,
                    timestamp=time.time(),
                    user_name=user_name,
                    entry_type=entry_type,
                    metadata=metadata or {},
                    importance=importance
                )
                
                # Add to user's conversation
                context.entries.append(entry)
                context.last_activity = time.time()
                
                # Add to global context
                self.global_context.append(entry)
                
                # Update topic keywords
                self._update_topic_keywords(context, content)
                
                # Detect current game
                self._detect_current_game(context, content)
                
                self.logger.debug(f"[ShortTermMemory] Added {entry_type} for {user_name}: '{content[:50]}...'")
                return True
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error adding entry: {e}")
            return False
    
    def get_recent_context(self, user_name: str, max_entries: int = 10) -> List[MemoryEntry]:
        """Get recent conversation context for a user"""
        try:
            with self.lock:
                if user_name not in self.conversations:
                    return []
                
                context = self.conversations[user_name]
                # Return most recent entries, up to max_entries
                recent_entries = list(context.entries)[-max_entries:]
                
                self.logger.debug(f"[ShortTermMemory] Retrieved {len(recent_entries)} recent entries for {user_name}")
                return recent_entries
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting recent context: {e}")
            return []
    
    def get_conversation_summary(self, user_name: str) -> str:
        """Get a summary of the conversation with a user"""
        try:
            with self.lock:
                if user_name not in self.conversations:
                    return ""
                
                context = self.conversations[user_name]
                
                # If we have a cached summary, return it
                if context.conversation_summary:
                    return context.conversation_summary
                
                # Generate summary from recent entries
                recent_entries = list(context.entries)[-10:]  # Last 10 entries
                if not recent_entries:
                    return ""
                
                # Simple summary generation
                topics = list(context.topic_keywords.keys())[:3]  # Top 3 topics
                game = context.current_game or "gaming"
                
                summary = f"Recent conversation with {user_name} about {game}"
                if topics:
                    summary += f", discussing: {', '.join(topics)}"
                
                # Cache the summary
                context.conversation_summary = summary
                
                return summary
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting conversation summary: {e}")
            return ""
    
    def get_user_context(self, user_name: str) -> Optional[ConversationContext]:
        """Get full conversation context for a user"""
        try:
            with self.lock:
                return self.conversations.get(user_name)
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting user context: {e}")
            return None
    
    def get_global_context(self, max_entries: int = 20) -> List[MemoryEntry]:
        """Get recent global conversation context across all users"""
        try:
            with self.lock:
                recent_global = list(self.global_context)[-max_entries:]
                self.logger.debug(f"[ShortTermMemory] Retrieved {len(recent_global)} global context entries")
                return recent_global
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting global context: {e}")
            return []
    
    def search_memory(self, query: str, user_name: Optional[str] = None, max_results: int = 5) -> List[MemoryEntry]:
        """Search memory entries by content"""
        try:
            with self.lock:
                query_lower = query.lower()
                results = []
                
                # Search in specific user's conversation or all conversations
                conversations_to_search = [self.conversations[user_name]] if user_name and user_name in self.conversations else self.conversations.values()
                
                for context in conversations_to_search:
                    for entry in context.entries:
                        if query_lower in entry.content.lower():
                            results.append(entry)
                
                # Sort by timestamp (most recent first) and importance
                results.sort(key=lambda e: (e.timestamp, e.importance), reverse=True)
                
                self.logger.debug(f"[ShortTermMemory] Found {len(results)} results for query: '{query}'")
                return results[:max_results]
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error searching memory: {e}")
            return []
    
    def cleanup_old_conversations(self):
        """Clean up old, inactive conversations"""
        try:
            with self.lock:
                current_time = time.time()
                timeout_seconds = self.conversation_timeout * 60
                
                users_to_remove = []
                for user_name, context in self.conversations.items():
                    if current_time - context.last_activity > timeout_seconds:
                        users_to_remove.append(user_name)
                
                for user_name in users_to_remove:
                    del self.conversations[user_name]
                    self.logger.info(f"[ShortTermMemory] Cleaned up inactive conversation for {user_name}")
                
                if users_to_remove:
                    self.logger.info(f"[ShortTermMemory] Cleaned up {len(users_to_remove)} inactive conversations")
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error during cleanup: {e}")
    
    def _update_topic_keywords(self, context: ConversationContext, content: str):
        """Update topic keywords for a conversation"""
        try:
            # Simple keyword extraction (could be enhanced with NLP)
            words = content.lower().split()
            
            # Filter out common words and extract meaningful keywords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
            
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Update keyword frequencies
            for keyword in keywords:
                context.topic_keywords[keyword] = context.topic_keywords.get(keyword, 0) + 1
            
            # Keep only top 20 keywords
            if len(context.topic_keywords) > 20:
                sorted_keywords = sorted(context.topic_keywords.items(), key=lambda x: x[1], reverse=True)
                context.topic_keywords = dict(sorted_keywords[:20])
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error updating topic keywords: {e}")
    
    def _detect_current_game(self, context: ConversationContext, content: str):
        """Detect and update the current game being discussed"""
        try:
            content_lower = content.lower()
            
            # Game detection patterns
            game_patterns = {
                'everquest': ['everquest', 'eq', 'norrath'],
                'world_of_warcraft': ['warcraft', 'wow', 'azeroth'],
                'final_fantasy': ['final fantasy', 'ff14', 'ffxiv'],
                'elder_scrolls': ['elder scrolls', 'skyrim', 'eso'],
                'minecraft': ['minecraft', 'creeper', 'enderman'],
                'league_of_legends': ['league of legends', 'lol', 'riot'],
                'dota': ['dota', 'dota2'],
                'counter_strike': ['counter strike', 'cs', 'csgo'],
                'valorant': ['valorant'],
                'apex_legends': ['apex', 'apex legends'],
                'fortnite': ['fortnite'],
                'overwatch': ['overwatch']
            }
            
            for game, patterns in game_patterns.items():
                if any(pattern in content_lower for pattern in patterns):
                    if context.current_game != game:
                        context.current_game = game
                        self.logger.info(f"[ShortTermMemory] Detected game change for {context.user_name}: {game}")
                    break
                    
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error detecting current game: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            with self.lock:
                total_entries = sum(len(context.entries) for context in self.conversations.values())
                
                stats = {
                    'active_conversations': len(self.conversations),
                    'total_entries': total_entries,
                    'global_context_size': len(self.global_context),
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'conversations': {
                        user: {
                            'entries': len(context.entries),
                            'last_activity': context.last_activity,
                            'current_game': context.current_game,
                            'top_topics': list(sorted(context.topic_keywords.items(), key=lambda x: x[1], reverse=True)[:5])
                        }
                        for user, context in self.conversations.items()
                    }
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting memory stats: {e}")
            return {}
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB (rough calculation)"""
        try:
            total_chars = 0
            
            # Count characters in all entries
            for context in self.conversations.values():
                for entry in context.entries:
                    total_chars += len(entry.content)
                    total_chars += len(str(entry.metadata))
            
            # Add global context
            for entry in self.global_context:
                total_chars += len(entry.content)
            
            # Rough estimate: 1 char ≈ 1 byte, plus overhead
            estimated_bytes = total_chars * 2  # 2x for overhead
            estimated_mb = estimated_bytes / (1024 * 1024)
            
            return round(estimated_mb, 2)
            
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error estimating memory usage: {e}")
            return 0.0
    
    def clear_user_memory(self, user_name: str) -> bool:
        """Clear memory for a specific user"""
        try:
            with self.lock:
                if user_name in self.conversations:
                    del self.conversations[user_name]
                    self.logger.info(f"[ShortTermMemory] Cleared memory for {user_name}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error clearing user memory: {e}")
            return False
    
    def clear_all_memory(self) -> bool:
        """Clear all memory (use with caution)"""
        try:
            with self.lock:
                self.conversations.clear()
                self.global_context.clear()
                self.logger.info("[ShortTermMemory] Cleared all memory")
                return True
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error clearing all memory: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics for Discord commands"""
        try:
            with self.lock:
                total_entries = sum(len(ctx.entries) for ctx in self.conversations.values())
                memory_usage_mb = self._estimate_memory_usage()
                
                return {
                    'active_conversations': len(self.conversations),
                    'total_entries': total_entries,
                    'memory_usage_mb': memory_usage_mb
                }
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting stats: {e}")
            return {'active_conversations': 0, 'total_entries': 0, 'memory_usage_mb': 0.0}

    def clear_conversation(self, user_name: str) -> bool:
        """Clear conversation for a specific user"""
        try:
            with self.lock:
                if user_name in self.conversations:
                    del self.conversations[user_name]
                    self.logger.info(f"[ShortTermMemory] Cleared conversation for {user_name}")
                    return True
                else:
                    self.logger.warning(f"[ShortTermMemory] No conversation found for {user_name}")
                    return False
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error clearing conversation for {user_name}: {e}")
            return False

    def get_active_conversations(self) -> Dict[str, int]:
        """Get active conversations with entry counts"""
        try:
            with self.lock:
                return {user: len(ctx.entries) for user, ctx in self.conversations.items()}
        except Exception as e:
            self.logger.error(f"[ShortTermMemory] Error getting active conversations: {e}")
            return {} 