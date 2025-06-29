#!/usr/bin/env python3
"""
Enhanced Conversation Memory Service
Integrates Short-Term Memory (STM) in RAM with Long-Term Memory (LTM) in RAG
Provides seamless conversation context management and memory persistence
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import logging
import json

@dataclass
class STMEntry:
    """Short-Term Memory entry - stored in RAM for fast access"""
    content: str
    timestamp: float
    user_name: str
    entry_type: str  # 'user_input', 'bot_response', 'system_event', 'visual_context'
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    conversation_id: str = ""
    visual_context: Optional[Dict[str, Any]] = None

@dataclass
class ConversationSession:
    """Active conversation session with STM and context"""
    session_id: str
    user_name: str
    stm_entries: deque = field(default_factory=lambda: deque(maxlen=50))  # Last 50 entries in RAM
    conversation_summary: str = ""
    current_topic: str = ""
    game_context: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    context_window_start: float = field(default_factory=time.time)
    visual_context_history: deque = field(default_factory=lambda: deque(maxlen=10))  # Last 10 visual contexts

class EnhancedConversationMemory:
    """
    Enhanced conversation memory system with STM in RAM and LTM in RAG
    
    Features:
    - Fast STM access in RAM for immediate context
    - Persistent LTM storage in RAG for long-term memory
    - Automatic context window management
    - Visual context integration
    - Conversation flow tracking
    - Memory consolidation and retrieval
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.settings = app_context.global_settings
        
        # STM Storage (RAM)
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.global_stm: deque = deque(maxlen=100)  # Global conversation flow
        
        # LTM Integration (RAG)
        self.rag_service = None
        self.ltm_collection = "conversation_memory"
        
        # Configuration
        self.stm_window_minutes = self.settings.get('CONVERSATION_MEMORY', {}).get('stm_window_minutes', 30)
        self.max_active_sessions = self.settings.get('CONVERSATION_MEMORY', {}).get('max_active_sessions', 10)
        self.consolidation_threshold = self.settings.get('CONVERSATION_MEMORY', {}).get('consolidation_threshold', 20)
        self.context_window_minutes = self.settings.get('CONVERSATION_MEMORY', {}).get('context_window_minutes', 60)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.cleanup_thread = None
        self.consolidation_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize RAG service
        self._initialize_rag_service()
        
        self.logger.info(f"[EnhancedConversationMemory] Initialized with STM window={self.stm_window_minutes}min, context window={self.context_window_minutes}min")
    
    def _initialize_rag_service(self):
        """Initialize RAG service for LTM storage"""
        try:
            from services.llamacpp_rag_service import LlamaCppRAGService
            self.rag_service = LlamaCppRAGService(self.settings)
            
            # Ensure LTM collection exists
            if self.rag_service and not self.rag_service.collection_exists(self.ltm_collection):
                self.rag_service._create_collection(self.ltm_collection)
                self.logger.info(f"[EnhancedConversationMemory] Created LTM collection: {self.ltm_collection}")
            
            self.logger.info("[EnhancedConversationMemory] RAG service initialized for LTM")
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Failed to initialize RAG service: {e}")
            self.rag_service = None
    
    def start_background_tasks(self):
        """Start background cleanup and consolidation tasks"""
        # Start cleanup thread
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("[EnhancedConversationMemory] Cleanup thread started")
        
        # Start consolidation thread
        if self.consolidation_thread is None or not self.consolidation_thread.is_alive():
            self.consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
            self.consolidation_thread.start()
            self.logger.info("[EnhancedConversationMemory] Consolidation thread started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.shutdown_event.set()
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        if self.consolidation_thread and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5.0)
        
        self.logger.info("[EnhancedConversationMemory] Background tasks stopped")
    
    def _cleanup_worker(self):
        """Background worker for cleaning up old sessions"""
        while not self.shutdown_event.wait(300):  # Check every 5 minutes
            try:
                self._cleanup_old_sessions()
            except Exception as e:
                self.logger.error(f"[EnhancedConversationMemory] Cleanup error: {e}")
    
    def _consolidation_worker(self):
        """Background worker for consolidating STM to LTM"""
        while not self.shutdown_event.wait(600):  # Check every 10 minutes
            try:
                self._consolidate_sessions_to_ltm()
            except Exception as e:
                self.logger.error(f"[EnhancedConversationMemory] Consolidation error: {e}")
    
    def add_conversation_entry(self, user_name: str, content: str, entry_type: str = 'user_input',
                             metadata: Optional[Dict[str, Any]] = None, importance: float = 1.0,
                             visual_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation entry to STM and potentially LTM
        
        Args:
            user_name: Name of the user
            content: Content of the entry
            entry_type: Type of entry
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            visual_context: Visual context if available
        
        Returns:
            True if added successfully
        """
        try:
            with self.lock:
                # Get or create session
                session_id = self._get_session_id(user_name)
                session = self._get_or_create_session(session_id, user_name)
                
                # Create STM entry
                entry = STMEntry(
                    content=content,
                    timestamp=time.time(),
                    user_name=user_name,
                    entry_type=entry_type,
                    metadata=metadata or {},
                    importance=importance,
                    conversation_id=session_id,
                    visual_context=visual_context
                )
                
                # Add to session STM
                session.stm_entries.append(entry)
                session.last_activity = time.time()
                
                # Add to global STM
                self.global_stm.append(entry)
                
                # Update visual context history if available
                if visual_context:
                    session.visual_context_history.append({
                        'timestamp': time.time(),
                        'context': visual_context
                    })
                
                # Update conversation summary
                self._update_conversation_summary(session)
                
                # Check if consolidation is needed
                if len(session.stm_entries) >= self.consolidation_threshold:
                    self._schedule_consolidation(session_id)
                
                self.logger.debug(f"[EnhancedConversationMemory] Added {entry_type} for {user_name}: '{content[:50]}...'")
                return True
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error adding entry: {e}")
            return False
    
    def get_conversation_context(self, user_name: str, include_ltm: bool = True, 
                               max_stm_entries: int = 10, max_ltm_results: int = 3) -> Dict[str, Any]:
        """
        Get comprehensive conversation context including STM and LTM
        
        Args:
            user_name: Name of the user
            include_ltm: Whether to include LTM results
            max_stm_entries: Maximum STM entries to include
            max_ltm_results: Maximum LTM results to include
        
        Returns:
            Dictionary with conversation context
        """
        try:
            with self.lock:
                session_id = self._get_session_id(user_name)
                session = self.active_sessions.get(session_id)
                
                context = {
                    'user_name': user_name,
                    'session_id': session_id,
                    'stm_entries': [],
                    'conversation_summary': "",
                    'current_topic': "",
                    'game_context': None,
                    'ltm_results': [],
                    'visual_context': None,
                    'context_window_start': time.time() - (self.context_window_minutes * 60)
                }
                
                # Get STM entries
                if session:
                    # Get recent STM entries within context window
                    recent_entries = []
                    cutoff_time = time.time() - (self.context_window_minutes * 60)
                    
                    for entry in session.stm_entries:
                        if entry.timestamp >= cutoff_time:
                            recent_entries.append(entry)
                    
                    # Take most recent entries
                    context['stm_entries'] = recent_entries[-max_stm_entries:]
                    context['conversation_summary'] = session.conversation_summary
                    context['current_topic'] = session.current_topic
                    context['game_context'] = session.game_context
                    
                    # Get latest visual context
                    if session.visual_context_history:
                        context['visual_context'] = session.visual_context_history[-1]
                
                # Get LTM results if requested and available
                if include_ltm and self.rag_service and session:
                    ltm_query = self._build_ltm_query(user_name, session)
                    if ltm_query:
                        ltm_results = self.rag_service.query(
                            self.ltm_collection, 
                            ltm_query, 
                            n_results=max_ltm_results
                        )
                        context['ltm_results'] = ltm_results
                
                self.logger.debug(f"[EnhancedConversationMemory] Retrieved context for {user_name}: {len(context['stm_entries'])} STM entries, {len(context['ltm_results'])} LTM results")
                return context
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error getting context: {e}")
            return {'user_name': user_name, 'error': str(e)}
    
    def _get_session_id(self, user_name: str) -> str:
        """Get session ID for user (special handling for VirtualAudio)"""
        if user_name == "VirtualAudio":
            return "VirtualAudio_Session"
        return f"{user_name}_{int(time.time() // (self.stm_window_minutes * 60))}"
    
    def _get_or_create_session(self, session_id: str, user_name: str) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id not in self.active_sessions:
            # Check if we need to remove old sessions
            if len(self.active_sessions) >= self.max_active_sessions:
                self._remove_oldest_session()
            
            self.active_sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_name=user_name
            )
            self.logger.info(f"[EnhancedConversationMemory] Created new session: {session_id}")
        
        return self.active_sessions[session_id]
    
    def _update_conversation_summary(self, session: ConversationSession):
        """Update conversation summary for session"""
        try:
            if len(session.stm_entries) < 3:
                return
            
            # Create summary from recent entries
            recent_entries = list(session.stm_entries)[-10:]  # Last 10 entries
            summary_parts = []
            
            # Group by user and bot interactions
            user_messages = [e.content for e in recent_entries if e.entry_type == 'user_input']
            bot_responses = [e.content for e in recent_entries if e.entry_type == 'bot_response']
            
            if user_messages:
                summary_parts.append(f"User topics: {', '.join(user_messages[-3:])}")
            
            if bot_responses:
                summary_parts.append(f"Bot responses: {', '.join(bot_responses[-3:])}")
            
            # Detect current topic
            if user_messages:
                session.current_topic = self._extract_topic(user_messages[-1])
            
            session.conversation_summary = " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error updating summary: {e}")
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text"""
        # Simple topic extraction - could be enhanced with NLP
        words = text.lower().split()
        topic_keywords = ['game', 'health', 'inventory', 'character', 'quest', 'combat', 'level']
        
        for keyword in topic_keywords:
            if keyword in words:
                return keyword
        
        return "general"
    
    def _build_ltm_query(self, user_name: str, session: ConversationSession) -> str:
        """Build query for LTM retrieval"""
        if not session.conversation_summary:
            return ""
        
        # Build query from conversation summary and recent topics
        query_parts = [session.conversation_summary]
        
        if session.current_topic:
            query_parts.append(f"topic: {session.current_topic}")
        
        if session.game_context:
            query_parts.append(f"game: {session.game_context}")
        
        return " ".join(query_parts)
    
    def _schedule_consolidation(self, session_id: str):
        """Schedule session consolidation to LTM"""
        try:
            # This would typically be done asynchronously
            # For now, we'll do it synchronously but could be enhanced
            session = self.active_sessions.get(session_id)
            if session and self.rag_service:
                self._consolidate_session_to_ltm(session)
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Consolidation scheduling error: {e}")
    
    def _consolidate_session_to_ltm(self, session: ConversationSession):
        """Consolidate session data to LTM"""
        try:
            if not self.rag_service:
                return
            
            # Create consolidation entry
            consolidation_data = {
                'session_id': session.session_id,
                'user_name': session.user_name,
                'conversation_summary': session.conversation_summary,
                'current_topic': session.current_topic,
                'game_context': session.game_context,
                'entry_count': len(session.stm_entries),
                'timestamp': time.time(),
                'entries': [
                    {
                        'content': entry.content,
                        'entry_type': entry.entry_type,
                        'timestamp': entry.timestamp,
                        'importance': entry.importance
                    }
                    for entry in session.stm_entries
                ]
            }
            
            # Store in RAG
            consolidation_text = f"Conversation with {session.user_name}: {session.conversation_summary}"
            metadata = {
                'user_name': session.user_name,
                'session_id': session.session_id,
                'consolidation_data': json.dumps(consolidation_data)
            }
            
            success = self.rag_service.ingest_text(
                self.ltm_collection,
                consolidation_text,
                metadata
            )
            
            if success:
                self.logger.info(f"[EnhancedConversationMemory] Consolidated session {session.session_id} to LTM")
                # Clear old STM entries after consolidation
                session.stm_entries.clear()
                session.conversation_summary = ""
            else:
                self.logger.error(f"[EnhancedConversationMemory] Failed to consolidate session {session.session_id}")
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Consolidation error: {e}")
    
    def _consolidate_sessions_to_ltm(self):
        """Consolidate all sessions that meet the threshold"""
        try:
            with self.lock:
                sessions_to_consolidate = []
                
                for session_id, session in self.active_sessions.items():
                    if len(session.stm_entries) >= self.consolidation_threshold:
                        sessions_to_consolidate.append(session_id)
                
                for session_id in sessions_to_consolidate:
                    session = self.active_sessions.get(session_id)
                    if session:
                        self._consolidate_session_to_ltm(session)
                        
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Batch consolidation error: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions"""
        try:
            with self.lock:
                current_time = time.time()
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    # Keep VirtualAudio session longer
                    if session_id == "VirtualAudio_Session":
                        timeout = self.stm_window_minutes * 2 * 60  # 2x longer for VirtualAudio
                    else:
                        timeout = self.stm_window_minutes * 60
                    
                    if current_time - session.last_activity > timeout:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    # Consolidate before removing
                    session = self.active_sessions.get(session_id)
                    if session and len(session.stm_entries) > 0:
                        self._consolidate_session_to_ltm(session)
                    
                    del self.active_sessions[session_id]
                    self.logger.info(f"[EnhancedConversationMemory] Removed old session: {session_id}")
                    
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Cleanup error: {e}")
    
    def _remove_oldest_session(self):
        """Remove the oldest session to make room for new one"""
        try:
            if not self.active_sessions:
                return
            
            # Don't remove VirtualAudio session if it's the only one
            if len(self.active_sessions) == 1 and "VirtualAudio_Session" in self.active_sessions:
                return
            
            # Find oldest session (excluding VirtualAudio if possible)
            oldest_session_id = None
            oldest_time = float('inf')
            
            for session_id, session in self.active_sessions.items():
                if session_id != "VirtualAudio_Session" and session.last_activity < oldest_time:
                    oldest_time = session.last_activity
                    oldest_session_id = session_id
            
            # If no non-VirtualAudio session found, remove VirtualAudio
            if oldest_session_id is None:
                oldest_session_id = "VirtualAudio_Session"
            
            if oldest_session_id:
                # Consolidate before removing
                session = self.active_sessions.get(oldest_session_id)
                if session and len(session.stm_entries) > 0:
                    self._consolidate_session_to_ltm(session)
                
                del self.active_sessions[oldest_session_id]
                self.logger.info(f"[EnhancedConversationMemory] Removed oldest session: {oldest_session_id}")
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Remove oldest session error: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            with self.lock:
                stats = {
                    'active_sessions': len(self.active_sessions),
                    'global_stm_entries': len(self.global_stm),
                    'ltm_collection': self.ltm_collection,
                    'rag_service_available': self.rag_service is not None,
                    'sessions': {}
                }
                
                for session_id, session in self.active_sessions.items():
                    stats['sessions'][session_id] = {
                        'user_name': session.user_name,
                        'stm_entries': len(session.stm_entries),
                        'last_activity': session.last_activity,
                        'current_topic': session.current_topic,
                        'game_context': session.game_context
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error getting stats: {e}")
            return {'error': str(e)}
    
    def clear_user_memory(self, user_name: str) -> bool:
        """Clear memory for a specific user"""
        try:
            with self.lock:
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    if session.user_name == user_name:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self.active_sessions[session_id]
                
                self.logger.info(f"[EnhancedConversationMemory] Cleared memory for user: {user_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error clearing user memory: {e}")
            return False
    
    def clear_all_memory(self) -> bool:
        """Clear all memory"""
        try:
            with self.lock:
                self.active_sessions.clear()
                self.global_stm.clear()
                self.logger.info("[EnhancedConversationMemory] Cleared all memory")
                return True
                
        except Exception as e:
            self.logger.error(f"[EnhancedConversationMemory] Error clearing all memory: {e}")
            return False 