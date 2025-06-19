"""
Conversational Memory Buffer Service for DanzarVLM

Implements short-term conversational memory similar to LangChain's ConversationBufferMemory
but without requiring vector databases. Stores recent conversation turns in memory.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

@dataclass
class ConversationTurn:
    """A single conversation turn between user and AI"""
    timestamp: float
    user_name: str
    user_query: str
    ai_response: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'user_name': self.user_name,
            'user_query': self.user_query,
            'ai_response': self.ai_response,
            'context': self.context
        }
    
    def get_formatted_turn(self) -> str:
        """Get formatted conversation turn for display"""
        return f"Human: {self.user_query}\nAI: {self.ai_response}"

class ConversationBuffer:
    """
    Simple conversational memory buffer that stores recent conversation history
    Based on LangChain's ConversationBufferMemory approach
    """
    
    def __init__(self, max_turns: int = 10, max_age_minutes: int = 60):
        """
        Initialize conversation buffer
        
        Args:
            max_turns: Maximum number of conversation turns to keep
            max_age_minutes: Maximum age of conversation turns in minutes
        """
        self.max_turns = max_turns
        self.max_age_seconds = max_age_minutes * 60
        self.conversation_turns: deque = deque(maxlen=max_turns)
        self.logger = logging.getLogger(f"{__name__}.ConversationBuffer")
        
        self.logger.info(f"[ConversationBuffer] Initialized with max_turns={max_turns}, max_age={max_age_minutes}min")
    
    def add_turn(self, user_name: str, user_query: str, ai_response: str, context: Dict[str, Any] = None) -> None:
        """Add a new conversation turn to the buffer"""
        if context is None:
            context = {}
            
        turn = ConversationTurn(
            timestamp=time.time(),
            user_name=user_name,
            user_query=user_query,
            ai_response=ai_response,
            context=context
        )
        
        self.conversation_turns.append(turn)
        self._cleanup_old_turns()
        
        self.logger.debug(f"[ConversationBuffer] Added turn from {user_name}: '{user_query[:50]}...'")
        self.logger.debug(f"[ConversationBuffer] Buffer now has {len(self.conversation_turns)} turns")
    
    def _cleanup_old_turns(self) -> None:
        """Remove conversation turns that are too old"""
        current_time = time.time()
        cutoff_time = current_time - self.max_age_seconds
        
        # Remove old turns from the left (oldest first)
        while self.conversation_turns and self.conversation_turns[0].timestamp < cutoff_time:
            old_turn = self.conversation_turns.popleft()
            self.logger.debug(f"[ConversationBuffer] Removed expired turn from {old_turn.user_name}")
    
    def get_last_question(self, user_name: str = None, exclude_meta: bool = True) -> Optional[str]:
        """
        Get the last question asked by the user
        
        Args:
            user_name: Filter by specific user (None for any user)
            exclude_meta: Exclude meta-conversational questions
        """
        self._cleanup_old_turns()
        
        # Meta-conversational patterns to exclude
        meta_patterns = [
            'what was the last', 'what did i ask', 'what have we been',
            'what are we talking', 'what were we discussing', 'conversation about',
            'what was our last', 'previous question', 'last question'
        ] if exclude_meta else []
        
        # Search backwards through conversation turns
        for turn in reversed(self.conversation_turns):
            # Filter by user if specified
            if user_name and turn.user_name.lower() != user_name.lower():
                continue
            
            query_lower = turn.user_query.lower()
            
            # Skip meta-conversational questions if requested
            if exclude_meta and any(pattern in query_lower for pattern in meta_patterns):
                continue
            
            return turn.user_query
        
        return None
    
    def get_conversation_summary(self, user_name: str = None, last_n_turns: int = 3) -> str:
        """
        Get a summary of recent conversation
        
        Args:
            user_name: Filter by specific user (None for any user)
            last_n_turns: Number of recent turns to include
        """
        self._cleanup_old_turns()
        
        if not self.conversation_turns:
            return "We haven't had any conversation yet. This is the start of our discussion!"
        
        # Get recent turns
        recent_turns = list(self.conversation_turns)[-last_n_turns:]
        
        # Filter by user if specified
        if user_name:
            recent_turns = [turn for turn in recent_turns if turn.user_name.lower() == user_name.lower()]
        
        if not recent_turns:
            return f"I don't have any recent conversation with {user_name or 'you'}."
        
        # Extract topics and context
        topics = set()
        games = set()
        
        for turn in recent_turns:
            query_lower = turn.user_query.lower()
            response_lower = turn.ai_response.lower()
            
            # Detect games mentioned
            game_terms = ['everquest', 'eq', 'rimworld', 'minecraft', 'wow', 'warcraft']
            for game in game_terms:
                if game in query_lower or game in response_lower:
                    games.add(game)
            
            # Detect topics
            topic_terms = ['class', 'race', 'skill', 'quest', 'zone', 'item', 'spell', 'guide']
            for topic in topic_terms:
                if topic in query_lower:
                    topics.add(topic + 's' if not topic.endswith('s') else topic)
        
        # Build summary
        summary_parts = []
        
        if games:
            games_str = ', '.join(games).title()
            summary_parts.append(f"We were discussing {games_str}")
        
        if topics:
            topics_str = ', '.join(topics)
            if games:
                summary_parts.append(f"specifically about {topics_str}")
            else:
                summary_parts.append(f"We were talking about {topics_str}")
        
        # Add last question context
        last_turn = recent_turns[-1]
        if not any(pattern in last_turn.user_query.lower() for pattern in [
            'what was the last', 'what did i ask', 'what have we been', 'conversation about'
        ]):
            summary_parts.append(f"Your last question was: '{last_turn.user_query}'")
        
        if summary_parts:
            return '. '.join(summary_parts) + '.'
        else:
            # Fallback to simple summary
            last_turn = recent_turns[-1]
            return f"We were just discussing: '{last_turn.user_query}'."
    
    def get_conversation_history(self, user_name: str = None, format_for_llm: bool = False) -> List[Dict[str, Any]]:
        """
        Get conversation history as a list
        
        Args:
            user_name: Filter by specific user (None for any user)
            format_for_llm: Format for LLM consumption
        """
        self._cleanup_old_turns()
        
        turns = list(self.conversation_turns)
        
        # Filter by user if specified
        if user_name:
            turns = [turn for turn in turns if turn.user_name.lower() == user_name.lower()]
        
        if format_for_llm:
            # Format for LLM prompt
            history = []
            for turn in turns:
                history.append({
                    'role': 'user',
                    'content': turn.user_query
                })
                history.append({
                    'role': 'assistant', 
                    'content': turn.ai_response
                })
            return history
        else:
            # Return raw turn data
            return [turn.to_dict() for turn in turns]
    
    def get_formatted_history(self, user_name: str = None, max_chars: int = 1000) -> str:
        """
        Get formatted conversation history as a string (similar to LangChain's buffer)
        
        Args:
            user_name: Filter by specific user
            max_chars: Maximum characters to return
        """
        self._cleanup_old_turns()
        
        turns = list(self.conversation_turns)
        
        # Filter by user if specified
        if user_name:
            turns = [turn for turn in turns if turn.user_name.lower() == user_name.lower()]
        
        if not turns:
            return ""
        
        # Build formatted history
        formatted_turns = []
        for turn in turns:
            formatted_turn = turn.get_formatted_turn()
            formatted_turns.append(formatted_turn)
        
        history_text = '\n\n'.join(formatted_turns)
        
        # Truncate if too long
        if len(history_text) > max_chars:
            history_text = history_text[:max_chars] + "...[truncated]"
        
        return history_text
    
    def clear_history(self, user_name: str = None) -> None:
        """Clear conversation history"""
        if user_name:
            # Remove turns for specific user
            self.conversation_turns = deque(
                [turn for turn in self.conversation_turns if turn.user_name.lower() != user_name.lower()],
                maxlen=self.max_turns
            )
            self.logger.info(f"[ConversationBuffer] Cleared history for user: {user_name}")
        else:
            # Clear all history
            self.conversation_turns.clear()
            self.logger.info("[ConversationBuffer] Cleared all conversation history")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        self._cleanup_old_turns()
        
        if not self.conversation_turns:
            return {
                'total_turns': 0,
                'unique_users': 0,
                'oldest_turn_age': 0,
                'buffer_utilization': 0.0
            }
        
        users = set(turn.user_name for turn in self.conversation_turns)
        oldest_turn = min(turn.timestamp for turn in self.conversation_turns)
        oldest_age = time.time() - oldest_turn
        
        return {
            'total_turns': len(self.conversation_turns),
            'unique_users': len(users),
            'oldest_turn_age_seconds': oldest_age,
            'buffer_utilization': len(self.conversation_turns) / self.max_turns,
            'max_turns': self.max_turns,
            'max_age_seconds': self.max_age_seconds
        }

class ConversationBufferService:
    """
    Service wrapper for ConversationBuffer with integration into DanzarVLM
    """
    
    def __init__(self, app_context, max_turns: int = 10, max_age_minutes: int = 60):
        """Initialize the conversation buffer service"""
        self.ctx = app_context
        self.logger = app_context.logger
        self.buffer = ConversationBuffer(max_turns=max_turns, max_age_minutes=max_age_minutes)
        
        self.logger.info("[ConversationBufferService] Initialized conversation buffer service")
    
    def is_conversational_query(self, query: str) -> bool:
        """
        Enhanced detection of conversational queries that should use memory buffer
        """
        query_lower = query.lower().strip()
        
        # Comprehensive conversational patterns
        conversational_patterns = [
            # Last question patterns
            'what was the last', 'what did i ask', 'last question', 'previous question',
            'what was i asking', 'what did we talk', 'what were we discussing',
            
            # Conversation history patterns  
            'what have we been', 'what are we talking', 'what were we talking',
            'what have we talked', 'conversation about', 'our conversation',
            'what was our last', 'we discussed', 'just said', 'what was that',
            
            # General memory patterns
            'what have we', 'what did we', 'what were we', 'remember when',
            'did i mention', 'did we talk', 'earlier you said', 'before you said'
        ]
        
        return any(pattern in query_lower for pattern in conversational_patterns)
    
    def handle_conversational_query(self, query: str, user_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        Handle conversational queries using the memory buffer
        
        Returns:
            Tuple of (response, metadata)
        """
        query_lower = query.lower().strip()
        
        # Determine query type and generate appropriate response
        if any(pattern in query_lower for pattern in [
            'what was the last', 'last question', 'what did i ask', 'previous question'
        ]):
            # User asking about their last question
            last_question = self.buffer.get_last_question(user_name, exclude_meta=True)
            if last_question:
                response = f"Your last question was: '{last_question}'"
            else:
                response = "I don't have record of a previous question in our conversation."
        
        else:
            # General conversation summary
            response = self.buffer.get_conversation_summary(user_name, last_n_turns=3)
        
        # Create metadata
        metadata = {
            'method': 'conversation_buffer_memory',
            'buffer_stats': self.buffer.get_stats(),
            'query_type': 'conversational',
            'processing_time': 0.001,  # Very fast since it's in-memory
            'success': True
        }
        
        self.logger.info(f"[ConversationBufferService] Handled conversational query for {user_name}: '{query}'")
        self.logger.debug(f"[ConversationBufferService] Response: {response}")
        
        return response, metadata
    
    def add_conversation_turn(self, user_name: str, user_query: str, ai_response: str, context: Dict[str, Any] = None) -> None:
        """Add a conversation turn to the buffer"""
        self.buffer.add_turn(user_name, user_query, ai_response, context or {})
        
        self.logger.debug(f"[ConversationBufferService] Added turn: {user_name} -> '{user_query[:50]}...'")
    
    def get_conversation_context_for_rag(self, user_name: str, max_chars: int = 500) -> str:
        """
        Get conversation context formatted for RAG system enhancement
        """
        history = self.buffer.get_formatted_history(user_name, max_chars=max_chars)
        
        if history:
            return f"Recent conversation context:\n{history}\n\nCurrent query:"
        else:
            return "" 