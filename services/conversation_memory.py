#!/usr/bin/env python3
"""
Short-Term Conversation Memory Service for DanzarVLM
Maintains context for ongoing conversations with automatic cleanup
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    user_name: str
    user_message: str
    bot_response: str
    timestamp: float
    user_sentiment: str = "neutral"  # positive, negative, neutral
    topic: str = ""  # detected topic
    importance: float = 0.5  # 0.0 to 1.0

class ConversationMemoryService:
    """
    Manages short-term conversation memory for more contextual responses
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = logging.getLogger("DanzarVLM.ConversationMemory")
        
        # Conversation storage per user
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))  # Max 20 turns per user
        self.conversation_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Global recent context (cross-user)
        self.recent_global_context = deque(maxlen=10)  # Last 10 interactions across all users
        self.global_lock = threading.Lock()
        
        # Topic tracking
        self.current_topics: Dict[str, str] = {}  # user -> current topic
        self.topic_transitions: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # user -> [(topic, timestamp), ...]
        
        # Configuration
        self.max_context_age = app_context.global_settings.get("CONVERSATION_MEMORY", {}).get("max_age_seconds", 3600)  # 1 hour
        self.context_decay_factor = app_context.global_settings.get("CONVERSATION_MEMORY", {}).get("decay_factor", 0.1)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("[ConversationMemory] Initialized with max age: %ds", self.max_context_age)

    def add_conversation_turn(self, user_name: str, user_message: str, bot_response: str) -> None:
        """Add a new conversation turn"""
        try:
            # Detect topic and sentiment quickly
            topic = self._detect_topic(user_message)
            sentiment = self._detect_sentiment(user_message)
            importance = self._calculate_importance(user_message, topic, sentiment)
            
            turn = ConversationTurn(
                user_name=user_name,
                user_message=user_message,
                bot_response=bot_response,
                timestamp=time.time(),
                user_sentiment=sentiment,
                topic=topic,
                importance=importance
            )
            
            # Add to user-specific conversation
            with self.conversation_locks[user_name]:
                self.conversations[user_name].append(turn)
            
            # Add to global recent context
            with self.global_lock:
                self.recent_global_context.append(turn)
            
            # Update topic tracking
            self._update_topic_tracking(user_name, topic)
            
            self.logger.debug(f"[ConversationMemory] Added turn for {user_name}: topic={topic}, sentiment={sentiment}")
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error adding conversation turn: {e}")

    def get_conversation_context(self, user_name: str, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context for a user"""
        try:
            with self.conversation_locks[user_name]:
                user_conversations = list(self.conversations[user_name])
            
            # Filter out old conversations
            current_time = time.time()
            recent_conversations = [
                turn for turn in user_conversations 
                if current_time - turn.timestamp < self.max_context_age
            ]
            
            # Sort by importance and recency, take the most relevant
            recent_conversations.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
            return recent_conversations[:max_turns]
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error getting conversation context: {e}")
            return []

    def get_topic_context(self, user_name: str, topic: str) -> List[ConversationTurn]:
        """Get conversation turns related to a specific topic"""
        try:
            with self.conversation_locks[user_name]:
                user_conversations = list(self.conversations[user_name])
            
            # Filter by topic and recency
            current_time = time.time()
            topic_conversations = [
                turn for turn in user_conversations 
                if turn.topic == topic and current_time - turn.timestamp < self.max_context_age
            ]
            
            # Sort by recency
            topic_conversations.sort(key=lambda x: x.timestamp, reverse=True)
            return topic_conversations[:3]  # Max 3 topic-related turns
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error getting topic context: {e}")
            return []

    def get_contextual_prompt(self, user_name: str, current_query: str) -> str:
        """Generate a contextual prompt based on conversation history"""
        try:
            context_turns = self.get_conversation_context(user_name, max_turns=3)
            
            if not context_turns:
                return ""
            
            # Build context string
            context_parts = []
            for turn in reversed(context_turns):  # Chronological order
                # Truncate long messages
                user_msg = turn.user_message[:100] + "..." if len(turn.user_message) > 100 else turn.user_message
                bot_msg = turn.bot_response[:100] + "..." if len(turn.bot_response) > 100 else turn.bot_response
                
                context_parts.append(f"User: {user_msg}")
                context_parts.append(f"AI: {bot_msg}")
            
            context_text = "\n".join(context_parts)
            
            # Check for topic continuity
            current_topic = self._detect_topic(current_query)
            if context_turns and current_topic == context_turns[0].topic:
                topic_note = f"\n[Continuing conversation about: {current_topic}]"
                context_text += topic_note
            
            return f"Recent conversation context:\n{context_text}\n\nCurrent question: {current_query}"
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error generating contextual prompt: {e}")
            return current_query

    def is_followup_question(self, user_name: str, query: str) -> bool:
        """Check if this query is likely a follow-up to recent conversation"""
        try:
            # Quick linguistic indicators
            followup_indicators = [
                'what about', 'how about', 'and', 'also', 'too', 'as well',
                'what else', 'anything else', 'more', 'other', 'another',
                'it', 'that', 'this', 'them', 'they', 'those'
            ]
            
            query_lower = query.lower()
            has_followup_indicator = any(indicator in query_lower for indicator in followup_indicators)
            
            # Check if user has recent context
            recent_context = self.get_conversation_context(user_name, max_turns=2)
            has_recent_context = len(recent_context) > 0 and (time.time() - recent_context[0].timestamp) < 300  # 5 minutes
            
            return has_followup_indicator and has_recent_context
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error checking followup: {e}")
            return False

    def _detect_topic(self, message: str) -> str:
        """Quickly detect the topic of a message"""
        message_lower = message.lower()
        
        # Gaming topics
        if any(word in message_lower for word in ['class', 'classes', 'warrior', 'wizard', 'cleric', 'paladin', 'monk', 'rogue']):
            return 'character_classes'
        elif any(word in message_lower for word in ['quest', 'mission', 'story', 'campaign']):
            return 'quests'
        elif any(word in message_lower for word in ['item', 'weapon', 'armor', 'equipment', 'gear']):
            return 'equipment'
        elif any(word in message_lower for word in ['spell', 'magic', 'cast', 'mana']):
            return 'magic'
        elif any(word in message_lower for word in ['level', 'experience', 'xp', 'skill']):
            return 'progression'
        elif any(word in message_lower for word in ['guild', 'group', 'party', 'raid']):
            return 'social'
        elif any(word in message_lower for word in ['everquest', 'eq']):
            return 'everquest_general'
        else:
            return 'general'

    def _detect_sentiment(self, message: str) -> str:
        """Quickly detect sentiment of a message"""
        message_lower = message.lower()
        
        positive_words = ['good', 'great', 'awesome', 'love', 'like', 'cool', 'nice', 'thanks', 'thank you']
        negative_words = ['bad', 'terrible', 'hate', 'dislike', 'awful', 'wrong', 'error', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _calculate_importance(self, message: str, topic: str, sentiment: str) -> float:
        """Calculate importance score for a conversation turn"""
        base_importance = 0.5
        
        # Topic-based adjustments
        important_topics = {'character_classes', 'equipment', 'quests'}
        if topic in important_topics:
            base_importance += 0.2
        
        # Sentiment adjustments
        if sentiment == 'negative':
            base_importance += 0.1  # Problems are important
        elif sentiment == 'positive':
            base_importance += 0.05
        
        # Message length (longer messages might be more important)
        if len(message) > 100:
            base_importance += 0.1
        
        # Question indicators
        if any(word in message.lower() for word in ['how', 'what', 'why', 'when', 'where']):
            base_importance += 0.1
        
        return min(1.0, base_importance)

    def _update_topic_tracking(self, user_name: str, topic: str) -> None:
        """Update topic tracking for user"""
        try:
            current_time = time.time()
            
            # Update current topic
            self.current_topics[user_name] = topic
            
            # Add to topic transitions
            self.topic_transitions[user_name].append((topic, current_time))
            
            # Keep only recent transitions (last hour)
            self.topic_transitions[user_name] = [
                (t, ts) for t, ts in self.topic_transitions[user_name]
                if current_time - ts < 3600
            ]
            
        except Exception as e:
            self.logger.error(f"[ConversationMemory] Error updating topic tracking: {e}")

    def _cleanup_loop(self) -> None:
        """Background cleanup of old conversations"""
        while True:
            try:
                time.sleep(300)  # Cleanup every 5 minutes
                current_time = time.time()
                
                # Clean up old conversations
                for user_name in list(self.conversations.keys()):
                    with self.conversation_locks[user_name]:
                        # Remove old turns
                        self.conversations[user_name] = deque(
                            [turn for turn in self.conversations[user_name] 
                             if current_time - turn.timestamp < self.max_context_age],
                            maxlen=20
                        )
                        
                        # Remove empty conversations
                        if not self.conversations[user_name]:
                            del self.conversations[user_name]
                
                # Clean up global context
                with self.global_lock:
                    self.recent_global_context = deque(
                        [turn for turn in self.recent_global_context 
                         if current_time - turn.timestamp < self.max_context_age],
                        maxlen=10
                    )
                
                # Clean up topic tracking
                for user_name in list(self.topic_transitions.keys()):
                    self.topic_transitions[user_name] = [
                        (topic, ts) for topic, ts in self.topic_transitions[user_name]
                        if current_time - ts < self.max_context_age
                    ]
                    if not self.topic_transitions[user_name]:
                        if user_name in self.current_topics:
                            del self.current_topics[user_name]
                        del self.topic_transitions[user_name]
                
                self.logger.debug("[ConversationMemory] Cleanup completed")
                
            except Exception as e:
                self.logger.error(f"[ConversationMemory] Cleanup error: {e}")

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_users': len(self.conversations),
            'total_conversations': sum(len(conv) for conv in self.conversations.values()),
            'active_topics': len(self.current_topics),
            'global_context_size': len(self.recent_global_context)
        } 