#!/usr/bin/env python3
"""
Vision-Conversation Coordinator Service
=======================================

Coordinates between vision integration service and conversational AI service
to ensure they work together instead of independently.
Now includes RAG memory integration for learning and context retrieval.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class CoordinationState(Enum):
    IDLE = "idle"
    VISION_COMMENTARY = "vision_commentary"
    CONVERSATION = "conversation"
    THINKING = "thinking"

@dataclass
class CoordinatedEvent:
    """A coordinated event that can be either vision or conversation"""
    event_type: str  # 'vision', 'conversation'
    content: str
    priority: str  # 'low', 'normal', 'high', 'critical'
    timestamp: float
    metadata: Dict[str, Any]

class VisionConversationCoordinator:
    """Coordinates vision commentary and conversational AI to work together with RAG memory integration."""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Coordination state
        self.coordination_state = CoordinationState.IDLE
        self.current_speaker = None
        self.last_activity_time = 0
        
        # Service references
        self.vision_integration_service = None
        self.conversational_ai_service = None
        
        # Memory services for RAG integration
        self.memory_service = getattr(self.app_context, 'memory_service', None)
        self.rag_service = getattr(self.app_context, 'rag_service_instance', None)
        
        # Coordination settings
        self.vision_cooldown = 3.0  # Seconds after conversation before vision commentary
        self.conversation_priority = 2.0  # Conversation gets priority over vision
        self.max_vision_frequency = 10.0  # Maximum vision commentary frequency
        self.last_vision_commentary = 0
        
        # Event queue for coordination
        self.event_queue = []
        self.max_queue_size = 10
        
        # Vision context for conversation
        self.recent_vision_events = []
        self.vision_context_window = 30.0  # Seconds to keep vision context
        
        # RAG memory settings
        self.coordination_collection = "vision_conversation_coordination"
        self.memory_importance_threshold = 0.7
        self.context_retrieval_limit = 5
        
        if self.logger:
            self.logger.info("[VisionConversationCoordinator] Coordinator initialized with RAG memory integration")
    
    async def initialize(self) -> bool:
        """Initialize the coordinator and connect to services."""
        try:
            if self.logger:
                self.logger.info("[VisionConversationCoordinator] Initializing...")
            
            # Connect to vision integration service
            self.vision_integration_service = getattr(self.app_context, 'vision_integration_service', None)
            if self.vision_integration_service:
                if self.logger:
                    self.logger.info("[VisionConversationCoordinator] ✅ Connected to Vision Integration Service")
            else:
                if self.logger:
                    self.logger.warning("[VisionConversationCoordinator] ⚠️ No Vision Integration Service available")
            
            # Connect to conversational AI service
            self.conversational_ai_service = getattr(self.app_context, 'conversational_ai_service', None)
            if self.conversational_ai_service:
                if self.logger:
                    self.logger.info("[VisionConversationCoordinator] ✅ Connected to Conversational AI Service")
            else:
                if self.logger:
                    self.logger.warning("[VisionConversationCoordinator] ⚠️ No Conversational AI Service available")
            
            # Initialize RAG collection for coordination memory
            await self._initialize_rag_collection()
            
            # Start coordination loop
            asyncio.create_task(self._coordination_loop())
            
            if self.logger:
                self.logger.info("[VisionConversationCoordinator] ✅ Coordinator initialized successfully with RAG memory")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] ❌ Initialization failed: {e}")
            return False
    
    async def _initialize_rag_collection(self):
        """Initialize RAG collection for storing coordination memories."""
        try:
            if not self.rag_service:
                if self.logger:
                    self.logger.warning("[VisionConversationCoordinator] ⚠️ No RAG service available - coordination memory disabled")
                return
            
            # Check if coordination collection exists
            if not self.rag_service.collection_exists(self.coordination_collection):
                if self.logger:
                    self.logger.info(f"[VisionConversationCoordinator] Creating RAG collection: {self.coordination_collection}")
                
                # Store initial coordination setup memory
                setup_memory = f"""Vision-Conversation Coordination System initialized.
This system coordinates between vision commentary and conversational AI to prevent interruptions and maintain context.
Key coordination rules:
- Conversation takes priority over vision commentary
- Vision commentary has cooldown period after conversation
- Vision context is shared with conversation responses
- All interactions are stored in RAG memory for learning
- System learns from past coordination patterns"""
                
                success = self.rag_service.ingest_text(
                    collection=self.coordination_collection,
                    text=setup_memory,
                    metadata={
                        'type': 'system_setup',
                        'timestamp': time.time(),
                        'importance': 1.0
                    }
                )
                
                if success:
                    if self.logger:
                        self.logger.info("[VisionConversationCoordinator] ✅ RAG collection initialized with coordination memory")
                else:
                    if self.logger:
                        self.logger.warning("[VisionConversationCoordinator] ⚠️ Failed to initialize RAG collection")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error initializing RAG collection: {e}")
    
    def _store_coordination_memory(self, event_type: str, content: str, metadata: Dict[str, Any], importance: float = 1.0):
        """Store coordination event in RAG memory."""
        try:
            if not self.rag_service:
                return
            
            # Create memory content
            memory_content = f"""COORDINATION EVENT - {event_type.upper()}
Content: {content}
State: {self.coordination_state.value}
Speaker: {self.current_speaker or 'none'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"""
            
            # Store in RAG
            success = self.rag_service.ingest_text(
                collection=self.coordination_collection,
                text=memory_content,
                metadata={
                    'event_type': event_type,
                    'coordination_state': self.coordination_state.value,
                    'current_speaker': self.current_speaker,
                    'timestamp': time.time(),
                    'importance': importance,
                    **metadata
                }
            )
            
            # Also store in memory service if available
            if self.memory_service:
                from services.memory_service import MemoryEntry
                memory_entry = MemoryEntry(
                    content=memory_content,
                    source=f"coordination_{event_type}",
                    timestamp=time.time(),
                    metadata={
                        'event_type': event_type,
                        'coordination_state': self.coordination_state.value,
                        'importance': importance,
                        **metadata
                    },
                    importance_score=importance
                )
                self.memory_service.store_memory(memory_entry)
            
            if success and self.logger:
                self.logger.debug(f"[VisionConversationCoordinator] Stored coordination memory: {event_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error storing coordination memory: {e}")
    
    async def _retrieve_coordination_context(self, query: str) -> str:
        """Retrieve relevant coordination context from RAG memory."""
        try:
            if not self.rag_service:
                return ""
            
            # Query RAG for relevant coordination memories
            results = self.rag_service.query(
                collection=self.coordination_collection,
                query_text=query,
                n_results=self.context_retrieval_limit
            )
            
            if results:
                context = f"Relevant coordination history:\n" + "\n".join(results)
                if self.logger:
                    self.logger.debug(f"[VisionConversationCoordinator] Retrieved {len(results)} coordination memories")
                return context
            
            return ""
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error retrieving coordination context: {e}")
            return ""
    
    def can_generate_vision_commentary(self) -> bool:
        """Check if vision commentary can be generated without interrupting conversation."""
        current_time = time.time()
        
        # Check if conversational AI is currently active
        if self.conversational_ai_service:
            try:
                # Check conversation state
                if hasattr(self.conversational_ai_service, 'conversation_state'):
                    conversation_state = self.conversational_ai_service.conversation_state
                    if conversation_state in ['speaking', 'thinking', 'listening']:
                        if self.logger:
                            self.logger.debug(f"[VisionConversationCoordinator] Skipping vision - Conversation state: {conversation_state}")
                        
                        # Store coordination decision in memory
                        self._store_coordination_memory(
                            'vision_skipped',
                            f"Vision commentary skipped due to conversation state: {conversation_state}",
                            {'reason': 'conversation_active', 'state': conversation_state},
                            importance=0.5
                        )
                        return False
                
                # Check recent conversation activity
                if hasattr(self.conversational_ai_service, 'last_speech_time'):
                    time_since_speech = current_time - self.conversational_ai_service.last_speech_time
                    if time_since_speech < self.vision_cooldown:
                        if self.logger:
                            self.logger.debug(f"[VisionConversationCoordinator] Skipping vision - Recent conversation ({time_since_speech:.1f}s ago)")
                        
                        # Store coordination decision in memory
                        self._store_coordination_memory(
                            'vision_skipped',
                            f"Vision commentary skipped due to recent conversation ({time_since_speech:.1f}s ago)",
                            {'reason': 'cooldown_period', 'time_since_speech': time_since_speech},
                            importance=0.5
                        )
                        return False
                        
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"[VisionConversationCoordinator] Error checking conversation state: {e}")
        
        # Check vision frequency limits
        if current_time - self.last_vision_commentary < self.max_vision_frequency:
            return False
        
        # Store successful vision permission in memory
        self._store_coordination_memory(
            'vision_allowed',
            "Vision commentary allowed - no conversation conflicts",
            {'reason': 'no_conflicts', 'time_since_last_vision': current_time - self.last_vision_commentary},
            importance=0.3
        )
        
        return True
    
    def add_vision_event(self, event_type: str, description: str, confidence: float):
        """Add a vision event to the context for conversation."""
        try:
            current_time = time.time()
            
            # Add to recent vision events
            self.recent_vision_events.append({
                'type': event_type,
                'description': description,
                'confidence': confidence,
                'timestamp': current_time
            })
            
            # Clean up old events
            self.recent_vision_events = [
                event for event in self.recent_vision_events
                if current_time - event['timestamp'] < self.vision_context_window
            ]
            
            # Store vision event in RAG memory
            self._store_coordination_memory(
                'vision_event',
                f"Vision event detected: {event_type} - {description} (confidence: {confidence:.2f})",
                {
                    'event_type': event_type,
                    'description': description,
                    'confidence': confidence,
                    'vision_events_count': len(self.recent_vision_events)
                },
                importance=0.8 if confidence > 0.7 else 0.5
            )
            
            if self.logger:
                self.logger.debug(f"[VisionConversationCoordinator] Added vision event: {event_type} - {description}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error adding vision event: {e}")
    
    def get_vision_context_for_conversation(self) -> str:
        """Get recent vision events as context for conversation responses."""
        if not self.recent_vision_events:
            return ""
        
        try:
            current_time = time.time()
            
            # Get recent events within context window
            recent_events = []
            for event in self.recent_vision_events[-3:]:  # Last 3 events
                if current_time - event['timestamp'] < self.vision_context_window:
                    recent_events.append(f"{event['type']}: {event['description']}")
            
            if recent_events:
                context = f"Recent vision events: {', '.join(recent_events)}"
                
                # Store context sharing in memory
                self._store_coordination_memory(
                    'context_shared',
                    f"Vision context shared with conversation: {context}",
                    {'shared_events': len(recent_events), 'context_window': self.vision_context_window},
                    importance=0.6
                )
                
                return context
            
            return ""
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"[VisionConversationCoordinator] Error getting vision context: {e}")
            return ""
    
    def notify_conversation_start(self, user_id: str):
        """Notify coordinator that conversation has started."""
        try:
            self.coordination_state = CoordinationState.CONVERSATION
            self.current_speaker = user_id
            self.last_activity_time = time.time()
            
            # Store conversation start in memory
            self._store_coordination_memory(
                'conversation_start',
                f"Conversation started by user: {user_id}",
                {'user_id': user_id, 'previous_state': self.coordination_state.value},
                importance=0.9
            )
            
            if self.logger:
                self.logger.debug(f"[VisionConversationCoordinator] Conversation started by {user_id}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error notifying conversation start: {e}")
    
    def notify_conversation_end(self):
        """Notify coordinator that conversation has ended."""
        try:
            previous_state = self.coordination_state.value
            self.coordination_state = CoordinationState.IDLE
            self.current_speaker = None
            
            # Store conversation end in memory
            self._store_coordination_memory(
                'conversation_end',
                f"Conversation ended, returning to idle state",
                {'previous_state': previous_state, 'cooldown_start': time.time()},
                importance=0.7
            )
            
            if self.logger:
                self.logger.debug("[VisionConversationCoordinator] Conversation ended")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error notifying conversation end: {e}")
    
    def notify_vision_commentary(self):
        """Notify coordinator that vision commentary is being generated."""
        try:
            self.coordination_state = CoordinationState.VISION_COMMENTARY
            self.last_vision_commentary = time.time()
            
            # Store vision commentary notification in memory
            self._store_coordination_memory(
                'vision_commentary',
                "Vision commentary being generated",
                {'commentary_count': len(self.recent_vision_events)},
                importance=0.6
            )
            
            if self.logger:
                self.logger.debug("[VisionConversationCoordinator] Vision commentary notification")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error notifying vision commentary: {e}")
    
    async def _coordination_loop(self):
        """Main coordination loop that manages state and learns from patterns."""
        try:
            if self.logger:
                self.logger.info("[VisionConversationCoordinator] Coordination loop started")
            
            loop_count = 0
            while not self.app_context.shutdown_event.is_set():
                try:
                    loop_count += 1
                    
                    # Process coordination logic
                    await self._process_coordination()
                    
                    # Periodic memory consolidation (every 100 loops)
                    if loop_count % 100 == 0:
                        await self._consolidate_coordination_memories()
                    
                    # Sleep to prevent excessive CPU usage
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionConversationCoordinator] Error in coordination loop: {e}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Coordination loop failed: {e}")
    
    async def _process_coordination(self):
        """Process coordination logic and learn from patterns."""
        try:
            current_time = time.time()
            
            # Check for state transitions that need memory storage
            if self.coordination_state == CoordinationState.IDLE:
                # Check if we should transition based on recent activity
                if current_time - self.last_activity_time > 60:  # 1 minute of inactivity
                    # Store idle period in memory
                    self._store_coordination_memory(
                        'idle_period',
                        f"System idle for {current_time - self.last_activity_time:.1f} seconds",
                        {'idle_duration': current_time - self.last_activity_time},
                        importance=0.3
                    )
            
            # Process event queue
            if self.event_queue:
                event = self.event_queue.pop(0)
                
                # Store event processing in memory
                self._store_coordination_memory(
                    'event_processed',
                    f"Processed {event.event_type} event: {event.content[:50]}...",
                    {
                        'event_type': event.event_type,
                        'priority': event.priority,
                        'queue_position': len(self.event_queue)
                    },
                    importance=0.5
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error in coordination processing: {e}")
    
    async def _consolidate_coordination_memories(self):
        """Periodically consolidate coordination memories for better learning."""
        try:
            if not self.rag_service:
                return
            
            # Query for recent coordination patterns
            consolidation_query = "coordination patterns vision conversation interaction"
            context = await self._retrieve_coordination_context(consolidation_query)
            
            if context:
                # Store consolidation summary
                consolidation_summary = f"""COORDINATION MEMORY CONSOLIDATION
Recent coordination patterns and learnings:
{context}

Key insights:
- Vision commentary frequency: {self.max_vision_frequency}s
- Conversation cooldown: {self.vision_cooldown}s
- Current state: {self.coordination_state.value}
- Recent vision events: {len(self.recent_vision_events)}"""
                
                self._store_coordination_memory(
                    'memory_consolidation',
                    consolidation_summary,
                    {
                        'consolidation_type': 'periodic',
                        'vision_events_count': len(self.recent_vision_events),
                        'coordination_state': self.coordination_state.value
                    },
                    importance=0.4
                )
                
                if self.logger:
                    self.logger.debug("[VisionConversationCoordinator] Memory consolidation completed")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error in memory consolidation: {e}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status including memory statistics."""
        try:
            status = {
                'coordination_state': self.coordination_state.value,
                'current_speaker': self.current_speaker,
                'last_activity_time': self.last_activity_time,
                'vision_cooldown': self.vision_cooldown,
                'max_vision_frequency': self.max_vision_frequency,
                'last_vision_commentary': self.last_vision_commentary,
                'recent_vision_events': len(self.recent_vision_events),
                'event_queue_size': len(self.event_queue),
                'rag_memory_enabled': self.rag_service is not None,
                'memory_service_enabled': self.memory_service is not None,
                'coordination_collection': self.coordination_collection
            }
            
            return status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionConversationCoordinator] Error getting status: {e}")
            return {'error': str(e)} 