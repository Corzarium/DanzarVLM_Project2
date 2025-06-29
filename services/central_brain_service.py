#!/usr/bin/env python3
"""
Central Brain Service - Main AI Intelligence Coordinator

This service acts as the central brain that receives visual sensor data
and coordinates with voice conversation to provide unified, contextual responses.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import threading

@dataclass
class VisualSensorData:
    """Data from visual sensors (YOLO, OCR, etc.)"""
    sensor_type: str  # 'yolo', 'ocr', 'template'
    label: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class VoiceContext:
    """Context from voice conversation"""
    user_message: str
    bot_response: str
    timestamp: float
    game_mentioned: Optional[str] = None

@dataclass
class BrainDecision:
    """Decision made by the central brain"""
    action_type: str  # 'commentary', 'response', 'question', 'alert'
    content: str
    priority: str  # 'low', 'normal', 'high', 'critical'
    timestamp: float
    context_sources: List[str]  # ['visual', 'voice', 'memory']

@dataclass
class VisualContext:
    """Represents current visual context from vision models"""
    timestamp: float
    detected_objects: List[Dict[str, Any]]  # YOLO detections
    ocr_text: List[str]  # OCR results
    ui_elements: List[Dict[str, Any]]  # Template matches
    scene_summary: str  # High-level scene description
    confidence: float  # Overall confidence in visual understanding
    game_context: Dict[str, Any]  # Game-specific context

@dataclass
class ConversationTurn:
    """Represents a conversation turn with visual context"""
    user_input: str
    visual_context: Optional[VisualContext]
    timestamp: float
    response: Optional[str] = None
    visual_references: Optional[List[str]] = None  # What visual elements were referenced
    
    def __post_init__(self):
        if self.visual_references is None:
            self.visual_references = []

class CentralBrainService:
    """
    Central AI brain that integrates vision and conversation.
    Acts as the main intelligence that can "see" and converse naturally.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Sensory data buffers
        self.visual_sensor_buffer: List[VisualSensorData] = []
        self.voice_context_buffer: List[VoiceContext] = []
        self.game_context: Dict[str, Any] = {}
        
        # Brain state
        self.is_active = False
        self.last_decision_time = 0
        self.decision_frequency = 5.0  # seconds between decisions
        self.max_buffer_size = 50
        
        # Services
        self.llm_service = None
        self.tts_service = None
        self.memory_service = None
        
        # Callbacks
        self.text_callback = None
        self.tts_callback = None
        
        # Visual context management
        self.current_visual_context: Optional[VisualContext] = None
        self.visual_context_history: deque = deque(maxlen=50)  # Keep last 50 visual contexts
        self.last_visual_update: float = 0
        self.visual_update_interval: float = 2.0  # Update visual context every 2 seconds
        
        # Conversation management
        self.conversation_history: deque = deque(maxlen=100)
        self.is_in_conversation: bool = False
        self.last_conversation_time: float = 0
        
        # Vision integration settings
        self.vision_integration_enabled: bool = True
        self.visual_context_threshold: float = 0.6  # Minimum confidence to include visual data
        self.max_visual_elements: int = 5  # Max visual elements to mention in response
        
        # Threading
        self.visual_context_lock = threading.Lock()
        self.conversation_lock = threading.Lock()
        
        self.logger.info("[CentralBrain] Initialized with vision integration")
    
    async def initialize(self) -> bool:
        """Initialize the central brain service."""
        try:
            self.logger.info("[CentralBrain] Initializing central brain...")
            
            # Get required services
            self.llm_service = getattr(self.app_context, 'llm_service', None)
            self.tts_service = getattr(self.app_context, 'tts_service', None)
            self.memory_service = getattr(self.app_context, 'memory_service', None)
            
            if not self.llm_service:
                self.logger.error("[CentralBrain] LLM service not available")
                return False
            
            # Set up visual context monitoring
            await self._start_visual_context_monitor()
            
            self.logger.info("[CentralBrain] Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Initialization failed: {e}", exc_info=True)
            return False
    
    async def _start_visual_context_monitor(self):
        """Start monitoring for visual context updates."""
        async def monitor_loop():
            while not self.app_context.shutdown_event.is_set():
                try:
                    # Check if vision integration is enabled
                    if self.vision_integration_enabled:
                        await self._update_visual_context()
                    
                    await asyncio.sleep(1.0)  # Check every second
                    
                except Exception as e:
                    self.logger.error(f"[CentralBrain] Visual context monitor error: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error
        
        # Start the monitor in the background
        asyncio.create_task(monitor_loop())
    
    async def _update_visual_context(self):
        """Update current visual context from vision models."""
        try:
            # Get latest vision data from app context
            vision_data = getattr(self.app_context, 'latest_vision_data', None)
            if not vision_data:
                return
            
            current_time = time.time()
            
            # Only update if enough time has passed
            if current_time - self.last_visual_update < self.visual_update_interval:
                return
            
            with self.visual_context_lock:
                # Extract visual information
                detected_objects = vision_data.get('yolo_detections', [])
                ocr_text = vision_data.get('ocr_results', [])
                ui_elements = vision_data.get('template_matches', [])
                
                # Create scene summary
                scene_summary = self._create_scene_summary(detected_objects, ocr_text, ui_elements)
                
                # Calculate overall confidence
                confidence = self._calculate_visual_confidence(detected_objects, ocr_text, ui_elements)
                
                # Get game context
                game_context = self._get_game_context()
                
                # Create new visual context
                new_context = VisualContext(
                    timestamp=current_time,
                    detected_objects=detected_objects,
                    ocr_text=ocr_text,
                    ui_elements=ui_elements,
                    scene_summary=scene_summary,
                    confidence=confidence,
                    game_context=game_context
                )
                
                # Update current context
                if self.current_visual_context:
                    self.visual_context_history.append(self.current_visual_context)
                
                self.current_visual_context = new_context
                self.last_visual_update = current_time
                
                self.logger.debug(f"[CentralBrain] Updated visual context: {scene_summary[:100]}...")
                
        except Exception as e:
            self.logger.error(f"[CentralBrain] Visual context update error: {e}")
    
    def _create_scene_summary(self, detected_objects: List, ocr_text: List, ui_elements: List) -> str:
        """Create a natural language summary of the current scene."""
        summary_parts = []
        
        # Add object detections
        if detected_objects:
            object_counts = {}
            for obj in detected_objects:
                obj_type = obj.get('label', 'unknown')
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            object_descriptions = []
            for obj_type, count in object_counts.items():
                if count == 1:
                    object_descriptions.append(f"a {obj_type}")
                else:
                    object_descriptions.append(f"{count} {obj_type}s")
            
            if object_descriptions:
                summary_parts.append(f"I can see {', '.join(object_descriptions)}")
        
        # Add OCR text (filtered for relevance)
        if ocr_text:
            relevant_text = [text for text in ocr_text if len(text.strip()) > 3 and 
                           not any(irrelevant in text.lower() for irrelevant in ['youtube', 'www.', 'http', '©', '®'])]
            if relevant_text:
                summary_parts.append(f"Text visible: {' '.join(relevant_text[:3])}")
        
        # Add UI elements
        if ui_elements:
            ui_descriptions = [elem.get('label', 'UI element') for elem in ui_elements[:3]]
            summary_parts.append(f"UI elements: {', '.join(ui_descriptions)}")
        
        if not summary_parts:
            return "The screen appears to be mostly empty or unclear."
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_visual_confidence(self, detected_objects: List, ocr_text: List, ui_elements: List) -> float:
        """Calculate overall confidence in visual understanding."""
        if not detected_objects and not ocr_text and not ui_elements:
            return 0.0
        
        total_confidence = 0.0
        total_weight = 0.0
        
        # Weight object detections
        for obj in detected_objects:
            conf = obj.get('confidence', 0.5)
            total_confidence += conf * 0.4  # 40% weight for objects
            total_weight += 0.4
        
        # Weight OCR results
        for text in ocr_text:
            if len(text.strip()) > 3:
                total_confidence += 0.8 * 0.3  # 30% weight for OCR
                total_weight += 0.3
                break  # Only count first good OCR result
        
        # Weight UI elements
        for elem in ui_elements:
            conf = elem.get('confidence', 0.7)
            total_confidence += conf * 0.3  # 30% weight for UI
            total_weight += 0.3
            break  # Only count first UI element
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_game_context(self) -> Dict[str, Any]:
        """Get current game context from app context."""
        context = {}
        
        # Get active game profile
        if hasattr(self.app_context, 'active_profile') and self.app_context.active_profile:
            context['game_name'] = self.app_context.active_profile.game_name
        
        # Get recent conversation context
        if self.conversation_history:
            recent_turns = list(self.conversation_history)[-3:]  # Last 3 turns
            context['recent_conversation'] = [
                turn.user_input for turn in recent_turns
            ]
        
        return context
    
    async def process_conversation(self, user_input: str, include_visual_context: bool = True) -> str:
        """
        Process a conversation turn with optional visual context integration.
        This is the main method for natural conversation with vision awareness.
        """
        try:
            current_time = time.time()
            
            # Get current visual context if requested and available
            visual_context = None
            if include_visual_context and self.vision_integration_enabled:
                with self.visual_context_lock:
                    visual_context = self.current_visual_context
            
            # Create conversation turn
            turn = ConversationTurn(
                user_input=user_input,
                visual_context=visual_context,
                timestamp=current_time
            )
            
            # Generate response with visual awareness
            response = await self._generate_visual_aware_response(turn)
            turn.response = response
            
            # Store in conversation history
            with self.conversation_lock:
                self.conversation_history.append(turn)
            
            self.is_in_conversation = True
            self.last_conversation_time = current_time
            
            return response
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Conversation processing error: {e}", exc_info=True)
            return "I'm having trouble processing that right now. Could you repeat?"
    
    async def _generate_visual_aware_response(self, turn: ConversationTurn) -> str:
        """Generate a response that naturally incorporates visual context."""
        try:
            # Build the prompt with visual context
            prompt = self._build_visual_aware_prompt(turn)
            
            # Get LLM service
            llm_service = self.app_context.get_service('llm_service')
            if not llm_service:
                return "I'm not connected to my language model right now."
            
            # Generate response
            response = await llm_service.generate_response(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Response generation error: {e}")
            return "I'm having trouble thinking about that right now."
    
    def _build_visual_aware_prompt(self, turn: ConversationTurn) -> str:
        """Build a prompt that naturally incorporates visual context."""
        prompt_parts = []
        
        # Use profile-based system prompt instead of hardcoded one
        system_prompt = self.app_context.active_profile.system_prompt_commentary
        
        prompt_parts.append(system_prompt)
        
        # Add visual context if available and relevant
        if turn.visual_context and turn.visual_context.confidence >= self.visual_context_threshold:
            visual_info = []
            
            # Add scene summary
            if turn.visual_context.scene_summary:
                visual_info.append(f"Current scene: {turn.visual_context.scene_summary}")
            
            # Add game context
            if turn.visual_context.game_context.get('game_name'):
                visual_info.append(f"Game: {turn.visual_context.game_context['game_name']}")
            
            # Add recent conversation context
            if turn.visual_context.game_context.get('recent_conversation'):
                recent = turn.visual_context.game_context['recent_conversation']
                visual_info.append(f"Recent conversation: {' '.join(recent[-2:])}")
            
            if visual_info:
                prompt_parts.append("Visual context:")
                prompt_parts.extend([f"- {info}" for info in visual_info])
        
        # Add user input
        prompt_parts.append(f"User: {turn.user_input}")
        prompt_parts.append("Danzar:")
        
        return "\n".join(prompt_parts)
    
    def get_visual_status(self) -> Dict[str, Any]:
        """Get current visual integration status."""
        with self.visual_context_lock:
            return {
                'vision_enabled': self.vision_integration_enabled,
                'current_context': self.current_visual_context.scene_summary if self.current_visual_context else None,
                'confidence': self.current_visual_context.confidence if self.current_visual_context else 0.0,
                'last_update': self.last_visual_update,
                'context_history_count': len(self.visual_context_history)
            }
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status."""
        with self.conversation_lock:
            return {
                'in_conversation': self.is_in_conversation,
                'conversation_history_count': len(self.conversation_history),
                'last_conversation_time': self.last_conversation_time
            }
    
    def update_vision_integration(self, enabled: bool):
        """Enable or disable vision integration."""
        self.vision_integration_enabled = enabled
        self.logger.info(f"[CentralBrain] Vision integration {'enabled' if enabled else 'disabled'}")
    
    def receive_visual_data(self, sensor_data: VisualSensorData):
        """Receive visual sensor data from vision pipeline."""
        try:
            self.visual_sensor_buffer.append(sensor_data)
            
            # Keep buffer size manageable
            if len(self.visual_sensor_buffer) > self.max_buffer_size:
                self.visual_sensor_buffer = self.visual_sensor_buffer[-self.max_buffer_size:]
            
            self.logger.debug(f"[CentralBrain] Received visual data: {sensor_data.sensor_type} - {sensor_data.label}")
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Error receiving visual data: {e}")
    
    def receive_voice_context(self, voice_context: VoiceContext):
        """Receive voice conversation context."""
        try:
            self.voice_context_buffer.append(voice_context)
            
            # Keep buffer size manageable
            if len(self.voice_context_buffer) > self.max_buffer_size:
                self.voice_context_buffer = self.voice_context_buffer[-self.max_buffer_size:]
            
            # Update game context if game is mentioned
            if voice_context.game_mentioned:
                self.game_context['current_game'] = voice_context.game_mentioned
                self.game_context['last_mentioned'] = time.time()
            
            self.logger.debug(f"[CentralBrain] Received voice context: {voice_context.user_message[:50]}...")
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Error receiving voice context: {e}")
    
    async def _brain_processing_loop(self):
        """Main brain processing loop."""
        try:
            while self.is_active:
                current_time = time.time()
                
                # Check if it's time to make a decision
                if current_time - self.last_decision_time >= self.decision_frequency:
                    await self._make_brain_decision()
                    self.last_decision_time = current_time
                
                await asyncio.sleep(1.0)  # Check every second
                
        except asyncio.CancelledError:
            self.logger.info("[CentralBrain] Brain processing loop cancelled")
        except Exception as e:
            self.logger.error(f"[CentralBrain] Brain processing loop error: {e}")
    
    async def _make_brain_decision(self):
        """Make a decision based on current sensory data and context."""
        try:
            if not self.visual_sensor_buffer and not self.voice_context_buffer:
                return  # No data to process
            
            # Analyze current situation
            analysis = self._analyze_current_situation()
            
            # Generate brain prompt
            brain_prompt = self._create_brain_prompt(analysis)
            
            # Get decision from LLM
            decision = await self._get_llm_decision(brain_prompt)
            
            if decision:
                await self._execute_decision(decision)
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Error making brain decision: {e}")
    
    def _analyze_current_situation(self) -> Dict[str, Any]:
        """Analyze current sensory data and context."""
        analysis = {
            'visual_events': [],
            'voice_context': [],
            'game_context': self.game_context.copy(),
            'time_span': 0
        }
        
        # Analyze visual events
        if self.visual_sensor_buffer:
            recent_visual = self.visual_sensor_buffer[-10:]  # Last 10 visual events
            analysis['visual_events'] = [
                {
                    'type': data.sensor_type,
                    'label': data.label,
                    'confidence': data.confidence,
                    'age': time.time() - data.timestamp
                }
                for data in recent_visual
            ]
            analysis['time_span'] = recent_visual[-1].timestamp - recent_visual[0].timestamp
        
        # Analyze voice context
        if self.voice_context_buffer:
            recent_voice = self.voice_context_buffer[-5:]  # Last 5 voice interactions
            analysis['voice_context'] = [
                {
                    'user_message': ctx.user_message,
                    'bot_response': ctx.bot_response,
                    'game_mentioned': ctx.game_mentioned,
                    'age': time.time() - ctx.timestamp
                }
                for ctx in recent_voice
            ]
        
        return analysis
    
    def _create_brain_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create a prompt for the central brain to make a decision."""
        prompt_parts = []
        
        # Add visual sensor summary
        if analysis['visual_events']:
            visual_summary = []
            for event in analysis['visual_events']:
                if event['age'] < 30:  # Only recent events
                    visual_summary.append(f"{event['type']}: {event['label']} ({event['confidence']:.1%})")
            
            if visual_summary:
                prompt_parts.append(f"VISUAL SENSORS: {' | '.join(visual_summary)}")
        
        # Add voice context
        if analysis['voice_context']:
            recent_voice = analysis['voice_context'][-1]  # Most recent
            prompt_parts.append(f"VOICE CONTEXT: User said '{recent_voice['user_message']}', I responded '{recent_voice['bot_response']}'")
        
        # Add game context
        if analysis['game_context'].get('current_game'):
            prompt_parts.append(f"GAME CONTEXT: Currently watching {analysis['game_context']['current_game']}")
        
        # Create the brain prompt
        context = "\n".join(prompt_parts)
        prompt = f"""CENTRAL BRAIN ANALYSIS:
{context}

As the central AI brain, analyze this sensory data and decide if you should:
1. Provide commentary about what you're seeing in the game
2. Respond to the voice conversation
3. Ask a clarifying question
4. Stay silent (if nothing interesting is happening)

Provide a natural, conversational response that integrates both visual and voice context. Focus on the actual game content."""

        return prompt
    
    async def _get_llm_decision(self, brain_prompt: str) -> Optional[BrainDecision]:
        """Get a decision from the LLM."""
        try:
            if not self.llm_service:
                return None
            
            # Get response from LLM
            response = await self.llm_service.handle_user_text_query(
                user_text=brain_prompt,
                user_name="CentralBrain"
            )
            
            if response and len(response.strip()) > 10:  # Minimum meaningful response
                decision = BrainDecision(
                    action_type='commentary',
                    content=response.strip(),
                    priority='normal',
                    timestamp=time.time(),
                    context_sources=['visual', 'voice']
                )
                return decision
            
            return None
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Error getting LLM decision: {e}")
            return None
    
    async def _execute_decision(self, decision: BrainDecision):
        """Execute a brain decision."""
        try:
            if not decision.content.strip():
                return
            
            # Send to text callback
            if self.text_callback:
                if asyncio.iscoroutinefunction(self.text_callback):
                    await self.text_callback(decision.content)
                else:
                    self.text_callback(decision.content)
            
            # Send to TTS callback
            if self.tts_callback:
                if asyncio.iscoroutinefunction(self.tts_callback):
                    await self.tts_callback(decision.content)
                else:
                    self.tts_callback(decision.content)
            
            # Store in memory
            if self.memory_service:
                try:
                    from services.memory_service import MemoryEntry
                    memory_entry = MemoryEntry(
                        content=f"Central brain decision: {decision.content}",
                        source="central_brain",
                        timestamp=time.time(),
                        metadata={
                            'action_type': decision.action_type,
                            'priority': decision.priority,
                            'context_sources': decision.context_sources
                        }
                    )
                    self.memory_service.store_memory(memory_entry)
                except Exception as e:
                    self.logger.error(f"[CentralBrain] Memory storage error: {e}")
            
            self.logger.info(f"[CentralBrain] Executed decision: {decision.content[:50]}...")
            
        except Exception as e:
            self.logger.error(f"[CentralBrain] Error executing decision: {e}")
    
    def update_game_context(self, context: Dict[str, Any]):
        """Update the game context."""
        self.game_context.update(context)
        self.logger.debug(f"[CentralBrain] Updated game context: {context}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get brain status."""
        return {
            'is_active': self.is_active,
            'visual_buffer_size': len(self.visual_sensor_buffer),
            'voice_buffer_size': len(self.voice_context_buffer),
            'game_context': self.game_context,
            'last_decision_time': self.last_decision_time
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("[CentralBrain] Cleaning up...")
        # Cleanup logic here 