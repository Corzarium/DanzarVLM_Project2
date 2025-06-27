"""
Conversational AI Service for DanzarAI
Handles turn-taking, game awareness, and BLIP/CLIP integration for detecting notable events
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from PIL import Image
import io
import base64

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    GAME_COMMENTARY = "game_commentary"


@dataclass
class GameEvent:
    """Represents a notable game event detected by vision analysis"""
    event_type: str
    description: str
    confidence: float
    timestamp: float
    frame_data: Optional[bytes] = None
    context: Optional[str] = None


@dataclass
class ConversationTurn:
    """Represents a turn in the conversation"""
    user_id: str
    user_name: str
    message: str
    timestamp: float
    response: Optional[str] = None
    game_context: Optional[str] = None


class ConversationalAIService:
    """
    Advanced conversational AI service with game awareness and turn-taking
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Conversation state management
        self.conversation_state = ConversationState.IDLE
        self.current_speaker = None
        self.last_speech_time = 0
        self.turn_timeout = 5.0  # seconds
        self.min_turn_gap = 0.5  # minimum gap between turns
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        self.max_history_length = 50
        
        # Game awareness
        self.current_game = "unknown"
        self.game_events: List[GameEvent] = []
        self.last_game_analysis = 0
        self.game_analysis_interval = 10.0  # seconds
        
        # Vision models for game event detection
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self.vision_models_loaded = False
        
        # Notable event keywords for different games
        self.game_event_keywords = {
            "everquest": [
                "death", "level up", "rare spawn", "boss", "raid", "loot", "combat",
                "healing", "buff", "debuff", "group", "guild", "trade", "auction"
            ],
            "generic": [
                "achievement", "victory", "defeat", "score", "level", "boss", "enemy",
                "item", "power-up", "health", "damage", "healing", "team", "player"
            ]
        }
        
        # Turn-taking parameters
        self.interruption_threshold = 0.8  # confidence threshold for interrupting
        self.urgent_keywords = ["help", "emergency", "danger", "boss", "rare", "death"]
        
    async def initialize(self) -> bool:
        """Initialize the conversational AI service"""
        try:
            self.logger.info("🚀 Initializing Conversational AI Service...")
            
            # Load vision models for game event detection
            if VISION_AVAILABLE:
                await self._load_vision_models()
            else:
                self.logger.warning("⚠️ Vision models not available - install torch and transformers")
            
            # Initialize conversation state
            self.conversation_state = ConversationState.IDLE
            self.logger.info("✅ Conversational AI Service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Conversational AI Service: {e}")
            return False
    
    async def _load_vision_models(self):
        """Load CLIP and BLIP models for vision analysis"""
        try:
            self.logger.info("🔍 Loading vision models for game event detection...")
            
            # Load CLIP model for image understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Load BLIP model for image captioning
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            self.vision_models_loaded = True
            self.logger.info("✅ Vision models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load vision models: {e}")
            self.vision_models_loaded = False
    
    def can_take_turn(self, user_id: str, message: str, urgency: float = 0.0) -> bool:
        """
        Determine if a user can take a turn in the conversation
        
        Args:
            user_id: Discord user ID
            message: User's message
            urgency: Urgency level (0.0-1.0)
            
        Returns:
            bool: True if user can take turn
        """
        current_time = time.time()
        
        # Check if we're in a speaking state
        if self.conversation_state == ConversationState.SPEAKING:
            # Allow interruption for urgent messages
            if urgency > self.interruption_threshold:
                self.logger.info(f"🚨 Urgent interruption allowed (urgency: {urgency:.2f})")
                return True
            
            # Check for urgent keywords
            message_lower = message.lower()
            if any(keyword in message_lower for keyword in self.urgent_keywords):
                self.logger.info(f"🚨 Urgent keyword detected: {message}")
                return True
            
            # Otherwise, wait for current speaker to finish
            return False
        
        # Check turn timeout
        if (self.current_speaker and 
            current_time - self.last_speech_time < self.turn_timeout):
            return False
        
        # Check minimum gap between turns
        if current_time - self.last_speech_time < self.min_turn_gap:
            return False
        
        return True
    
    async def process_user_message(self, user_id: str, user_name: str, message: str, 
                                 game_context: Optional[str] = None) -> Optional[str]:
        """
        Process a user message and generate an appropriate response
        
        Args:
            user_id: Discord user ID
            user_name: Discord username
            message: User's message
            game_context: Current game context
            
        Returns:
            Optional[str]: AI response, or None if no response needed
        """
        try:
            current_time = time.time()
            
            # Check if user can take turn
            if not self.can_take_turn(user_id, message):
                self.logger.info(f"⏳ User {user_name} must wait for turn")
                return None
            
            # Update conversation state
            self.conversation_state = ConversationState.LISTENING
            self.current_speaker = user_id
            self.last_speech_time = current_time
            
            # Add to conversation history
            turn = ConversationTurn(
                user_id=user_id,
                user_name=user_name,
                message=message,
                timestamp=current_time,
                game_context=game_context
            )
            self.conversation_history.append(turn)
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Generate response with game awareness
            response = await self._generate_contextual_response(turn)
            turn.response = response
            
            # Update state
            self.conversation_state = ConversationState.THINKING
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error processing user message: {e}")
            return None
    
    async def _generate_contextual_response(self, turn: ConversationTurn) -> str:
        """Generate a contextual response based on conversation and game state"""
        try:
            # Build context for the LLM
            context_parts = []
            
            # Add game context
            if turn.game_context:
                context_parts.append(f"Current game context: {turn.game_context}")
            
            # Add recent conversation history
            recent_turns = self.conversation_history[-5:]  # Last 5 turns
            if recent_turns:
                context_parts.append("Recent conversation:")
                for t in recent_turns:
                    context_parts.append(f"- {t.user_name}: {t.message}")
                    if t.response:
                        context_parts.append(f"- Danzar: {t.response}")
            
            # Add detected game events
            recent_events = [e for e in self.game_events 
                           if time.time() - e.timestamp < 60]  # Last minute
            if recent_events:
                context_parts.append("Recent game events:")
                for event in recent_events:
                    context_parts.append(f"- {event.description} (confidence: {event.confidence:.2f})")
            
            # Build the prompt
            context = "\n".join(context_parts)
            prompt = f"""You are Danzar, an upbeat and witty gaming assistant. 

Context:
{context}

User: {turn.message}

Respond naturally and conversationally, incorporating game context when relevant. Keep responses concise but engaging."""

            # Get response from LLM service
            if hasattr(self.app_context, 'llm_service') and self.app_context.llm_service:
                response = await self.app_context.llm_service.generate_response(
                    prompt, max_tokens=150, temperature=0.7
                )
                return response or "I'm here to help with your gaming adventures!"
            else:
                return "I'm here to help with your gaming adventures!"
                
        except Exception as e:
            self.logger.error(f"❌ Error generating contextual response: {e}")
            return "I'm here to help with your gaming adventures!"
    
    async def analyze_game_frame(self, frame_data: bytes, game_type: str = "generic") -> Optional[GameEvent]:
        """
        Analyze a game frame to detect notable events using BLIP/CLIP
        
        Args:
            frame_data: Image data as bytes
            game_type: Type of game being played
            
        Returns:
            Optional[GameEvent]: Detected game event, or None
        """
        try:
            if not self.vision_models_loaded:
                return None
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_data))
            
            # Get image caption using BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Analyze caption for notable events
            event_keywords = self.game_event_keywords.get(game_type, self.game_event_keywords["generic"])
            caption_lower = caption.lower()
            
            detected_events = []
            for keyword in event_keywords:
                if keyword in caption_lower:
                    # Calculate confidence based on keyword presence and context
                    confidence = 0.5  # Base confidence
                    
                    # Boost confidence for multiple keywords
                    keyword_count = caption_lower.count(keyword)
                    confidence += min(keyword_count * 0.2, 0.3)
                    
                    # Boost confidence for urgent keywords
                    if keyword in self.urgent_keywords:
                        confidence += 0.2
                    
                    detected_events.append((keyword, confidence))
            
            if detected_events:
                # Get the highest confidence event
                best_event = max(detected_events, key=lambda x: x[1])
                event_type, confidence = best_event
                
                # Create game event
                game_event = GameEvent(
                    event_type=event_type,
                    description=f"Detected {event_type}: {caption}",
                    confidence=confidence,
                    timestamp=time.time(),
                    frame_data=frame_data,
                    context=caption
                )
                
                self.game_events.append(game_event)
                
                # Trim old events
                current_time = time.time()
                self.game_events = [e for e in self.game_events 
                                  if current_time - e.timestamp < 300]  # Keep last 5 minutes
                
                self.logger.info(f"🎮 Game event detected: {event_type} (confidence: {confidence:.2f})")
                return game_event
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing game frame: {e}")
            return None
    
    async def generate_game_commentary(self, game_event: GameEvent) -> Optional[str]:
        """Generate commentary for a detected game event"""
        try:
            if game_event.confidence < 0.6:  # Only comment on high-confidence events
                return None
            
            # Build commentary prompt
            prompt = f"""You are Danzar, providing live gaming commentary. 

Game event: {game_event.description}
Event type: {game_event.event_type}
Context: {game_event.context}

Generate a brief, exciting commentary about this game event. Keep it under 100 words and make it engaging."""

            # Get commentary from LLM
            if hasattr(self.app_context, 'llm_service') and self.app_context.llm_service:
                commentary = await self.app_context.llm_service.generate_response(
                    prompt, max_tokens=100, temperature=0.8
                )
                return commentary
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating game commentary: {e}")
            return None
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status"""
        return {
            "state": self.conversation_state.value,
            "current_speaker": self.current_speaker,
            "last_speech_time": self.last_speech_time,
            "conversation_length": len(self.conversation_history),
            "recent_events": len([e for e in self.game_events 
                                if time.time() - e.timestamp < 60]),
            "vision_models_loaded": self.vision_models_loaded
        }
    
    def set_game_type(self, game_type: str):
        """Set the current game type for better event detection"""
        self.current_game = game_type
        self.logger.info(f"🎮 Game type set to: {game_type}")
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.logger.info("🗑️ Conversation history cleared") 