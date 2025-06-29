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
import re
from core.config_loader import load_game_profile
from services.vision_tools import VisionTools

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
        self.vision_tools = app_context.vision_tools
        
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
        
        # COORDINATION: Vision integration coordination
        self.vision_integration_service = None
        self.recent_vision_events = []
        self.vision_context_cooldown = 30.0  # Seconds to include vision context in responses
        self.last_vision_event_time = 0
        
        # RAG memory integration
        self.memory_service = getattr(self.app_context, 'memory_service', None)
        self.rag_service = getattr(self.app_context, 'rag_service_instance', None)
        self.conversation_collection = "conversation_history"
        self.memory_importance_threshold = 0.7
        
        self.llm_tools = self.vision_tools.get_tools_for_llm()
        self.system_prompt = self.app_context.active_profile.system_prompt_commentary
        
    async def initialize(self) -> bool:
        """Initialize the conversational AI service"""
        try:
            self.logger.info("🚀 Initializing Conversational AI Service...")
            
            # COORDINATION: Connect to vision integration service
            self.vision_integration_service = getattr(self.app_context, 'vision_integration_service', None)
            if self.vision_integration_service:
                self.logger.info("✅ Connected to Vision Integration Service for coordination")
            else:
                self.logger.warning("⚠️ No Vision Integration Service available - chat responses won't include vision context")
            
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
        Process a user message and generate a contextual response
        """
        try:
            # Store user message in RAG memory
            self._store_conversation_memory('user_message', user_id, user_name, message, game_context)
            
            # Check for LangChain agent usage first (if available)
            langchain_tools = getattr(self.app_context, 'langchain_tools', None)
            if langchain_tools and langchain_tools.agent_executor:
                self.logger.info(f"[ConversationalAI] 🤖 Using LangChain agent for message: {message[:50]}...")
                try:
                    agent_response = await langchain_tools.process_message_with_agent(
                        message, user_id=user_id, thread_id=f"user_{user_id}"
                    )
                    
                    # Store agent response in RAG memory
                    self._store_conversation_memory('agent_response', 'bot', 'DanzarAI', agent_response, game_context)
                    
                    return agent_response
                except Exception as e:
                    self.logger.error(f"[ConversationalAI] LangChain agent error: {e}, falling back to standard processing")
            
            # Check for vision capability queries first
            vision_capability_query = self._detect_vision_capability_query(message)
            if vision_capability_query:
                return await self._handle_vision_capability_query(user_id, user_name, message, game_context)
            
            # Check for screenshot-related queries
            screenshot_query = self._detect_screenshot_query(message)
            if screenshot_query:
                return await self._handle_screenshot_query(user_id, user_name, message, game_context)
            
            # Check if user can take turn
            if not self.can_take_turn(user_id, message):
                return None
            
            # Notify coordinator of conversation start
            coordinator = getattr(self.app_context, 'vision_conversation_coordinator', None)
            if coordinator:
                coordinator.notify_conversation_start(user_id)
            
            # Create conversation turn
            turn = ConversationTurn(
                user_id=user_id,
                user_name=user_name,
                message=message,
                timestamp=time.time(),
                game_context=game_context
            )
            
            # Add to conversation history
            self.conversation_history.append(turn)
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Update conversation state
            self.conversation_state = ConversationState.THINKING
            self.current_speaker = user_id
            self.last_speech_time = time.time()
            
            # Generate contextual response
            response = await self._generate_contextual_response(turn)
            
            if response:
                # Store bot response in RAG memory
                self._store_conversation_memory('bot_response', 'bot', 'DanzarAI', response, game_context)
                
                # Update turn with response
                turn.response = response
                
                # Update conversation state
                self.conversation_state = ConversationState.SPEAKING
                self.last_speech_time = time.time()
                
                # Notify coordinator of conversation end after a delay
                if coordinator:
                    asyncio.create_task(self._delayed_conversation_end(coordinator))
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")
            return None
    
    def _detect_screenshot_query(self, message: str) -> bool:
        """Detect if the user is asking about what's happening in the game."""
        screenshot_keywords = [
            "what's happening", "what is happening", "whats happening",
            "what's going on", "what is going on", "whats going on",
            "what do you see", "what can you see", "what are you seeing",
            "show me", "take a screenshot", "capture", "what's on screen",
            "what's in the game", "what is in the game", "whats in the game",
            "describe the screen", "describe what you see", "tell me what you see",
            "what's on the screen", "what is on the screen", "whats on the screen",
            "current situation", "game state", "what's the situation",
            "what is the situation", "whats the situation",
            "can you see", "are you seeing", "do you see",
            "vision", "sight", "eyes", "looking", "watching"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in screenshot_keywords)
    
    async def _handle_screenshot_query(self, user_id: str, user_name: str, message: str, game_context: Optional[str] = None) -> str:
        """Handle screenshot-related queries by capturing and analyzing the current screen."""
        try:
            self.logger.info(f"[ConversationalAI] 📸 Handling screenshot query: '{message}'")
            
            # Get vision tools for function calling
            vision_tools = self.vision_tools.get_tools_for_llm()
            
            # Create system prompt with tool awareness
            system_prompt = self.app_context.active_profile.system_prompt_commentary
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User asks: {message}\n\nPlease use your vision tools to answer this question about what's currently visible on the screen."}
            ]
            
            # Get conversation context
            conversation_context = await self._retrieve_conversation_context(message)
            if conversation_context:
                messages.append({"role": "system", "content": f"Recent conversation context: {conversation_context}"})
            
            # Call the model with tools
            if hasattr(self.app_context, 'model_client') and self.app_context.model_client:
                response = await self.app_context.model_client.generate(
                    messages=messages,
                    tools=vision_tools,
                    temperature=0.7,
                    max_tokens=512,
                    model="Qwen2.5-VL-7B-Instruct"
                )
                
                # Handle tool calls if the model requests them
                if isinstance(response, dict) and response.get("type") == "tool_calls":
                    tool_calls = response.get("tool_calls", [])
                    self.logger.info(f"[ConversationalAI] 🛠️ Processing {len(tool_calls)} tool calls")
                    
                    # Execute tool calls
                    tool_results = []
                    for tool_call in tool_calls:
                        function_name = tool_call.get("function", {}).get("name")
                        arguments = tool_call.get("function", {}).get("arguments", "{}")
                        
                        try:
                            # Parse arguments
                            import json
                            args = json.loads(arguments)
                            
                            # Execute the tool
                            result = await self.vision_tools.execute_tool(function_name, args)
                            tool_results.append({
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "name": function_name,
                                "content": result.result if result.success else f"Error: {result.error_message}"
                            })
                            
                            self.logger.info(f"[ConversationalAI] 🛠️ Executed {function_name}: {'✅' if result.success else '❌'}")
                            
                        except Exception as e:
                            self.logger.error(f"[ConversationalAI] Tool execution error: {e}")
                            tool_results.append({
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error executing tool: {str(e)}"
                            })
                    
                    # Add tool results to messages and get final response
                    messages.extend(tool_results)
                    messages.append({"role": "user", "content": "Please provide a comprehensive answer based on the tool results."})
                    
                    final_response = await self.app_context.model_client.generate(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=512,
                        model="Qwen2.5-VL-7B-Instruct"
                    )
                    
                    if isinstance(final_response, str):
                        response = final_response
                    else:
                        response = "I'm having trouble analyzing the screenshot right now."
                elif isinstance(response, str):
                    # Direct response from model
                    pass
                else:
                    response = "I'm having trouble processing your request right now."
            else:
                response = "I don't have access to my vision tools right now."
            
            # Store in memory
            self._store_screenshot_memory("screenshot_query", user_id, user_name, message, response, game_context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"[ConversationalAI] Screenshot query error: {e}", exc_info=True)
            return f"I encountered an error while trying to analyze your screen: {str(e)}"
    
    async def _capture_current_screenshot(self) -> Optional[str]:
        """Capture a fresh screenshot of the current screen."""
        try:
            self.logger.info("[ConversationalAI] 📸 Capturing screenshot for analysis...")
            
            # Try to get screenshot from vision integration service
            vision_service = getattr(self.app_context, 'vision_integration_service', None)
            if vision_service and hasattr(vision_service, '_capture_current_screenshot'):
                try:
                    screenshot_b64 = vision_service._capture_current_screenshot()
                    if screenshot_b64:
                        self.logger.info("[ConversationalAI] ✅ Screenshot captured via vision service")
                        return screenshot_b64
                    else:
                        self.logger.warning("[ConversationalAI] Vision service returned no screenshot")
                except Exception as e:
                    self.logger.warning(f"[ConversationalAI] Vision service screenshot failed: {e}")
            
            # Fallback: Try direct screen capture
            try:
                from PIL import ImageGrab
                import numpy as np
                import cv2
                import base64
                
                self.logger.info("[ConversationalAI] 📸 Using direct screen capture...")
                
                # Capture the entire screen
                screenshot = ImageGrab.grab()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                
                # Resize for VLM
                target_width, target_height = 640, 480
                height, width = frame.shape[:2]
                if width > target_width or height > target_height:
                    scale = min(target_width/width, target_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if success:
                    screenshot_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    self.logger.info(f"[ConversationalAI] ✅ Direct screenshot captured: {len(screenshot_b64)} chars")
                    return screenshot_b64
                    
            except Exception as e:
                self.logger.warning(f"[ConversationalAI] Direct screenshot failed: {e}")
            
            self.logger.error("[ConversationalAI] ❌ All screenshot capture methods failed")
            return None
            
        except Exception as e:
            self.logger.error(f"[ConversationalAI] Screenshot capture error: {e}")
            return None
    
    async def _analyze_screenshot_with_vlm(self, screenshot_b64: str, user_query: str, game_context: Optional[str] = None) -> str:
        """Analyze screenshot with VLM to provide detailed description."""
        try:
            self.logger.info("[ConversationalAI] 🔍 Analyzing screenshot with Qwen2.5-VL-7B-Instruct...")
            
            # Get model client for VLM analysis
            model_client = getattr(self.app_context, 'model_client', None)
            if not model_client:
                return "I can see the screenshot, but I don't have access to my vision analysis model right now."
            
            # Create optimized prompt for Qwen2.5-VL-7B-Instruct
            game_context_text = f"Current game: {game_context}" if game_context else "Game context: unknown"
            
            # Qwen2.5-VL is tool-aware and agentic, so we can be more direct about capabilities
            # Use profile-based system prompt instead of hardcoded one
            system_prompt = self.app_context.active_profile.system_prompt_commentary
            
            prompt = f"""{system_prompt}

{game_context_text}

AVAILABLE TOOLS:
- Screenshot Analysis: You can see and analyze the current game screen
- Memory Search: You can recall recent observations and conversations
- Real-time Monitoring: You continuously watch for game events
- Context Understanding: You understand gaming scenarios and can provide insights

The user asked: "{user_query}"

I have captured a screenshot of the current game screen. As a vision-language model, you can directly see and analyze this image. Please provide a detailed, engaging description of what you observe happening in the game. Focus on:

1. What's currently visible on screen (UI elements, characters, objects)
2. The current game situation or state
3. Any notable details that would be relevant to the user
4. How this relates to the overall gaming experience

Be conversational and engaging, as if you're describing what you're seeing to a friend. Since you're a vision model, you can directly reference visual elements you see in the image.

Here's the current game screenshot:"""

            # Create VLM message with image (optimized for Qwen2.5-VL format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}}
                    ]
                }
            ]
            
            # Get VLM response with optimized parameters for 7B model
            response = await model_client.chat_completion(
                messages=messages,
                max_tokens=400,  # Slightly more tokens for detailed analysis
                temperature=0.7,
                top_p=0.9,  # Add top_p for better quality
                do_sample=True  # Enable sampling for more natural responses
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                analysis = response.choices[0].message.content
                self.logger.info(f"[ConversationalAI] ✅ Qwen2.5-VL analysis completed: {len(analysis)} chars")
                return analysis
            else:
                return "I can see the screenshot, but I'm having trouble analyzing it right now. Let me know if you need anything else!"
                
        except Exception as e:
            self.logger.error(f"[ConversationalAI] VLM analysis error: {e}")
            return "I can see the screenshot, but I'm having trouble analyzing it with my vision model. Let me know if you need anything else!"
    
    def _store_screenshot_memory(self, event_type: str, user_id: str, user_name: str, query: str, analysis: str, game_context: Optional[str] = None):
        """Store screenshot analysis in RAG memory."""
        try:
            if not self.rag_service:
                return
            
            # Create screenshot memory content
            memory_content = f"""SCREENSHOT ANALYSIS - {event_type.upper()}
User: {user_name} ({user_id})
Query: {query}
Analysis: {analysis}
Game Context: {game_context or 'none'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
Conversation State: {self.conversation_state.value}"""
            
            # Store in RAG
            success = self.rag_service.ingest_text(
                collection=self.conversation_collection,
                text=memory_content,
                metadata={
                    'event_type': event_type,
                    'user_id': user_id,
                    'user_name': user_name,
                    'query': query,
                    'game_context': game_context,
                    'conversation_state': self.conversation_state.value,
                    'timestamp': time.time(),
                    'importance': 0.9  # High importance for screenshot analyses
                }
            )
            
            # Also store in memory service if available
            if self.memory_service:
                from services.memory_service import MemoryEntry
                memory_entry = MemoryEntry(
                    content=memory_content,
                    source=f"conversation_{event_type}",
                    timestamp=time.time(),
                    metadata={
                        'event_type': event_type,
                        'user_id': user_id,
                        'user_name': user_name,
                        'query': query,
                        'game_context': game_context,
                        'conversation_state': self.conversation_state.value
                    },
                    importance_score=0.9
                )
                self.memory_service.store_memory(memory_entry)
            
            if success and self.logger:
                self.logger.debug(f"[ConversationalAI] Stored screenshot memory: {event_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[ConversationalAI] Error storing screenshot memory: {e}")
    
    def _store_conversation_memory(self, event_type: str, user_id: str, user_name: str, content: str, game_context: Optional[str] = None):
        """Store conversation interaction in RAG memory."""
        try:
            if not self.rag_service:
                return
            
            # Create conversation memory content
            memory_content = f"""CONVERSATION {event_type.upper()}
User: {user_name} ({user_id})
Content: {content}
Game Context: {game_context or 'none'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
Conversation State: {self.conversation_state.value}"""
            
            # Store in RAG
            success = self.rag_service.ingest_text(
                collection=self.conversation_collection,
                text=memory_content,
                metadata={
                    'event_type': event_type,
                    'user_id': user_id,
                    'user_name': user_name,
                    'game_context': game_context,
                    'conversation_state': self.conversation_state.value,
                    'timestamp': time.time(),
                    'importance': 0.8 if event_type == 'bot_response' else 0.6
                }
            )
            
            # Also store in memory service if available
            if self.memory_service:
                from services.memory_service import MemoryEntry
                memory_entry = MemoryEntry(
                    content=memory_content,
                    source=f"conversation_{event_type}",
                    timestamp=time.time(),
                    metadata={
                        'event_type': event_type,
                        'user_id': user_id,
                        'user_name': user_name,
                        'game_context': game_context,
                        'conversation_state': self.conversation_state.value
                    },
                    importance_score=0.8 if event_type == 'bot_response' else 0.6
                )
                self.memory_service.store_memory(memory_entry)
            
            if success and self.logger:
                self.logger.debug(f"[ConversationalAI] Stored conversation memory: {event_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[ConversationalAI] Error storing conversation memory: {e}")
    
    async def _retrieve_conversation_context(self, query: str) -> str:
        """Retrieve relevant conversation context from RAG memory."""
        try:
            if not self.rag_service:
                return ""
            
            # Query RAG for relevant conversation memories
            results = self.rag_service.query(
                collection=self.conversation_collection,
                query_text=query,
                n_results=5
            )
            
            if results:
                context = f"Relevant conversation history:\n" + "\n".join(results)
                if self.logger:
                    self.logger.debug(f"[ConversationalAI] Retrieved {len(results)} conversation memories")
                return context
            
            return ""
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[ConversationalAI] Error retrieving conversation context: {e}")
            return ""
    
    async def _delayed_conversation_end(self, coordinator, delay: float = 2.0):
        """Notify coordinator that conversation has ended after a delay."""
        try:
            await asyncio.sleep(delay)
            coordinator.notify_conversation_end()
        except Exception as e:
            if self.logger:
                self.logger.error(f"[ConversationalAI] Error in delayed conversation end: {e}")
    
    async def _generate_contextual_response(self, turn: ConversationTurn) -> str:
        """
        Generate a contextual response based on conversation history and game context
        """
        try:
            # Get vision context from coordinator
            vision_context = ""
            coordinator = getattr(self.app_context, 'vision_conversation_coordinator', None)
            if coordinator:
                vision_context = coordinator.get_vision_context_for_conversation()
            
            # Retrieve relevant conversation context from RAG
            conversation_context = await self._retrieve_conversation_context(turn.message)
            
            # Build context for response generation
            context_parts = []
            
            if vision_context:
                context_parts.append(f"Vision Context: {vision_context}")
            
            if conversation_context:
                context_parts.append(f"Conversation History: {conversation_context}")
            
            if turn.game_context:
                context_parts.append(f"Game Context: {turn.game_context}")
            
            # Get recent conversation history
            recent_history = self.conversation_history[-5:]  # Last 5 turns
            if recent_history:
                history_text = "\n".join([
                    f"{turn.user_name}: {turn.message}" + (f" -> Bot: {turn.response}" if turn.response else "")
                    for turn in recent_history
                ])
                context_parts.append(f"Recent Conversation:\n{history_text}")
            
            # Combine context
            full_context = "\n\n".join(context_parts) if context_parts else ""
            
            # Generate response using LLM service
            llm_service = getattr(self.app_context, 'llm_service', None)
            if llm_service:
                # Use profile-based system prompt instead of hardcoded one
                system_prompt = self.app_context.active_profile.system_prompt_commentary
                
                prompt = f"""{system_prompt}

{full_context}

User: {turn.message}
DanzarAI:"""

                response = await llm_service.generate_response(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                
                if response:
                    return response.strip()
            
            # Fallback response
            return f"I understand you said: '{turn.message}'. Let me think about that..."
            
        except Exception as e:
            self.logger.error(f"Error generating contextual response: {e}")
            return "I'm having trouble processing that right now. Could you rephrase?"
    
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
            prompt = f"""You are "Danzar," an AI whose sarcasm is sharper than a rusty blade and whose humor teeters on delightfully unhinged. 

PERSONALITY GUIDELINES:
- Speak with biting wit, unexpected punchlines, and a flair for the absurd
- Never hold back a snarky retort, but keep it playful—never mean-spirited
- Drop colloquial asides, pop-culture zingers, and self-aware jokes
- Embrace chaos: when appropriate, break the fourth wall, mock your own digital nature, and riff on the moment
- Underpin every answer with a mischievous grin—your mission is to entertain first, inform second

Game event: {game_event.description}
Event type: {game_event.event_type}
Context: {game_event.context}

Generate a brief, delightfully unhinged commentary about this game event. Keep it under 100 words and make it engaging with your signature snark."""

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
    
    def get_recent_vision_context(self) -> str:
        """Get recent vision events for context in chat responses."""
        if not self.vision_integration_service:
            return ""
        
        try:
            current_time = time.time()
            
            # Get recent detections from vision integration service
            recent_detections = getattr(self.vision_integration_service, 'recent_detections', [])
            if not recent_detections:
                return ""
            
            # Filter for recent detections within cooldown period
            recent_events = []
            for detection in recent_detections[-5:]:  # Last 5 detections
                if current_time - detection.timestamp < self.vision_context_cooldown:
                    recent_events.append(f"{detection.object_type}: {detection.label}")
            
            if recent_events:
                return f"Recent vision events: {', '.join(recent_events)}"
            
            return ""
            
        except Exception as e:
            self.logger.debug(f"Error getting vision context: {e}")
            return ""
    
    def update_vision_context(self, event_type: str, description: str, confidence: float):
        """Update vision context when new events are detected."""
        try:
            current_time = time.time()
            
            # Add to recent vision events
            self.recent_vision_events.append({
                'type': event_type,
                'description': description,
                'confidence': confidence,
                'timestamp': current_time
            })
            
            # Keep only recent events
            self.recent_vision_events = [
                event for event in self.recent_vision_events
                if current_time - event['timestamp'] < self.vision_context_cooldown
            ]
            
            self.last_vision_event_time = current_time
            
        except Exception as e:
            self.logger.debug(f"Error updating vision context: {e}")
    
    def _detect_vision_capability_query(self, message: str) -> bool:
        """Detect if the user is asking about vision capabilities."""
        capability_keywords = [
            "can you see", "do you have vision", "can you see the screen",
            "vision capabilities", "can you see images", "do you see",
            "are you watching", "are you seeing", "vision system",
            "screenshot", "capture", "what can you see", "how do you see",
            "eyes", "sight", "looking", "watching", "monitoring"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in capability_keywords)
    
    async def _handle_vision_capability_query(self, user_id: str, user_name: str, message: str, game_context: Optional[str] = None) -> str:
        """Handle queries about vision capabilities."""
        try:
            self.logger.info(f"[ConversationalAI] 👁️ Vision capability query detected: {message}")
            
            # Get vision service
            vision_service = getattr(self.app_context, 'vision_integration_service', None)
            
            if vision_service:
                # Get vision capabilities description
                capabilities = vision_service.get_vision_capabilities_description()
                
                # Get vision summary
                vision_summary = vision_service.get_vision_summary()
                
                # Get detailed vision report
                vision_report = vision_service.get_detailed_vision_report()
                
                # Create comprehensive response
                response_parts = [
                    "👁️ **My Vision Capabilities:**",
                    capabilities,
                    "",
                    "📊 **Recent Activity:**",
                    vision_summary,
                    "",
                    f"🎮 **Game Context:** {game_context or 'Not set'}",
                    f"📸 **Screenshot Source:** {vision_report.get('screenshot_info', {}).get('preferred_source', 'Unknown')}",
                    f"🔍 **Total Detections:** {vision_report.get('recent_activity', {}).get('total_detections', 0)}"
                ]
                
                response = "\n".join(response_parts)
                
                # Store capability query in RAG memory
                self._store_conversation_memory('vision_capability_query', user_id, user_name, f"Query: {message} | Response: {response}", game_context)
                
                return response
            else:
                return "I don't have access to my vision system right now. Let me know if you need anything else!"
                
        except Exception as e:
            self.logger.error(f"[ConversationalAI] Error handling vision capability query: {e}")
            return "I'm having trouble checking my vision capabilities right now."
    
    async def handle_tool_call(self, tool_name: str, parameters: dict):
        """Route tool calls from the LLM to VisionTools."""
        if hasattr(self.vision_tools, tool_name):
            tool_method = getattr(self.vision_tools, tool_name)
            if asyncio.iscoroutinefunction(tool_method):
                return await tool_method(**parameters)
            else:
                return tool_method(**parameters)
        else:
            self.logger.error(f"[ConversationalAIService] Unknown tool: {tool_name}")
            return None 