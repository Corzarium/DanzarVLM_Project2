# services/vision_aware_conversation_service.py
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import threading

# Import CLIP vision enhancer
try:
    from services.clip_vision_enhancer import CLIPVisionEnhancer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPVisionEnhancer = None

# Import enhanced conversation memory
try:
    from services.enhanced_conversation_memory import EnhancedConversationMemory
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    ENHANCED_MEMORY_AVAILABLE = False
    EnhancedConversationMemory = None

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
    clip_insights: Optional[Dict[str, Any]] = None  # CLIP semantic understanding

@dataclass
class ConversationTurn:
    """Represents a conversation turn with visual context"""
    user_input: str
    visual_context: Optional[VisualContext]
    timestamp: float
    response: Optional[str] = None
    visual_references: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.visual_references is None:
            self.visual_references = []

class VisionAwareConversationService:
    """
    Service that enables natural conversation with vision awareness.
    The VLM can "see" and naturally reference visual elements in conversation.
    Enhanced with CLIP for semantic visual understanding and proper memory management.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Enhanced Memory System - STM in RAM, LTM in RAG
        self.conversation_memory = None
        if ENHANCED_MEMORY_AVAILABLE and EnhancedConversationMemory:
            try:
                self.conversation_memory = EnhancedConversationMemory(app_context)
                if self.conversation_memory:
                    self.conversation_memory.start_background_tasks()
                    self.logger.info("[VisionAwareConversation] Enhanced conversation memory initialized")
            except Exception as e:
                self.logger.error(f"[VisionAwareConversation] Failed to initialize enhanced memory: {e}")
                self.conversation_memory = None
        
        # Visual context management - SLOWED DOWN to prevent system overload
        self.current_visual_context: Optional[VisualContext] = None
        self.visual_context_history: deque = deque(maxlen=20)  # Reduced from 50
        self.last_visual_update: float = 0
        self.visual_update_interval: float = 10.0  # Increased to 10.0 seconds to match 1 FPS vision processing
        
        # Legacy conversation management (fallback if enhanced memory not available)
        self.conversation_history: deque = deque(maxlen=50)  # Reduced from 100 for better focus
        self.conversation_summary: str = ""  # Track conversation summary
        self.last_conversation_time: float = 0
        self.conversation_context_window: float = 300.0  # 5 minutes of context
        
        # Vision integration settings
        self.vision_integration_enabled: bool = True
        self.visual_context_threshold: float = 0.6
        self.max_visual_elements: int = 3  # Reduced from 5
        
        # CLIP integration
        self.clip_enhancer = None
        self.clip_enabled = CLIP_AVAILABLE
        
        # Threading
        self.visual_context_lock = threading.Lock()
        self.conversation_lock = threading.Lock()
        
        # Initialize CLIP if available
        if self.clip_enabled and CLIPVisionEnhancer:
            try:
                self.clip_enhancer = CLIPVisionEnhancer(self.app_context)
                self.logger.info("[VisionAwareConversation] CLIP vision enhancer initialized")
            except Exception as e:
                self.logger.error(f"[VisionAwareConversation] Failed to initialize CLIP: {e}")
                self.clip_enhancer = None
                self.clip_enabled = False
        
        self.logger.info(f"[VisionAwareConversation] Initialized with CLIP: {self.clip_enabled}")
        self.logger.info(f"[VisionAwareConversation] Enhanced memory: {self.conversation_memory is not None}")
        self.logger.info(f"[VisionAwareConversation] Visual update interval: {self.visual_update_interval}s")
    
    async def initialize(self) -> bool:
        """Initialize the vision-aware conversation service."""
        try:
            self.logger.info("[VisionAwareConversation] Initializing...")
            
            # Set up visual context monitoring with reduced frequency
            await self._start_visual_context_monitor()
            
            self.logger.info("[VisionAwareConversation] Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Initialization failed: {e}", exc_info=True)
            return False
    
    async def _start_visual_context_monitor(self):
        """Start monitoring for visual context updates with reduced frequency."""
        async def monitor_loop():
            while not self.app_context.shutdown_event.is_set():
                try:
                    if self.vision_integration_enabled:
                        await self._update_visual_context()
                    
                    # Increased sleep time to reduce processing load
                    await asyncio.sleep(2.0)  # Increased from 1.0 to 2.0 seconds
                    
                except Exception as e:
                    self.logger.error(f"[VisionAwareConversation] Visual context monitor error: {e}")
                    await asyncio.sleep(10.0)  # Increased from 5.0 to 10.0 seconds
        
        asyncio.create_task(monitor_loop())
    
    async def _update_visual_context(self):
        """Update current visual context from vision models with CLIP enhancement."""
        try:
            # Get latest vision data from app context
            vision_data = getattr(self.app_context, 'latest_vision_data', None)
            if not vision_data:
                return
            
            current_time = time.time()
            
            # Only update if enough time has passed (increased interval)
            if current_time - self.last_visual_update < self.visual_update_interval:
                return
            
            with self.visual_context_lock:
                # Extract visual information
                detected_objects = vision_data.get('yolo_detections', [])
                ocr_text = vision_data.get('ocr_results', [])
                ui_elements = vision_data.get('template_matches', [])
                current_frame = vision_data.get('current_frame')
                
                # Get game context
                game_context = self._get_game_context()
                game_type = game_context.get('game_name', 'generic_game').lower()
                
                # Enhanced CLIP integration - more frequent processing
                clip_insights = None
                if self.clip_enhancer and current_frame is not None:
                    try:
                        # Process CLIP more frequently for better visual understanding
                        if current_time % 5 < 1:  # Every 5 seconds instead of 10
                            clip_insights = self.clip_enhancer.enhance_visual_context(
                                current_frame, detected_objects, ocr_text, game_type
                            )
                            self.logger.info(f"[VisionAwareConversation] CLIP enhanced context with {len(clip_insights.get('clip_insights', []))} insights")
                    except Exception as e:
                        self.logger.error(f"[VisionAwareConversation] CLIP enhancement error: {e}")
                
                # Create enhanced scene summary with CLIP insights
                scene_summary = self._create_enhanced_scene_summary(
                    detected_objects, ocr_text, ui_elements, clip_insights
                )
                
                # Calculate overall confidence
                confidence = self._calculate_enhanced_visual_confidence(
                    detected_objects, ocr_text, ui_elements, clip_insights
                )
                
                # Create new visual context
                new_context = VisualContext(
                    timestamp=current_time,
                    detected_objects=detected_objects,
                    ocr_text=ocr_text,
                    ui_elements=ui_elements,
                    scene_summary=scene_summary,
                    confidence=confidence,
                    game_context=game_context,
                    clip_insights=clip_insights
                )
                
                # Update current context
                if self.current_visual_context:
                    self.visual_context_history.append(self.current_visual_context)
                
                self.current_visual_context = new_context
                self.last_visual_update = current_time
                
                # Store in app context for other services to access
                if hasattr(self.app_context, 'current_visual_context'):
                    self.app_context.current_visual_context = new_context
                
                self.logger.info(f"[VisionAwareConversation] Updated visual context: {scene_summary[:100]}...")
                
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Visual context update error: {e}")
    
    def _create_enhanced_scene_summary(self, detected_objects: List, ocr_text: List, 
                                     ui_elements: List, clip_insights: Optional[Dict]) -> str:
        """Create an enhanced scene summary using CLIP insights."""
        summary_parts = []
        
        # Add CLIP insights if available
        if clip_insights and clip_insights.get('clip_insights'):
            clip_descriptions = clip_insights.get('visual_descriptions', [])
            if clip_descriptions:
                summary_parts.append(f"Visual analysis: {', '.join(clip_descriptions[:3])}")
        
        # Add detected objects
        if detected_objects:
            obj_summary = []
            for obj in detected_objects[:3]:  # Limit to top 3
                label = obj.get('label', 'unknown')
                conf = obj.get('confidence', 0)
                obj_summary.append(f"{label} ({conf:.2f})")
            if obj_summary:
                summary_parts.append(f"Detected: {', '.join(obj_summary)}")
        
        # Add OCR text
        if ocr_text:
            text_summary = []
            for text in ocr_text[:3]:  # Limit to top 3
                if len(text.strip()) > 2:  # Only meaningful text
                    text_summary.append(f'"{text.strip()}"')
            if text_summary:
                summary_parts.append(f"Text: {', '.join(text_summary)}")
        
        # Add UI elements
        if ui_elements:
            ui_summary = []
            for ui in ui_elements[:2]:  # Limit to top 2
                label = ui.get('label', 'ui_element')
                ui_summary.append(label)
            if ui_summary:
                summary_parts.append(f"UI: {', '.join(ui_summary)}")
        
        # Create final summary
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "No significant visual elements detected"
    
    def _calculate_enhanced_visual_confidence(self, detected_objects: List, ocr_text: List, 
                                            ui_elements: List, clip_insights: Optional[Dict]) -> float:
        """Calculate enhanced confidence using CLIP insights."""
        base_confidence = self._calculate_visual_confidence(detected_objects, ocr_text, ui_elements)
        
        # Boost confidence if CLIP provides strong insights
        if clip_insights and clip_insights.get('clip_insights'):
            clip_confidence = max([insight['confidence'] for insight in clip_insights['clip_insights']], default=0.0)
            # Blend base confidence with CLIP confidence
            enhanced_confidence = (base_confidence * 0.6) + (clip_confidence * 0.4)
            return min(enhanced_confidence, 1.0)
        
        return base_confidence
    
    def _calculate_visual_confidence(self, detected_objects: List, ocr_text: List, ui_elements: List) -> float:
        """Calculate overall confidence in visual understanding."""
        if not detected_objects and not ocr_text and not ui_elements:
            return 0.0
        
        total_confidence = 0.0
        total_weight = 0.0
        
        # Weight object detections
        for obj in detected_objects:
            conf = obj.get('confidence', 0.5)
            total_confidence += conf * 0.4
            total_weight += 0.4
        
        # Weight OCR results
        for text in ocr_text:
            if len(text.strip()) > 3:
                total_confidence += 0.8 * 0.3
                total_weight += 0.3
                break
        
        # Weight UI elements
        for elem in ui_elements:
            conf = elem.get('confidence', 0.7)
            total_confidence += conf * 0.3
            total_weight += 0.3
            break
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_game_context(self) -> Dict[str, Any]:
        """Get current game context from app context."""
        context = {}
        
        # Get active game profile
        if hasattr(self.app_context, 'active_profile') and self.app_context.active_profile:
            context['game_name'] = self.app_context.active_profile.game_name
            context['game_settings'] = self.app_context.active_profile.settings
        
        # Get recent conversation context
        if self.conversation_history:
            recent_turns = list(self.conversation_history)[-3:]
            context['recent_conversation'] = [
                turn.user_input for turn in recent_turns
            ]
        
        return context
    
    async def process_conversation(self, user_input: str, include_visual_context: bool = True) -> str:
        """Process a conversation turn with visual awareness."""
        try:
            current_time = time.time()
            
            # Get current visual context if requested
            visual_context = None
            if include_visual_context and self.current_visual_context:
                visual_context = self.current_visual_context
            
            # Create conversation turn
            turn = ConversationTurn(
                user_input=user_input,
                visual_context=visual_context,
                timestamp=current_time
            )
            
            # Store in enhanced memory system
            if self.conversation_memory:
                # Add user input to STM
                self.conversation_memory.add_conversation_entry(
                    user_name="VirtualAudio",
                    content=user_input,
                    entry_type='user_input',
                    visual_context=visual_context.__dict__ if visual_context else None
                )
            
            # Generate response with visual awareness
            response = await self._generate_visual_aware_response(turn)
            
            # Store response in enhanced memory system
            if self.conversation_memory:
                self.conversation_memory.add_conversation_entry(
                    user_name="VirtualAudio",
                    content=response,
                    entry_type='bot_response',
                    visual_context=visual_context.__dict__ if visual_context else None
                )
            
            # Update conversation history (legacy fallback)
            with self.conversation_lock:
                self.conversation_history.append(turn)
                self.last_conversation_time = current_time
                self._update_conversation_summary(turn)
            
            self.logger.info(f"[VisionAwareConversation] Processed conversation turn: {len(user_input)} chars input, {len(response)} chars response")
            return response
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Conversation processing error: {e}")
            return "I'm having trouble processing that right now. Could you try again?"
    
    async def process_user_message(self, user_id: str, user_name: str, message: str, 
                                 game_context: Optional[str] = None) -> Optional[str]:
        """Process a user message with enhanced memory and visual context."""
        try:
            current_time = time.time()
            
            # Check for screenshot-related queries first
            if self._detect_screenshot_query(message):
                self.logger.info(f"[VisionAwareConversation] 📸 Detected screenshot query: '{message}'")
                return await self._handle_screenshot_query(user_id, user_name, message, game_context)
            
            # Get current visual context
            visual_context = None
            if self.current_visual_context:
                visual_context = self.current_visual_context
            
            # Store user message in enhanced memory
            if self.conversation_memory:
                self.conversation_memory.add_conversation_entry(
                    user_name=user_name,
                    content=message,
                    entry_type='user_input',
                    metadata={'user_id': user_id, 'game_context': game_context},
                    visual_context=visual_context.__dict__ if visual_context else None
                )
            
            # Get conversation context from enhanced memory
            conversation_context = None
            if self.conversation_memory:
                conversation_context = self.conversation_memory.get_conversation_context(
                    user_name=user_name,
                    include_ltm=True,
                    max_stm_entries=10,
                    max_ltm_results=3
                )
                self.logger.info(f"[VisionAwareConversation] Retrieved context for {user_name}: {len(conversation_context.get('stm_entries', []))} STM entries, {len(conversation_context.get('ltm_results', []))} LTM results")
            
            # Create conversation turn with enhanced context
            turn = ConversationTurn(
                user_input=message,
                visual_context=visual_context,
                timestamp=current_time
            )
            
            # Generate response with enhanced context
            response = await self._generate_enhanced_context_response(turn, conversation_context)
            
            # Store response in enhanced memory
            if self.conversation_memory:
                self.conversation_memory.add_conversation_entry(
                    user_name=user_name,
                    content=response,
                    entry_type='bot_response',
                    metadata={'user_id': user_id, 'game_context': game_context},
                    visual_context=visual_context.__dict__ if visual_context else None
                )
            
            self.logger.info(f"[VisionAwareConversation] Processed message for {user_name}: {len(message)} chars input, {len(response)} chars response")
            return response
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] User message processing error: {e}")
            return "I'm having trouble processing that right now. Could you try again?"
    
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
            self.logger.info(f"[VisionAwareConversation] 📸 Handling screenshot query: '{message}'")
            
            # Get vision tools for function calling
            vision_tools = None
            langchain_tools = getattr(self.app_context, 'langchain_tools', None)
            if langchain_tools and hasattr(langchain_tools, 'get_tools_for_llm'):
                vision_tools = langchain_tools.get_tools_for_llm()
            
            # Create system prompt with tool awareness
            system_prompt = self.app_context.active_profile.system_prompt_chat
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User asks: {message}\n\nPlease use your vision tools to answer this question about what's currently visible on the screen."}
            ]
            
            # Get conversation context
            conversation_context = None
            if self.conversation_memory:
                conversation_context = self.conversation_memory.get_conversation_context(
                    user_name=user_name,
                    include_ltm=True,
                    max_stm_entries=5,
                    max_ltm_results=2
                )
            
            if conversation_context:
                stm_entries = conversation_context.get('stm_entries', [])
                if stm_entries:
                    recent_context = []
                    for entry in stm_entries[-3:]:
                        if entry.entry_type == 'user_input':
                            recent_context.append(f"User: {entry.content}")
                        elif entry.entry_type == 'bot_response':
                            recent_context.append(f"You: {entry.content}")
                    
                    if recent_context:
                        messages.append({"role": "system", "content": f"Recent conversation context: {' '.join(recent_context)}"})
            
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
                    self.logger.info(f"[VisionAwareConversation] 🛠️ Processing {len(tool_calls)} tool calls")
                    
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
                            if langchain_tools and hasattr(langchain_tools, 'execute_tool'):
                                result = await langchain_tools.execute_tool(function_name, args)
                                tool_results.append({
                                    "tool_call_id": tool_call.get("id"),
                                    "role": "tool",
                                    "name": function_name,
                                    "content": result.result if result.success else f"Error: {result.error_message}"
                                })
                                
                                self.logger.info(f"[VisionAwareConversation] 🛠️ Executed {function_name}: {'✅' if result.success else '❌'}")
                            
                        except Exception as e:
                            self.logger.error(f"[VisionAwareConversation] Tool execution error: {e}")
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
            if self.conversation_memory:
                self.conversation_memory.add_conversation_entry(
                    user_name=user_name,
                    content=f"Screenshot query: {message}",
                    entry_type='user_input',
                    metadata={'user_id': user_id, 'game_context': game_context, 'query_type': 'screenshot'},
                    visual_context=None
                )
                
                self.conversation_memory.add_conversation_entry(
                    user_name=user_name,
                    content=response,
                    entry_type='bot_response',
                    metadata={'user_id': user_id, 'game_context': game_context, 'query_type': 'screenshot'},
                    visual_context=None
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Screenshot query error: {e}", exc_info=True)
            return f"I encountered an error while trying to analyze your screen: {str(e)}"
    
    async def _generate_enhanced_context_response(self, turn: ConversationTurn, conversation_context: Optional[Dict[str, Any]]) -> str:
        """Generate a response using the LangChain agent for tool awareness."""
        try:
            # Try to use LangChain agent if available
            langchain_tools = getattr(self.app_context, 'langchain_tools', None)
            if langchain_tools and hasattr(langchain_tools, 'agent') and langchain_tools.agent:
                self.logger.info("[VisionAwareConversation] Using LangChain agent for tool-aware response")
                
                # Build context for the agent
                context_info = self._build_agent_context(turn, conversation_context)
                
                # Use the LangChain agent
                response = await langchain_tools.agent.ainvoke({
                    "input": f"{context_info}\n\nUser: {turn.user_input}"
                })
                
                return response.get('output', 'I had trouble processing that request.')
            
            # Fallback to direct LLM if LangChain not available
            self.logger.warning("[VisionAwareConversation] LangChain agent not available, using direct LLM")
            return await self._generate_direct_llm_response(turn, conversation_context)
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Response generation error: {e}")
            return "I'm having trouble thinking about that right now."
    
    def _build_agent_context(self, turn: ConversationTurn, conversation_context: Optional[Dict[str, Any]]) -> str:
        """Build context information for the LangChain agent."""
        context_parts = []
        
        # Visual context
        if turn.visual_context:
            visual = turn.visual_context
            context_parts.append(f"Current visual context: {visual.scene_summary}")
            
            if visual.detected_objects:
                objects = [f"{obj.get('label', 'unknown')}" for obj in visual.detected_objects[:3]]
                context_parts.append(f"Detected objects: {', '.join(objects)}")
            
            if visual.ocr_text:
                texts = [f'"{text}"' for text in visual.ocr_text[:3] if text.strip()]
                if texts:
                    context_parts.append(f"Screen text: {', '.join(texts)}")
        
        # Conversation context
        if conversation_context:
            stm_entries = conversation_context.get('stm_entries', [])
            if stm_entries:
                recent_context = []
                for entry in stm_entries[-3:]:  # Last 3 entries
                    if entry.entry_type == 'user_input':
                        recent_context.append(f"User: {entry.content}")
                    elif entry.entry_type == 'bot_response':
                        recent_context.append(f"You: {entry.content}")
                
                if recent_context:
                    context_parts.append("Recent conversation:")
                    context_parts.extend(recent_context)
        
        # Game context
        game_context = self._get_game_context()
        if game_context:
            context_parts.append(f"Game: {game_context.get('game_name', 'Unknown')}")
        
        return "\n".join(context_parts) if context_parts else "No additional context available."
    
    async def _generate_direct_llm_response(self, turn: ConversationTurn, conversation_context: Optional[Dict[str, Any]]) -> str:
        """Fallback method using direct LLM calls."""
        try:
            # Build the prompt with visual context and conversation memory
            prompt = self._build_enhanced_visual_aware_prompt(turn)
            
            # Get LLM service
            llm_service = self.app_context.get_service('llm_service')
            if not llm_service:
                return "I'm not connected to my language model right now."
            
            # Generate response using the LLM service
            response = await llm_service.handle_user_text_query(
                user_text=prompt,
                user_name="VisionAwareUser"
            )
            
            return response if response else "I'm having trouble thinking about that right now."
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Direct LLM response generation error: {e}")
            return "I'm having trouble thinking about that right now."
    
    def _update_conversation_summary(self, turn: ConversationTurn):
        """Update conversation summary to maintain short-term memory."""
        try:
            # Get recent conversation turns (last 5 turns)
            recent_turns = list(self.conversation_history)[-5:]
            
            # Create a summary of recent conversation
            summary_parts = []
            for t in recent_turns:
                if t.user_input and t.response:
                    summary_parts.append(f"User: {t.user_input[:100]}...")
                    summary_parts.append(f"Danzar: {t.response[:100]}...")
            
            self.conversation_summary = "\n".join(summary_parts)
            
            # Clean up old conversation history (older than 5 minutes)
            current_time = time.time()
            self.conversation_history = deque([
                t for t in self.conversation_history 
                if current_time - t.timestamp < self.conversation_context_window
            ], maxlen=50)
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Error updating conversation summary: {e}")
    
    async def _generate_visual_aware_response(self, turn: ConversationTurn) -> str:
        """Generate a response that naturally incorporates visual context and conversation memory."""
        try:
            # Build the prompt with visual context and conversation memory
            prompt = self._build_enhanced_visual_aware_prompt(turn)
            
            # Get LLM service
            llm_service = self.app_context.get_service('llm_service')
            if not llm_service:
                return "I'm not connected to my language model right now."
            
            # Generate response using the LLM service
            response = await llm_service.handle_user_text_query(
                user_text=prompt,
                user_name="VisionAwareUser"
            )
            
            return response if response else "I'm having trouble thinking about that right now."
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Response generation error: {e}")
            return "I'm having trouble thinking about that right now."
    
    def _build_enhanced_visual_aware_prompt(self, turn: ConversationTurn) -> str:
        """Build an enhanced prompt that includes visual context and CLIP insights."""
        prompt_parts = []
        
        # System context
        system_prompt = self.app_context.active_profile.system_prompt_commentary if self.app_context and hasattr(self.app_context, 'active_profile') else "You are DANZAR, a vision-capable gaming assistant with a witty personality."
        prompt_parts.append(system_prompt)
        
        # Visual context
        if turn.visual_context:
            visual = turn.visual_context
            prompt_parts.append(f"\nCURRENT VISUAL CONTEXT:")
            prompt_parts.append(f"- Scene: {visual.scene_summary}")
            prompt_parts.append(f"- Confidence: {visual.confidence:.2f}")
            
            # Add CLIP insights if available
            if visual.clip_insights and visual.clip_insights.get('clip_insights'):
                clip_insights = visual.clip_insights['clip_insights']
                if clip_insights:
                    prompt_parts.append(f"- Visual Understanding: {', '.join([insight['description'] for insight in clip_insights[:3]])}")
            
            # Add detected objects
            if visual.detected_objects:
                objects = [f"{obj.get('label', 'unknown')} ({obj.get('confidence', 0):.2f})" 
                          for obj in visual.detected_objects[:3]]
                prompt_parts.append(f"- Detected Objects: {', '.join(objects)}")
            
            # Add OCR text
            if visual.ocr_text:
                texts = [f'"{text}"' for text in visual.ocr_text[:3] if text.strip()]
                if texts:
                    prompt_parts.append(f"- Screen Text: {', '.join(texts)}")
        
        # Conversation context
        if self.conversation_summary:
            prompt_parts.append(f"\nCONVERSATION CONTEXT:")
            prompt_parts.append(self.conversation_summary)
        
        # Recent conversation history
        recent_turns = list(self.conversation_history)[-3:]  # Last 3 turns
        if recent_turns:
            prompt_parts.append(f"\nRECENT CONVERSATION:")
            for i, recent_turn in enumerate(recent_turns):
                if recent_turn != turn:  # Don't include current turn
                    prompt_parts.append(f"- User: {recent_turn.user_input}")
                    if recent_turn.response:
                        prompt_parts.append(f"- You: {recent_turn.response}")
        
        # Game context
        game_context = self._get_game_context()
        if game_context:
            prompt_parts.append(f"\nGAME CONTEXT:")
            prompt_parts.append(f"- Game: {game_context.get('game_name', 'Unknown')}")
            prompt_parts.append(f"- Context: {game_context.get('context', 'General gameplay')}")
        
        # User message
        prompt_parts.append(f"\nUSER MESSAGE: {turn.user_input}")
        
        # Instructions
        prompt_parts.append(f"\nINSTRUCTIONS:")
        prompt_parts.append("- Respond naturally and conversationally")
        prompt_parts.append("- Reference what you can see in the game when relevant")
        prompt_parts.append("- Provide helpful gaming advice and commentary")
        prompt_parts.append("- Keep responses concise but informative")
        prompt_parts.append("- If you can't see anything relevant, ask for clarification")
        
        return "\n".join(prompt_parts)
    
    async def query_visual_concept(self, query: str) -> Dict[str, Any]:
        """
        Query CLIP about a specific visual concept in the current frame.
        
        Args:
            query: Natural language query about what to look for
            
        Returns:
            CLIP's response about the query
        """
        if not self.clip_enhancer:
            return {'confidence': 0.0, 'description': 'CLIP not available'}
        
        try:
            # Get current frame from vision data
            vision_data = getattr(self.app_context, 'latest_vision_data', None)
            if not vision_data or not vision_data.get('current_frame'):
                return {'confidence': 0.0, 'description': 'No current frame available'}
            
            current_frame = vision_data['current_frame']
            
            # Query CLIP
            result = self.clip_enhancer.query_visual_concept(current_frame, query)
            return result
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Error querying visual concept: {e}")
            return {'confidence': 0.0, 'description': f'Error querying: {str(e)}'}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics including enhanced memory system."""
        stats = {
            'enhanced_memory_available': self.conversation_memory is not None,
            'clip_enabled': self.clip_enabled,
            'vision_integration_enabled': self.vision_integration_enabled,
            'current_visual_context': self.current_visual_context is not None,
            'conversation_history_length': len(self.conversation_history),
            'visual_context_history_length': len(self.visual_context_history)
        }
        
        # Add enhanced memory stats if available
        if self.conversation_memory:
            try:
                memory_stats = self.conversation_memory.get_memory_stats()
                stats['enhanced_memory_stats'] = memory_stats
            except Exception as e:
                stats['enhanced_memory_error'] = str(e)
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.conversation_memory:
                self.conversation_memory.stop_background_tasks()
                self.logger.info("[VisionAwareConversation] Enhanced memory cleanup completed")
            
            if self.clip_enhancer:
                # Add any CLIP cleanup if needed
                pass
            
            self.logger.info("[VisionAwareConversation] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"[VisionAwareConversation] Cleanup error: {e}") 