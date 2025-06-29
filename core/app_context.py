import threading
import asyncio
from typing import Optional, TYPE_CHECKING
import logging
from services.vision_tools import VisionTools

# Import type hints conditionally to avoid circular imports
if TYPE_CHECKING:
    from core.game_profile import GameProfile
    from services.tts_service import TTSService
    from services.memory_service import MemoryService
    from services.model_client import ModelClient
    from services.llm_service import LLMService
    from services.faster_whisper_stt_service import FasterWhisperSTTService
    from services.simple_voice_receiver import SimpleVoiceReceiver
    from services.vad_voice_receiver import VADVoiceReceiver
    from services.offline_vad_voice_receiver import OfflineVADVoiceReceiver
    from services.langchain_tools_service import DanzarLangChainTools
    from services.langchain_model_client import LangChainOllamaWrapper

class AppContext:
    """Application context for managing shared resources and services."""
    
    def __init__(self, global_settings: dict, active_profile: 'GameProfile', logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger_instance
        self.shutdown_event = threading.Event()
        
        # Event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop running, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Add missing attributes for LLM service compatibility
        self.rag_service_instance = None
        self.is_in_conversation = threading.Event()
        
        # Service instances
        self.tts_service: Optional['TTSService'] = None
        self.memory_service: Optional['MemoryService'] = None
        self.model_client: Optional['ModelClient'] = None
        self.llm_service: Optional['LLMService'] = None
        
        # Voice processing components
        self.whisper_model = None
        self.vad_model = None
        self.faster_whisper_stt_service: Optional['FasterWhisperSTTService'] = None
        self.simple_voice_receiver: Optional['SimpleVoiceReceiver'] = None
        self.vad_voice_receiver: Optional['VADVoiceReceiver'] = None
        self.offline_vad_voice_receiver: Optional['OfflineVADVoiceReceiver'] = None
        
        # Vision and LangChain tools
        self.vision_tools = VisionTools(self)
        self.langchain_tools: Optional['DanzarLangChainTools'] = None
        self.langchain_model: Optional['LangChainOllamaWrapper'] = None
        
        logger_instance.info("[AppContext] Initialized.")
    
    async def initialize_langchain_tools(self) -> bool:
        """Initialize LangChain tools and agent."""
        try:
            # Import here to avoid circular imports
            from services.langchain_tools_service import DanzarLangChainTools
            from services.langchain_model_client import LangChainOllamaWrapper
            
            self.logger.info("[AppContext] 🔧 Initializing LangChain tools...")
            
            # Create LangChain tools service
            self.langchain_tools = DanzarLangChainTools(self)
            
            # Create LangChain model wrapper if model client is available
            if self.model_client:
                self.langchain_model = LangChainOllamaWrapper(self.model_client)
                
                # Initialize agent
                success = await self.langchain_tools.initialize_agent(self.langchain_model)
                if success:
                    self.logger.info("[AppContext] ✅ LangChain tools and agent initialized successfully")
                    return True
                else:
                    self.logger.error("[AppContext] ❌ Failed to initialize LangChain agent")
                    return False
            else:
                self.logger.warning("[AppContext] ⚠️ Model client not available, LangChain tools will be limited")
                return True
                
        except Exception as e:
            self.logger.error(f"[AppContext] ❌ Error initializing LangChain tools: {e}")
            return False
    
    def get_langchain_tools_info(self) -> dict:
        """Get information about LangChain tools status."""
        if self.langchain_tools:
            return self.langchain_tools.get_tools_info()
        else:
            return {
                "total_tools": 0,
                "tool_names": [],
                "agent_ready": False,
                "vision_tools_available": bool(self.vision_tools),
                "memory_tools_available": bool(self.memory_service),
                "rag_tools_available": bool(self.rag_service_instance)
            } 