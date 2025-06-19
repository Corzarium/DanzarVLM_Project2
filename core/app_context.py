import threading
from typing import Optional, TYPE_CHECKING
import logging

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

class AppContext:
    """Application context for managing shared resources and services."""
    
    def __init__(self, global_settings: dict, active_profile: 'GameProfile', logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger_instance
        self.shutdown_event = threading.Event()
        
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
        
        logger_instance.info("[AppContext] Initialized.") 