import os
import logging
import asyncio
from typing import Optional

class SmartTTSService:
    """Smart TTS service that switches between providers based on configuration"""
    
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.SmartTTSService")
        self.config = self.ctx.global_settings.get('TTS_SERVER', {})
        
        # Get provider from config
        self.provider = self.config.get('provider', 'pyttsx3').lower()
        
        # Initialize the appropriate TTS service
        self.tts_service = None
        self._initialize_provider()
        
        # Initialize async services if needed
        self._initialize_async_services()

    def _initialize_provider(self):
        """Initialize the configured TTS provider"""
        try:
            if self.provider == 'pyttsx3':
                from .tts_service_fast import FastTTSService
                self.tts_service = FastTTSService(self.ctx)
                self.logger.info("[SmartTTS] Using pyttsx3 (Windows SAPI) - Fastest option")
                
            elif self.provider == 'piper':
                from .tts_service_piper import PiperTTSService
                self.tts_service = PiperTTSService(self.ctx)
                self.logger.info("[SmartTTS] Using Piper TTS - GPU-accelerated neural voices")
                
            elif self.provider == 'openai':
                from .tts_service_openai import OpenAITTSService
                self.tts_service = OpenAITTSService(self.ctx)
                self.logger.info("[SmartTTS] Using OpenAI TTS - Cloud-based, very fast")
                
            elif self.provider in ['chatterbox', 'custom']:
                from .tts_service import TTSService
                self.tts_service = TTSService(self.ctx)
                self.logger.info("[SmartTTS] Using Chatterbox TTS - Legacy (slow)")
                
            else:
                self.logger.error(f"[SmartTTS] Unknown provider '{self.provider}', falling back to pyttsx3")
                from .tts_service_fast import FastTTSService
                self.tts_service = FastTTSService(self.ctx)
                
        except ImportError as e:
            self.logger.error(f"[SmartTTS] Failed to import provider '{self.provider}': {e}")
            self.logger.info("[SmartTTS] Falling back to pyttsx3 TTS service")
            try:
                from .tts_service_fast import FastTTSService
                self.tts_service = FastTTSService(self.ctx)
            except Exception:
                self.logger.error("[SmartTTS] Even pyttsx3 fallback failed, using basic TTS")
                from .tts_service import TTSService
                self.tts_service = TTSService(self.ctx)
        except Exception as e:
            self.logger.error(f"[SmartTTS] Error initializing provider '{self.provider}': {e}")
            self.tts_service = None

    def _initialize_async_services(self):
        """Initialize async services like Piper TTS"""
        if self.provider == 'piper' and self.tts_service and hasattr(self.tts_service, 'initialize'):
            try:
                # Run async initialization in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    initialize_method = getattr(self.tts_service, 'initialize', None)
                    if initialize_method:
                        success = loop.run_until_complete(initialize_method())
                    else:
                        success = True  # No initialization needed
                    if success:
                        self.logger.info("[SmartTTS] Piper TTS initialized successfully")
                    else:
                        self.logger.error("[SmartTTS] Piper TTS initialization failed")
                        self.tts_service = None
                finally:
                    loop.close()
            except Exception as e:
                self.logger.error(f"[SmartTTS] Error initializing Piper TTS: {e}")
                self.tts_service = None

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio using the configured provider"""
        if not self.tts_service:
            self.logger.error("[SmartTTS] No TTS service available")
            return None
            
        return self.tts_service.generate_audio(text)

    def switch_provider(self, new_provider: str) -> bool:
        """Dynamically switch TTS provider"""
        old_provider = self.provider
        self.provider = new_provider.lower()
        
        try:
            self._initialize_provider()
            self.logger.info(f"[SmartTTS] Switched from '{old_provider}' to '{new_provider}'")
            return True
        except Exception as e:
            self.logger.error(f"[SmartTTS] Failed to switch to '{new_provider}': {e}")
            # Revert to old provider
            self.provider = old_provider
            self._initialize_provider()
            return False

    def get_provider_info(self) -> dict:
        """Get information about current provider"""
        return {
            'provider': self.provider,
            'available': self.tts_service is not None,
            'speed_rating': self._get_speed_rating(),
            'description': self._get_provider_description()
        }

    def _get_speed_rating(self) -> str:
        """Get speed rating for current provider"""
        ratings = {
            'piper': 'Very Fast (1-3s)',
            'pyttsx3': 'Instant',
            'openai': 'Very Fast (2-3s)',
            'chatterbox': 'Slow (60s+)',
            'custom': 'Variable'
        }
        return ratings.get(self.provider, 'Unknown')

    def _get_provider_description(self) -> str:
        """Get description for current provider"""
        descriptions = {
            'piper': 'Piper TTS - GPU-accelerated, local, high quality neural voices',
            'pyttsx3': 'Windows SAPI voices - local, instant response',
            'openai': 'OpenAI TTS API - cloud-based, high quality',
            'chatterbox': 'Custom Chatterbox server - slow but configurable',
            'custom': 'Custom TTS implementation'
        }
        return descriptions.get(self.provider, 'Unknown provider') 