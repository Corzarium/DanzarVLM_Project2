"""
Voice Chat Service for DanzarAI Discord Bot
Handles voice recognition, STT processing, LLM conversation, and TTS playback
"""

import asyncio
import io
import tempfile
import logging
from collections import deque
from typing import Optional, Dict, Any, Callable, List
import webrtcvad
import whisper
import openai
from discord.ext import voice_recv
from discord.ext.voice_recv.extras import SpeechRecognitionSink
import discord

from services.tts_service import TTSService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from core.config_loader import load_global_settings


class VoiceChatService:
    """
    Voice Chat Service for DanzarAI Discord Integration
    
    Manages the full voice conversation pipeline:
    1. Voice Activity Detection (VAD)
    2. Speech-to-Text (STT) with Whisper
    3. LLM Processing with context memory
    4. Text-to-Speech (TTS) via existing Chatterbox service
    5. Turn-taking coordination
    """
    
    def __init__(self, 
                 tts_service: TTSService,
                 llm_service: LLMService,
                 memory_service: MemoryService):
        """Initialize voice chat service with existing DanzarAI services."""
        self.logger = logging.getLogger(__name__)
        self.tts_service = tts_service
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.config = load_global_settings() or {}
        
        # Voice chat components
        self.context_buffer = deque(maxlen=self.config.get('SHORT_TERM_SIZE', 6))
        self.vad = None
        self.whisper_model = None
        self.openai_client = None
        self.voice_sink = None
        
        # State management
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.current_voice_client: Optional[discord.VoiceClient] = None
        
        self.logger.info("[VoiceChatService] Initialized voice chat service")
    
    async def initialize(self) -> bool:
        """Initialize voice processing components."""
        try:
            # Initialize VAD
            vad_mode = self.config.get('OWW_VAD_THRESHOLD', 1)  # 1 = moderate filtering
            self.vad = webrtcvad.Vad(int(vad_mode))
            self.logger.info(f"[VoiceChatService] VAD initialized with mode {vad_mode}")
            
            # Initialize Whisper model
            whisper_model_size = 'medium'  # Force Whisper model to always use medium
            self.whisper_model = whisper.load_model(whisper_model_size)
            self.logger.info(f"[VoiceChatService] Whisper model '{whisper_model_size}' loaded")
            
            # Initialize OpenAI client for llama.cpp server
            llm_endpoint = self.config.get('LLM_SERVER', {}).get('endpoint', 'http://localhost:8080/v1')
            
            self.openai_client = openai.AsyncOpenAI(
                base_url=llm_endpoint,
                api_key="not-needed"  # llama.cpp server doesn't require API key
            )
            self.logger.info(f"[VoiceChatService] OpenAI client initialized for {llm_endpoint}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Failed to initialize: {e}", exc_info=True)
            return False
    
    def create_voice_sink(self) -> SpeechRecognitionSink:
        """Create voice recognition sink for Discord voice receive."""
        try:
            # Create speech recognition sink with callback
            sink = SpeechRecognitionSink(
                process_cb=self._on_speech_detected,
                default_recognizer="whisper"
            )
            self.voice_sink = sink
            self.logger.info("[VoiceChatService] Voice recognition sink created")
            return sink
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Failed to create voice sink: {e}", exc_info=True)
            raise
    
    async def start_listening(self, voice_client: discord.VoiceClient) -> bool:
        """Start listening for voice input."""
        try:
            if not self.voice_sink:
                self.voice_sink = self.create_voice_sink()
            
            self.current_voice_client = voice_client
            voice_client.listen(self.voice_sink)
            self.is_listening = True
            
            self.logger.info("[VoiceChatService] Started voice listening")
            return True
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Failed to start listening: {e}", exc_info=True)
            return False
    
    def stop_listening(self) -> None:
        """Stop listening for voice input."""
        try:
            if self.current_voice_client and self.is_listening:
                self.current_voice_client.stop_listening()
                self.is_listening = False
                self.logger.info("[VoiceChatService] Stopped voice listening")
                
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Error stopping listening: {e}")
    
    async def _on_speech_detected(self, user: discord.User, audio_data: bytes) -> None:
        """Handle detected speech from Discord user."""
        try:
            if self.is_processing or self.is_speaking:
                self.logger.debug(f"[VoiceChatService] Ignoring speech from {user.name} - busy processing")
                return
            
            self.is_processing = True
            self.logger.info(f"[VoiceChatService] Processing speech from {user.name}")
            
            # Process the speech in a separate task to avoid blocking
            asyncio.create_task(self._process_speech_pipeline(user, audio_data))
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Error in speech detection: {e}", exc_info=True)
            self.is_processing = False
    
    async def _process_speech_pipeline(self, user: discord.User, audio_data: bytes) -> None:
        """Complete speech processing pipeline: STT -> LLM -> TTS."""
        try:
            # Step 1: Speech-to-Text
            transcribed_text = await self._transcribe_audio(audio_data)
            if not transcribed_text or not transcribed_text.strip():
                self.logger.debug("[VoiceChatService] No text transcribed from audio")
                return
            
            self.logger.info(f"[VoiceChatService] Transcribed: '{transcribed_text}'")
            
            # Step 2: Add user message to context
            self.context_buffer.append({
                "role": "user", 
                "content": transcribed_text,
                "name": user.display_name
            })
            
            # Step 3: Get LLM response
            llm_response = await self._get_llm_response(transcribed_text, user.display_name)
            if not llm_response or not llm_response.strip():
                self.logger.warning("[VoiceChatService] No response from LLM")
                return
            
            # Step 4: Add assistant response to context
            self.context_buffer.append({
                "role": "assistant",
                "content": llm_response
            })
            
            # Step 5: Stop listening during TTS playback
            self.stop_listening()
            self.is_speaking = True
            
            # Step 6: Generate and play TTS
            await self._play_tts_response(llm_response)
            
            # Step 7: Resume listening after TTS completes
            self.is_speaking = False
            if self.current_voice_client:
                await self.start_listening(self.current_voice_client)
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Error in speech pipeline: {e}", exc_info=True)
        finally:
            self.is_processing = False
    
    async def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data using Whisper."""
        try:
            if not self.whisper_model:
                self.logger.error("[VoiceChatService] Whisper model not initialized")
                return None
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(
                temp_file_path,
                language=self.config.get('LANGUAGE', 'en')
            )
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            transcribed_text = result.get('text', '').strip()
            return transcribed_text if transcribed_text else None
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] STT transcription error: {e}", exc_info=True)
            return None
    
    async def _get_llm_response(self, user_text: str, user_name: str) -> Optional[str]:
        """Get response from LLM using context buffer."""
        try:
            # Use existing LLM service if available
            if self.llm_service:
                response = await self.llm_service.handle_user_text_query(user_text, user_name)
                if isinstance(response, str):
                    return self._strip_think_tags(response)
                return None
            
            # Fallback to direct OpenAI call (llama.cpp server)
            if not self.openai_client:
                self.logger.error("[VoiceChatService] No LLM service available")
                return None
            
            # Prepare messages for OpenAI API
            messages = list(self.context_buffer)
            if not messages:
                messages = [{"role": "user", "content": user_text}]
            
            # Get model name from config
            model_name = self.config.get('DEFAULT_CONVERSATIONAL_LLM_MODEL', 'mimo-vl-7b-rl')
            
            # Make request to llama.cpp server
            response = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=self.config.get('conversational_max_tokens', 150),
                temperature=self.config.get('conversational_temperature', 0.7)
            )
            
            if response and response.choices:
                return self._strip_think_tags(response.choices[0].message.content)
            
            return None
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] LLM request error: {e}", exc_info=True)
            return None
    
    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM responses."""
        if not text:
            return text
        
        import re
        # Remove think tags and their content
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        # Fallback if response is empty after stripping
        if not clean_text and text.strip():
            clean_text = "Let me think about that for a moment."
        
        return clean_text
    
    async def _play_tts_response(self, response_text: str) -> None:
        """Generate and play TTS audio response."""
        try:
            if not self.current_voice_client:
                self.logger.warning("[VoiceChatService] No voice client for TTS playback")
                return
            
            # Use existing TTS service
            if not self.tts_service:
                self.logger.error("[VoiceChatService] TTS service not available")
                return
            
            # Generate TTS audio
            tts_audio = self.tts_service.generate_audio(response_text)
            if not tts_audio:
                self.logger.warning("[VoiceChatService] Failed to generate TTS audio")
                return
            
            # Save audio to temporary file for Discord playback
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(tts_audio)
                temp_file_path = temp_file.name
            
            # Play audio through Discord
            source = discord.FFmpegPCMAudio(temp_file_path)
            
            # Create event for completion tracking
            playback_complete = asyncio.Event()
            
            def after_playback(error):
                if error:
                    self.logger.error(f"[VoiceChatService] TTS playback error: {error}")
                else:
                    self.logger.info("[VoiceChatService] TTS playback completed")
                
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"[VoiceChatService] Failed to cleanup temp file: {cleanup_error}")
                
                playback_complete.set()
            
            # Start playback
            self.current_voice_client.play(source, after=after_playback)
            
            # Wait for playback to complete
            timeout = self.config.get('TTS_PLAYBACK_TIMEOUT_S', 30)
            await asyncio.wait_for(playback_complete.wait(), timeout=timeout)
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] TTS playback error: {e}", exc_info=True)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation context."""
        return {
            "context_length": len(self.context_buffer),
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "is_speaking": self.is_speaking,
            "recent_messages": list(self.context_buffer)[-3:] if self.context_buffer else []
        }
    
    def clear_context(self) -> None:
        """Clear conversation context buffer."""
        self.context_buffer.clear()
        self.logger.info("[VoiceChatService] Conversation context cleared")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.stop_listening()
            
            # Close OpenAI client if needed
            if hasattr(self.openai_client, 'close'):
                await self.openai_client.close()
            
            self.logger.info("[VoiceChatService] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"[VoiceChatService] Cleanup error: {e}", exc_info=True) 