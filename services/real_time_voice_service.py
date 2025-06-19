#!/usr/bin/env python3
"""
Real-Time Voice Service for DanzarVLM
Neuro-sama style direct Discord voice pipeline
"""

import asyncio
import logging
import tempfile
import time
import os
from typing import Optional, Dict, Any, Callable
import numpy as np
import torch

class RealTimeVoiceService:
    """
    Real-time voice service using direct Discord voice integration
    Similar to Neuro-sama's architecture
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Voice processing components
        self.voice_client = None
        self.audio_sink = None
        self.stt_engine = None
        self.llm_client = None
        self.tts_engine = None
        
        # Real-time processing
        self.is_listening = False
        self.audio_buffer = []
        self.processing_lock = asyncio.Lock()
        
        # Callbacks
        self.on_transcription: Optional[Callable] = None
        self.on_response: Optional[Callable] = None
        
    async def initialize(self) -> bool:
        """Initialize the real-time voice service"""
        try:
            self.logger.info("🚀 Initializing Real-Time Voice Service...")
            
            # Initialize STT engine (Vosk for low latency)
            await self._initialize_stt()
            
            # Initialize LLM client
            await self._initialize_llm()
            
            # Initialize TTS engine
            await self._initialize_tts()
            
            self.logger.info("✅ Real-Time Voice Service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Real-Time Voice Service: {e}")
            return False
    
    async def _initialize_stt(self):
        """Initialize streaming STT engine"""
        try:
            # Try Vosk first (best for real-time)
            try:
                import vosk
                import json
                
                # Download model if needed
                model_path = "models/vosk-model-en-us-0.22"
                if not os.path.exists(model_path):
                    self.logger.info("📥 Downloading Vosk model...")
                    # You'd implement model download here
                
                self.stt_engine = vosk.Model(model_path)
                self.logger.info("✅ Vosk STT engine initialized")
                return
                
            except ImportError:
                self.logger.warning("⚠️ Vosk not available, falling back to faster-whisper")
            
            # Fallback to faster-whisper
            try:
                from faster_whisper import WhisperModel
                
                self.stt_engine = WhisperModel(
                    model_size_or_path="base",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16"
                )
                self.logger.info("✅ Faster-Whisper STT engine initialized")
                
            except ImportError:
                self.logger.error("❌ No STT engine available")
                raise
                
        except Exception as e:
            self.logger.error(f"❌ STT initialization failed: {e}")
            raise
    
    async def _initialize_llm(self):
        """Initialize local LLM client"""
        try:
            # Use existing LLM service from app context
            if hasattr(self.app_context, 'llm_service'):
                self.llm_client = self.app_context.llm_service
                self.logger.info("✅ Using existing LLM service")
            else:
                self.logger.error("❌ No LLM service available")
                raise Exception("LLM service not found")
                
        except Exception as e:
            self.logger.error(f"❌ LLM initialization failed: {e}")
            raise
    
    async def _initialize_tts(self):
        """Initialize streaming TTS engine"""
        try:
            # Use existing TTS service
            if hasattr(self.app_context, 'tts_service'):
                self.tts_engine = self.app_context.tts_service
                self.logger.info("✅ Using existing TTS service")
            else:
                self.logger.error("❌ No TTS service available")
                raise Exception("TTS service not found")
                
        except Exception as e:
            self.logger.error(f"❌ TTS initialization failed: {e}")
            raise
    
    async def connect_to_voice_channel(self, voice_client):
        """Connect to Discord voice channel with custom audio sink"""
        try:
            self.voice_client = voice_client
            
            # Create custom audio sink for real-time processing
            self.audio_sink = RealTimeAudioSink(self._process_audio_chunk)
            
            # Start listening
            self.voice_client.listen(self.audio_sink)
            self.is_listening = True
            
            self.logger.info("🎤 Connected to voice channel with real-time processing")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to voice channel: {e}")
            return False
    
    async def _process_audio_chunk(self, audio_data: bytes, user_id: int):
        """Process incoming audio chunk in real-time"""
        try:
            if not self.is_listening:
                return
            
            async with self.processing_lock:
                # Convert audio data to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to buffer
                self.audio_buffer.extend(audio_array)
                
                # Process if we have enough data (e.g., 1 second)
                sample_rate = 48000  # Discord's sample rate
                min_samples = sample_rate * 1  # 1 second
                
                if len(self.audio_buffer) >= min_samples:
                    # Extract chunk for processing
                    chunk = np.array(self.audio_buffer[:min_samples])
                    self.audio_buffer = self.audio_buffer[min_samples:]
                    
                    # Process chunk asynchronously
                    asyncio.create_task(self._process_speech_chunk(chunk, user_id))
                    
        except Exception as e:
            self.logger.error(f"❌ Audio chunk processing error: {e}")
    
    async def _process_speech_chunk(self, audio_chunk: np.ndarray, user_id: int):
        """Process a speech chunk through STT -> LLM -> TTS pipeline"""
        try:
            start_time = time.time()
            
            # 1. Speech-to-Text
            transcription = await self._transcribe_chunk(audio_chunk)
            if not transcription or len(transcription.strip()) < 3:
                return
            
            stt_time = time.time() - start_time
            self.logger.info(f"🎯 STT: '{transcription}' ({stt_time:.2f}s)")
            
            # Call transcription callback
            if self.on_transcription:
                await self.on_transcription(transcription, user_id)
            
            # 2. LLM Processing
            llm_start = time.time()
            response = await self._generate_response(transcription)
            llm_time = time.time() - llm_start
            
            self.logger.info(f"🧠 LLM: '{response[:50]}...' ({llm_time:.2f}s)")
            
            # 3. Text-to-Speech
            tts_start = time.time()
            await self._synthesize_and_play(response)
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            self.logger.info(f"⚡ Total pipeline: {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
            # Call response callback
            if self.on_response:
                await self.on_response(response, user_id)
                
        except Exception as e:
            self.logger.error(f"❌ Speech chunk processing error: {e}")
    
    async def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk using STT engine"""
        try:
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Convert numpy array to audio file
                import soundfile as sf
                sf.write(tmp_file.name, audio_chunk, 48000)
                
                # Transcribe based on engine type
                if hasattr(self.stt_engine, 'transcribe'):
                    # Faster-Whisper
                    segments, _ = self.stt_engine.transcribe(tmp_file.name)
                    transcription = " ".join([segment.text for segment in segments])
                else:
                    # Vosk or other
                    # Implementation depends on engine
                    transcription = "Transcription not implemented for this engine"
                
                return transcription.strip()
                
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return ""
    
    async def _generate_response(self, text: str) -> str:
        """Generate LLM response"""
        try:
            if hasattr(self.llm_client, 'generate'):
                response = await self.llm_client.generate(text)
            elif hasattr(self.llm_client, 'process_query'):
                response = await self.llm_client.process_query(text, source="RealTimeVoice")
            else:
                response = "LLM processing not available"
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ LLM generation error: {e}")
            return "I'm having trouble processing that right now."
    
    async def _synthesize_and_play(self, text: str):
        """Synthesize speech and play through Discord"""
        try:
            if hasattr(self.tts_engine, 'generate_audio'):
                # Generate audio
                audio_data = await self.tts_engine.generate_audio(text)
                
                # Play through Discord voice client
                if self.voice_client and hasattr(self.voice_client, 'play_audio'):
                    await self.voice_client.play_audio(audio_data)
                else:
                    self.logger.warning("⚠️ Voice client not available for audio playback")
                    
        except Exception as e:
            self.logger.error(f"❌ TTS synthesis error: {e}")
    
    def set_transcription_callback(self, callback: Callable):
        """Set callback for transcription events"""
        self.on_transcription = callback
    
    def set_response_callback(self, callback: Callable):
        """Set callback for response events"""
        self.on_response = callback
    
    async def disconnect(self):
        """Disconnect from voice channel"""
        try:
            self.is_listening = False
            
            if self.voice_client:
                await self.voice_client.disconnect()
                
            self.logger.info("🔇 Disconnected from voice channel")
            
        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "RealTimeVoiceService",
            "listening": self.is_listening,
            "connected": self.voice_client is not None,
            "stt_engine": type(self.stt_engine).__name__ if self.stt_engine else None,
            "buffer_size": len(self.audio_buffer)
        }


class RealTimeAudioSink:
    """Custom audio sink for real-time Discord voice processing"""
    
    def __init__(self, process_callback):
        self.process_callback = process_callback
        
    def write(self, data, user):
        """Called when audio data is received"""
        try:
            if user and hasattr(user, 'id'):
                # Process audio data asynchronously
                asyncio.create_task(self.process_callback(data, user.id))
        except Exception as e:
            logging.error(f"❌ Audio sink write error: {e}")
    
    def cleanup(self):
        """Cleanup audio sink"""
        pass 