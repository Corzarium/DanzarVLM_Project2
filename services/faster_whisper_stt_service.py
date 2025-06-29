#!/usr/bin/env python3
"""
Whisper STT Service Client for DanzarVLM
Provides speech-to-text by calling external Whisper STT server
"""

import logging
import os
import tempfile
import time
import wave
import base64
import io
import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, Union
import numpy as np

class WhisperSTTService:
    """Client service for Whisper STT via external server"""
    
    def __init__(self, app_context, model_size: str = "base"):
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Server configuration
        self.endpoint = self.app_context.global_settings.get('EXTERNAL_SERVERS', {}).get('WHISPER_STT_SERVER', {}).get('endpoint', 'http://localhost:8083/transcribe')
        self.server_timeout = self.app_context.global_settings.get('EXTERNAL_SERVERS', {}).get('WHISPER_STT_SERVER', {}).get('timeout', 30)
        self.server_enabled = self.app_context.global_settings.get('EXTERNAL_SERVERS', {}).get('WHISPER_STT_SERVER', {}).get('enabled', True)
        
        # Model configuration
        self.model_size = model_size
        
        # Audio settings
        self.sample_rate = 16000
        
        # HTTP session
        self.session = None
        self.server_available = False
        
        self.logger.info(f"[WhisperSTTService] Initializing client for model: {model_size}")
        self.logger.info(f"[WhisperSTTService] Server endpoint: {self.endpoint}")
        
    async def initialize(self) -> bool:
        """Initialize the Whisper STT service client"""
        try:
            if not self.server_enabled:
                self.logger.warning("Whisper STT server is disabled in configuration")
                return False
                
            self.logger.info(f"🚀 Initializing Whisper STT service client...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.server_timeout)
            )
            
            # Test server connectivity
            await self._test_server_connection()
            
            if self.server_available:
                self.logger.info("✅ Whisper STT service client initialized successfully")
                return True
            else:
                self.logger.error("❌ Failed to connect to Whisper STT server")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Whisper STT service client: {e}")
            return False
    
    async def _test_server_connection(self):
        """Test connection to the external server"""
        try:
            if not self.session:
                return
                
            # Try to get server health status
            async with self.session.get(f"{self.endpoint.replace('/transcribe', '/health')}") as response:
                if response.status == 200:
                    self.server_available = True
                    self.logger.info("✅ Whisper STT server is available")
                else:
                    self.server_available = False
                    self.logger.warning(f"⚠️ Whisper STT server health check failed: {response.status}")
                    
        except Exception as e:
            self.server_available = False
            self.logger.warning(f"⚠️ Could not connect to Whisper STT server: {e}")
    
    async def _call_server(self, audio_data: np.ndarray, language: str = "en", task: str = "transcribe") -> str:
        """
        Call the external Whisper STT server
        
        Args:
            audio_data: Audio data as numpy array
            language: Language code
            task: Transcription task (transcribe/translate)
            
        Returns:
            Transcribed text
        """
        try:
            if not self.server_available or not self.session:
                return "Error: Server not available"
            
            # Convert audio to WAV bytes
            audio_bytes = self._numpy_to_wav_bytes(audio_data)
            if not audio_bytes:
                return "Error: Failed to convert audio to WAV"
            
            # Prepare multipart form data with file upload
            form_data = aiohttp.FormData()
            form_data.add_field('file', audio_bytes, 
                              filename='audio.wav',
                              content_type='audio/wav')
            form_data.add_field('language', language)
            form_data.add_field('task', task)
            
            # Make the request
            async with self.session.post(
                self.endpoint,
                data=form_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    if "transcription" in result:
                        return result["transcription"].strip()
                    else:
                        return "Error: Invalid response format from server"
                else:
                    error_text = await response.text()
                    self.logger.error(f"Server error {response.status}: {error_text}")
                    return f"Error: Server returned status {response.status}"
                    
        except asyncio.TimeoutError:
            self.logger.error("Timeout calling Whisper STT server")
            return "Error: Server timeout"
        except Exception as e:
            self.logger.error(f"❌ Error calling Whisper STT server: {e}")
            return f"Error: {str(e)}"
    
    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> Optional[bytes]:
        """Convert numpy audio array to WAV bytes"""
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                return wav_buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Error converting audio to WAV: {e}")
            return None
    
    async def transcribe_audio_data(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data using external Whisper STT server
        
        Args:
            audio_data: Audio data as numpy array (16kHz, mono, float32)
            
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            if not self.server_available:
                self.logger.error("❌ Whisper STT server not available")
                return None
            
            # Basic audio analysis
            audio_duration = len(audio_data) / self.sample_rate
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Audio analysis - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # Check for silence
            if audio_rms < 0.01:
                self.logger.info("🔇 Audio too quiet, likely silence")
                return None
            
            # Call external server
            result = await self._call_server(audio_data, language="en", task="transcribe")
            
            if result.startswith("Error:"):
                self.logger.error(f"❌ Transcription failed: {result}")
                return None
            
            # Clean up transcription
            cleaned_result = self._clean_transcription(result)
            
            if cleaned_result and len(cleaned_result.strip()) > 0:
                self.logger.info(f"✅ Transcription successful: '{cleaned_result}'")
                return cleaned_result
            else:
                self.logger.info("🔇 No speech detected in audio")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None
    
    def _clean_transcription(self, text: str) -> str:
        """Clean up transcription text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common transcription artifacts
        text = text.replace("[BLANK_AUDIO]", "")
        text = text.replace("[MUSIC]", "")
        text = text.replace("[NOISE]", "")
        
        # Remove timestamps if present
        import re
        text = re.sub(r'\[\d+:\d+\.\d+\]', '', text)
        
        return text.strip()
    
    def _preprocess_discord_audio(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """
        Preprocess Discord audio for better transcription
        
        Args:
            audio_data: Raw audio data
            original_sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Resample if needed
            if original_sample_rate != self.sample_rate:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * self.sample_rate / original_sample_rate))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Audio preprocessing failed: {e}")
            return audio_data
    
    async def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file using external Whisper STT server
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not os.path.exists(audio_file_path):
                self.logger.error(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            # Read audio file
            import wave
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe
            return await self.transcribe_audio_data(audio_data)
            
        except Exception as e:
            self.logger.error(f"❌ File transcription error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.server_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "Whisper STT Client",
            "server_available": self.server_available,
            "server_endpoint": self.endpoint,
            "server_enabled": self.server_enabled,
            "model_size": self.model_size,
            "session_active": self.session is not None
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None 