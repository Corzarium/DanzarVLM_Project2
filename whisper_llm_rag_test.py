#!/usr/bin/env python3
"""
Whisper + LLM + RAG Integration Test
Demonstrates the complete pipeline: Audio → Whisper STT → LLM/RAG → Response

This test uses:
- Whisper for Speech-to-Text
- LM Studio for LLM processing
- RAG for knowledge retrieval
- Memory service for context
"""

import asyncio
import logging
import time
import numpy as np
import tempfile
import os
from typing import Optional

# Audio capture
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

# Whisper STT
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# Core imports
from core.config_loader import load_global_settings
from core.game_profile import GameProfile

# Service imports
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.model_client import ModelClient
from services.tts_service import TTSService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhisperLLMRAGTest")

class SimpleAppContext:
    """Minimal app context for testing"""
    def __init__(self, global_settings: dict, active_profile: GameProfile):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger
        
        # Service instances (will be initialized)
        self.tts_service: Optional[TTSService] = None
        self.memory_service: Optional[MemoryService] = None
        self.model_client: Optional[ModelClient] = None
        self.llm_service: Optional[LLMService] = None

class WhisperLLMRAGSystem:
    """Complete Whisper + LLM + RAG system"""
    
    def __init__(self):
        self.logger = logger
        
        # Load configuration
        self.settings = load_global_settings() or {}
        
        # Create game profile for testing
        self.profile = GameProfile(
            game_name="test_voice_assistant",
            vlm_model="qwen2.5:7b",  # LM Studio model
            system_prompt_commentary="You are DanzarAI, a helpful AI assistant with access to knowledge and memory.",
            user_prompt_template_commentary="User said: {user_text}. Provide a helpful response.",
            vlm_max_tokens=300,
            vlm_temperature=0.7,
            vlm_max_commentary_sentences=3,
            conversational_llm_model="qwen2.5:7b"
        )
        
        # Create app context
        self.app_context = SimpleAppContext(self.settings, self.profile)
        
        # Initialize components
        self.whisper_model = None
        self.input_device = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.dtype = np.float32
        
        # Speech detection
        self.speech_threshold = 0.01
        self.min_speech_duration = 1.0
        self.max_silence_duration = 2.0
        
        # State
        self.is_recording = False
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize Whisper
            if not WHISPER_AVAILABLE:
                self.logger.error("❌ Whisper not available - install with: pip install openai-whisper")
                return False
            
            self.logger.info("🔧 Loading Whisper model...")
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None,
                whisper.load_model,
                "base"
            )
            self.logger.info("✅ Whisper model loaded")
            
            # Initialize services
            self.logger.info("🔧 Initializing services...")
            
            # Model Client
            self.app_context.model_client = ModelClient(app_context=self.app_context)
            self.logger.info("✅ Model Client initialized")
            
            # Memory Service
            self.app_context.memory_service = MemoryService(self.app_context)
            self.logger.info("✅ Memory Service initialized")
            
            # TTS Service
            self.app_context.tts_service = TTSService(self.app_context)
            self.logger.info("✅ TTS Service initialized")
            
            # LLM Service (with full RAG integration)
            self.app_context.llm_service = LLMService(
                app_context=self.app_context,
                audio_service=None,
                rag_service=None,  # Will use memory service for RAG
                model_client=self.app_context.model_client
            )
            self.logger.info("✅ LLM Service initialized")
            
            # Initialize audio device
            if not self._initialize_audio():
                return False
            
            self.logger.info("🎯 Whisper + LLM + RAG system ready!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _initialize_audio(self):
        """Initialize audio input device"""
        if not SOUNDDEVICE_AVAILABLE:
            self.logger.error("❌ sounddevice not available - install with: pip install sounddevice")
            return False
        
        try:
            # List and select virtual audio device
            self.logger.info("🎵 Available Audio Input Devices:")
            devices = sd.query_devices()
            
            virtual_devices = []
            for i, device in enumerate(devices):
                try:
                    if isinstance(device, dict):
                        max_channels = device.get('max_input_channels', 0)
                        device_name = device.get('name', f'Device {i}')
                    else:
                        max_channels = getattr(device, 'max_input_channels', 0)
                        device_name = getattr(device, 'name', f'Device {i}')
                    
                    if max_channels > 0:
                        self.logger.info(f"  {i}: {device_name} (channels: {max_channels})")
                        
                        # Look for virtual audio cables
                        if any(keyword in device_name.lower() for keyword in 
                              ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'stereo mix']):
                            virtual_devices.append((i, device_name))
                            self.logger.info(f"      ⭐ VIRTUAL AUDIO DEVICE")
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not process device {i}: {e}")
            
            # Select device
            if virtual_devices:
                self.input_device = virtual_devices[0][0]
                device_name = virtual_devices[0][1]
                self.logger.info(f"🎯 Auto-selected virtual audio device: {device_name}")
            else:
                try:
                    self.input_device = sd.default.device[0]
                    self.logger.warning("⚠️  Using default input device (no virtual audio found)")
                except:
                    self.input_device = 0
                    self.logger.warning("⚠️  Using device 0 as fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Audio initialization failed: {e}")
            return False
    
    def detect_speech(self, audio_chunk: np.ndarray) -> tuple[bool, bool]:
        """Simple speech detection"""
        try:
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            current_time = time.time()
            is_speech = rms > self.speech_threshold
            speech_ended = False
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.logger.info(f"🎤 Speech started (RMS: {rms:.4f})")
                self.last_speech_time = current_time
            elif self.is_speaking and self.last_speech_time:
                silence_duration = current_time - self.last_speech_time
                if silence_duration > self.max_silence_duration:
                    if self.speech_start_time:
                        speech_duration = current_time - self.speech_start_time
                        if speech_duration >= self.min_speech_duration:
                            speech_ended = True
                            self.logger.info(f"🎤 Speech ended (duration: {speech_duration:.2f}s)")
                    
                    self.is_speaking = False
                    self.speech_start_time = None
                    self.last_speech_time = None
            
            return is_speech, speech_ended
            
        except Exception as e:
            self.logger.error(f"❌ Speech detection error: {e}")
            return False, False
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            return None
        
        try:
            # Validate and normalize audio
            if len(audio_data) == 0:
                return None
            
            audio_duration = len(audio_data) / self.sample_rate
            if audio_duration < 0.5:
                return None
            
            # Normalize
            audio_max = np.max(np.abs(audio_data))
            if audio_max > 0:
                normalized_audio = audio_data * (0.8 / audio_max)
            else:
                normalized_audio = audio_data
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                try:
                    import scipy.io.wavfile as wavfile
                    audio_int16 = np.clip(normalized_audio * 32767, -32767, 32767).astype(np.int16)
                    wavfile.write(temp_file.name, self.sample_rate, audio_int16)
                except ImportError:
                    # Fallback without scipy
                    import wave
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        audio_int16 = np.clip(normalized_audio * 32767, -32767, 32767).astype(np.int16)
                        wav_file.writeframes(audio_int16.tobytes())
                
                temp_file_path = temp_file.name
            
            # Transcribe with Whisper
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    temp_file_path,
                    language='en',
                    temperature=0.0,
                    best_of=1
                )
            )
            
            # Clean up
            os.unlink(temp_file_path)
            
            if result and "text" in result:
                text = str(result["text"]).strip()
                if len(text) > 2:  # Basic filtering
                    self.logger.info(f"📝 Transcription: '{text}'")
                    return text
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None
    
    async def process_transcription(self, transcription: str):
        """Process transcription with LLM/RAG pipeline"""
        try:
            self.logger.info(f"🧠 Processing: '{transcription}'")
            
            # Use the full LLM service pipeline
            if self.app_context.llm_service:
                response = await self.app_context.llm_service.handle_user_text_query(
                    user_text=transcription,
                    user_name="VoiceUser"
                )
                
                if response and len(response.strip()) > 0:
                    self.logger.info(f"🤖 Response: {response}")
                    
                    # Generate TTS
                    if self.app_context.tts_service:
                        try:
                            self.logger.info("🔊 Generating TTS...")
                            tts_text = self._strip_markdown_for_tts(response)
                            
                            loop = asyncio.get_event_loop()
                            tts_audio = await loop.run_in_executor(
                                None,
                                self.app_context.tts_service.generate_audio,
                                tts_text
                            )
                            
                            if tts_audio:
                                self.logger.info("✅ TTS generated successfully")
                                # Could save to file or play here
                            else:
                                self.logger.warning("⚠️ TTS generation failed")
                        except Exception as e:
                            self.logger.error(f"❌ TTS error: {e}")
                else:
                    self.logger.info("🤖 No response generated")
            else:
                self.logger.error("❌ LLM service not available")
                
        except Exception as e:
            self.logger.error(f"❌ Processing error: {e}")
    
    def _strip_markdown_for_tts(self, text: str) -> str:
        """Remove markdown formatting for TTS"""
        import re
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        return text.strip()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for real-time processing"""
        if status:
            self.logger.warning(f"⚠️ Audio status: {status}")
        
        try:
            # Convert to mono if needed
            if len(indata.shape) > 1:
                audio_mono = np.mean(indata, axis=1)
            else:
                audio_mono = indata.copy()
            
            # Detect speech
            is_speech, speech_ended = self.detect_speech(audio_mono.flatten())
            
            # Buffer speech audio
            if self.is_speaking:
                self.audio_buffer.extend(audio_mono.flatten())
            
            # Process complete speech segments
            if speech_ended and len(self.audio_buffer) > 0:
                speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                self.audio_buffer.clear()
                
                # Schedule async processing
                asyncio.create_task(self._process_speech_segment(speech_audio))
                
        except Exception as e:
            self.logger.error(f"❌ Audio callback error: {e}")
    
    async def _process_speech_segment(self, audio_data: np.ndarray):
        """Process a complete speech segment"""
        try:
            # Transcribe
            transcription = await self.transcribe_audio(audio_data)
            
            if transcription:
                # Process with LLM/RAG
                await self.process_transcription(transcription)
            else:
                self.logger.info("🔇 No clear speech detected")
                
        except Exception as e:
            self.logger.error(f"❌ Speech processing error: {e}")
    
    async def start_listening(self):
        """Start listening for audio input"""
        if not SOUNDDEVICE_AVAILABLE:
            self.logger.error("❌ Audio not available")
            return
        
        try:
            self.logger.info("🎤 Starting audio capture...")
            self.logger.info("💡 Speak into your microphone or virtual audio cable")
            self.logger.info("💡 Press Ctrl+C to stop")
            
            # Start audio stream
            with sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                callback=self.audio_callback
            ):
                self.is_recording = True
                self.logger.info("✅ Audio capture started")
                
                # Keep running until interrupted
                try:
                    while True:
                        await asyncio.sleep(0.1)
                except KeyboardInterrupt:
                    self.logger.info("🛑 Stopping audio capture...")
                    self.is_recording = False
                    
        except Exception as e:
            self.logger.error(f"❌ Audio capture error: {e}")

async def main():
    """Main test function"""
    logger.info("🚀 Starting Whisper + LLM + RAG Integration Test")
    
    # Create system
    system = WhisperLLMRAGSystem()
    
    # Initialize
    if not await system.initialize():
        logger.error("❌ System initialization failed")
        return
    
    # Start listening
    await system.start_listening()
    
    logger.info("✅ Test completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"�� Test failed: {e}") 