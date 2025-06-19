"""
VAD-based Voice Receiver for Discord
Implements proper Voice Activity Detection using WebRTC VAD and correct Vosk usage patterns.
Works with discord-ext-voice-recv for voice receiving capabilities.
"""

import asyncio
import logging
import time
import numpy as np
from collections import deque
from typing import Optional, Callable, Dict, Any
import webrtcvad
import vosk
import json
import discord
import audioop
import threading

try:
    from discord.ext import voice_recv
    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False
    voice_recv = None

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None


class VADVoiceSink:
    """
    A voice sink that works with discord-ext-voice-recv:
     • Downmixes 48 kHz stereo → 16 kHz mono
     • Uses WebRTC VAD to detect utterance boundaries
     • Runs Vosk STT on each finished utterance
    """
    def __init__(self, model_path: str, vad_aggressiveness: int = 2, speech_callback: Optional[Callable] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.speech_callback = speech_callback
        
        # Load Vosk
        self.logger.info(f"🔧 Loading Vosk model from: {model_path}")
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.logger.info("✅ Vosk recognizer initialized for VAD receiver")
        
        # Init VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.logger.info(f"✅ WebRTC VAD initialized (aggressiveness: {vad_aggressiveness})")
        
        # Resampler state
        self._resample_state = None
        
        # Per-user buffers
        self._buffers = {}  # user_id -> bytearray
        self._user_names = {}  # user_id -> display_name
        self._lock = threading.Lock()

    def wants_opus(self) -> bool:
        """Return False to receive PCM data instead of Opus."""
        return False

    def write(self, user, data):
        """
        Called by discord-ext-voice-recv for every ~20 ms of audio.
        data.pcm is 48 kHz stereo PCM.
        """
        try:
            # Store user name for logging
            self._user_names[user.id] = user.display_name
            
            # 1) Stereo → mono (downmix)
            mono = audioop.tomono(data.pcm, 2, 1, 1)
            
            # 2) Resample 48 kHz → 16 kHz
            pcm16k, self._resample_state = audioop.ratecv(
                mono, 2, 1, 48000, 16000, self._resample_state
            )
            
            # 3) Process into VAD/STT
            self._process_audio(user.id, pcm16k)
            
        except Exception as e:
            self.logger.error(f"❌ Error in VAD audio processing: {e}")

    def cleanup(self):
        """Cleanup when sink is finished."""
        try:
            self._buffers.clear()
            self._user_names.clear()
            self.logger.info("✅ VAD Voice Sink cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up sink: {e}")

    def _process_audio(self, user_id: int, pcm16k: bytes):
        """Process 16kHz mono PCM audio through VAD and Vosk."""
        with self._lock:
            buf = self._buffers.get(user_id, bytearray())
            
            # Break into 30 ms frames for VAD (480 samples * 2 bytes = 960 bytes)
            frame_size = int(16000 * 30 / 1000) * 2  # 30ms at 16kHz, 2 bytes per sample
            
            for i in range(0, len(pcm16k), frame_size):
                frame = pcm16k[i:i+frame_size]
                if len(frame) < frame_size:
                    continue
                    
                try:
                    is_speech = self.vad.is_speech(frame, 16000)
                    
                    if is_speech:
                        # Speech detected - add to buffer
                        buf.extend(frame)
                    else:
                        # Silence detected - end of utterance?
                        if buf:
                            # Process accumulated speech buffer
                            self._finalize_utterance(user_id, bytes(buf))
                            buf = bytearray()
                            
                except Exception as e:
                    self.logger.error(f"❌ VAD frame processing error: {e}")
                    
            self._buffers[user_id] = buf

    def _finalize_utterance(self, user_id: int, speech_data: bytes):
        """Process complete utterance through Vosk STT."""
        try:
            user_name = self._user_names.get(user_id, f"User{user_id}")
            
            # Feed speech data to Vosk
            if self.recognizer.AcceptWaveform(speech_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
            else:
                result = json.loads(self.recognizer.FinalResult())
                text = result.get("text", "").strip()
            
            if text:
                self.logger.info(f"✅ VAD transcription: '{text}' from {user_name}")
                
                # Call speech callback if provided
                if self.speech_callback:
                    # Schedule callback in main event loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self.speech_callback(np.array([]), user_name, text))
                    except RuntimeError:
                        # No running loop, call synchronously
                        asyncio.run(self.speech_callback(np.array([]), user_name, text))
            else:
                self.logger.debug(f"🔇 VAD utterance complete but no text detected for {user_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Error finalizing VAD utterance: {e}")

class VADVoiceReceiver:
    """
    VAD-based voice receiver that properly handles Discord's 48kHz stereo to 16kHz mono conversion.
    Uses discord-ext-voice-recv for voice receiving capabilities.
    """
    
    def __init__(self, app_context, speech_callback: Optional[Callable] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        # VAD settings
        self.vad_aggressiveness = 2  # Proven setting from successful Discord STT bots
        self.model_path = "models/vosk-model-small-en-us-0.15"
        
        # Audio sink
        self.sink = None
        
        self.logger.info("✅ VAD Voice Receiver initialized")

    async def initialize(self) -> bool:
        """Initialize the VAD voice receiver."""
        try:
            if not VOICE_RECV_AVAILABLE:
                self.logger.error("❌ discord-ext-voice-recv not available")
                return False
            # VAD voice receiver is initialized on-demand when starting to listen
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize VAD voice receiver: {e}")
            return False

    async def start_listening(self, voice_client) -> bool:
        """Start listening with VAD-based processing."""
        try:
            if not VOICE_RECV_AVAILABLE:
                self.logger.error("❌ discord-ext-voice-recv not available")
                return False
                
            self.logger.info("🎯 Starting VAD-based voice listening...")
            
            # Create VAD sink with proper audio format conversion
            self.sink = VADVoiceSink(
                model_path=self.model_path,
                vad_aggressiveness=self.vad_aggressiveness,
                speech_callback=self.speech_callback,
                logger=self.logger
            )
            
            # Start listening with the VAD sink
            voice_client.listen(self.sink)
            
            self.logger.info("✅ Started recording with VAD-based processing (48kHz→16kHz conversion)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start VAD listening: {e}")
            return False

    async def stop_listening(self, voice_client) -> bool:
        """Stop listening and cleanup."""
        try:
            if voice_client and hasattr(voice_client, 'is_listening') and voice_client.is_listening():
                voice_client.stop_listening()
            
            if self.sink:
                self.sink.cleanup()
                self.sink = None
            
            self.logger.info("✅ VAD voice receiver stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping VAD voice receiver: {e}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.sink:
                self.sink.cleanup()
                self.sink = None
            self.logger.info("✅ VAD voice receiver cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up VAD voice receiver: {e}") 