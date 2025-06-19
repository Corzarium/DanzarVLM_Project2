# discord_integration/audio_sink.py
import discord
import discord.voice_client
import logging
import threading
import time
from typing import Any, Dict, Optional, TYPE_CHECKING, Callable, Tuple
import numpy as np
import asyncio
import queue
from services.vad_service import VoiceActivityDetector
from scipy import signal  # For resampling

if TYPE_CHECKING:
    from ..DanzarVLM import AppContext


class DanzarAudioSink:
    """Compatibility audio sink for standard discord.py (no voice receiving)"""
    
    def __init__(self, app_context: 'AppContext'):
        self.ctx = app_context
        self.logger = app_context.logger
        self.is_listening = False  # Disabled for standard discord.py
        self._processing_enabled = False
        
        self.logger.warning("[DanzarAudioSink] Voice receiving disabled - standard discord.py does not support voice input")
        
        # Logging stats (kept for compatibility)
        self._total_chunks_received = 0
        self._last_activity_log = 0
        self._active_users = set()

    def write(self, data: bytes, user_id: int) -> None:
        """Compatibility method - no voice data will be received in standard discord.py"""
        # This method exists for compatibility but won't be called
        pass

    def stop_listening(self) -> None:
        """Stop voice packet processing"""
        self.is_listening = False
        self._processing_enabled = False
        self.logger.info("[DanzarAudioSink] Voice listening stopped (compatibility mode)")

    def cleanup(self) -> None:
        """Clean up audio sink resources"""
        self.stop_listening()
        self.logger.info("[DanzarAudioSink] Audio sink cleaned up")


class ModernAudioRecorder:
    """Audio recorder compatible with standard discord.py"""
    
    def __init__(self, app_context: 'AppContext'):
        self.app_context = app_context
        self.logger = app_context.logger
        self.audio_sink: Optional[DanzarAudioSink] = None
        self.is_recording = False
        self._recording_thread: Optional[threading.Thread] = None
        
        self.logger.info("[ModernAudioRecorder] Initialized with discord.py compatibility mode")

    def start_recording(self, voice_client, callback_finished=None):
        """Start recording simulation (discord.py doesn't support voice receiving)"""
        try:
            if not voice_client:
                self.logger.error("[ModernAudioRecorder] No voice client provided")
                return False

            self.logger.info(f"[ModernAudioRecorder] start_recording called for channel '{voice_client.channel.name if voice_client.channel else 'Unknown'}'")
            
            # Check available voice methods (will be empty in standard discord.py)
            voice_methods = [method for method in dir(voice_client) if 'record' in method.lower() or 'sink' in method.lower()]
            self.logger.info(f"[ModernAudioRecorder] Available voice methods: {voice_methods}")
            
            # Create compatibility audio sink
            self.audio_sink = DanzarAudioSink(self.app_context)
            
            self.logger.warning("[ModernAudioRecorder] Voice input disabled - discord.py does not support voice receiving")
            self.logger.info("[ModernAudioRecorder] For voice input, consider using py-cord or external voice solutions")
            
            self.is_recording = True
            
            if callback_finished:
                callback_finished(self.audio_sink)
                
            return True
            
        except Exception as e:
            self.logger.error(f"[ModernAudioRecorder] Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording simulation"""
        try:
            if self.audio_sink:
                self.audio_sink.stop_listening()
                self.audio_sink = None
            
            self.is_recording = False
            self.logger.info("[ModernAudioRecorder] Recording stopped")
            
        except Exception as e:
            self.logger.error(f"[ModernAudioRecorder] Error stopping recording: {e}")

    def cleanup(self):
        """Clean up recorder resources"""
        self.stop_recording()
        self.logger.info("[ModernAudioRecorder] Audio recorder cleaned up")


class DanzarVoiceReceiver:
    """Voice receiver compatibility layer for discord.py"""
    
    def __init__(self, app_context: 'AppContext'):
        self.app_context = app_context
        self.logger = app_context.logger
        
        self.logger.warning("[DanzarVoiceReceiver] Voice receiving disabled with standard discord.py")
        self.logger.info("[DanzarVoiceReceiver] Text-based interaction available")


class VoiceAudioSink:
    def __init__(self, 
                 vad_callback: Optional[Callable[[bool, bool], None]] = None,
                 sample_rate: int = 48000,
                 channels: int = 2,
                 buffer_size: int = 960,  # 20ms at 48kHz
                 vad_threshold: float = 0.3,
                 vad_trigger_level: int = 2,
                 vad_hold_time: float = 0.3):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.audio_buffer = queue.Queue()
        self.is_recording = False
        self.vad_callback = vad_callback
        
        # Initialize VAD with 16kHz sample rate
        self.vad = VoiceActivityDetector(
            sample_rate=16000,  # VAD expects 16kHz
            frame_duration_ms=20,  # 20ms frames
            threshold=vad_threshold,
            trigger_level=vad_trigger_level,
            hold_time=vad_hold_time
        )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        self.logger.info(f"[VoiceAudioSink] Initialized with sample_rate={sample_rate}, channels={channels}")

    def write(self, data: bytes):
        """Process incoming audio data."""
        try:
            if not self.is_recording:
                self.logger.debug("[VoiceAudioSink] Received data but not recording")
                return
                
            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Reshape if stereo
            if self.channels == 2:
                audio_data = audio_data.reshape(-1, 2)
            
            self.logger.info(f"[VoiceAudioSink] Received audio data: shape={audio_data.shape}, "
                           f"min={audio_data.min():.3f}, max={audio_data.max():.3f}, "
                           f"mean={np.abs(audio_data).mean():.3f}")
            
            # Resample for VAD
            vad_audio = self._resample_audio(audio_data)
            
            # Add to buffer
            self.audio_buffer.put(audio_data)
            
            # Process with VAD
            is_speaking, speech_ended = self.vad.process_audio(vad_audio)
            
            if is_speaking or speech_ended:
                self.logger.info(f"[VoiceAudioSink] VAD detected: is_speaking={is_speaking}, speech_ended={speech_ended}")
            
            # Call VAD callback if provided
            if self.vad_callback:
                self.vad_callback(is_speaking, speech_ended)
                
        except Exception as e:
            self.logger.error(f"[VoiceAudioSink] Error in write: {e}", exc_info=True)

    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio from 48kHz to 16kHz."""
        try:
            # Convert to mono if stereo
            if self.channels == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample from 48kHz to 16kHz
            number_of_samples = round(len(audio_data) * 16000 / self.sample_rate)
            resampled = signal.resample(audio_data, number_of_samples)
            
            self.logger.debug(f"[VoiceAudioSink] Resampled audio: original_shape={audio_data.shape}, "
                            f"resampled_shape={resampled.shape}")
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"[VoiceAudioSink] Error resampling audio: {e}", exc_info=True)
            return audio_data

    def _process_audio(self):
        """Process audio in background thread."""
        while True:
            try:
                # Process audio from buffer
                if not self.audio_buffer.empty():
                    audio_data = self.audio_buffer.get()
                    # Resample for VAD
                    vad_audio = self._resample_audio(audio_data)
                    # Process with VAD
                    is_speaking, speech_ended = self.vad.process_audio(vad_audio)
                    
                    if is_speaking or speech_ended:
                        self.logger.info(f"[VoiceAudioSink] Background VAD detected: is_speaking={is_speaking}, "
                                       f"speech_ended={speech_ended}")
                    
                    # Call VAD callback if provided
                    if self.vad_callback:
                        self.vad_callback(is_speaking, speech_ended)
                        
            except Exception as e:
                self.logger.error(f"[VoiceAudioSink] Error in process_audio: {e}", exc_info=True)
                
            time.sleep(0.01)  # Small sleep to prevent CPU hogging

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get the current audio buffer data."""
        if self.audio_buffer.empty():
            self.logger.debug("[VoiceAudioSink] No audio data available")
            return None
            
        audio_data = self.audio_buffer.get()
        self.logger.debug(f"[VoiceAudioSink] Retrieved audio data: shape={audio_data.shape}, "
                         f"min={audio_data.min():.3f}, max={audio_data.max():.3f}")
        return audio_data
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        self.audio_buffer.queue.clear()
        self.vad.reset()
        self.speech_started = False
        self.logger.debug("[VoiceAudioSink] Buffer cleared")

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_buffer.queue.clear()
        self.vad.reset()
        self.logger.info("[VoiceAudioSink] Recording started")

    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        self.logger.info("[VoiceAudioSink] Recording stopped")