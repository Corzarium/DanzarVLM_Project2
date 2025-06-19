"""
Voice Activity Detection (VAD) service using webrtcvad.
"""
import asyncio
import logging
import webrtcvad
import numpy as np
from typing import Optional, Callable, AsyncGenerator
from collections import deque

logger = logging.getLogger(__name__)

class VADService:
    """Voice Activity Detection service using webrtcvad."""
    
    def __init__(self, settings: dict):
        """Initialize VAD service with settings."""
        self.settings = settings
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Most aggressive mode
        
        # Audio processing parameters
        self.frame_duration = settings["VAD_FRAME_DURATION"]
        self.padding_duration = settings["VAD_PADDING_DURATION"]
        self.sample_rate = settings["SAMPLE_RATE"]
        self.channels = settings["CHANNELS"]
        self.chunk_size = settings["CHUNK_SIZE"]
        
        # Voice activity state
        self.is_speaking = False
        self.silence_frames = 0
        self.silence_threshold = settings["SILENCE_THRESHOLD"]
        self.silence_duration = settings["SILENCE_DURATION"]
        self.silence_frames_threshold = int(self.silence_duration * self.sample_rate / self.chunk_size)
        
        # Audio buffer for processing
        self.audio_buffer = deque(maxlen=int(self.padding_duration * self.sample_rate / 1000))
        
    async def process_audio(self, audio_data: bytes) -> bool:
        """
        Process incoming audio data and detect voice activity.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            bool: True if voice activity is detected, False otherwise
        """
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Add to buffer
            self.audio_buffer.extend(audio_array)
            
            # Check if we have enough data for a frame
            if len(self.audio_buffer) < self.chunk_size:
                return self.is_speaking
            
            # Process frame
            frame = np.array(list(self.audio_buffer)[:self.chunk_size], dtype=np.int16)
            frame_bytes = frame.tobytes()
            
            # Check if frame contains speech
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            # Update speaking state
            if is_speech:
                self.is_speaking = True
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.silence_frames >= self.silence_frames_threshold:
                    self.is_speaking = False
            
            return self.is_speaking
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
    
    async def detect_speech(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[tuple[bool, bytes], None]:
        """
        Process audio stream and yield (is_speaking, audio_data) tuples.
        
        Args:
            audio_stream: AsyncGenerator yielding audio data in bytes
            
        Yields:
            tuple[bool, bytes]: (is_speaking, audio_data)
        """
        try:
            async for audio_data in audio_stream:
                is_speaking = await self.process_audio(audio_data)
                yield is_speaking, audio_data
                
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            raise
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.silence_frames = 0
        self.audio_buffer.clear() 