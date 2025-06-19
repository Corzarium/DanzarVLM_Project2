import numpy as np
import logging
from typing import Tuple, Optional
import webrtcvad
import time

class VoiceActivityDetector:
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 20,
                 padding_duration_ms: int = 300,
                 threshold: float = 0.3,
                 trigger_level: int = 2,
                 hold_time: float = 0.3):
        """
        Initialize the Voice Activity Detector.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            frame_duration_ms: Frame duration in milliseconds (default: 20)
            padding_duration_ms: Padding duration in milliseconds (default: 300)
            threshold: Energy threshold for speech detection (default: 0.3)
            trigger_level: Number of consecutive frames needed to trigger speech (default: 2)
            hold_time: Time to hold speech state after last detection (default: 0.3)
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.threshold = threshold
        self.trigger_level = trigger_level
        self.hold_time = hold_time
        
        # Calculate frame size
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        # Initialize state
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.last_speech_time = 0
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        self.logger.info(f"[VAD] Initialized with sample_rate={sample_rate}, "
                        f"frame_duration_ms={frame_duration_ms}, "
                        f"threshold={threshold}, "
                        f"trigger_level={trigger_level}, "
                        f"hold_time={hold_time}")

    def process_audio(self, audio_data: np.ndarray) -> Tuple[bool, bool]:
        """
        Process audio data and detect voice activity.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (is_speaking, speech_ended)
        """
        try:
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32768).astype(np.int16)
            
            # Calculate frame energy
            frame_energy = np.mean(np.abs(audio_data))
            
            # Log frame details
            self.logger.debug(f"[VAD] Frame energy: {frame_energy:.3f}, "
                            f"threshold: {self.threshold}, "
                            f"speech_frames: {self.speech_frames}, "
                            f"silence_frames: {self.silence_frames}")
            
            # Check if frame contains speech
            is_speech = frame_energy > self.threshold
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                self.last_speech_time = time.time()
                
                # Detect speech start
                if not self.is_speaking and self.speech_frames >= self.trigger_level:
                    self.is_speaking = True
                    self.logger.info(f"[VAD] Speech detected: energy={frame_energy:.3f}, "
                                   f"frames={self.speech_frames}")
                    return True, False
                    
            else:
                self.silence_frames += 1
                
                # Check for speech end
                if self.is_speaking:
                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration > self.hold_time:
                        self.is_speaking = False
                        self.speech_frames = 0
                        self.logger.info(f"[VAD] Speech ended: silence_duration={silence_duration:.2f}s")
                        return False, True
            
            return self.is_speaking, False
            
        except Exception as e:
            self.logger.error(f"[VAD] Error processing audio: {e}", exc_info=True)
            return False, False

    def reset(self):
        """Reset the VAD state."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.last_speech_time = 0
        self.logger.debug("[VAD] State reset") 