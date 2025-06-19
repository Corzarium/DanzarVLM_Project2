#!/usr/bin/env python3
"""
Vosk Speech-to-Text Service for DanzarAI
Provides fast, offline, real-time speech recognition with streaming API
"""

import json
import logging
import os
import tempfile
import time
import wave
from typing import Optional, Dict, Any
import numpy as np

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    vosk = None


class VoskSTTService:
    """Fast offline speech-to-text service using Vosk."""
    
    def __init__(self, app_context, model_path: Optional[str] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Model configuration
        self.model_path = model_path or "models/vosk-model-small-en-us-0.15"
        self.model = None
        self.recognizer = None
        
        # Audio settings optimized for real-time processing
        self.sample_rate = 16000  # Vosk works best with 16kHz
        self.chunk_size = 4000    # 250ms chunks for responsive processing
        
        # Performance settings - optimized for distributed Discord setup
        self.confidence_threshold = 0.1  # Much lower for Discord's compressed audio
        self.min_speech_length = 0.2     # Shorter minimum for fragmented audio
        
        self.logger.info("[VoskSTTService] Initializing Vosk STT service...")
        
    def initialize(self) -> bool:
        """Initialize the Vosk model and recognizer."""
        if not VOSK_AVAILABLE:
            self.logger.error("❌ Vosk not available - install with: pip install vosk")
            return False
            
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Vosk model not found at: {self.model_path}")
                self.logger.info("💡 Download model with:")
                self.logger.info("   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                self.logger.info("   unzip vosk-model-small-en-us-0.15.zip -d models/")
                return False
            
            # Load model
            self.logger.info(f"🔧 Loading Vosk model from: {self.model_path}")
            self.model = vosk.Model(self.model_path)
            
            # Create recognizer
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            # Configure recognizer for better performance
            self.recognizer.SetWords(True)  # Enable word-level timestamps
            
            self.logger.info("✅ Vosk STT service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Vosk STT: {e}")
            return False
    
    def transcribe_audio_data(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data using Vosk.
        
        Args:
            audio_data: Audio data as numpy array (any sample rate, will be resampled)
            
        Returns:
            Transcribed text or None if no speech detected
        """
        if not self.model or not self.recognizer:
            self.logger.error("❌ Vosk STT not initialized")
            return None
            
        try:
            # Validate audio data
            if len(audio_data) == 0:
                return None
                
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Detect source sample rate from audio length and duration
            # Discord typically uses 48kHz, virtual audio often 44.1kHz
            estimated_duration = len(audio_data) / 48000  # Try Discord rate first
            if estimated_duration < 0.1 or estimated_duration > 30:
                estimated_duration = len(audio_data) / 44100  # Try virtual audio rate
                original_sample_rate = 44100
            else:
                original_sample_rate = 48000
            
            duration = len(audio_data) / original_sample_rate
            max_volume = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Vosk processing: {duration:.2f}s, max: {max_volume:.4f}, RMS: {rms:.4f}, rate: {original_sample_rate}Hz")
            
            # More lenient quality checks for better sensitivity
            if duration < 0.2:  # Further reduced for Discord fragments
                self.logger.info(f"🔇 Audio too short for Vosk: {duration:.2f}s < 0.2s")
                return None
                
            if max_volume < 0.003:  # Further reduced for Discord compression
                self.logger.info(f"🔇 Audio volume too low for Vosk: {max_volume:.4f}")
                return None
            
            # Better resampling to 16kHz for Vosk
            if original_sample_rate != self.sample_rate:
                # Use proper resampling ratio
                resample_ratio = self.sample_rate / original_sample_rate
                target_length = int(len(audio_data) * resample_ratio)
                
                # Simple but effective resampling
                indices = np.linspace(0, len(audio_data) - 1, target_length)
                audio_16k = np.interp(indices, np.arange(len(audio_data)), audio_data)
            else:
                audio_16k = audio_data.copy()
            
            # Discord audio corruption fixes (based on VB-Audio forum research)
            try:
                # 1. Remove DC offset that Discord often introduces
                audio_16k = audio_16k - np.mean(audio_16k)
                
                # 2. Apply aggressive pre-emphasis to counteract Discord's compression
                pre_emphasis = 0.95  # More aggressive than standard 0.97
                if len(audio_16k) > 1:
                    audio_16k = np.append(audio_16k[0], audio_16k[1:] - pre_emphasis * audio_16k[:-1])
                
                # 3. Apply bandpass filter for speech frequencies (Discord corrupts these)
                try:
                    from scipy import signal
                    # Design bandpass filter for speech (300-3400 Hz at 16kHz)
                    nyquist = self.sample_rate / 2
                    low = 300 / nyquist
                    high = 3400 / nyquist
                    if low < 1.0 and high < 1.0:  # Valid frequency range
                        sos = signal.butter(4, [low, high], btype='band', output='sos')
                        audio_16k = signal.sosfilt(sos, audio_16k)
                        self.logger.debug("🎵 Applied speech bandpass filter")
                except ImportError:
                    self.logger.debug("🔇 scipy not available, skipping bandpass filter")
                except Exception as e:
                    self.logger.debug(f"🔇 Bandpass filter failed: {e}")
                
                # 4. Spectral subtraction for Discord noise reduction
                # Simple approach: reduce very quiet sections more aggressively
                noise_threshold = np.sqrt(np.mean(np.square(audio_16k))) * 0.1
                audio_abs = np.abs(audio_16k)
                noise_mask = audio_abs < noise_threshold
                audio_16k[noise_mask] *= 0.1  # Reduce noise by 90%
                
                # 5. Adaptive gain control for Discord's variable levels
                window_size = int(self.sample_rate * 0.1)  # 100ms windows
                if len(audio_16k) > window_size:
                    for i in range(0, len(audio_16k) - window_size, window_size // 2):
                        window = audio_16k[i:i + window_size]
                        window_rms = np.sqrt(np.mean(np.square(window)))
                        if window_rms > 0.01:  # Only adjust if there's significant signal
                            target_rms = 0.15
                            gain = target_rms / window_rms
                            gain = np.clip(gain, 0.5, 3.0)  # Limit gain range
                            audio_16k[i:i + window_size] = window * gain
                
                # 6. Final normalization optimized for Vosk
                current_max = np.max(np.abs(audio_16k))
                if current_max > 0:
                    # Normalize to 80% to leave headroom
                    audio_16k = audio_16k * (0.8 / current_max)
                
                self.logger.debug(f"🎵 Applied Discord audio corruption fixes")
                
            except Exception as e:
                self.logger.warning(f"🔇 Discord audio fix failed, using basic processing: {e}")
                # Fallback to basic normalization
                if max_volume > 0:
                    normalization_factor = min(0.9 / max_volume, 3.0)
                    audio_16k = audio_16k * normalization_factor
            
            # Convert to 16-bit PCM format for Vosk
            audio_int16 = np.clip(audio_16k * 32767, -32767, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Process with Vosk
            self.logger.info(f"🎯 Processing {len(audio_bytes)} bytes with Vosk...")
            
            # Debug: Save audio sample for analysis (optional)
            if self.logger.level <= logging.DEBUG:
                debug_path = f"debug_audio_{int(time.time())}.wav"
                try:
                    import wave
                    with wave.open(debug_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_bytes)
                    self.logger.debug(f"🔍 Saved debug audio to {debug_path}")
                except Exception as e:
                    self.logger.debug(f"⚠️ Could not save debug audio: {e}")
            
            # Reset recognizer for new audio
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            
            # Process audio in smaller chunks for better Discord fragment handling
            chunk_size = self.chunk_size  # Use smaller chunks for Discord
            results = []
            
            # Process entire audio as one chunk first (better for continuous speech)
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Complete phrase detected
                result = json.loads(self.recognizer.Result())
                if result.get('text', '').strip():
                    results.append(result['text'].strip())
                    self.logger.info(f"🎯 Vosk complete result: '{result['text']}'")
            
            # Then process in chunks for any remaining audio
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                
                if self.recognizer.AcceptWaveform(chunk):
                    # Complete phrase detected
                    result = json.loads(self.recognizer.Result())
                    if result.get('text', '').strip():
                        results.append(result['text'].strip())
                        self.logger.info(f"🎯 Vosk partial result: '{result['text']}'")
            
            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            if final_result.get('text', '').strip():
                results.append(final_result['text'].strip())
                self.logger.info(f"🎯 Vosk final result: '{final_result['text']}'")
            
            # Combine all results
            if results:
                combined_text = ' '.join(results).strip()
                
                # More lenient quality filtering for Discord
                if len(combined_text) < 1:  # Accept even single characters
                    self.logger.info(f"🚫 Vosk result empty: '{combined_text}'")
                    return None
                
                # Check for confidence if available - much more lenient
                confidence = final_result.get('confidence', 1.0)
                if confidence < self.confidence_threshold:
                    self.logger.info(f"🚫 Vosk confidence too low: {confidence:.3f} < {self.confidence_threshold}")
                    # For Discord, still return the result but log the low confidence
                    self.logger.info(f"🎯 Accepting low-confidence result due to Discord compression: '{combined_text}'")
                
                self.logger.info(f"✅ Vosk transcription: '{combined_text}' (confidence: {confidence:.3f})")
                return combined_text
            else:
                self.logger.info("🔇 Vosk detected no speech")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Vosk transcription error: {e}", exc_info=True)
            return None
    
    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio from a file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Read audio file
            with wave.open(audio_file_path, 'rb') as wf:
                # Validate format
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != self.sample_rate:
                    self.logger.warning(f"⚠️ Audio file format not optimal for Vosk: {wf.getnchannels()}ch, {wf.getsampwidth()*8}bit, {wf.getframerate()}Hz")
                
                # Read all frames
                frames = wf.readframes(wf.getnframes())
                
                # Convert to numpy array
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                return self.transcribe_audio_data(audio_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error reading audio file {audio_file_path}: {e}")
            return None
    
    def create_streaming_recognizer(self) -> Optional[Any]:
        """
        Create a streaming recognizer for real-time processing.
        
        Returns:
            Vosk recognizer instance for streaming
        """
        if not self.model:
            return None
            
        try:
            recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            recognizer.SetWords(True)
            return recognizer
        except Exception as e:
            self.logger.error(f"❌ Failed to create streaming recognizer: {e}")
            return None
    
    def process_streaming_chunk(self, recognizer: Any, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Process a streaming audio chunk.
        
        Args:
            recognizer: Vosk recognizer instance
            audio_chunk: Raw audio bytes (16-bit PCM, 16kHz)
            
        Returns:
            Dictionary with 'partial', 'final', and 'text' keys
        """
        try:
            if recognizer.AcceptWaveform(audio_chunk):
                # Complete phrase
                result = json.loads(recognizer.Result())
                return {
                    'partial': False,
                    'final': True,
                    'text': result.get('text', '').strip(),
                    'confidence': result.get('confidence', 1.0)
                }
            else:
                # Partial result
                result = json.loads(recognizer.PartialResult())
                return {
                    'partial': True,
                    'final': False,
                    'text': result.get('partial', '').strip(),
                    'confidence': 1.0  # Partial results don't have confidence
                }
                
        except Exception as e:
            self.logger.error(f"❌ Streaming chunk processing error: {e}")
            return {'partial': False, 'final': False, 'text': '', 'confidence': 0.0}
    
    def get_final_result(self, recognizer: Any) -> str:
        """Get final result from streaming recognizer."""
        try:
            result = json.loads(recognizer.FinalResult())
            return result.get('text', '').strip()
        except Exception as e:
            self.logger.error(f"❌ Error getting final result: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.recognizer = None
        self.logger.info("🧹 Vosk STT service cleaned up") 