#!/usr/bin/env python3
"""
Faster-Whisper Speech-to-Text Service for DanzarAI
Provides ultra-fast, accurate speech recognition with GPU acceleration
Up to 4x faster than OpenAI Whisper with same accuracy
"""

import logging
import os
import tempfile
import time
import wave
from typing import Optional, Dict, Any, Union
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


class FasterWhisperSTTService:
    """Ultra-fast offline speech-to-text service using faster-whisper with CTranslate2."""
    
    def __init__(self, app_context, model_size: str = "large", device: str = "auto"):
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Model configuration - upgraded to large for maximum accuracy
        self.model_size = model_size
        self.device = device
        self.model = None
        
        # Audio settings optimized for real-time processing
        self.sample_rate = 16000  # faster-whisper works best with 16kHz
        
        # Performance settings - optimized for Discord's compressed audio
        self.confidence_threshold = 0.05  # Lower for Discord's compressed audio (was 0.1)
        self.min_speech_length = 0.2     # Minimum speech duration (was 0.3)
        
        # Transcription parameters optimized for higher accuracy
        self.transcribe_params = {
            "language": "en",
            "task": "transcribe",
            "beam_size": 3,  # Increased from 1 for better accuracy
            "best_of": 3,    # Increased from 1 for better accuracy
            "temperature": [0.0, 0.2, 0.4],  # Multiple temperatures for better results
            "condition_on_previous_text": False,  # Prevent context bias
            "initial_prompt": None,  # No prompt bias
            "word_timestamps": False,  # Disable for speed
            "prepend_punctuations": "\"'¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：\")]}、",
            "vad_filter": False,  # DISABLE built-in VAD - use our own speech detection
            "vad_parameters": None,  # Not needed when VAD is disabled
            "compression_ratio_threshold": 2.0,  # Lower threshold for better quality
            "logprob_threshold": -0.5,  # More lenient for better recall
            "no_speech_threshold": 0.8   # Higher threshold to reduce false positives
        }
        
        self.logger.info(f"[FasterWhisperSTTService] Initializing with model: {model_size}, device: {device}")
        
    def initialize(self) -> bool:
        """Initialize the faster-whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            self.logger.error("❌ faster-whisper not available - install with: pip install faster-whisper")
            return False
            
        try:
            # Determine optimal device and compute type
            device, compute_type = self._get_optimal_device_config()
            
            self.logger.info(f"🔧 Loading faster-whisper model '{self.model_size}' on {device} with {compute_type}")
            
            # Load model with optimal settings
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4,  # Optimal for most systems
                num_workers=1   # Single worker for real-time processing
            )
            
            self.logger.info("✅ faster-whisper STT service initialized successfully")
            
            # Test transcription to warm up the model
            self._warmup_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize faster-whisper STT: {e}")
            return False
    
    def _get_optimal_device_config(self) -> tuple[str, str]:
        """Determine optimal device and compute type configuration."""
        if self.device == "auto":
            # Auto-detect best configuration
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"  # Best balance of speed/accuracy on GPU
                    self.logger.info("🚀 Using GPU acceleration with FP16")
                else:
                    device = "cpu"
                    compute_type = "int8"  # Fastest on CPU
                    self.logger.info("🔧 Using CPU with INT8 quantization")
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                self.logger.info("🔧 Using CPU with INT8 quantization (torch not available)")
        else:
            device = self.device
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
        
        return device, compute_type
    
    def _warmup_model(self):
        """Warm up the model with a short audio sample."""
        try:
            # Create a short silence for warmup
            warmup_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Save as WAV
                try:
                    import scipy.io.wavfile as wavfile
                    audio_int16 = (warmup_audio * 32767).astype(np.int16)
                    wavfile.write(temp_file.name, self.sample_rate, audio_int16)
                except ImportError:
                    import wave
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        audio_int16 = (warmup_audio * 32767).astype(np.int16)
                        wav_file.writeframes(audio_int16.tobytes())
                
                temp_path = temp_file.name
            
            # Run warmup transcription
            segments, _ = self.model.transcribe(temp_path, **self.transcribe_params)
            list(segments)  # Force execution
            
            # Cleanup
            os.unlink(temp_path)
            
            self.logger.info("🔥 Model warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model warmup failed (not critical): {e}")
    
    def transcribe_audio_data(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Enhanced transcription with intelligent audio chunking for long segments.
        
        Args:
            audio_data: Audio data as numpy array (16kHz, mono, float32)
            
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            if not self.model:
                self.logger.error("❌ Whisper model not initialized")
                return None
            
            # Basic audio analysis
            audio_duration = len(audio_data) / self.sample_rate
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Audio analysis - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # Spectral analysis for speech detection
            try:
                from scipy import signal
                f, Pxx = signal.welch(audio_data, self.sample_rate, nperseg=1024)
                
                # Calculate spectral centroid (weighted average frequency)
                spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
                
                # Calculate energy in speech frequency range (300-3400 Hz)
                speech_mask = (f >= 300) & (f <= 3400)
                speech_energy = np.sum(Pxx[speech_mask])
                total_energy = np.sum(Pxx)
                speech_ratio = speech_energy / (total_energy + 1e-8)
                
                # Zero crossing rate
                zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
                zcr = len(zero_crossings) / len(audio_data)
                
            except ImportError:
                # Fallback without scipy
                spectral_centroid = 1000  # Assume reasonable value
                speech_ratio = 0.5
                zcr = 0.1
            
            self.logger.info(f"🎵 Spectral analysis - Centroid: {spectral_centroid:.0f}Hz, Speech ratio: {speech_ratio:.3f}, ZCR: {zcr:.4f}")
            
            # ENHANCED quality checks with chunking support
            
            # 1. Volume thresholds
            if audio_max_volume < 0.12:
                self.logger.info(f"🔇 Audio volume too low for speech (max: {audio_max_volume:.4f} < 0.12)")
                return None
            
            if audio_rms < 0.025:
                self.logger.info(f"🔇 Audio RMS too low for speech (RMS: {audio_rms:.4f} < 0.025)")
                return None
            
            # 2. Duration checks with chunking support
            if audio_duration < 0.8:
                self.logger.info(f"🔇 Audio too short for meaningful speech ({audio_duration:.2f}s < 0.8s)")
                return None
            
            # NEW: Handle long audio with intelligent chunking
            if audio_duration > 15.0:  # Increased from 10.0s to 15.0s
                self.logger.info(f"🔄 Audio segment long ({audio_duration:.2f}s), using intelligent chunking")
                return self._transcribe_with_chunking(audio_data)
            
            # 3. Spectral analysis for speech detection
            if spectral_centroid < 200 or spectral_centroid > 4000:
                self.logger.info(f"🔇 Spectral centroid outside speech range ({spectral_centroid:.0f}Hz)")
                return None
            
            if speech_ratio < 0.3:
                self.logger.info(f"🔇 Insufficient energy in speech frequencies ({speech_ratio:.3f} < 0.3)")
                return None
            
            # 4. Zero crossing rate
            if zcr < 0.01 or zcr > 0.3:
                self.logger.info(f"🔇 Zero crossing rate outside speech range ({zcr:.4f})")
                return None
            
            # 5. Signal-to-noise ratio estimation
            snr_proxy = audio_max_volume / (audio_rms + 1e-8)
            if snr_proxy < 3.0:
                self.logger.info(f"🔇 Poor signal-to-noise ratio ({snr_proxy:.2f} < 3.0)")
                return None
            
            self.logger.info(f"✅ Audio passed quality checks - proceeding with transcription")
            
            # Use the working file-based transcription approach
            return self._transcribe_single_segment(audio_data)
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None

    def _transcribe_with_chunking(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe long audio by splitting into intelligent chunks.
        
        Args:
            audio_data: Long audio segment to chunk and transcribe
            
        Returns:
            Combined transcription from all chunks
        """
        try:
            audio_duration = len(audio_data) / self.sample_rate
            self.logger.info(f"🔄 Chunking audio segment: {audio_duration:.2f}s")
            
            # Split into overlapping chunks for better continuity
            chunk_duration = 10.0  # 10 second chunks
            overlap_duration = 2.0  # 2 second overlap
            
            chunk_samples = int(chunk_duration * self.sample_rate)
            overlap_samples = int(overlap_duration * self.sample_rate)
            step_samples = chunk_samples - overlap_samples
            
            chunks = []
            transcriptions = []
            
            # Create chunks with overlap
            start = 0
            chunk_num = 0
            while start < len(audio_data):
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                
                # Only process chunks that are long enough
                chunk_duration_actual = len(chunk) / self.sample_rate
                if chunk_duration_actual >= 1.0:  # At least 1 second
                    chunks.append((chunk_num, chunk, start / self.sample_rate))
                    chunk_num += 1
                
                start += step_samples
                if end >= len(audio_data):
                    break
            
            self.logger.info(f"🔄 Created {len(chunks)} chunks for processing")
            
            # Process each chunk
            for chunk_num, chunk_audio, start_time in chunks:
                try:
                    self.logger.info(f"🔄 Processing chunk {chunk_num + 1}/{len(chunks)} (start: {start_time:.1f}s)")
                    
                    # Transcribe this chunk
                    chunk_transcription = self._transcribe_single_segment(chunk_audio)
                    
                    if chunk_transcription and len(chunk_transcription.strip()) > 0:
                        # Clean up the transcription
                        chunk_transcription = chunk_transcription.strip()
                        
                        # Avoid duplicate phrases from overlapping chunks
                        if not self._is_duplicate_content(chunk_transcription, transcriptions):
                            transcriptions.append(chunk_transcription)
                            self.logger.info(f"✅ Chunk {chunk_num + 1} transcription: '{chunk_transcription}'")
                        else:
                            self.logger.info(f"🔄 Chunk {chunk_num + 1} skipped (duplicate content)")
                    else:
                        self.logger.info(f"🔇 Chunk {chunk_num + 1} produced no transcription")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing chunk {chunk_num + 1}: {e}")
                    continue
            
            # Combine transcriptions intelligently
            if transcriptions:
                # Join with spaces and clean up
                combined = " ".join(transcriptions)
                combined = self._clean_combined_transcription(combined)
                
                self.logger.info(f"✅ Combined chunked transcription: '{combined}'")
                return combined
            else:
                self.logger.info("🔇 No valid transcriptions from any chunks")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in chunked transcription: {e}")
            return None

    def _transcribe_single_segment(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe a single audio segment using file-based approach.
        
        Args:
            audio_data: Audio segment to transcribe
            
        Returns:
            Transcription text or None
        """
        try:
            # Use the working file-based transcription approach
            self.logger.info("🎯 Using working file-based transcription...")
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Convert to 16-bit PCM WAV
            audio_int16 = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
            
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_file_path, self.sample_rate, audio_int16)
            
            self.logger.info(f"🎵 Transcribing file: {temp_file_path}")
            
            # Use simple, working parameters with hallucination suppression
            simple_params = {
                "language": "en",
                "task": "transcribe",
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "word_timestamps": False,
                "vad_filter": False,
                "suppress_tokens": [-1]
            }
            
            # Transcribe with simple, working parameters
            start_time = time.time()
            segments, info = self.model.transcribe(
                temp_file_path,
                **simple_params
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Collect all segments
            transcription_parts = []
            for segment in segments:
                if segment.text.strip():
                    transcription_parts.append(segment.text.strip())
            
            processing_time = time.time() - start_time
            
            if transcription_parts:
                full_transcription = " ".join(transcription_parts)
                
                # Check for hallucinations
                audio_duration = len(audio_data) / self.sample_rate
                audio_rms = np.sqrt(np.mean(np.square(audio_data)))
                
                if self._is_likely_hallucination(full_transcription, audio_duration, audio_rms):
                    self.logger.info(f"🚫 Filtered out likely hallucination: '{full_transcription}'")
                    return None
                
                self.logger.info(f"✅ File transcription: '{full_transcription}'")
                self.logger.info(f"📊 Language: {info.language} (confidence: {info.language_probability:.3f})")
                return full_transcription
            else:
                self.logger.info("🔇 No speech detected in file")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Single segment transcription error: {e}")
            return None

    def _is_duplicate_content(self, new_transcription: str, existing_transcriptions: list) -> bool:
        """
        Check if new transcription is duplicate content from overlapping chunks.
        
        Args:
            new_transcription: New transcription to check
            existing_transcriptions: List of existing transcriptions
            
        Returns:
            True if content appears to be duplicate
        """
        if not existing_transcriptions:
            return False
        
        new_words = set(new_transcription.lower().split())
        
        # Check against recent transcriptions (last 2)
        for existing in existing_transcriptions[-2:]:
            existing_words = set(existing.lower().split())
            
            # Calculate word overlap
            if len(new_words) > 0 and len(existing_words) > 0:
                overlap = len(new_words.intersection(existing_words))
                overlap_ratio = overlap / min(len(new_words), len(existing_words))
                
                # If more than 70% word overlap, consider duplicate
                if overlap_ratio > 0.7:
                    return True
        
        return False

    def _clean_combined_transcription(self, combined: str) -> str:
        """
        Clean up combined transcription from multiple chunks.
        
        Args:
            combined: Raw combined transcription
            
        Returns:
            Cleaned transcription
        """
        # Remove extra whitespace
        combined = " ".join(combined.split())
        
        # Remove common transcription artifacts
        artifacts = [
            " . ", " , ", " ? ", " ! ",
            "  ", "   "  # Multiple spaces
        ]
        
        for artifact in artifacts:
            combined = combined.replace(artifact, " ")
        
        # Capitalize first letter
        if combined:
            combined = combined[0].upper() + combined[1:] if len(combined) > 1 else combined.upper()
        
        return combined.strip()

    def _preprocess_discord_audio(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """Enhanced preprocessing specifically for Discord's compressed audio."""
        try:
            # Resample to 16kHz if needed
            if original_sample_rate != self.sample_rate:
                resample_ratio = self.sample_rate / original_sample_rate
                target_length = int(len(audio_data) * resample_ratio)
                indices = np.linspace(0, len(audio_data) - 1, target_length)
                audio_16k = np.interp(indices, np.arange(len(audio_data)), audio_data)
            else:
                audio_16k = audio_data.copy()
            
            # Discord audio corruption fixes
            try:
                from scipy import signal
                
                # 1. Remove DC offset
                audio_16k = audio_16k - np.mean(audio_16k)
                
                # 2. Pre-emphasis for Discord compression
                pre_emphasis = 0.97
                if len(audio_16k) > 1:
                    audio_16k = np.append(audio_16k[0], audio_16k[1:] - pre_emphasis * audio_16k[:-1])
                
                # 3. Bandpass filter for speech (300-3400 Hz)
                nyquist = self.sample_rate / 2
                low = 300 / nyquist
                high = 3400 / nyquist
                if 0 < low < 1 and 0 < high < 1:
                    sos = signal.butter(4, [low, high], btype='band', output='sos')
                    audio_16k = signal.sosfilt(sos, audio_16k)
                
                # 4. Noise reduction
                noise_threshold = np.sqrt(np.mean(np.square(audio_16k))) * 0.1
                noise_mask = np.abs(audio_16k) < noise_threshold
                audio_16k[noise_mask] *= 0.2  # Reduce noise by 80%
                
            except ImportError:
                # Basic processing without scipy
                audio_16k = audio_16k - np.mean(audio_16k)
            
            # Final normalization
            max_val = np.max(np.abs(audio_16k))
            if max_val > 0:
                audio_16k = audio_16k * (0.8 / max_val)
            
            return audio_16k.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Audio preprocessing failed: {e}")
            return audio_data.astype(np.float32)
    
    def _is_likely_hallucination(self, text: str, audio_duration: float, audio_rms: float) -> bool:
        """
        Enhanced hallucination detection based on Calm-Whisper research and Discord audio patterns.
        
        Args:
            text: Transcribed text to analyze
            audio_duration: Duration of audio in seconds
            audio_rms: RMS level of audio
            
        Returns:
            True if text is likely a hallucination
        """
        text_lower = text.lower().strip()
        
        # 1. Common Whisper hallucinations (based on research and observations) - LESS AGGRESSIVE
        common_hallucinations = [
            "thank you",
            "thank you.",
            "thanks for watching",
            "thanks for watching.",
            "subscribe",
            "subscribe.",
            "like and subscribe",
            "like and subscribe.",
            "music",
            "music.",
            "[music]",
            "(music)",
            "♪",
            "♫"
        ]
        
        # Check for exact matches
        if text_lower in common_hallucinations:
            self.logger.info(f"🚫 Exact hallucination match: '{text}'")
            return True
        
        # 2. Very short responses (likely noise interpretation)
        if len(text_lower) <= 3:
            self.logger.info(f"🚫 Text too short: '{text}'")
            return True
        
        # 3. Repetitive patterns
        words = text_lower.split()
        if len(words) >= 2:
            # Check for word repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 50% of the time
            max_word_count = max(word_counts.values()) if word_counts else 0
            repetition_ratio = max_word_count / len(words) if words else 0
            
            if repetition_ratio > 0.5:
                most_repeated_word = max(word_counts.keys(), key=lambda k: word_counts[k])
                self.logger.info(f"🚫 Repetitive pattern: '{most_repeated_word}' appears {max_word_count}/{len(words)} times")
                return True
        
        # 4. Character-level repetition
        if len(text) > 5:
            char_counts = {}
            for char in text_lower:
                if char.isalpha():
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            if char_counts:
                max_char_count = max(char_counts.values())
                total_chars = sum(char_counts.values())
                char_repetition_ratio = max_char_count / total_chars if total_chars > 0 else 0
                
                if char_repetition_ratio > 0.6:  # 60% same character
                    self.logger.info(f"🚫 Character repetition detected: {char_repetition_ratio:.1%}")
                    return True
        
        # 5. Audio-based hallucination detection
        # Very quiet audio producing text is suspicious
        if audio_rms < 0.01 and len(text) > 5:
            self.logger.info(f"🚫 Quiet audio ({audio_rms:.4f}) producing long text: '{text}'")
            return True
        
        # Very short audio producing long text is suspicious
        if audio_duration < 1.0 and len(text) > 20:
            self.logger.info(f"🚫 Short audio ({audio_duration:.2f}s) producing long text: '{text}'")
            return True
        
        # 6. Specific Discord audio hallucination patterns
        discord_patterns = [
            "sound of the",
            "the sound of",
            "human speech",
            "not a human",
            "clear human speech",
            "speech is not"
        ]
        
        for pattern in discord_patterns:
            if pattern in text_lower:
                self.logger.info(f"🚫 Discord audio pattern: '{pattern}' in '{text}'")
                return True
        
        # 7. Punctuation-only or mostly punctuation
        non_alpha_chars = sum(1 for c in text if not c.isalpha() and not c.isspace())
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        if non_alpha_chars > alpha_chars and len(text) > 3:
            self.logger.info(f"🚫 Mostly punctuation: '{text}'")
            return True
        
        # If we get here, it's probably legitimate speech
        return False
    
    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio from a file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.model:
            self.logger.error("❌ faster-whisper STT not initialized")
            return None
            
        try:
            self.logger.info(f"🎵 Transcribing file: {audio_file_path}")
            
            segments, info = self.model.transcribe(audio_file_path, **self.transcribe_params)
            
            # Collect all segments
            transcription_parts = []
            for segment in segments:
                if segment.text.strip():
                    transcription_parts.append(segment.text.strip())
            
            if transcription_parts:
                result = ' '.join(transcription_parts).strip()
                self.logger.info(f"✅ File transcription: '{result}'")
                return result
            else:
                self.logger.info("🔇 No speech detected in file")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error transcribing file {audio_file_path}: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.logger.info("🧹 faster-whisper STT service cleaned up") 