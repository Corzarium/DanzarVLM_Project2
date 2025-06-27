# services/audio_service.py
import time
import numpy as np
import requests
import os
import collections # For deque
from typing import Optional, List, Dict # Added List and Dict for clarity
import io # For in-memory audio buffer for the beep
import logging
import queue
import tensorflow as tf
from collections import deque

# --- Real Implementations (Ensure these are installed) ---
from openwakeword.model import Model as OWWModel
import whisper # openai-whisper
from pydub import AudioSegment # Make sure this is imported
from pydub.generators import Sine # For generating beep sound
import torch # For whisper fp16

from .tts_service_smart import SmartTTSService # Smart TTS service for multiple providers

# from ..DanzarVLM import AppContext # For type hinting if AppContext is in root

class AudioService:
    def __init__(self, app_context): # app_context: AppContext
        self.ctx = app_context  # Store as self.ctx to match other services
        self.logger = app_context.logger
        self.logger.info("[AudioService] Instance created.")
        self.logger.info("[AudioService] Initializing audio systems (TensorFlow Lite Wake Word, STT)...")
        
        # Initialize components
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        self.stt_model = None
        self.tts_service = SmartTTSService(app_context)
        
        # Audio processing buffers
        self.raw_discord_audio_buffer = bytearray()
        self.wake_word_audio_buffer = deque()

        self.is_capturing_for_stt = False
        self.stt_audio_capture_chunks: List[np.ndarray] = []
        self.stt_user_speaking_id: Optional[int] = None
        self.stt_consecutive_silent_oww_chunks = 0
        self.stt_vad_grace_period_oww_chunks = 0

        # Update configuration BEFORE initializing audio systems
        self._update_derived_constants_from_config()
        
        # Now initialize audio systems with proper configuration
        self.initialize_audio_systems()


    def _update_derived_constants_from_config(self):
        """Update derived constants from global settings and active profile"""
        gs = self.ctx.global_settings
        profile = self.ctx.active_profile
        
        # Audio processing constants
        self.TARGET_SR = gs.get("AUDIO_TARGET_SAMPLE_RATE", 16000)
        self.RMS_VAD_THRESHOLD_VALUE = gs.get("RMS_VAD_THRESHOLD", 100)
        
        # Wake word constants
        self.WAKE_WORD_EXPECTED_SAMPLES = 1536  # 16 * 96 for TensorFlow Lite model
        self.WAKE_WORD_THRESHOLD = float(gs.get("OWW_THRESHOLD", 0.01))
        
        # Get wake word model path and name
        oww_path = profile.oww_model_path_override or gs.get("OWW_CUSTOM_MODEL_PATH")
        if oww_path:
            default_oww_name = os.path.splitext(os.path.basename(oww_path))[0]
            self.WAKE_WORD_MODEL_PATH = oww_path
            self.WAKE_WORD_MODEL_NAME = profile.oww_model_name_override or gs.get("OWW_CUSTOM_MODEL_NAME", default_oww_name)
        else:
            self.WAKE_WORD_MODEL_PATH = None
            self.WAKE_WORD_MODEL_NAME = profile.oww_model_name_override or gs.get("OWW_CUSTOM_MODEL_NAME", "wakeword")
        
        # STT constants
        self.STT_MAX_BUFFER_SAMPLES = int(self.TARGET_SR * gs.get("STT_MAX_AUDIO_BUFFER_SECONDS", 15))
        self.STT_MIN_SPEECH_DURATION_SAMPLES = int(self.TARGET_SR * gs.get("STT_MIN_SPEECH_DURATION_SAMPLES_FACTOR", 0.2))
        self.STT_SILENCE_CHUNKS_TO_END = gs.get("STT_SILENCE_FRAMES_TO_END_SPEECH", 50)
        self.STT_GRACE_PERIOD_CHUNKS = int(self.TARGET_SR * gs.get("STT_VAD_GRACE_PERIOD_DURATION_S", 0.85) / 1280)
        
        self.logger.info(
            f"[AudioService] Audio config updated: "
            f"TargetSR={self.TARGET_SR}, WakeWordSamples={self.WAKE_WORD_EXPECTED_SAMPLES}, "
            f"RMSThresh={self.RMS_VAD_THRESHOLD_VALUE}, WakeWordModel='{self.WAKE_WORD_MODEL_NAME}', "
            f"WakeWordThresh={self.WAKE_WORD_THRESHOLD}, STTGraceChunks={self.STT_GRACE_PERIOD_CHUNKS}"
        )

    def initialize_audio_systems(self):
        """Initialize TensorFlow Lite wake word model and Whisper STT"""
        try:
            # Initialize wake word detection with TensorFlow Lite
            if self.WAKE_WORD_MODEL_PATH:
                self.logger.info(f"[AudioService] Loading TensorFlow Lite wake word model: {self.WAKE_WORD_MODEL_PATH}")
                try:
                    self.tflite_interpreter = tf.lite.Interpreter(model_path=self.WAKE_WORD_MODEL_PATH)
                    self.tflite_interpreter.allocate_tensors()
                    
                    self.tflite_input_details = self.tflite_interpreter.get_input_details()
                    self.tflite_output_details = self.tflite_interpreter.get_output_details()
                    
                    input_shape = self.tflite_input_details[0]['shape']
                    expected_samples = input_shape[1] * input_shape[2]
                    
                    if expected_samples != self.WAKE_WORD_EXPECTED_SAMPLES:
                        self.logger.warning(f"[AudioService] Model expects {expected_samples} samples, but configured for {self.WAKE_WORD_EXPECTED_SAMPLES}")
                        self.WAKE_WORD_EXPECTED_SAMPLES = expected_samples
                    
                    # Initialize audio buffer for wake word detection
                    self.wake_word_audio_buffer = deque(maxlen=self.WAKE_WORD_EXPECTED_SAMPLES * 2)
                    
                    self.logger.info(f"[AudioService] TensorFlow Lite wake word model loaded successfully!")
                    self.logger.info(f"[AudioService] Model input shape: {input_shape}, expected samples: {expected_samples}")
                    
                except Exception as e:
                    self.logger.error(f"[AudioService] Failed to load TensorFlow Lite model '{self.WAKE_WORD_MODEL_PATH}': {e}")
                    self.tflite_interpreter = None
            
            # Initialize Whisper STT
            whisper_model = self.ctx.global_settings.get("WHISPER_MODEL_SIZE", "large")
            self.logger.info(f"[AudioService] Loading Whisper STT model: {whisper_model}")
            try:
                self.stt_model = whisper.load_model(whisper_model)
                self.logger.info(f"[AudioService] Whisper STT model '{whisper_model}' loaded successfully.")
            except Exception as e:
                self.logger.error(f"[AudioService] Failed to load Whisper model: {e}")
                
        except Exception as e:
            self.logger.error(f"[AudioService] Audio systems initialization error: {e}")
            
        self.logger.info("[AudioService] Audio systems initialization complete.")

    def _resample_discord_audio_chunk(self, pcm_data_48k_stereo: bytes) -> Optional[np.ndarray]:
        # ... (this method remains the same) ...
        if not pcm_data_48k_stereo: return None
        try:
            discord_segment = AudioSegment(data=pcm_data_48k_stereo, sample_width=2, frame_rate=48000, channels=2)
            mono_segment = discord_segment.set_channels(1)
            resampled_segment = mono_segment.set_frame_rate(self.TARGET_SR)
            if resampled_segment and resampled_segment.raw_data:
                return np.frombuffer(resampled_segment.raw_data, dtype=np.int16)
            else:
                self.logger.warning("resampled_segment or resampled_segment.raw_data is None, returning empty array")
                return np.array([])
        except ImportError: # Should not happen if pydub is installed for beep
            self.logger.error("[AudioService] ERROR: pydub library not found for resampling.", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"[AudioService] Error during audio resampling: {e}", exc_info=True)
            return None

    def _generate_beep_audio(self) -> Optional[bytes]:
        """Generates a short beep sound as WAV bytes."""
        gs = self.ctx.global_settings
        if not gs.get("PLAY_WAKE_WORD_BEEP", False):
            return None

        frequency = gs.get("WAKE_WORD_BEEP_FREQUENCY_HZ", 1000)
        duration_ms = gs.get("WAKE_WORD_BEEP_DURATION_MS", 150)
        volume_dbfs = gs.get("WAKE_WORD_BEEP_VOLUME_DBFS", -20.0) # Use float for dBFS

        try:
            self.logger.debug(f"[AudioService] Generating beep: {frequency}Hz, {duration_ms}ms, {volume_dbfs}dBFS")
            # Generate a sine wave
            beep_segment = Sine(frequency).to_audio_segment(duration=duration_ms)
            # Adjust volume (apply_gain uses dBFS, so negative values reduce volume)
            beep_segment = beep_segment.apply_gain(volume_dbfs)
            
            # Export to WAV format in an in-memory buffer
            # FFmpegPCMAudio in Discord bot can handle WAV from BytesIO
            buffer = io.BytesIO()
            beep_segment.export(buffer, format="wav")
            self.logger.debug(f"[AudioService] Beep generated successfully ({len(buffer.getvalue())} bytes).")
            return buffer.getvalue()
        except NameError: # If Sine or AudioSegment from pydub is not imported
            self.logger.error("[AudioService] Failed to generate beep: pydub components (Sine, AudioSegment) not imported correctly.", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"[AudioService] Failed to generate beep audio: {e}", exc_info=True)
            return None

    def process_discord_audio_chunk(self, raw_discord_pcm: bytes, user_id: int):
        """Process Discord audio chunk for wake word detection and STT"""
        # Enhanced entry logging with comprehensive status
        self.logger.debug(f"[AudioService.DISCORD_CHUNK] User {user_id}: {len(raw_discord_pcm)} bytes | OWW={self.tflite_interpreter is not None} | STT={self.stt_model is not None} | Capturing={self.is_capturing_for_stt} | STTUser={self.stt_user_speaking_id}")
        
        if not self.tflite_interpreter:
            self.logger.warning("[AudioService] Wake word model not loaded - skipping audio processing")
            return

        self.raw_discord_audio_buffer.extend(raw_discord_pcm)
        buffer_size_before = len(self.raw_discord_audio_buffer) - len(raw_discord_pcm)
        
        # Log buffer activity periodically
        if hasattr(self, '_last_buffer_log_time'):
            if time.time() - self._last_buffer_log_time >= 15:  # Every 15 seconds
                self.logger.info(f"[AudioService.BUFFER_STATUS] Raw buffer: {len(self.raw_discord_audio_buffer)} bytes | Wake word buffer: {len(self.wake_word_audio_buffer)} samples | User {user_id}")
                self._last_buffer_log_time = time.time()
        else:
            self._last_buffer_log_time = time.time()

        DISCORD_PACKET_DURATION_MS = 20
        BYTES_PER_DISCORD_PACKET = int(48000 * (DISCORD_PACKET_DURATION_MS / 1000) * 2 * 2)

        packets_to_process = len(self.raw_discord_audio_buffer) // BYTES_PER_DISCORD_PACKET
        if packets_to_process > 0:
            self.logger.debug(f"[AudioService.PACKET_PROCESSING] Processing {packets_to_process} Discord packets from user {user_id}")

        while len(self.raw_discord_audio_buffer) >= BYTES_PER_DISCORD_PACKET:
            chunk_48k_stereo = bytes(self.raw_discord_audio_buffer[:BYTES_PER_DISCORD_PACKET])
            self.raw_discord_audio_buffer = self.raw_discord_audio_buffer[BYTES_PER_DISCORD_PACKET:]

            resampled_16k_mono_np = self._resample_discord_audio_chunk(chunk_48k_stereo)
            if resampled_16k_mono_np is None or resampled_16k_mono_np.size == 0:
                continue

            self.wake_word_audio_buffer.extend(resampled_16k_mono_np)

            while len(self.wake_word_audio_buffer) >= self.WAKE_WORD_EXPECTED_SAMPLES:
                # Convert deque to numpy array for processing
                chunk_samples = list(self.wake_word_audio_buffer)[:self.WAKE_WORD_EXPECTED_SAMPLES]
                oww_chunk_to_process = np.array(chunk_samples, dtype=np.int16)
                
                # Remove processed samples from buffer
                for _ in range(self.WAKE_WORD_EXPECTED_SAMPLES):
                    self.wake_word_audio_buffer.popleft()

                if not self.is_capturing_for_stt:
                    try:
                        score = self._predict_wake_word(oww_chunk_to_process)
                        
                        # Log wake word prediction results (periodic for non-detection, immediate for detection)
                        if hasattr(self, '_last_oww_log_time'):
                            time_since_last_log = time.time() - self._last_oww_log_time
                            should_log_prediction = time_since_last_log >= 30  # Every 30 seconds for normal activity
                        else:
                            should_log_prediction = True
                            self._last_oww_log_time = time.time()
                            
                        if score >= self.WAKE_WORD_THRESHOLD:
                            self.logger.info(f"🎯 WAKE WORD '{self.WAKE_WORD_MODEL_NAME}' DETECTED! User: {user_id}, Score: {score:.4f} (threshold: {self.WAKE_WORD_THRESHOLD})")
                            self._last_oww_log_time = time.time()  # Reset log timer on detection
                        elif should_log_prediction:
                            self.logger.info(f"[AudioService.OWW_MONITORING] User {user_id}: Score={score:.4f}, Threshold={self.WAKE_WORD_THRESHOLD}, Samples={len(oww_chunk_to_process)}")
                            self._last_oww_log_time = time.time()

                        if score >= self.WAKE_WORD_THRESHOLD:
                            
                            # --- PLAY BEEP ON WAKE WORD ---
                            if self.ctx.global_settings.get("PLAY_WAKE_WORD_BEEP", False):
                                beep_audio_bytes = self._generate_beep_audio()
                                if beep_audio_bytes:
                                    try:
                                        self.ctx.tts_queue.put_nowait(beep_audio_bytes)
                                        self.logger.debug("[AudioService] Wake word confirmation beep queued for playback.")
                                    except queue.Full: # Make sure 'queue' module is imported
                                        self.logger.warning("[AudioService] TTS queue full, wake word beep dropped.")
                            # --- END BEEP ---

                            self.is_capturing_for_stt = True
                            self.stt_user_speaking_id = user_id
                            # ... (rest of the STT capture initiation) ...
                            self.stt_audio_capture_chunks.clear()
                            self.stt_consecutive_silent_oww_chunks = 0
                            self.stt_vad_grace_period_oww_chunks = self.STT_GRACE_PERIOD_CHUNKS

                            self.ctx.is_in_conversation.set()
                            self.ctx.last_interaction_time = time.time()
                            self.stt_audio_capture_chunks.append(oww_chunk_to_process) # Add the wake word chunk itself
                            continue # Skip further processing of this chunk for STT VAD if it was a wake word trigger
                    except Exception as e_oww:
                        self.logger.error(f"[AudioService] Error during OWW prediction: {e_oww}", exc_info=True)
                        continue 

                # STT Capture Logic (VAD) - remains the same
                if self.is_capturing_for_stt:
                    if self.stt_user_speaking_id != user_id: # Only process audio from the user who triggered wake word
                        continue

                    self.stt_audio_capture_chunks.append(oww_chunk_to_process)
                    self.ctx.last_interaction_time = time.time() # Keep updating for conversation timeout

                    # Simple RMS VAD on the OWW chunk
                    rms_energy = np.sqrt(np.mean(oww_chunk_to_process.astype(np.float32)**2))
                    is_speech_in_oww_chunk = rms_energy > self.RMS_VAD_THRESHOLD_VALUE
                    # self.logger.debug(f"[AudioService_VAD] RMS: {rms_energy:.2f}, Speech: {is_speech_in_oww_chunk}")


                    if self.stt_vad_grace_period_oww_chunks > 0:
                        self.stt_vad_grace_period_oww_chunks -= 1
                        self.stt_consecutive_silent_oww_chunks = 0 # Reset silence counter during grace period
                        # self.logger.debug(f"[AudioService_VAD] In grace period ({self.stt_vad_grace_period_oww_chunks} chunks left).")
                    elif not is_speech_in_oww_chunk:
                        self.stt_consecutive_silent_oww_chunks += 1
                        # self.logger.debug(f"[AudioService_VAD] Silent chunk detected ({self.stt_consecutive_silent_oww_chunks}/{self.STT_SILENCE_CHUNKS_TO_END}).")
                    else: # Speech detected
                        self.stt_consecutive_silent_oww_chunks = 0 # Reset silence counter
                        # self.logger.debug(f"[AudioService_VAD] Speech chunk detected, silence counter reset.")


                    current_stt_duration_samples = sum(len(arr) for arr in self.stt_audio_capture_chunks)

                    end_stt_capture = False
                    if self.stt_consecutive_silent_oww_chunks >= self.STT_SILENCE_CHUNKS_TO_END:
                        self.logger.info("[AudioService] End of speech detected by VAD (prolonged silence).")
                        end_stt_capture = True
                    elif current_stt_duration_samples >= self.STT_MAX_BUFFER_SAMPLES:
                        self.logger.info("[AudioService] Max STT audio buffer duration reached.")
                        end_stt_capture = True

                    if end_stt_capture:
                        self.is_capturing_for_stt = False # Stop capturing
                        full_stt_audio_np = np.concatenate(self.stt_audio_capture_chunks) if self.stt_audio_capture_chunks else np.array([], dtype=np.int16)
                        self.stt_audio_capture_chunks.clear()
                        speaking_user_id = self.stt_user_speaking_id # Store before resetting

                        # Reset STT state for next interaction
                        self.stt_user_speaking_id = None
                        self.stt_consecutive_silent_oww_chunks = 0
                        self.stt_vad_grace_period_oww_chunks = 0 # Reset grace period

                        if len(full_stt_audio_np) >= self.STT_MIN_SPEECH_DURATION_SAMPLES:
                            self.logger.info(f"[AudioService] STT audio captured ({len(full_stt_audio_np)/self.TARGET_SR:.2f}s for user {speaking_user_id}). Transcribing...")
                            transcribed_text = self.transcribe_audio(full_stt_audio_np)

                            if transcribed_text:
                                self.logger.info(f"[AudioService] Transcription for user {speaking_user_id}: '{transcribed_text}'")
                                if self.ctx.llm_service_instance:
                                     # LLMService will handle clearing the conversation flag after processing
                                     self.ctx.llm_service_instance.handle_user_text_query_sync(transcribed_text, str(speaking_user_id))
                                else:
                                    self.logger.error("[AudioService] llm_service_instance not available in app_context! Cannot send transcription.")
                                    # Manually clear if LLM service is missing, to prevent getting stuck
                                    if self.ctx.is_in_conversation.is_set():
                                        self.ctx.is_in_conversation.clear()
                                        self.logger.info("[AudioService] Conversation flag cleared (LLM service missing).")
                            else: # Empty or failed transcription
                                self.logger.info("[AudioService] Transcription resulted in empty text or failed. Ending conversation state.")
                                if self.ctx.is_in_conversation.is_set():
                                    self.ctx.is_in_conversation.clear()
                                    self.logger.info("[AudioService] Conversation flag cleared (empty/failed STT).")
                        else: # Audio too short
                            self.logger.info(f"[AudioService] Captured STT audio too short ({len(full_stt_audio_np)/self.TARGET_SR:.2f}s). Ignoring. Ending conversation state.")
                            if self.ctx.is_in_conversation.is_set():
                                self.ctx.is_in_conversation.clear()
                                self.logger.info("[AudioService] Conversation flag cleared (STT audio too short).")

    def transcribe_audio(self, audio_data_np: np.ndarray) -> Optional[str]:
        # ... (this method remains the same) ...
        if not self.stt_model:
            self.logger.warning("[AudioService] STT model (Whisper) not loaded, cannot transcribe.")
            return None
        if audio_data_np.size == 0:
            self.logger.debug("[AudioService] Empty audio data passed to transcribe_audio.")
            return None

        self.logger.info(f"[AudioService] Transcribing audio data of shape {audio_data_np.shape} using Whisper...")
        try:
            audio_f32 = audio_data_np.astype(np.float32) / 32768.0

            use_fp16 = torch.cuda.is_available()
            if use_fp16:
                self.logger.debug("[AudioService] CUDA available, using fp16 for Whisper.")
            else:
                self.logger.debug("[AudioService] CUDA not available, using fp32 for Whisper.")

            result = self.stt_model.transcribe(audio_f32, language="en", fp16=use_fp16)

            if result and "text" in result:
                if isinstance(result["text"], str):
                    transcribed_text = result["text"].strip()
                else:
                    self.logger.error(f"[AudioService] Transcription 'text' is not a string: {type(result['text'])}. Raw output: {result}")
                    return None
            else:
                self.logger.error(f"[AudioService] Transcription result missing 'text' key. Raw output: {result}")
                return None

            self.logger.debug(f"[AudioService] Whisper raw transcription: '{transcribed_text}'")
            return transcribed_text if transcribed_text else None
        except Exception as e:
            self.logger.error(f"[AudioService] ERROR: Whisper transcription failed: {e}", exc_info=True)
            return None

    def fetch_tts_audio(self, text: str) -> Optional[bytes]:
        """Get TTS audio for text"""
        try:
            return self.tts_service.generate_audio(text)
        except Exception as e:
            self.logger.error(f"[AudioService] TTS failed: {e}")
            return None

    def reinitialize_for_profile(self, new_profile):
        # ... (this method remains the same) ...
        self.logger.info(f"[AudioService] Re-initializing audio systems for new profile: {new_profile.game_name}")
        # Cancel any ongoing STT capture
        if self.is_capturing_for_stt:
            self.logger.info("[AudioService] Ongoing STT capture cancelled due to profile change.")
            self.is_capturing_for_stt = False
            self.stt_audio_capture_chunks.clear()
            self.stt_user_speaking_id = None
            # Ensure conversation state is reset if it was active due to STT
            if self.ctx.is_in_conversation.is_set(): # Check if the event is actually set
                self.ctx.is_in_conversation.clear()
                self.logger.info("[AudioService] Conversation flag cleared due to profile change during STT.")
        
        # Re-initialize OWW and STT models (which also updates derived constants)
        self.initialize_audio_systems()

    def _predict_wake_word(self, audio_samples: np.ndarray) -> float:
        """Run wake word detection using TensorFlow Lite model"""
        if (not self.tflite_interpreter or not self.tflite_input_details or 
            not self.tflite_output_details or len(audio_samples) != self.WAKE_WORD_EXPECTED_SAMPLES):
            return 0.0
        
        try:
            # Normalize audio to [-1, 1]
            normalized_audio = audio_samples.astype(np.float32) / 32768.0
            
            # Reshape to match model input [1, 16, 96]
            input_shape = self.tflite_input_details[0]['shape']
            input_data = normalized_audio.reshape(input_shape)
            
            # Set input tensor
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], input_data)
            
            # Run inference
            self.tflite_interpreter.invoke()
            
            # Get output
            output_data = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            score = float(output_data[0][0])
            
            return score
            
        except Exception as e:
            self.logger.error(f"[AudioService] Error during wake word prediction: {e}")
            return 0.0