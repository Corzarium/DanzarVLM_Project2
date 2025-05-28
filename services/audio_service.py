# services/audio_service.py
import time
import numpy as np
import requests
import os
import collections # For deque
from typing import Optional, List, Dict # Added List and Dict for clarity
import io # For in-memory audio buffer for the beep

# --- Real Implementations (Ensure these are installed) ---
from openwakeword.model import Model as OWWModel
import whisper # openai-whisper
from pydub import AudioSegment # Make sure this is imported
from pydub.generators import Sine # For generating beep sound
import torch # For whisper fp16
import queue # For app_context.tts_queue Full exception

# from ..DanzarVLM import AppContext # For type hinting if AppContext is in root

class AudioService:
    def __init__(self, app_context): # app_context: AppContext
        # ... (rest of __init__ remains the same) ...
        self.app_context = app_context
        self.oww_model: Optional[OWWModel] = None
        self.stt_model: Optional[whisper.Whisper] = None
        self.logger = self.app_context.logger
        self.logger.info("[AudioService] Instance created.")

        self.raw_discord_audio_buffer = bytearray()
        self.oww_processing_buffer = np.array([], dtype=np.int16)

        self.is_capturing_for_stt = False
        self.stt_audio_capture_chunks: List[np.ndarray] = []
        self.stt_user_speaking_id: Optional[int] = None
        self.stt_consecutive_silent_oww_chunks = 0
        self.stt_vad_grace_period_oww_chunks = 0

        self._update_derived_constants_from_config()


    def _update_derived_constants_from_config(self):
        # ... (this method remains the same) ...
        gs = self.app_context.global_settings
        profile = self.app_context.active_profile

        self.TARGET_SR = gs.get("AUDIO_TARGET_SAMPLE_RATE", 16000)
        self.OWW_EXPECTED_CHUNK_SAMPLES = gs.get("OWW_CHUNK_SAMPLES", 1280)
        self.RMS_VAD_THRESHOLD_VALUE = gs.get("RMS_VAD_THRESHOLD", 100)
        self.OWW_VAD_THRESHOLD_VALUE = gs.get("OWW_VAD_THRESHOLD", 0.05)
        self.OWW_WAKEWORD_THRESHOLD = gs.get("OWW_THRESHOLD", 0.5)

        oww_path = profile.oww_model_path_override or gs.get("OWW_CUSTOM_MODEL_PATH")
        if oww_path:
            default_oww_name = os.path.splitext(os.path.basename(oww_path))[0]
            self.OWW_ACTIVE_MODEL_NAME = profile.oww_model_name_override or gs.get("OWW_CUSTOM_MODEL_NAME", default_oww_name)
        else:
            self.OWW_ACTIVE_MODEL_NAME = profile.oww_model_name_override or gs.get("OWW_CUSTOM_MODEL_NAME", "wakeword") # Fallback

        self.STT_MIN_SPEECH_SAMPLES = int(self.TARGET_SR * gs.get("STT_MIN_SPEECH_DURATION_SAMPLES_FACTOR", 0.4))
        self.STT_MAX_BUFFER_SAMPLES = int(self.TARGET_SR * gs.get("STT_MAX_AUDIO_BUFFER_SECONDS", 15))
        self.STT_SILENCE_CHUNKS_TO_END = gs.get("STT_SILENCE_FRAMES_TO_END_SPEECH", 25)
        self.STT_GRACE_PERIOD_CHUNKS = int(self.TARGET_SR * gs.get("STT_VAD_GRACE_PERIOD_DURATION_S", 0.85) / self.OWW_EXPECTED_CHUNK_SAMPLES)

        self.logger.debug(
            f"[AudioService] Derived audio constants updated: "
            f"TargetSR={self.TARGET_SR}, OWWChunk={self.OWW_EXPECTED_CHUNK_SAMPLES}, "
            f"RMSThresh={self.RMS_VAD_THRESHOLD_VALUE}, OWWActiveModel='{self.OWW_ACTIVE_MODEL_NAME}', "
            f"OWWThresh={self.OWW_WAKEWORD_THRESHOLD}, STTGraceChunks={self.STT_GRACE_PERIOD_CHUNKS}"
        )

    def initialize_audio_systems(self):
        # ... (this method remains the same) ...
        self.logger.info("[AudioService] Initializing audio systems (OWW, STT)...")
        gs = self.app_context.global_settings
        profile = self.app_context.active_profile
        self._update_derived_constants_from_config()

        oww_model_path = profile.oww_model_path_override if profile.oww_model_path_override else gs.get("OWW_CUSTOM_MODEL_PATH")

        if oww_model_path and os.path.exists(oww_model_path):
            try:
                self.logger.info(f"[AudioService] Attempting to load OWW model from: {oww_model_path}")
                self.oww_model = OWWModel(
                    wakeword_models=[oww_model_path],
                    enable_speex_noise_suppression=gs.get("OWW_ENABLE_SPEEX", True),
                    vad_threshold=self.OWW_VAD_THRESHOLD_VALUE
                )
                if hasattr(self.oww_model, 'chunk_size') and self.oww_model.chunk_size and self.oww_model.chunk_size != self.OWW_EXPECTED_CHUNK_SAMPLES:
                    self.logger.warning(f"[AudioService] OWW model's internal chunk_size ({self.oww_model.chunk_size}) differs from configured OWW_EXPECTED_CHUNK_SAMPLES ({self.OWW_EXPECTED_CHUNK_SAMPLES}). Using model's size.")
                    self.OWW_EXPECTED_CHUNK_SAMPLES = self.oww_model.chunk_size
                    self._update_derived_constants_from_config() # Recalculate based on new chunk size

                self.logger.info(f"[AudioService] OWW model loaded successfully from {oww_model_path}. Active model key for prediction checks: '{self.OWW_ACTIVE_MODEL_NAME}'")
            except ImportError as e_imp:
                 self.logger.error(f"[AudioService] ERROR: ImportError during OWW init: {e_imp}", exc_info=True)
            except Exception as e:
                self.logger.error(f"[AudioService] ERROR: Failed to load/initialize OWW model from '{oww_model_path}': {e}", exc_info=True)
        elif oww_model_path:
             self.logger.error(f"[AudioService] ERROR: OWW model path '{oww_model_path}' configured but file not found.")
        else:
            self.logger.warning("[AudioService] WARNING: OWW_CUSTOM_MODEL_PATH not configured. Wake word detection will be disabled.")

        whisper_model_size = gs.get("WHISPER_MODEL_SIZE", "base.en")
        try:
            self.logger.info(f"[AudioService] Attempting to load Whisper STT model: {whisper_model_size}")
            self.stt_model = whisper.load_model(whisper_model_size)
            self.logger.info(f"[AudioService] Whisper STT model '{whisper_model_size}' loaded successfully.")
        except ImportError:
            self.logger.error("[AudioService] ERROR: openai-whisper library or torch not found.", exc_info=True)
        except Exception as e:
            self.logger.error(f"[AudioService] ERROR: Failed to load Whisper model '{whisper_model_size}': {e}", exc_info=True)

        self.logger.info("[AudioService] Audio systems initialization attempt complete.")

    def _resample_discord_audio_chunk(self, pcm_data_48k_stereo: bytes) -> Optional[np.ndarray]:
        # ... (this method remains the same) ...
        if not pcm_data_48k_stereo: return None
        try:
            discord_segment = AudioSegment(data=pcm_data_48k_stereo, sample_width=2, frame_rate=48000, channels=2)
            mono_segment = discord_segment.set_channels(1)
            resampled_segment = mono_segment.set_frame_rate(self.TARGET_SR)
            return np.frombuffer(resampled_segment.raw_data, dtype=np.int16)
        except ImportError: # Should not happen if pydub is installed for beep
            self.logger.error("[AudioService] ERROR: pydub library not found for resampling.", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"[AudioService] Error during audio resampling: {e}", exc_info=True)
            return None

    def _generate_beep_audio(self) -> Optional[bytes]:
        """Generates a short beep sound as WAV bytes."""
        gs = self.app_context.global_settings
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
        # ... (initial part of the method is the same) ...
        self.logger.info(f"[AudioService.PROCESS_CHUNK_ENTRY] User ID: {user_id}, Data len: {len(raw_discord_pcm)}, OWW Model Loaded: {self.oww_model is not None}, STT Model Loaded: {self.stt_model is not None}, IsCapturingSTT: {self.is_capturing_for_stt}, STTUser: {self.stt_user_speaking_id}")
        if not self.oww_model:
            return

        self.raw_discord_audio_buffer.extend(raw_discord_pcm)

        DISCORD_PACKET_DURATION_MS = 20
        BYTES_PER_DISCORD_PACKET = int(48000 * (DISCORD_PACKET_DURATION_MS / 1000) * 2 * 2)

        while len(self.raw_discord_audio_buffer) >= BYTES_PER_DISCORD_PACKET:
            chunk_48k_stereo = bytes(self.raw_discord_audio_buffer[:BYTES_PER_DISCORD_PACKET])
            self.raw_discord_audio_buffer = self.raw_discord_audio_buffer[BYTES_PER_DISCORD_PACKET:]

            resampled_16k_mono_np = self._resample_discord_audio_chunk(chunk_48k_stereo)
            if resampled_16k_mono_np is None or resampled_16k_mono_np.size == 0:
                continue

            self.oww_processing_buffer = np.concatenate((self.oww_processing_buffer, resampled_16k_mono_np))

            while len(self.oww_processing_buffer) >= self.OWW_EXPECTED_CHUNK_SAMPLES:
                oww_chunk_to_process = self.oww_processing_buffer[:self.OWW_EXPECTED_CHUNK_SAMPLES].copy()
                self.oww_processing_buffer = self.oww_processing_buffer[self.OWW_EXPECTED_CHUNK_SAMPLES:]

                if not self.is_capturing_for_stt:
                    try:
                        raw_prediction_output = self.oww_model.predict(oww_chunk_to_process)
                        # ... (OWW debug logging as before) ...
                        self.logger.info(f"[AudioService.OWW_DEBUG] Raw prediction_output type: {type(raw_prediction_output)}")
                        self.logger.info(f"[AudioService.OWW_DEBUG] Raw prediction_output content: {raw_prediction_output}")

                        current_prediction_dict: Optional[Dict[str, float]] = None 
                        if isinstance(raw_prediction_output, list): 
                            if raw_prediction_output:
                                current_prediction_dict = raw_prediction_output[0] 
                        elif isinstance(raw_prediction_output, dict): 
                            current_prediction_dict = raw_prediction_output

                        if current_prediction_dict:
                            self.logger.info(
                                f"[AudioService.OWW_PREDICT] User: {user_id}, "
                                f"TargetModel: '{self.OWW_ACTIVE_MODEL_NAME}', "
                                f"Score: {current_prediction_dict.get(self.OWW_ACTIVE_MODEL_NAME, 0.0):.4f}, "
                                f"Threshold: {self.OWW_WAKEWORD_THRESHOLD}, "
                                f"FullPrediction: {current_prediction_dict}"
                            )

                            if self.OWW_ACTIVE_MODEL_NAME in current_prediction_dict and \
                               current_prediction_dict[self.OWW_ACTIVE_MODEL_NAME] >= self.OWW_WAKEWORD_THRESHOLD:

                                self.logger.info(f"!!! WAKE WORD '{self.OWW_ACTIVE_MODEL_NAME}' DETECTED !!! User: {user_id}, Score: {current_prediction_dict[self.OWW_ACTIVE_MODEL_NAME]:.3f}")
                                
                                # --- PLAY BEEP ON WAKE WORD ---
                                if self.app_context.global_settings.get("PLAY_WAKE_WORD_BEEP", False):
                                    beep_audio_bytes = self._generate_beep_audio()
                                    if beep_audio_bytes:
                                        try:
                                            self.app_context.tts_queue.put_nowait(beep_audio_bytes)
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

                                self.app_context.is_in_conversation.set()
                                self.app_context.last_interaction_time = time.time()
                                self.stt_audio_capture_chunks.append(oww_chunk_to_process) # Add the wake word chunk itself
                                continue # Skip further processing of this chunk for STT VAD if it was a wake word trigger
                        else:
                             self.logger.warning(f"[AudioService.OWW_PREDICT] OWW predict() did not return a usable prediction structure. Output: {raw_prediction_output}")

                    except Exception as e_oww:
                        self.logger.error(f"[AudioService] Error during OWW prediction: {e_oww}", exc_info=True)
                        continue 

                # STT Capture Logic (VAD) - remains the same
                if self.is_capturing_for_stt:
                    if self.stt_user_speaking_id != user_id: # Only process audio from the user who triggered wake word
                        continue

                    self.stt_audio_capture_chunks.append(oww_chunk_to_process)
                    self.app_context.last_interaction_time = time.time() # Keep updating for conversation timeout

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

                        if len(full_stt_audio_np) >= self.STT_MIN_SPEECH_SAMPLES:
                            self.logger.info(f"[AudioService] STT audio captured ({len(full_stt_audio_np)/self.TARGET_SR:.2f}s for user {speaking_user_id}). Transcribing...")
                            transcribed_text = self.transcribe_audio(full_stt_audio_np)

                            if transcribed_text:
                                self.logger.info(f"[AudioService] Transcription for user {speaking_user_id}: '{transcribed_text}'")
                                if self.app_context.llm_service_instance:
                                     # LLMService will handle clearing the conversation flag after processing
                                     self.app_context.llm_service_instance.handle_user_text_query(transcribed_text, str(speaking_user_id))
                                else:
                                    self.logger.error("[AudioService] llm_service_instance not available in app_context! Cannot send transcription.")
                                    # Manually clear if LLM service is missing, to prevent getting stuck
                                    if self.app_context.is_in_conversation.is_set():
                                        self.app_context.is_in_conversation.clear()
                                        self.logger.info("[AudioService] Conversation flag cleared (LLM service missing).")
                            else: # Empty or failed transcription
                                self.logger.info("[AudioService] Transcription resulted in empty text or failed. Ending conversation state.")
                                if self.app_context.is_in_conversation.is_set():
                                    self.app_context.is_in_conversation.clear()
                                    self.logger.info("[AudioService] Conversation flag cleared (empty/failed STT).")
                        else: # Audio too short
                            self.logger.info(f"[AudioService] Captured STT audio too short ({len(full_stt_audio_np)/self.TARGET_SR:.2f}s). Ignoring. Ending conversation state.")
                            if self.app_context.is_in_conversation.is_set():
                                self.app_context.is_in_conversation.clear()
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
            transcribed_text = result["text"].strip()

            self.logger.debug(f"[AudioService] Whisper raw transcription: '{transcribed_text}'")
            return transcribed_text if transcribed_text else None
        except Exception as e:
            self.logger.error(f"[AudioService] ERROR: Whisper transcription failed: {e}", exc_info=True)
            return None

    def fetch_tts_audio(self, text: str) -> Optional[bytes]:
        # ... (this method remains the same) ...
        tts_url = self.app_context.global_settings.get("TTS_SERVER_URL")
        if not tts_url or not text:
            self.logger.debug(f"[AudioService] TTS request skipped: URL: '{tts_url}', Text: '{text[:20]}...'")
            return None

        self.logger.info(f"[AudioService] Requesting TTS for: '{text[:50]}...' from {tts_url}")
        try:
            tts_timeout = self.app_context.global_settings.get("TTS_REQUEST_TIMEOUT_S", 20)
            response = requests.post(tts_url, json={"text": text}, timeout=tts_timeout)
            response.raise_for_status()
            self.logger.info(f"[AudioService] TTS audio received (length: {len(response.content)} bytes).")
            return response.content
        except requests.exceptions.Timeout:
            self.logger.error(f"[AudioService] TTS request timed out ({tts_timeout}s) for '{text[:30]}...'.")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[AudioService] TTS request failed for '{text[:30]}...': {e}")
            return None
        except Exception as e_other:
            self.logger.error(f"[AudioService] Unexpected error during TTS request: {e_other}", exc_info=True)
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
            if self.app_context.is_in_conversation.is_set(): # Check if the event is actually set
                self.app_context.is_in_conversation.clear()
                self.logger.info("[AudioService] Conversation flag cleared due to profile change during STT.")
        
        # Re-initialize OWW and STT models (which also updates derived constants)
        self.initialize_audio_systems()