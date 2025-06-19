#!/usr/bin/env python3
"""
DanzarVLM - AI Game Commentary and Interaction Suite with Voice Activity Detection
Enhanced with real-time VAD for natural conversation flow and faster-whisper STT
"""

import asyncio
import argparse
import logging
import os
import sys
import signal
import time
import threading
import tempfile
import atexit
import subprocess
from typing import Optional, Dict, Any, List
import keyboard
import discord
from discord.ext import commands
import torch
import numpy as np
from collections import deque
import io
import queue

# faster-whisper for efficient STT
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# Standard whisper for fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# Virtual audio support
try:
    import sounddevice as sd
    VIRTUAL_AUDIO_AVAILABLE = True
except ImportError:
    VIRTUAL_AUDIO_AVAILABLE = False
    sd = None

# Offline voice receiver (100% local processing)
try:
    from services.offline_vad_voice_receiver import OfflineVADVoiceReceiver
    OFFLINE_VOICE_AVAILABLE = True
except ImportError:
    OFFLINE_VOICE_AVAILABLE = False
    OfflineVADVoiceReceiver = None

# Core imports
from core.config_loader import load_global_settings
from core.game_profile import GameProfile

# Service imports  
from services.tts_service import TTSService
from services.memory_service import MemoryService
from services.model_client import ModelClient
from services.llm_service import LLMService
from services.faster_whisper_stt_service import FasterWhisperSTTService
from services.simple_voice_receiver import SimpleVoiceReceiver
from services.vad_voice_receiver import VADVoiceReceiver
from services.short_term_memory_service import ShortTermMemoryService

# Setup logging with Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/danzar_voice.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("DanzarVLM")

# Suppress some noisy loggers
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('discord.gateway').setLevel(logging.INFO)
logging.getLogger('discord.voice_client').setLevel(logging.INFO)

class SingleInstanceLock:
    """Ensures only one instance of DanzarAI can run at a time."""
    
    def __init__(self, lockfile_path: str = "danzar_voice.lock"):
        self.lockfile_path = lockfile_path
        self.lockfile = None
        
    def acquire(self) -> bool:
        """Acquire the lock. Returns True if successful, False if another instance is running."""
        try:
            # Check if lock file already exists
            if os.path.exists(self.lockfile_path):
                # Try to read existing PID
                try:
                    with open(self.lockfile_path, 'r') as f:
                        existing_pid = f.read().strip()
                    
                    # Check if process with that PID is still running
                    if self._is_process_running(existing_pid):
                        logger.warning(f"Another instance is running with PID: {existing_pid}")
                        return False
                    else:
                        # Process is dead, remove stale lock file
                        os.unlink(self.lockfile_path)
                        logger.info("Removed stale lock file")
                except Exception:
                    # If we can't read the file, assume it's stale and remove it
                    try:
                        os.unlink(self.lockfile_path)
                    except:
                        pass
            
            # Create new lock file
            self.lockfile = open(self.lockfile_path, 'w')
            
            # Try to acquire exclusive lock (Windows compatible)
            if os.name == 'nt':  # Windows
                import msvcrt
                try:
                    msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
                except IOError:
                    self.lockfile.close()
                    return False
            else:  # Unix/Linux
                try:
                    import fcntl
                    fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (ImportError, IOError):
                    # If fcntl is not available, just proceed (basic file existence check)
                    pass
            
            # Write PID to lockfile
            self.lockfile.write(str(os.getpid()))
            self.lockfile.flush()
            
            # Register cleanup on exit
            atexit.register(self.release)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False
    
    def _is_process_running(self, pid_str: str) -> bool:
        """Check if a process with given PID is still running."""
        try:
            pid = int(pid_str)
            if os.name == 'nt':  # Windows
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                      capture_output=True, text=True)
                return str(pid) in result.stdout
            else:  # Unix/Linux
                os.kill(pid, 0)  # Send signal 0 to check if process exists
                return True
        except (ValueError, OSError, subprocess.SubprocessError):
            return False
    
    def release(self):
        """Release the lock."""
        if self.lockfile:
            try:
                if os.name == 'nt':  # Windows
                    import msvcrt
                    try:
                        msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                    except:
                        pass
                else:  # Unix/Linux
                    try:
                        import fcntl
                        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
                    except ImportError:
                        pass
                
                self.lockfile.close()
                
                # Remove lockfile
                if os.path.exists(self.lockfile_path):
                    os.unlink(self.lockfile_path)
                    
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self.lockfile = None

# Global lock instance
app_lock = SingleInstanceLock()

class AppContext:
    """Application context for managing shared resources and services."""
    
    def __init__(self, global_settings: dict, active_profile: GameProfile, logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger_instance
        self.shutdown_event = threading.Event()
        
        # Add missing attributes for LLM service compatibility
        self.rag_service_instance: Optional[Any] = None
        self.is_in_conversation = threading.Event()
        self.active_profile_change_subscribers = []
        
        # Service instances
        self.tts_service: Optional[TTSService] = None
        self.memory_service: Optional[MemoryService] = None
        self.model_client: Optional[ModelClient] = None
        self.llm_service: Optional[LLMService] = None
        self.short_term_memory_service: Optional[ShortTermMemoryService] = None
        self.faster_whisper_stt_service: Optional[FasterWhisperSTTService] = None
        self.simple_voice_receiver: Optional[SimpleVoiceReceiver] = None
        self.vad_voice_receiver: Optional[VADVoiceReceiver] = None
        self.offline_vad_voice_receiver: Optional[Any] = None  # Type will be set when available
        self.qwen_omni_service: Optional[Any] = None  # Qwen2.5-Omni service instance
        self.llamacpp_qwen_service: Optional[Any] = None  # LlamaCpp Qwen service instance
        
        logger_instance.info("[AppContext] Initialized.")

    def get_service(self, service_name: str):
        """Get a service instance by name."""
        return getattr(self, f"{service_name}_service", None)

    def set_service(self, service_name: str, service_instance):
        """Set a service instance."""
        setattr(self, f"{service_name}_service", service_instance)

    def get_profile_setting(self, key: str, default=None):
        """Get a setting from the active profile."""
        if self.active_profile and hasattr(self.active_profile, 'ocr_settings'):
            return getattr(self.active_profile, key, default)
        return default

    def get_global_setting(self, key: str, default=None):
        """Get a setting from global settings."""
        return self.global_settings.get(key, default)

    def update_active_profile(self, new_profile: GameProfile):
        """Update the active profile."""
        self.active_profile = new_profile
        self.logger.info(f"[AppContext] Active profile updated to: {new_profile.game_name}")

class AudioFeedbackPrevention:
    """Comprehensive audio feedback prevention system to stop TTS output from being captured as input."""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Track when we're playing TTS to ignore input during that time
        self.tts_playing = False
        self.tts_start_time = None
        self.tts_duration = 0.0
        
        # Track our own TTS output text to filter it out if detected
        self.recent_tts_outputs = deque(maxlen=20)  # Increased from 10 to 20
        self.tts_output_timestamps = deque(maxlen=20)
        
        # Audio signature tracking (simple approach)
        self.last_tts_audio_signature = None
        
        # Reasonable silence periods after TTS
        self.post_tts_silence_duration = 3.0  # Reduced from 8.0 to 3.0 seconds
        
        # Global TTS blocking - when ANY TTS is active, block ALL input
        self.global_tts_block = False
        self.global_tts_block_start = None
        self.global_tts_block_duration = 0.0
        
        self.logger.info("🛡️ Audio Feedback Prevention System initialized with balanced blocking")
    
    def start_tts_playback(self, tts_text: str, estimated_duration: Optional[float] = None):
        """Call this when starting TTS playback to prevent feedback."""
        self.tts_playing = True
        self.tts_start_time = time.time()
        
        # More reasonable duration estimation
        if estimated_duration is None:
            word_count = len(tts_text.split())
            # More reasonable: 3 words per second + 2 second buffer
            self.tts_duration = max(2.0, word_count / 3.0 + 2.0)
        else:
            self.tts_duration = estimated_duration + 1.0  # Reduced buffer from 3.0 to 1.0
        
        # GLOBAL TTS BLOCK - block ALL audio input
        self.global_tts_block = True
        self.global_tts_block_start = time.time()
        self.global_tts_block_duration = self.tts_duration + self.post_tts_silence_duration
        
        # Store the TTS text for comparison (split into words for better matching)
        tts_clean = tts_text.lower().strip()
        self.recent_tts_outputs.append(tts_clean)
        self.tts_output_timestamps.append(time.time())
        
        # Store key phrases instead of all 3-word combinations to reduce false positives
        words = tts_clean.split()
        if len(words) >= 4:
            # Only store the first and last few words as key phrases
            first_phrase = " ".join(words[:3])
            last_phrase = " ".join(words[-3:])
            self.recent_tts_outputs.append(first_phrase)
            self.recent_tts_outputs.append(last_phrase)
            self.tts_output_timestamps.append(time.time())
            self.tts_output_timestamps.append(time.time())
        
        self.logger.info(f"🛡️ TTS BLOCK ACTIVATED - blocking input for {self.global_tts_block_duration:.1f}s")
        self.logger.info(f"🛡️ TTS text stored: '{tts_text[:50]}...'")
    
    def stop_tts_playback(self):
        """Call this when TTS playback finishes."""
        if self.tts_playing:
            actual_duration = time.time() - self.tts_start_time if self.tts_start_time else 0
            self.logger.info(f"🛡️ TTS playback stopped - actual duration: {actual_duration:.1f}s")
            self.logger.info(f"🛡️ Continuing block for {self.post_tts_silence_duration:.1f}s more...")
        
        self.tts_playing = False
        self.tts_start_time = None
        # NOTE: Don't stop global_tts_block here - let it expire naturally
    
    def should_ignore_input(self) -> tuple[bool, str]:
        """
        Check if we should ignore audio input right now to prevent feedback.
        Returns (should_ignore, reason)
        """
        current_time = time.time()
        
        # 1. GLOBAL TTS BLOCK - highest priority
        if self.global_tts_block and self.global_tts_block_start:
            elapsed = current_time - self.global_tts_block_start
            if elapsed < self.global_tts_block_duration:
                return True, f"GLOBAL TTS BLOCK ({elapsed:.1f}s/{self.global_tts_block_duration:.1f}s)"
            else:
                # Global block expired
                self.global_tts_block = False
                self.global_tts_block_start = None
                self.logger.info("🛡️ Global TTS block expired - input allowed")
        
        # 2. Check if TTS is currently playing
        if self.tts_playing and self.tts_start_time:
            elapsed = current_time - self.tts_start_time
            if elapsed < self.tts_duration:
                return True, f"TTS playing ({elapsed:.1f}s/{self.tts_duration:.1f}s)"
            else:
                # TTS should be done, but add post-silence period
                if elapsed < (self.tts_duration + self.post_tts_silence_duration):
                    return True, f"Post-TTS silence ({elapsed:.1f}s/{self.tts_duration + self.post_tts_silence_duration:.1f}s)"
                else:
                    # TTS is definitely done
                    self.stop_tts_playback()
        
        # 3. Check recent TTS timestamps for additional safety
        for tts_time in self.tts_output_timestamps:
            time_since_tts = current_time - tts_time
            if time_since_tts < self.post_tts_silence_duration:
                return True, f"Recent TTS cooldown ({time_since_tts:.1f}s < {self.post_tts_silence_duration:.1f}s)"
        
        return False, "Input allowed"
    
    def is_likely_tts_echo(self, transcription: str) -> tuple[bool, str]:
        """
        Check if a transcription is likely an echo of our own TTS output.
        Returns (is_echo, reason)
        """
        if not transcription or len(transcription.strip()) < 3:
            return False, "Too short to analyze"
        
        transcription_clean = transcription.lower().strip()
        
        # Check against recent TTS outputs with aggressive matching for exact/substring matches
        for i, tts_output in enumerate(self.recent_tts_outputs):
            # Exact match - definitely an echo
            if transcription_clean == tts_output:
                return True, f"EXACT match with recent TTS output #{i}"
            
            # Substring match - very likely an echo
            if len(transcription_clean) > 5 and len(tts_output) > 5:
                if transcription_clean in tts_output or tts_output in transcription_clean:
                    return True, f"SUBSTRING match with recent TTS output #{i}"
            
            # Word similarity - only for very high similarity (80%+)
            if len(transcription_clean) > 10 and len(tts_output) > 10:
                trans_words = set(transcription_clean.split())
                tts_words = set(tts_output.split())
                
                if len(trans_words) > 0 and len(tts_words) > 0:
                    intersection = len(trans_words.intersection(tts_words))
                    union = len(trans_words.union(tts_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.8:  # Raised back to 80% for word similarity
                        return True, f"HIGH similarity ({similarity:.1%}) with recent TTS output #{i}"
                
                # Word match - only for very high word overlap (70%+)
                if len(trans_words) >= 3 and len(tts_words) >= 3:
                    matching_words = sum(1 for word in trans_words if word in tts_words)
                    match_ratio = matching_words / len(trans_words)
                    if match_ratio >= 0.5:  # 50% word match
                        return True, f"WORD match ({match_ratio:.1%}) with recent TTS output #{i}"
        
        # Enhanced TTS echo patterns - more specific to avoid false positives
        echo_patterns = [
            # Exact TTS response patterns (very specific)
            "i heard you say", "i'm danzar", "danzarai",
            "gaming assistant", "how can i help",
            "that's interesting", "let me help you", "i understand",
            # EverQuest TTS-specific phrases (from actual TTS responses)
            "first they mentioned", "user is confident", "prep expansions ready",
            "detailed spreadsheet for platinum", "in-game currency",
            "they want to crush goals", "looking at the tools provided",
            "there's info on necromancer", "both are", "they mentioned",
            # Common TTS response starters
            "looking to maximize their", "so they're probably",
            "the user is confident with their"
        ]
        
        # Check for pattern matches - only exact or very close matches
        for pattern in echo_patterns:
            # Exact match
            if transcription_clean == pattern:
                return True, f"EXACT TTS echo pattern: '{pattern}'"
            # Very close match (pattern is 70%+ of the transcription)
            if pattern in transcription_clean and len(pattern) / len(transcription_clean) > 0.7:
                return True, f"DOMINANT TTS echo pattern: '{pattern}'"
        
        # Check for common single-word TTS echoes only if they're the entire transcription
        single_word_echoes = ["danzar", "everquest", "necromancer", "wizard", "expansions", "platinum"]
        if transcription_clean in single_word_echoes:
            return True, f"SINGLE-WORD TTS echo: '{transcription_clean}'"
        
        return False, "Not detected as echo"
    
    def cleanup_old_entries(self):
        """Clean up old TTS entries to prevent memory buildup."""
        current_time = time.time()
        cutoff_time = current_time - 60.0  # Keep last 60 seconds (increased from 30)
        
        # Remove old timestamps
        while self.tts_output_timestamps and self.tts_output_timestamps[0] < cutoff_time:
            self.tts_output_timestamps.popleft()
            if self.recent_tts_outputs:
                self.recent_tts_outputs.popleft()

class WhisperAudioCapture:
    """Captures audio from virtual audio cables and processes with Whisper STT only."""
    
    def __init__(self, app_context: AppContext, callback_func=None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.callback_func = callback_func
        
        # Audio settings optimized for Whisper
        self.sample_rate = 16000  # Whisper's native sample rate
        self.channels = 1  # Mono for Whisper
        self.chunk_size = 1024  # Smaller chunks for real-time processing
        self.dtype = np.float32
        
        # Speech detection settings (simple level-based) - BALANCED
        self.speech_threshold = 0.10  # Lower threshold for easier detection (was 0.15)
        self.min_speech_duration = 1.0  # Shorter minimum duration (was 1.5)
        self.max_silence_duration = 2.5  # Longer silence tolerance (was 2.0)
        
        # State tracking
        self.is_recording = False
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
        
        # Audio device
        self.input_device = None
        self.stream = None
        
        # Processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
        # Transcription results queue for Discord bot
        self.transcription_queue = queue.Queue()
        
        # Whisper model
        self.whisper_model = None
        
    def list_audio_devices(self):
        """List all available audio input devices."""
        if not VIRTUAL_AUDIO_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available - install with: pip install sounddevice")
            return []
            
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
                
                if max_channels > 0:  # Input device
                    self.logger.info(f"  {i}: {device_name} (channels: {max_channels})")
                    
                    # Look for virtual audio cables
                    if any(keyword in device_name.lower() for keyword in 
                          ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'stereo mix', 'wave out mix', 'what u hear']):
                        virtual_devices.append((i, device_name))
                        self.logger.info(f"      ⭐ VIRTUAL AUDIO DEVICE DETECTED")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not process device {i}: {e}")
                continue
        
        if virtual_devices:
            self.logger.info("🎯 Recommended Virtual Audio Devices:")
            for device_id, device_name in virtual_devices:
                self.logger.info(f"  Device {device_id}: {device_name}")
        else:
            self.logger.warning("⚠️  No virtual audio devices detected. Install VB-Cable or enable Stereo Mix.")
            
        return virtual_devices
    
    def select_input_device(self, device_id: Optional[int] = None):
        """Select audio input device."""
        if not VIRTUAL_AUDIO_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available")
            return False
            
        if device_id is None:
            # Use the working VB-Audio device (Device 8: CABLE Output VB-Audio Virtual Cable)
            working_device_id = 8
            
            # Check if the working device is available
            virtual_devices = self.list_audio_devices()
            working_device_found = False
            
            for dev_id, dev_info in virtual_devices:
                if dev_id == working_device_id:
                    device_id = working_device_id
                    working_device_found = True
                    self.logger.info(f"✅ Selected audio device: {dev_info['name']}")
                    self.logger.info(f"🎯 Using working VB-Audio device: {dev_info['name']}")
                    break
            
            if not working_device_found and virtual_devices:
                # Fallback to first available virtual device
                device_id = virtual_devices[0][0]
                self.logger.info(f"🎯 Auto-selected virtual audio device: {device_id}")
            elif not working_device_found:
                # Fall back to default device
                try:
                    device_id = sd.default.device[0]
                    self.logger.warning(f"⚠️  Using default input device: {device_id}")
                except Exception:
                    device_id = 0  # Fallback to device 0
                    self.logger.warning("⚠️  Using device 0 as fallback")
        
        try:
            device_info = sd.query_devices(device_id, 'input')
            self.input_device = device_id
            device_name = device_info.get('name', f'Device {device_id}') if isinstance(device_info, dict) else str(device_info)
            self.logger.info(f"✅ Selected audio device: {device_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to select device {device_id}: {e}")
            return False
    
    async def initialize_whisper(self, model_size: str = "base"):
        """Initialize Whisper model."""
        try:
            self.logger.info(f"🔧 Loading Whisper model '{model_size}'...")
            self.logger.info("💡 Available models: tiny, base, small, medium, large")
            self.logger.info("💡 Accuracy: tiny < base < small < medium < large")
            self.logger.info("💡 Speed: large < medium < small < base < tiny")
            
            loop = asyncio.get_event_loop()
            start_time = time.time()
            if WHISPER_AVAILABLE and whisper:
                self.whisper_model = await loop.run_in_executor(
                    None,
                    whisper.load_model,
                    model_size
                )
            else:
                raise Exception("Whisper not available")
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ Whisper model '{model_size}' loaded successfully in {load_time:.1f}s")
            
            # Show model info
            if hasattr(self.whisper_model, 'dims'):
                dims = self.whisper_model.dims
                self.logger.info(f"📊 Model info: {dims.n_audio_ctx} audio tokens, {dims.n_text_ctx} text tokens")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Whisper initialization failed: {e}")
            return False
    
    def detect_speech(self, audio_chunk: np.ndarray) -> tuple[bool, bool]:
        """Simple level-based speech detection."""
        try:
            # Calculate RMS (Root Mean Square) for volume detection
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
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for sounddevice stream."""
        if status:
            self.logger.warning(f"⚠️  Audio callback status: {status}")
        
        try:
            # Convert stereo to mono if needed
            if len(indata.shape) > 1:
                audio_mono = np.mean(indata, axis=1)
            else:
                audio_mono = indata.copy()
            
            # Add audio to queue for processing
            self.audio_queue.put(audio_mono.flatten())
        except Exception as e:
            self.logger.error(f"❌ Audio callback error: {e}")
    
    def processing_worker(self):
        """Worker thread for processing audio chunks."""
        self.logger.info("🎯 Whisper audio processing worker started")
        
        while self.is_recording:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Detect speech using simple level-based detection
                is_speech, speech_ended = self.detect_speech(audio_chunk)
                
                # Always add to buffer when speaking
                if self.is_speaking:
                    self.audio_buffer.extend(audio_chunk)
                
                # Process complete speech segments
                if speech_ended and len(self.audio_buffer) > 0:
                    # Check if we should ignore this input due to TTS feedback prevention
                    # Note: We need to access the bot's feedback prevention through callback
                    # This is a limitation of the current architecture
                    # For now, we'll rely on the transcription queue processing to handle TTS blocking
                    
                    # Get accumulated audio
                    speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer.clear()
                    
                    # Process with Whisper using direct call instead of async callback
                    if self.callback_func:
                        try:
                            self.logger.info("🎯 Processing speech audio directly...")
                            
                            # Call the transcription directly in this thread
                            # This avoids the event loop threading issues
                            import asyncio
                            
                            # Create a new event loop for this thread
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Run the callback in this thread's event loop
                                result = loop.run_until_complete(self.callback_func(speech_audio))
                                
                                loop.close()
                                self.logger.info("✅ Speech processing completed successfully")
                                
                            except Exception as loop_error:
                                self.logger.error(f"❌ Event loop error: {loop_error}")
                                
                                # Use direct synchronous processing with queue
                                try:
                                    self.logger.info("🔄 Processing with STT service...")
                                    
                                    # Check if LlamaCpp Qwen can handle audio directly
                                    llamacpp_config = self.app_context.global_settings.get('LLAMACPP_QWEN', {})
                                    self.logger.info(f"🔍 Debug - LlamaCpp config: enabled={llamacpp_config.get('enabled')}, use_for_audio={llamacpp_config.get('use_for_audio')}")
                                    self.logger.info(f"🔍 Debug - Has service: {hasattr(self.app_context, 'llamacpp_qwen_service')}, Service exists: {getattr(self.app_context, 'llamacpp_qwen_service', None) is not None}")
                                    if (llamacpp_config.get('enabled') and llamacpp_config.get('use_for_audio') and
                                        hasattr(self.app_context, 'llamacpp_qwen_service') and self.app_context.llamacpp_qwen_service):
                                        
                                        self.logger.info("🎵 Using LlamaCpp Qwen for synchronous audio processing...")
                                        transcription = self._transcribe_with_llamacpp_sync(speech_audio)
                                        
                                        if transcription and len(transcription.strip()) > 0:
                                            self.logger.info(f"✅ LlamaCpp Qwen transcription: '{transcription}'")
                                            
                                            # Put transcription in queue for Discord bot to process
                                            self.transcription_queue.put({
                                                'transcription': transcription,
                                                'timestamp': time.time(),
                                                'user': 'VirtualAudio'
                                            })
                                            self.logger.info("📤 Added LlamaCpp Qwen transcription to queue for Discord processing")
                                        else:
                                            self.logger.info("🔇 No transcription from LlamaCpp Qwen")
                                    
                                    # Fallback to Qwen2.5-Omni if enabled
                                    elif (self.app_context.global_settings.get('QWEN_OMNI_SERVICE', {}).get('enabled') and 
                                          self.app_context.global_settings.get('QWEN_OMNI_SERVICE', {}).get('use_for_audio') and
                                          hasattr(self.app_context, 'qwen_omni_service') and self.app_context.qwen_omni_service):
                                        
                                        self.logger.info("🎵 Using Qwen2.5-Omni for synchronous audio processing...")
                                        transcription = self._transcribe_with_omni_sync(speech_audio)
                                        
                                        if transcription and len(transcription.strip()) > 0:
                                            self.logger.info(f"✅ Qwen2.5-Omni transcription: '{transcription}'")
                                            
                                            # Put transcription in queue for Discord bot to process
                                            self.transcription_queue.put({
                                                'transcription': transcription,
                                                'timestamp': time.time(),
                                                'user': 'VirtualAudio'
                                            })
                                            self.logger.info("📤 Added Qwen2.5-Omni transcription to queue for Discord processing")
                                        else:
                                            self.logger.info("🔇 No transcription from Qwen2.5-Omni")
                                    
                                    # Fallback to faster-whisper
                                    elif hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service:
                                        transcription = self.app_context.faster_whisper_stt_service.transcribe_audio_data(speech_audio)
                                        if transcription and len(transcription.strip()) > 0:
                                            self.logger.info(f"✅ faster-whisper transcription: '{transcription}'")
                                            
                                            # Put transcription in queue for Discord bot to process
                                            self.transcription_queue.put({
                                                'transcription': transcription,
                                                'timestamp': time.time(),
                                                'user': 'VirtualAudio'
                                            })
                                            self.logger.info("📤 Added faster-whisper transcription to queue for Discord processing")
                                        else:
                                            self.logger.info("🔇 No transcription from faster-whisper")
                                    else:
                                        self.logger.warning("⚠️ No STT service available - audio detected but cannot transcribe")
                                except Exception as sync_error:
                                    self.logger.error(f"❌ Synchronous processing failed: {sync_error}")
                        except Exception as e:
                            self.logger.error(f"❌ Processing error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing worker error: {e}")
        
        self.logger.info("🎯 Whisper audio processing worker stopped")
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data using available STT service."""
        try:
            # Basic audio validation
            if len(audio_data) == 0:
                return None
            
            # Calculate audio metrics
            audio_duration = len(audio_data) / 16000  # Assuming 16kHz
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Audio preprocessing - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # Quality checks
            if audio_max_volume < 0.01:
                self.logger.warning(f"🔇 Audio volume too low (max: {audio_max_volume:.4f})")
                return None
            
            if audio_duration < 0.3:
                self.logger.warning(f"🔇 Audio too short ({audio_duration:.2f}s)")
                return None
            
            # Check if Qwen2.5-Omni can handle audio directly
            qwen_config = self.app_context.global_settings.get('QWEN_OMNI_SERVICE', {})
            if (qwen_config.get('enabled') and qwen_config.get('use_for_audio') and
                hasattr(self.app_context, 'qwen_omni_service') and self.app_context.qwen_omni_service):
                
                self.logger.info("🎵 Using Qwen2.5-Omni for direct audio processing...")
                return await self._transcribe_with_omni(audio_data)
            
            # Use faster-whisper STT service
            elif (hasattr(self.app_context, 'faster_whisper_stt_service') and 
                self.app_context.faster_whisper_stt_service):
                
                self.logger.info("🎯 Using faster-whisper for transcription...")
                result = self.app_context.faster_whisper_stt_service.transcribe_audio_data(audio_data)
                
                if result and len(result.strip()) > 0:
                    self.logger.info(f"✅ faster-whisper transcription: '{result}'")
                    return result
                else:
                    self.logger.info("🔇 faster-whisper returned no result")
                    return None
            else:
                self.logger.error("❌ No STT service available")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None

    async def _transcribe_with_omni(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Qwen2.5-Omni multimodal model."""
        try:
            # Save audio to temporary file for Omni
            import tempfile
            import scipy.io.wavfile as wavfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                audio_int16 = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
                wavfile.write(temp_file.name, 16000, audio_int16)
                temp_file_path = temp_file.name
            
            self.logger.info(f"🎵 Saved audio to {temp_file_path}, processing with Qwen2.5-Omni...")
            
            # Use Qwen2.5-Omni to process audio directly
            response = await self.app_context.qwen_omni_service.generate_response(
                text="Please transcribe the speech in this audio accurately.",
                audio_path=temp_file_path
            )
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            if response and len(response.strip()) > 0:
                # Extract just the transcription from the response
                transcription = response.strip()
                
                # Remove any assistant commentary, keep just the transcription
                if "transcription:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif "says:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif '"' in transcription:
                    # Extract quoted text if present
                    import re
                    quoted = re.findall(r'"([^"]*)"', transcription)
                    if quoted:
                        transcription = quoted[0]
                
                self.logger.info(f"✅ Qwen2.5-Omni transcription: '{transcription}'")
                return transcription
            else:
                self.logger.info("🔇 Qwen2.5-Omni returned no transcription")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Qwen2.5-Omni transcription error: {e}")
            return None

    def _transcribe_with_omni_sync(self, audio_data: np.ndarray) -> Optional[str]:
        """Synchronous version of Qwen2.5-Omni transcription for worker threads."""
        try:
            # Save audio to temporary file for Omni
            import tempfile
            import scipy.io.wavfile as wavfile
            import asyncio
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                audio_int16 = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
                wavfile.write(temp_file.name, 16000, audio_int16)
                temp_file_path = temp_file.name
            
            self.logger.info(f"🎵 Saved audio to {temp_file_path}, processing with Qwen2.5-Omni sync...")
            
            # Create a new event loop for this thread and run the async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async Qwen2.5-Omni call in this thread's event loop
                response = loop.run_until_complete(
                    self.app_context.qwen_omni_service.generate_response(
                        text="Please transcribe the speech in this audio accurately.",
                        audio_path=temp_file_path
                    )
                )
            finally:
                loop.close()
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            if response and len(response.strip()) > 0:
                # Extract just the transcription from the response
                transcription = response.strip()
                
                # Remove any assistant commentary, keep just the transcription
                if "transcription:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif "says:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif '"' in transcription:
                    # Extract quoted text if present
                    import re
                    quoted = re.findall(r'"([^"]*)"', transcription)
                    if quoted:
                        transcription = quoted[0]
                
                self.logger.info(f"✅ Qwen2.5-Omni sync transcription: '{transcription}'")
                return transcription
            else:
                self.logger.info("🔇 Qwen2.5-Omni sync returned no transcription")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Qwen2.5-Omni sync transcription error: {e}")
            return None

    def _transcribe_with_llamacpp_sync(self, audio_data: np.ndarray) -> Optional[str]:
        """Synchronous version of LlamaCpp Qwen transcription - DISABLED for GGUF."""
        # LlamaCpp GGUF doesn't support audio transcription, return None to fall back to Whisper
        self.logger.info("🎵 LlamaCpp audio transcription disabled - falling back to Whisper")
        return None
    
    def start_recording(self):
        """Start recording from audio device."""
        if not VIRTUAL_AUDIO_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available")
            return False
            
        if self.is_recording:
            self.logger.warning("⚠️  Already recording")
            return True
        
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info("✅ Whisper audio recording started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.logger.info("🛑 Whisper audio recording stopped")

class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD for real-time speech processing."""
    
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.app_context: Optional[Any] = None  # Will be set later for configuration access
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms chunks at 16kHz
        # VAD thresholds optimized for distributed Discord setup (double compression)
        self.speech_threshold = 0.15  # Very low for Discord's compressed audio
        self.silence_threshold = 0.1   # Very low for compressed audio
        self.min_speech_duration = 0.2  # Very short minimum - catch brief words
        self.max_silence_duration = 3.0  # Longer pauses for network delays
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 10))  # 10 second buffer
        
    async def initialize(self):
        """Initialize the VAD model."""
        try:
            # Check if VAD is disabled in configuration
            if hasattr(self, 'app_context') and self.app_context:
                if self.app_context.global_settings.get('DISABLE_SILERO_VAD', False):
                    self.logger.info("🔇 Silero VAD disabled in configuration, using basic audio detection")
                    return True
                if self.app_context.global_settings.get('DISABLE_VAD', False):
                    self.logger.info("🔇 VAD disabled in configuration, skipping initialization")
                    return True
            
            self.logger.info("🔧 Loading Silero VAD model...")
            
            # Load Silero VAD model
            loop = asyncio.get_event_loop()
            self.model, _ = await loop.run_in_executor(
                None,
                lambda: torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
            )
            
            self.logger.info("✅ Silero VAD model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load VAD model: {e}")
            self.logger.info("🔇 Falling back to basic audio detection without VAD")
            return True  # Return True to continue without VAD
    
    def process_audio_chunk(self, audio_data: bytes) -> tuple[bool, bool]:
        """
        Process audio chunk and return (is_speech, speech_ended).
        
        Returns:
            is_speech: True if current chunk contains speech
            speech_ended: True if a complete speech segment just ended
        """
        if not self.model:
            return False, False
            
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed (Discord uses 48kHz, VAD expects 16kHz)
            if len(audio_np) > 0:
                # Simple downsampling (Discord 48kHz -> VAD 16kHz)
                audio_np = audio_np[::3]  # Take every 3rd sample
                
                # Ensure we have enough samples
                if len(audio_np) < self.chunk_size:
                    # Pad with zeros if too short
                    audio_np = np.pad(audio_np, (0, self.chunk_size - len(audio_np)))
                elif len(audio_np) > self.chunk_size:
                    # Truncate if too long
                    audio_np = audio_np[:self.chunk_size]
                
                # Add to buffer
                self.audio_buffer.extend(audio_np)
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_np).float()
                
                # Get VAD probability
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                
                current_time = time.time()
                
                # For distributed Discord setup: if volume is good, override VAD
                audio_max = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0.0
                volume_override = audio_max > 0.3  # Good volume suggests speech
                
                is_speech = speech_prob > self.speech_threshold or volume_override
                speech_ended = False
                
                if volume_override and speech_prob <= self.speech_threshold:
                    self.logger.info(f"🎯 Volume override: treating as speech (vol: {audio_max:.3f}, prob: {speech_prob:.3f})")
                
                if is_speech:
                    if not self.is_speaking:
                        # Speech started
                        self.is_speaking = True
                        self.speech_start_time = current_time
                        self.logger.info(f"🎤 Speech started (prob: {speech_prob:.3f})")
                    
                    self.last_speech_time = current_time
                    
                elif self.is_speaking and self.last_speech_time is not None:
                    # Check if silence duration exceeds threshold
                    silence_duration = current_time - self.last_speech_time
                    
                    if silence_duration > self.max_silence_duration and self.speech_start_time is not None:
                        # Speech ended
                        speech_duration = current_time - self.speech_start_time
                        
                        if speech_duration >= self.min_speech_duration:
                            speech_ended = True
                            self.logger.info(f"🎤 Speech ended (duration: {speech_duration:.2f}s, silence: {silence_duration:.2f}s)")
                        else:
                            self.logger.info(f"🔇 Speech too short (duration: {speech_duration:.2f}s < {self.min_speech_duration:.2f}s)")
                        
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.last_speech_time = None
                
                return is_speech, speech_ended
                
        except Exception as e:
            self.logger.error(f"❌ VAD processing error: {e}")
            
        return False, False
    
    def get_speech_audio(self) -> Optional[np.ndarray]:
        """Get the accumulated speech audio from buffer."""
        if len(self.audio_buffer) > 0:
            audio_array = np.array(list(self.audio_buffer))
            self.audio_buffer.clear()
            return audio_array
        return None

class DanzarVoiceBot(commands.Bot):
    """Enhanced Discord bot with Voice Activity Detection for natural conversation."""
    
    def __init__(self, settings: dict, app_context: AppContext):
        # Set up proper intents for voice functionality
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None
        )
        
        self.settings = settings
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Voice processing components
        self.whisper_model = None
        self.vad_model = None
        self.faster_whisper_stt_service: Optional[FasterWhisperSTTService] = None
        self.simple_voice_receiver: Optional[SimpleVoiceReceiver] = None
        self.vad_voice_receiver: Optional[VADVoiceReceiver] = None
        self.offline_vad_voice_receiver = None  # Type will be set when available
        
        # Service references
        self.tts_service = None
        self.llm_service = None
        
        # Voice connection tracking
        self.connections: Dict[int, discord.VoiceClient] = {}
        self.recording_tasks: Dict[int, asyncio.Task] = {}
        
        # VAD-based voice receiver
        self.vad_voice_receiver: Optional[VADVoiceReceiver] = None
        
        # Initialize VAD detector
        self.vad = VoiceActivityDetector(self.logger)
        self.vad.app_context = self.app_context  # Pass app_context for configuration access
        
        # Virtual audio capture
        self.virtual_audio: Optional[WhisperAudioCapture] = None
        self.use_virtual_audio = settings.get('USE_VIRTUAL_AUDIO', False)
        
        # ===== AUDIO FEEDBACK PREVENTION SYSTEM =====
        self.feedback_prevention = AudioFeedbackPrevention(self.logger)
        
        # ===== TTS QUEUE SYSTEM FOR ORDERED PLAYBACK =====
        self.tts_queue = asyncio.Queue()
        self.tts_queue_active = False
        self.tts_queue_task = None
        
        # LLM search mode setting
        self.llm_search_mode = False
        
        # Add commands
        self.add_commands()

    async def initialize_services(self):
        """Initialize all required services."""
        try:
            # Initialize TTS Service
            self.tts_service = TTSService(self.app_context)
            # Note: TTSService doesn't have initialize method, it's ready after construction
            self.app_context.tts_service = self.tts_service
            self.logger.info("✅ TTS Service initialized")
            
            # Initialize Memory Service
            memory_service = MemoryService(self.app_context)
            # Note: MemoryService doesn't have initialize method, it's ready after construction
            self.app_context.memory_service = memory_service
            self.logger.info("✅ Memory Service initialized")
            
            # Initialize Real-Time Voice Service (Neuro-sama style)
            realtime_voice_config = self.app_context.global_settings.get('REAL_TIME_VOICE', {})
            if realtime_voice_config.get('enabled', False):
                try:
                    from services.real_time_voice_service import RealTimeVoiceService
                    self.app_context.realtime_voice_service = RealTimeVoiceService(self.app_context)
                    if await self.app_context.realtime_voice_service.initialize():
                        self.logger.info("✅ Real-Time Voice Service initialized")
                        
                        # Set up callbacks
                        self.app_context.realtime_voice_service.set_transcription_callback(self._on_realtime_transcription)
                        self.app_context.realtime_voice_service.set_response_callback(self._on_realtime_response)
                    else:
                        self.logger.error("❌ Real-Time Voice Service initialization failed")
                        self.app_context.realtime_voice_service = None
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize Real-Time Voice Service: {e}")
                    self.app_context.realtime_voice_service = None
            else:
                self.app_context.realtime_voice_service = None
                self.logger.info("ℹ️ Real-Time Voice Service disabled in configuration")
            
            # Initialize Qwen2.5-Omni Service if enabled
            qwen_omni_config = self.app_context.global_settings.get('QWEN_OMNI_SERVICE', {})
            if qwen_omni_config.get('enabled', False):
                try:
                    from services.qwen_omni_service import QwenOmniService
                    self.app_context.qwen_omni_service = QwenOmniService(self.app_context)
                    if await self.app_context.qwen_omni_service.initialize():
                        self.logger.info("✅ Qwen2.5-Omni Service initialized")
                        # Use Qwen2.5-Omni as the primary model client
                        model_client = self.app_context.qwen_omni_service
                    else:
                        self.logger.error("❌ Qwen2.5-Omni Service initialization failed")
                        model_client = ModelClient(app_context=self.app_context)
                except Exception as e:
                    self.logger.error(f"❌ Qwen2.5-Omni Service error: {e}")
                    model_client = ModelClient(app_context=self.app_context)
            else:
                # Initialize LlamaCpp Qwen Service if enabled
                llamacpp_config = self.app_context.global_settings.get('LLAMACPP_QWEN', {})
                if llamacpp_config.get('enabled', False):
                    try:
                        from services.llamacpp_qwen_service import LlamaCppQwenService
                        self.app_context.llamacpp_qwen_service = LlamaCppQwenService(self.app_context)
                        if await self.app_context.llamacpp_qwen_service.initialize():
                            self.logger.info("✅ LlamaCpp Qwen Service initialized")
                            # Use LlamaCpp Qwen as the primary model client
                            model_client = self.app_context.llamacpp_qwen_service
                        else:
                            self.logger.error("❌ LlamaCpp Qwen Service initialization failed")
                            model_client = ModelClient(app_context=self.app_context)
                    except Exception as e:
                        self.logger.error(f"❌ LlamaCpp Qwen Service error: {e}")
                        model_client = ModelClient(app_context=self.app_context)
                else:
                    # Initialize Model Client with proper parameters
                    model_client = ModelClient(app_context=self.app_context)
                    self.logger.info("✅ Standard Model Client initialized")
            
            # Note: ModelClient doesn't have initialize method, it's ready after construction
            self.app_context.model_client = model_client
            self.logger.info("✅ Model Client initialized")
            
            # Initialize RAG Service (Qdrant connection)
            try:
                from services.lmstudio_qdrant_rag_service import LMStudioQdrantRAGService
                rag_service = LMStudioQdrantRAGService(self.app_context.global_settings)
                self.app_context.rag_service_instance = rag_service
                self.logger.info("✅ LM Studio + Qdrant RAG Service initialized")
            except Exception as e:
                self.logger.error(f"❌ RAG Service initialization failed: {e}")
                rag_service = None
            
            # Initialize faster-whisper STT Service with maximum accuracy
            self.app_context.faster_whisper_stt_service = FasterWhisperSTTService(
                self.app_context, 
                model_size="large",  # Upgraded to 'large' for maximum accuracy (RTX 4070 SUPER has 12GB VRAM)
                device="auto"        # Auto-detect GPU/CPU with optimal settings
            )
            if await asyncio.get_event_loop().run_in_executor(None, self.app_context.faster_whisper_stt_service.initialize):
                self.logger.info("✅ faster-whisper STT Service initialized")
            else:
                self.logger.error("❌ faster-whisper STT Service initialization failed")
            
            # Initialize Simple Voice Receiver for Discord voice capture
            self.app_context.simple_voice_receiver = SimpleVoiceReceiver(
                self.app_context, 
                speech_callback=self.process_simple_discord_audio
            )
            if self.app_context.simple_voice_receiver.initialize():
                self.logger.info("✅ Simple Voice Receiver initialized")
            else:
                self.logger.error("❌ Simple Voice Receiver initialization failed")
            
            # Initialize VAD Voice Receiver (improved version)
            self.app_context.vad_voice_receiver = VADVoiceReceiver(
                self.app_context,
                speech_callback=self.process_vad_transcription
            )
            if await self.app_context.vad_voice_receiver.initialize():
                self.logger.info("✅ VAD Voice Receiver initialized")
            else:
                self.logger.error("❌ VAD Voice Receiver initialization failed")
            
            # Initialize Offline VAD Voice Receiver (100% local processing)
            if OFFLINE_VOICE_AVAILABLE and OfflineVADVoiceReceiver:
                self.app_context.offline_vad_voice_receiver = OfflineVADVoiceReceiver(
                    self.app_context,
                    speech_callback=self.process_offline_voice_response
                )
                if await self.app_context.offline_vad_voice_receiver.initialize():
                    self.logger.info("✅ Offline VAD Voice Receiver initialized (100% local)")
                else:
                    self.logger.error("❌ Offline VAD Voice Receiver initialization failed")
            else:
                self.logger.info("ℹ️ Offline voice receiver not available (missing dependencies)")
            
            # Initialize LLM Service with required parameters
            self.llm_service = LLMService(
                app_context=self.app_context,
                audio_service=None,  # Not needed for voice bot
                rag_service=rag_service,    # Use the initialized RAG service
                model_client=model_client
            )
            # Note: LLMService doesn't have initialize method, it's ready after construction
            self.app_context.llm_service = self.llm_service
            self.logger.info("✅ LLM Service initialized")
            
            # Initialize Short-Term Memory Service
            self.app_context.short_term_memory_service = ShortTermMemoryService(self.app_context)
            self.logger.info("✅ Short-Term Memory Service initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")

    def add_commands(self):
        """Add Discord commands."""
        
        @self.command(name='join')
        async def join_command(ctx):
            """Join voice channel and start Windows audio capture for STT."""
            voice_client = None
            try:
                if not ctx.author.voice:
                    await ctx.send("❌ You need to be in a voice channel!")
                    return

                channel = ctx.author.voice.channel
                self.logger.info(f"📞 !join used by {ctx.author.name}")
                self.logger.info(f"Target channel: {channel.name}")
                self.logger.info(f"Attempting connection to {channel.name}")

                # Connect with py-cord's native VoiceClient
                voice_client = await channel.connect()
                self.logger.info(f"Successfully connected to {channel.name} with py-cord VoiceClient")
                
                # Try to use Real-Time Voice Service if available
                if (hasattr(self.app_context, 'realtime_voice_service') and 
                    self.app_context.realtime_voice_service):
                    
                    self.logger.info("🚀 Attempting to use Real-Time Voice Service...")
                    if await self.app_context.realtime_voice_service.connect_to_voice_channel(voice_client):
                        await ctx.send(f"✅ **Connected to {channel.name}**\n"
                                     f"⚡ **Real-Time Voice Processing Active**\n"
                                     f"🎤 **Direct Discord Audio Capture**\n"
                                     f"🎯 **Neuro-sama Style Pipeline**\n"
                                     f"💬 **Speak and I'll respond instantly!**")
                        self.logger.info("✅ Real-Time Voice Service connected successfully")
                        return
                    else:
                        self.logger.warning("⚠️ Real-Time Voice Service connection failed, falling back to VB-Audio")
                else:
                    self.logger.info("ℹ️ Real-Time Voice Service not available, using VB-Audio")
                
                # Start Windows audio capture for STT instead of Discord voice
                if not self.virtual_audio:
                    self.virtual_audio = WhisperAudioCapture(self.app_context, self.process_virtual_audio_sync)
                    # Initialize faster-whisper for STT
                    if hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service:
                        self.logger.info("✅ Using faster-whisper for STT")
                    else:
                        self.logger.warning("⚠️ faster-whisper STT service not available")

                # Use the working VB-Audio device (device 8) to avoid DirectSound errors
                working_device_id = 8  # CABLE Output (VB-Audio Virtual Cable) - confirmed working
                device_selected = False
                
                try:
                    # Test if the working device is available
                    if VIRTUAL_AUDIO_AVAILABLE and sd:
                        devices = sd.query_devices()
                        if working_device_id < len(devices):
                            device_info = sd.query_devices(working_device_id, 'input')
                            if hasattr(device_info, 'name'):
                                device_name = device_info.name
                            elif isinstance(device_info, dict):
                                device_name = device_info.get('name', f'Device {working_device_id}')
                            else:
                                device_name = f'Device {working_device_id}'
                            
                            if self.virtual_audio.select_input_device(working_device_id):
                                device_selected = True
                                self.logger.info(f"🎯 Using working VB-Audio device: {device_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not use working VB-Audio device 8: {e}")

                # Fallback to auto-detection if working VB-Audio device 8 not available
                if not device_selected and VIRTUAL_AUDIO_AVAILABLE and self.virtual_audio:
                    virtual_devices = self.virtual_audio.list_audio_devices()
                    if virtual_devices:
                        # Use first available virtual audio device
                        device_id = virtual_devices[0][0]
                        if self.virtual_audio.select_input_device(device_id):
                            device_selected = True
                            device_name = virtual_devices[0][1]
                            self.logger.info(f"🎯 Auto-selected virtual audio device: {device_name}")

                # Start recording if device selected
                if device_selected and self.virtual_audio.start_recording():
                    await ctx.send(f"✅ **Connected to {channel.name}**\n"
                                 f"🎤 **Windows Audio Capture Active**\n"
                                 f"🎯 **faster-whisper STT Ready**\n"
                                 f"🔊 **Listening to VB-Audio Point (avoiding DirectSound)**\n"
                                 f"💬 **Speak and I'll respond in text and voice!**\n"
                                 f"💡 **Make sure audio is playing through VB-Audio Point**")
                    self.logger.info("✅ Windows audio capture started successfully")
                else:
                    await ctx.send(f"⚠️ **Connected to {channel.name}** but audio capture failed\n"
                                 f"❌ **Could not start Windows audio capture**\n"
                                 f"🔧 **Install VB-Audio Point: https://vb-audio.com/Voicemeeter/index.htm**\n"
                                 f"🔧 **Or check your audio devices with `!virtual list`**")
                    self.logger.warning("⚠️ Failed to start Windows audio capture")

            except Exception as e:
                self.logger.error(f"❌ Error in join command: {e}")
                await ctx.send(f"❌ Error joining voice channel: {e}")
                if voice_client:
                    await voice_client.disconnect()

        @self.command(name='leave')
        async def leave_command(ctx):
            """Stop recording and leave voice channel."""
            self.logger.info(f"🛑 !leave used by {ctx.author.name}")
            
            if ctx.guild.id in self.connections:
                # Cancel recording task
                if ctx.guild.id in self.recording_tasks:
                    self.recording_tasks[ctx.guild.id].cancel()
                    del self.recording_tasks[ctx.guild.id]
                
                # Disconnect
                vc = self.connections[ctx.guild.id]
                await vc.disconnect()
                del self.connections[ctx.guild.id]
                
                await ctx.send("👋 **Disconnected from voice channel.**")
                self.logger.info("✅ Successfully disconnected from voice")
            else:
                await ctx.send("❌ I am currently not connected here.")

        @self.command(name='test')
        async def test_command(ctx):
            """Test command to verify bot is working."""
            self.logger.info(f"🧪 !test used by {ctx.author.name}")
            await ctx.send("🧪 **DanzarAI Voice Bot with VAD is working!** Use `!join` to start natural conversation.")

        @self.command(name='status')
        async def status_command(ctx):
            """Show bot status."""
            self.logger.info(f"!status used by {ctx.author.name}")
            
            status_msg = "🤖 **DanzarAI VAD Status:**\n"
            
            if self.use_virtual_audio:
                status_msg += "🎵 **Mode: Virtual Audio Capture**\n"
                if self.virtual_audio and self.virtual_audio.is_recording:
                    status_msg += "🎙️ Virtual Audio Recording: **Active**\n"
                    if self.virtual_audio.input_device is not None:
                        status_msg += f"🎯 Input Device: **{self.virtual_audio.input_device}**\n"
                else:
                    status_msg += "🔇 Virtual Audio Recording: **Inactive**\n"
            else:
                status_msg += "🎵 **Mode: Discord Voice Capture**\n"
                if ctx.guild.id in self.connections:
                    vc = self.connections[ctx.guild.id]
                    if vc.is_connected():
                        channel_name = getattr(vc.channel, 'name', 'Unknown Channel')
                        status_msg += f"🔊 Connected to: **{channel_name}**\n"
                        status_msg += f"🎙️ VAD Recording: **Active**\n"
                    else:
                        status_msg += "🔇 Voice connection lost\n"
                else:
                    status_msg += "🔇 Not connected to voice\n"
            
            status_msg += f"🧠 Whisper Model: **{'Loaded' if self.whisper_model else 'Not Loaded'}**\n"
            status_msg += f"🎯 faster-whisper STT: **{'Ready' if hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service else 'Not Ready'}**\n"
            status_msg += f"🎯 VAD Model: **{'Loaded' if hasattr(self.vad, 'model') and self.vad.model else 'Not Loaded'}**\n"
            status_msg += f"🤖 LLM Service: **{'Ready' if self.llm_service else 'Not Ready'}**\n"
            status_msg += f"🔊 TTS Service: **{'Ready' if self.tts_service else 'Not Ready'}**\n"
            status_msg += f"🎯 Guilds: **{len(self.guilds)}**\n"
            status_msg += f"🎵 Virtual Audio Available: **{'Yes' if VIRTUAL_AUDIO_AVAILABLE else 'No'}**\n"
            
            await ctx.send(status_msg)

        @self.command(name='offline')
        async def offline_command(ctx):
            """Show offline voice processing capabilities."""
            self.logger.info(f"!offline used by {ctx.author.name}")
            
            status_msg = "🔒 **DanzarAI Offline Voice Processing:**\n\n"
            
            if OFFLINE_VOICE_AVAILABLE:
                status_msg += "✅ **Offline Voice Available**\n"
                status_msg += "🎤 **STT**: Vosk (offline Kaldi recognizer)\n"
                status_msg += "🧠 **LLM**: Local Transformers model\n"
                status_msg += "🔊 **TTS**: Silero (offline neural TTS)\n"
                status_msg += "📡 **Voice Receive**: discord-ext-voice-recv\n\n"
                
                if (hasattr(self.app_context, 'offline_vad_voice_receiver') and 
                    self.app_context.offline_vad_voice_receiver):
                    status_msg += "🎯 **Status**: Initialized and ready\n"
                    status_msg += "🔒 **Privacy**: 100% local - no data leaves your machine\n"
                    status_msg += "⚡ **Performance**: GPU accelerated if available\n\n"
                    status_msg += "💡 **Use `!join` to start offline voice processing**"
                else:
                    status_msg += "⚠️ **Status**: Not initialized\n"
            else:
                status_msg += "❌ **Offline Voice Not Available**\n"
                status_msg += "📦 **Missing**: discord-ext-voice-recv extension\n"
                status_msg += "🔧 **Install**: `pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git`\n"
                status_msg += "📚 **Docs**: [AssemblyAI Discord Voice Bot Tutorial](https://assemblyai.com/blog/build-a-discord-voice-bot-to-add-chatgpt-to-your-voice-channel)"
            
            await ctx.send(status_msg)

        @self.command(name='virtual')
        async def virtual_command(ctx, action: str = "status", device_id: Optional[int] = None):
            """Virtual audio commands: start, stop, list, status."""
            self.logger.info(f"!virtual {action} used by {ctx.author.name}")
            
            if not VIRTUAL_AUDIO_AVAILABLE:
                await ctx.send("❌ **Virtual audio not available!** Install sounddevice: `pip install sounddevice`")
                return
            
            if action.lower() == "list":
                # List available devices
                if not self.virtual_audio:
                    self.virtual_audio = WhisperAudioCapture(self.app_context, self.process_virtual_audio_sync)
                
                virtual_devices = self.virtual_audio.list_audio_devices()
                
                if virtual_devices:
                    device_list = "🎵 **Virtual Audio Devices:**\n"
                    for device_id, device_name in virtual_devices:
                        device_list += f"  **{device_id}**: {device_name}\n"
                    device_list += "\nUse `!virtual start [device_id]` to start recording."
                    await ctx.send(device_list)
                else:
                    await ctx.send("⚠️ **No virtual audio devices found!**\n"
                                  "Install VB-Cable or enable Windows Stereo Mix.")
            
            elif action.lower() == "start":
                # Start virtual audio recording
                if not self.virtual_audio:
                    self.virtual_audio = WhisperAudioCapture(self.app_context, self.process_virtual_audio_sync)
                    # Initialize Whisper
                    await self.virtual_audio.initialize_whisper("base")
                
                if self.virtual_audio.is_recording:
                    await ctx.send("⚠️ **Virtual audio recording already active!**")
                    return
                
                # Select device
                if not self.virtual_audio.select_input_device(device_id):
                    await ctx.send("❌ **Failed to select audio device!** Use `!virtual list` to see available devices.")
                    return
                
                # Start recording
                if self.virtual_audio.start_recording():
                    device_name = f"Device {self.virtual_audio.input_device}" if self.virtual_audio.input_device is not None else "Unknown"
                    await ctx.send(f"✅ **Virtual audio recording started!**\n"
                                  f"🎯 **Device**: {device_name}\n"
                                  f"🎤 **I'm listening for speech from virtual audio cables!**\n"
                                  f"🛑 **Use `!virtual stop` when done.**")
                else:
                    await ctx.send("❌ **Failed to start virtual audio recording!**")
            
            elif action.lower() == "stop":
                # Stop virtual audio recording
                if self.virtual_audio and self.virtual_audio.is_recording:
                    self.virtual_audio.stop_recording()
                    await ctx.send("🛑 **Virtual audio recording stopped.**")
                else:
                    await ctx.send("⚠️ **Virtual audio recording not active.**")
            
            elif action.lower() == "status":
                # Show virtual audio status
                if self.virtual_audio:
                    if self.virtual_audio.is_recording:
                        device_name = f"Device {self.virtual_audio.input_device}" if self.virtual_audio.input_device is not None else "Unknown"
                        await ctx.send(f"✅ **Virtual Audio Status: Active**\n"
                                      f"🎯 **Device**: {device_name}\n"
                                      f"🎤 **Whisper Model**: {'Loaded' if self.virtual_audio.whisper_model else 'Not Loaded'}")
                    else:
                        await ctx.send("🔇 **Virtual Audio Status: Inactive**")
                else:
                    await ctx.send("🔇 **Virtual Audio Status: Not Initialized**")
            
            else:
                await ctx.send("❓ **Unknown virtual audio command!**\n"
                              "Available commands: `list`, `start [device_id]`, `stop`, `status`")

        @self.command(name='memory')
        async def memory_command(ctx, action: str = "status", user: Optional[str] = None):
            """Memory management commands"""
            try:
                if action == "status":
                    if self.app_context.short_term_memory_service:
                        stats = self.app_context.short_term_memory_service.get_stats()
                        await ctx.send(f"📊 **Memory Status**\n"
                                     f"Active conversations: {stats['active_conversations']}\n"
                                     f"Total entries: {stats['total_entries']}\n"
                                     f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
                    else:
                        await ctx.send("❌ Memory service not available")
                        
                elif action == "clear":
                    if user:
                        if self.app_context.short_term_memory_service:
                            self.app_context.short_term_memory_service.clear_conversation(user)
                            await ctx.send(f"🗑️ Cleared memory for user: {user}")
                        else:
                            await ctx.send("❌ Memory service not available")
                    else:
                        await ctx.send("❌ Please specify a user: `!memory clear <username>`")
                        
                elif action == "list":
                    if self.app_context.short_term_memory_service:
                        conversations = self.app_context.short_term_memory_service.get_active_conversations()
                        if conversations:
                            conv_list = "\n".join([f"• {user} ({count} entries)" for user, count in conversations.items()])
                            await ctx.send(f"💭 **Active Conversations**\n{conv_list}")
                        else:
                            await ctx.send("📭 No active conversations")
                    else:
                        await ctx.send("❌ Memory service not available")
                        
                else:
                    await ctx.send("❌ Invalid action. Use: `status`, `clear <user>`, or `list`")
                    
            except Exception as e:
                self.logger.error(f"Memory command error: {e}")
                await ctx.send("❌ Memory command failed")

        @self.command(name='search')
        async def search_command(ctx, mode: str = "status"):
            """Toggle LLM-guided search mode"""
            try:
                if mode == "status":
                    current_mode = getattr(self, 'llm_search_mode', False)
                    status = "🔍 **ENABLED**" if current_mode else "❌ **DISABLED**"
                    await ctx.send(f"🧠 **LLM-Guided Search Mode**: {status}\n"
                                 f"When enabled, the LLM will admit when it doesn't know something\n"
                                 f"and formulate its own search queries to find the answer.\n"
                                 f"Use `!search enable` or `!search disable` to toggle.")
                    
                elif mode == "enable":
                    self.llm_search_mode = True
                    await ctx.send("🔍 **LLM-Guided Search Mode ENABLED**\n"
                                 "The bot will now admit when it doesn't know something\n"
                                 "and search for answers using its own formulated queries.")
                    
                elif mode == "disable":
                    self.llm_search_mode = False
                    await ctx.send("❌ **LLM-Guided Search Mode DISABLED**\n"
                                 "The bot will use standard RAG responses.")
                    
                else:
                    await ctx.send("❌ Invalid mode. Use: `status`, `enable`, or `disable`")
                    
            except Exception as e:
                self.logger.error(f"Search command error: {e}")
                await ctx.send("❌ Search command failed")

        @self.command(name='tts')
        async def tts_command(ctx, mode: str = "status"):
            """Control TTS settings"""
            if mode == "status":
                discord_only = self.app_context.global_settings.get('TTS_SERVER', {}).get('discord_only_mode', False)
                status = "Discord-only (local muted)" if discord_only else "Normal (local + Discord)"
                await ctx.send(f"🔊 **TTS Mode**: {status}")
                
            elif mode == "discord-only":
                # Enable Discord-only mode
                if 'TTS_SERVER' not in self.app_context.global_settings:
                    self.app_context.global_settings['TTS_SERVER'] = {}
                self.app_context.global_settings['TTS_SERVER']['discord_only_mode'] = True
                await ctx.send("🔊 **TTS Discord-Only Mode ENABLED**\\n"
                             "TTS will play through Discord voice channel only.\\n"
                             "Local audio output is muted.")
                
            elif mode == "normal":
                # Disable Discord-only mode
                if 'TTS_SERVER' not in self.app_context.global_settings:
                    self.app_context.global_settings['TTS_SERVER'] = {}
                self.app_context.global_settings['TTS_SERVER']['discord_only_mode'] = False
                await ctx.send("🔊 **TTS Normal Mode ENABLED**\\n"
                             "TTS will play through both local speakers and Discord.")
                
            else:
                await ctx.send("❌ Invalid TTS mode. Use: `!tts status`, `!tts discord-only`, or `!tts normal`")

        @self.command(name='llm')
        async def llm_command(ctx, mode: str = "status"):
            """Control LLM processing mode"""
            if mode == "status":
                tool_aware = self.app_context.global_settings.get('USE_TOOL_AWARE_LLM', False)
                smart_rag = self.app_context.global_settings.get('USE_SMART_RAG', True)
                
                current_mode = "Unknown"
                if tool_aware:
                    current_mode = "Tool-Aware (LLM decides when to use tools)"
                elif smart_rag:
                    current_mode = "Smart RAG (Always uses RAG)"
                else:
                    current_mode = "Traditional (Basic LLM only)"
                
                await ctx.send(f"🧠 **LLM Processing Mode**: {current_mode}")
                
            elif mode == "tool-aware":
                # Enable tool-aware mode
                self.app_context.global_settings['USE_TOOL_AWARE_LLM'] = True
                self.app_context.global_settings['USE_SMART_RAG'] = True  # Keep as fallback
                await ctx.send("🧠 **Tool-Aware LLM Mode ENABLED**\\n"
                             "✅ LLM will decide when to use RAG and search tools\\n"
                             "✅ Simple greetings will bypass RAG completely\\n"
                             "✅ Game questions will trigger knowledge base search")
                
            elif mode == "smart-rag":
                # Enable Smart RAG mode (disable tool-aware)
                self.app_context.global_settings['USE_TOOL_AWARE_LLM'] = False
                self.app_context.global_settings['USE_SMART_RAG'] = True
                await ctx.send("🧠 **Smart RAG Mode ENABLED**\\n"
                             "✅ Uses intent classification for RAG decisions\\n"
                             "⚠️ May still search for simple greetings")
                
            elif mode == "traditional":
                # Disable both advanced modes
                self.app_context.global_settings['USE_TOOL_AWARE_LLM'] = False
                self.app_context.global_settings['USE_SMART_RAG'] = False
                await ctx.send("🧠 **Traditional LLM Mode ENABLED**\\n"
                             "✅ Basic LLM responses only\\n"
                             "❌ No RAG or knowledge base access")
                
            else:
                await ctx.send("❌ **Invalid mode**. Use: `!llm status`, `!llm tool-aware`, `!llm smart-rag`, or `!llm traditional`")

        @self.command(name='whisper')
        async def whisper_command(ctx, action: str = "status", model: str = None):
            """Control Whisper model settings for STT accuracy"""
            try:
                if action == "status":
                    if (hasattr(self.app_context, 'faster_whisper_stt_service') and 
                        self.app_context.faster_whisper_stt_service):
                        current_model = self.app_context.faster_whisper_stt_service.model_size
                        device = self.app_context.faster_whisper_stt_service.device
                        
                        # Check if GPU is available
                        gpu_available = "Unknown"
                        try:
                            import torch
                            gpu_available = "Yes" if torch.cuda.is_available() else "No"
                        except ImportError:
                            gpu_available = "No (PyTorch not available)"
                        
                        await ctx.send(f"🎤 **Whisper STT Status**\n"
                                     f"**Current Model**: {current_model}\n"
                                     f"**Device**: {device}\n"
                                     f"**GPU Available**: {gpu_available}\n"
                                     f"**Available Models**: tiny, base, small, medium, large\n"
                                     f"**Accuracy**: tiny < base < small < medium < large\n"
                                     f"**Speed**: large < medium < small < base < tiny\n\n"
                                     f"Use `!whisper set <model>` to change model")
                    else:
                        await ctx.send("❌ Whisper STT service not available")
                        
                elif action == "set":
                    if not model:
                        await ctx.send("❌ Please specify a model: `!whisper set <model>`\n"
                                     "Available: tiny, base, small, medium, large")
                        return
                        
                    valid_models = ["tiny", "base", "small", "medium", "large"]
                    if model not in valid_models:
                        await ctx.send(f"❌ Invalid model '{model}'\n"
                                     f"Available: {', '.join(valid_models)}")
                        return
                    
                    # Reinitialize the STT service with new model
                    await ctx.send(f"🔄 **Switching to Whisper '{model}' model...**\n"
                                 "This may take a moment to download and load.")
                    
                    try:
                        # Create new STT service with the requested model
                        new_stt_service = FasterWhisperSTTService(
                            self.app_context, 
                            model_size=model,
                            device="auto"
                        )
                        
                        # Initialize in executor to avoid blocking
                        success = await asyncio.get_event_loop().run_in_executor(
                            None, new_stt_service.initialize
                        )
                        
                        if success:
                            # Replace the old service
                            if (hasattr(self.app_context, 'faster_whisper_stt_service') and 
                                self.app_context.faster_whisper_stt_service):
                                self.app_context.faster_whisper_stt_service.cleanup()
                            
                            self.app_context.faster_whisper_stt_service = new_stt_service
                            
                            await ctx.send(f"✅ **Successfully switched to Whisper '{model}' model**\n"
                                         f"🎯 **STT accuracy updated**\n"
                                         f"💡 **Try speaking to test the new model**")
                            self.logger.info(f"✅ Whisper model switched to '{model}' by {ctx.author.name}")
                        else:
                            await ctx.send(f"❌ **Failed to load Whisper '{model}' model**\n"
                                         "Check logs for details. Keeping current model.")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error switching Whisper model: {e}")
                        await ctx.send(f"❌ **Error switching to '{model}' model**: {str(e)}")
                        
                elif action == "benchmark":
                    await ctx.send("🧪 **Whisper Model Benchmark**\n"
                                 "**tiny**: ~32x realtime, lowest accuracy, 39 MB\n"
                                 "**base**: ~16x realtime, good accuracy, 74 MB\n"
                                 "**small**: ~6x realtime, better accuracy, 244 MB\n"
                                 "**medium**: ~2x realtime, high accuracy, 769 MB\n"
                                 "**large**: ~1x realtime, highest accuracy, 1550 MB\n\n"
                                 "💡 **Recommendation**: 'medium' for best balance of speed/accuracy")
                    
                else:
                    await ctx.send("❌ **Invalid action**. Use: `!whisper status`, `!whisper set <model>`, or `!whisper benchmark`")
                    
            except Exception as e:
                self.logger.error(f"Whisper command error: {e}")
                await ctx.send("❌ Whisper command failed")

        @self.command(name='cleanup')
        async def cleanup_command(ctx, action: str = "status"):
            """Knowledge base cleanup commands"""
            try:
                if action == "status":
                    if self.llm_service and hasattr(self.llm_service, 'cleanup_enabled'):
                        cleanup_enabled = self.llm_service.cleanup_enabled
                        cleanup_interval = self.llm_service.cleanup_interval_hours
                        retention_policy = self.app_context.global_settings.get("RAG_RETENTION_POLICY", {})
                        
                        cleanup_after_days = retention_policy.get("cleanup_after_days", 7)
                        permanent_collections = retention_policy.get("permanent_collections", [])
                        permanent_keywords = retention_policy.get("permanent_keywords", [])
                        
                        status = "🧹 **ENABLED**" if cleanup_enabled else "❌ **DISABLED**"
                        await ctx.send(f"🧹 **Intelligent Knowledge Base Cleanup**: {status}\n"
                                     f"**Cleanup Interval**: Every {cleanup_interval} hours\n"
                                     f"**Retention Policy**: Smart preservation (keeps valuable knowledge forever)\n"
                                     f"**Permanent Collections**: {', '.join(permanent_collections[:3])}{'...' if len(permanent_collections) > 3 else ''}\n"
                                     f"**Cleanup Only**: Low-value entries after {cleanup_after_days} days\n"
                                     f"**Protected Keywords**: {', '.join(permanent_keywords[:3])}{'...' if len(permanent_keywords) > 3 else ''}\n\n"
                                     f"🛡️ **Game guides, strategies, and detailed content are preserved forever**\n"
                                     f"🗑️ **Only errors, debug info, and short entries are cleaned up**\n\n"
                                     f"Use `!cleanup run` to manually trigger cleanup now")
                    else:
                        await ctx.send("❌ Cleanup system not available (LLM service not initialized)")
                        
                elif action == "run":
                    if self.llm_service and hasattr(self.llm_service, '_perform_knowledge_base_cleanup'):
                        await ctx.send("🧹 **Starting manual knowledge base cleanup...**\n"
                                     "This may take a few minutes depending on database size.")
                        
                        try:
                            # Run cleanup in background to avoid blocking Discord
                            cleanup_task = asyncio.create_task(
                                self.llm_service._perform_knowledge_base_cleanup()
                            )
                            
                            # Wait with timeout
                            await asyncio.wait_for(cleanup_task, timeout=300)  # 5 minute timeout
                            
                            await ctx.send("✅ **Knowledge base cleanup completed successfully!**\n"
                                         "🎯 **Performance improvements**: Duplicates removed, data consolidated\n"
                                         "📊 **Check logs for detailed cleanup statistics**")
                            
                        except asyncio.TimeoutError:
                            await ctx.send("⏰ **Cleanup is taking longer than expected**\n"
                                         "The process is still running in the background.\n"
                                         "Check logs for progress updates.")
                        except Exception as e:
                            self.logger.error(f"Manual cleanup error: {e}")
                            await ctx.send(f"❌ **Cleanup failed**: {str(e)}")
                    else:
                        await ctx.send("❌ Manual cleanup not available (cleanup system not initialized)")
                        
                elif action == "info":
                    await ctx.send("🧹 **Intelligent Knowledge Base Cleanup System**\n\n"
                                 "**What it does:**\n"
                                 "• 🔍 **Removes duplicates** - Eliminates redundant entries using similarity analysis\n"
                                 "• 🛡️ **Smart retention** - Preserves valuable knowledge forever, cleans up junk\n"
                                 "• 🔄 **Consolidates similar entries** - Merges related information for efficiency\n"
                                 "• 🗑️ **Removes low-quality data** - Filters out errors, debug info, and meaningless content\n\n"
                                 "**🛡️ What's Protected Forever:**\n"
                                 "• 📚 **Game guides & strategies** - EverQuest, WoW, Rimworld content\n"
                                 "• 📖 **Detailed explanations** - Tutorials, comprehensive answers (>100 chars)\n"
                                 "• 🎯 **Valuable searches** - Quest info, class guides, spell details\n"
                                 "• 🏆 **High-quality content** - Anything with detailed, comprehensive information\n\n"
                                 "**🗑️ What Gets Cleaned Up (after 7 days):**\n"
                                 "• ❌ **Error messages** - Failed requests, timeouts, debug info\n"
                                 "• 📝 **Short entries** - Brief, low-value responses (<100 chars)\n"
                                 "• 🔧 **Test data** - Temporary entries and debugging content\n\n"
                                 "**Benefits:**\n"
                                 "• ⚡ **Faster retrieval** - Cleaner database, better search performance\n"
                                 "• 🎯 **Better accuracy** - High-quality knowledge base with relevant results\n"
                                 "• 🧠 **Permanent learning** - Valuable game knowledge preserved forever\n"
                                 "• 💾 **Efficient storage** - Only junk gets removed, good stuff stays")
                        
                else:
                    await ctx.send("❌ **Invalid action**. Use: `!cleanup status`, `!cleanup run`, or `!cleanup info`")
                    
            except Exception as e:
                self.logger.error(f"Cleanup command error: {e}")
                await ctx.send("❌ Cleanup command failed")

    def get_configured_text_channel(self):
        """Get the configured text channel from settings."""
        try:
            text_channel_id = self.settings.get('DISCORD_TEXT_CHANNEL_ID')
            if text_channel_id:
                channel = self.get_channel(int(text_channel_id))
                if channel:
                    self.logger.info(f"✅ Using configured text channel: {channel.name} (ID: {text_channel_id})")
                    return channel
                else:
                    self.logger.warning(f"⚠️ Configured text channel ID {text_channel_id} not found")
            else:
                self.logger.warning("⚠️ DISCORD_TEXT_CHANNEL_ID not configured")
            
            # Fallback to first available text channel
            for guild in self.guilds:
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).send_messages:
                        self.logger.info(f"🔄 Fallback to text channel: {channel.name} (ID: {channel.id})")
                        return channel
            
            return None
        except Exception as e:
            self.logger.error(f"❌ Error getting configured text channel: {e}")
            return None

    async def setup_hook(self):
        """Setup hook called after login but before connecting to gateway."""
        self.logger.info("Loading Whisper and VAD models in setup_hook...")
        try:
            # Check if Qwen2.5-Omni can handle audio directly
            qwen_config = self.settings.get('QWEN_OMNI_SERVICE', {})
            if qwen_config.get('enabled') and qwen_config.get('skip_whisper'):
                self.logger.info("🎵 Qwen2.5-Omni will handle audio directly - skipping Whisper initialization")
                self.whisper_model = None
                self.use_omni_for_audio = True
            elif WHISPER_AVAILABLE and whisper:
                # Use the configured model size from settings
                model_size = self.settings.get('WHISPER_MODEL_SIZE', 'base.en')
                self.whisper_model = await asyncio.get_running_loop().run_in_executor(
                    None, 
                    whisper.load_model, 
                    model_size
                )
                self.logger.info(f"✅ Whisper model '{model_size}' loaded successfully")
                self.use_omni_for_audio = False
            else:
                self.logger.info("ℹ️ Standard Whisper not available, using faster-whisper instead")
                self.whisper_model = None
                self.use_omni_for_audio = False
            
            # Initialize VAD
            vad_success = await self.vad.initialize()
            if not vad_success:
                self.logger.warning("VAD initialization failed, falling back to basic recording")
                
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self.whisper_model = None

    def _check_windows_audio_optimization(self):
        """Check and suggest Windows audio optimizations for Discord processing."""
        try:
            if os.name == 'nt':  # Windows only
                self.logger.info("🔧 Checking Windows audio system optimization...")
                
                # Check if audiodg.exe process exists and suggest optimization
                try:
                    import subprocess
                    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq audiodg.exe'], 
                                          capture_output=True, text=True)
                    if 'audiodg.exe' in result.stdout:
                        self.logger.info("🎵 Windows Audio Device Graph service detected")
                        self.logger.info("💡 TIP: For better Discord audio quality, consider running this PowerShell command as Administrator:")
                        self.logger.info("💡      Get-Process audiodg | Set-Process -Priority High")
                        self.logger.info("💡 This can help resolve Discord audio processing issues")
                    else:
                        self.logger.warning("⚠️ Windows Audio Device Graph service not found")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not check audiodg.exe status: {e}")
                    
                # Suggest other Windows audio optimizations
                self.logger.info("🎵 Additional Windows audio tips:")
                self.logger.info("   • Disable Windows audio enhancements in Sound settings")
                self.logger.info("   • Set Discord input/output to 48000 Hz, 16-bit")
                self.logger.info("   • Close other audio applications (Spotify, etc.)")
                self.logger.info("   • Check microphone levels in Windows Sound settings")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Audio optimization check failed: {e}")

    async def on_ready(self):
        """Called when bot is ready."""
        self.logger.info(f"DanzarAI Voice Bot ready as {self.user}")
        self.logger.info(f"Available commands: {', '.join([f'!{cmd.name}' for cmd in self.commands])}")
        self.logger.info("🧹 New: !cleanup - Manage knowledge base cleanup and optimization")
        self.logger.info(f"Connected to {len(self.guilds)} guild(s)")
        
        # Check and optimize Windows audio system if possible
        self._check_windows_audio_optimization()
        
        # Initialize services after bot is ready
        await self.initialize_services()
        
        # Start transcription queue processor
        asyncio.create_task(self.process_transcription_queue())
        
        # Load models if not already loaded (backup to setup_hook)
        if not hasattr(self, 'whisper_model') or not self.whisper_model:
            self.logger.info("Loading Whisper and VAD models in on_ready...")
            try:
                # Check if Qwen2.5-Omni can handle audio directly
                qwen_config = self.settings.get('QWEN_OMNI_SERVICE', {})
                if qwen_config.get('enabled') and qwen_config.get('skip_whisper'):
                    self.logger.info("🎵 Qwen2.5-Omni will handle audio directly - skipping Whisper initialization in on_ready")
                    self.whisper_model = None
                    self.use_omni_for_audio = True
                elif WHISPER_AVAILABLE and whisper:
                    # Use the configured model size from settings
                    model_size = self.settings.get('WHISPER_MODEL_SIZE', 'base.en')
                    self.whisper_model = await asyncio.get_running_loop().run_in_executor(
                        None, 
                        whisper.load_model, 
                        model_size
                    )
                    self.logger.info(f"Whisper model '{model_size}' loaded successfully in on_ready")
                    self.use_omni_for_audio = False
                else:
                    self.logger.info("ℹ️ Standard Whisper not available, using faster-whisper instead")
                    self.whisper_model = None
                    self.use_omni_for_audio = False
                
                # Initialize VAD
                vad_success = await self.vad.initialize()
                if vad_success:
                    self.logger.info("VAD model initialized successfully in on_ready")
                else:
                    self.logger.warning("VAD initialization failed in on_ready")
                    
            except Exception as e:
                self.logger.error(f"Failed to load models in on_ready: {e}")
                self.whisper_model = None
        
        # Auto-join voice channel if configured
        await self._auto_join_voice_channel()

    async def _auto_join_voice_channel(self):
        """Automatically join the configured voice channel on startup"""
        try:
            voice_channel_id = self.settings.get('DISCORD_VOICE_CHANNEL_ID')
            if not voice_channel_id:
                self.logger.info("🔇 No voice channel configured for auto-join")
                return
            
            # Find the voice channel
            voice_channel = self.get_channel(int(voice_channel_id))
            if not voice_channel:
                self.logger.error(f"❌ Voice channel {voice_channel_id} not found")
                return
            
            if not hasattr(voice_channel, 'connect'):
                self.logger.error(f"❌ Channel {voice_channel_id} is not a voice channel")
                return
            
            self.logger.info(f"🎤 Auto-joining voice channel: {voice_channel.name}")
            
            # Connect to voice channel
            voice_client = await voice_channel.connect()
            self.logger.info(f"✅ Successfully connected to {voice_channel.name}")
            
            # Start Windows audio capture for STT
            if not self.virtual_audio:
                self.virtual_audio = WhisperAudioCapture(self.app_context, self.process_virtual_audio_sync)
                
                # Check if faster-whisper STT service is available
                if hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service:
                    self.logger.info("✅ Using faster-whisper for STT")
                else:
                    self.logger.warning("⚠️ faster-whisper STT service not available")

            # Use the working VB-Audio device (device 8) to avoid DirectSound errors
            working_device_id = 8  # CABLE Output (VB-Audio Virtual Cable) - confirmed working
            device_selected = False
            
            try:
                # Test if the working device is available
                if VIRTUAL_AUDIO_AVAILABLE and sd:
                    devices = sd.query_devices()
                    if working_device_id < len(devices):
                        device_info = sd.query_devices(working_device_id, 'input')
                        if hasattr(device_info, 'name'):
                            device_name = device_info.name
                        elif isinstance(device_info, dict):
                            device_name = device_info.get('name', f'Device {working_device_id}')
                        else:
                            device_name = f'Device {working_device_id}'
                        
                        if self.virtual_audio.select_input_device(working_device_id):
                            device_selected = True
                            self.logger.info(f"🎯 Using working VB-Audio device: {device_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not use working VB-Audio device 8: {e}")

            # Fallback to auto-detection if working VB-Audio device 8 not available
            if not device_selected and VIRTUAL_AUDIO_AVAILABLE and self.virtual_audio:
                virtual_devices = self.virtual_audio.list_audio_devices()
                if virtual_devices:
                    # Use first available virtual audio device
                    device_id = virtual_devices[0][0]
                    if self.virtual_audio.select_input_device(device_id):
                        device_selected = True
                        device_name = virtual_devices[0][1]
                        self.logger.info(f"🎯 Auto-selected virtual audio device: {device_name}")

            # Start recording if device selected
            if device_selected and self.virtual_audio.start_recording():
                self.logger.info("✅ Windows audio capture started successfully")
                
                # Get configured text channel for status message
                text_channel = self.get_configured_text_channel()
                if text_channel:
                    await text_channel.send(f"🤖 **DanzarAI Auto-Started**\n"
                                          f"✅ **Connected to {voice_channel.name}**\n"
                                          f"🎤 **Windows Audio Capture Active**\n"
                                          f"🎯 **faster-whisper STT Ready**\n"
                                          f"🔊 **Listening to VB-Audio Point**\n"
                                          f"💬 **Speak and I'll respond in text and voice!**")
            else:
                self.logger.warning("⚠️ Failed to start Windows audio capture")
                
                # Get configured text channel for status message
                text_channel = self.get_configured_text_channel()
                if text_channel:
                    await text_channel.send(f"⚠️ **DanzarAI Connected with Issues**\n"
                                          f"✅ **Connected to {voice_channel.name}**\n"
                                          f"❌ **Audio capture failed**\n"
                                          f"🔧 **Install VB-Audio Point or check audio devices**\n"
                                          f"💡 **Use `!virtual list` to see available devices**")
            
        except Exception as e:
            self.logger.error(f"❌ Auto-join failed: {e}")
            
            # Try to send error message to text channel
            try:
                text_channel = self.get_configured_text_channel()
                if text_channel:
                    await text_channel.send(f"❌ **DanzarAI Auto-Join Failed**\n"
                                          f"Error: {str(e)}\n"
                                          f"💡 **Use `!join` to manually connect**")
            except:
                pass  # Don't fail if we can't send the error message

    async def on_command_error(self, context, exception):
        """Handle command errors."""
        if isinstance(exception, commands.CommandNotFound):
            self.logger.warning(f"❓ Unknown command: {context.message.content}")
            await context.send(f"❓ Unknown command. Available commands: `!join`, `!leave`, `!test`, `!status`")
        else:
            self.logger.error(f"❌ Command error: {exception}")
            await context.send(f"❌ An error occurred: {str(exception)}")

    # Raw voice recording is now handled by the RawAudioReceiver class
    # No need for separate VAD recording method

    def process_virtual_audio_sync(self, audio_data: np.ndarray):
        """Process audio from Whisper audio capture - synchronous version for threading."""
        try:
            self.logger.info("🎵 Processing Whisper audio segment")
            
            # This method is called from the audio worker thread
            # It should NOT make any async Discord API calls
            # Instead, it puts results in the queue for the main Discord bot to process
            
            # The actual transcription is already handled in the worker thread
            # This method is just a placeholder for compatibility
            self.logger.info("🎯 Audio processing delegated to worker thread queue system")
            
        except Exception as e:
            self.logger.error(f"❌ Error in virtual audio sync processing: {e}")

    async def process_simple_discord_audio(self, audio_data: np.ndarray, user_name: str):
        """Process audio from simple Discord voice capture."""
        try:
            self.logger.info(f"🎵 Processing simple Discord audio from {user_name}")
            
            # Get configured text channel
            text_channel = self.get_configured_text_channel()
            if not text_channel:
                self.logger.warning("⚠️  No configured text channel available for Discord audio responses")
                return
            
            # Process the audio segment with the existing pipeline
            await self.process_speech_segment(audio_data, text_channel, user_name)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing simple Discord audio: {e}")

    async def process_vad_transcription(self, dummy_audio: np.ndarray, user_name: str, transcription: str):
        """Process transcription from VAD voice receiver with full LLM/RAG integration."""
        try:
            self.logger.info(f"🎯 Processing VAD transcription from {user_name}: '{transcription}'")
            
            # ===== ENHANCED FEEDBACK PREVENTION CHECK =====
            # Check if we should ignore this input due to TTS playback
            should_ignore, ignore_reason = self.feedback_prevention.should_ignore_input()
            if should_ignore:
                self.logger.info(f"🛡️ Ignoring input due to feedback prevention: {ignore_reason}")
                return
            
            # Check if this transcription is likely an echo of our own TTS
            is_echo, echo_reason = self.feedback_prevention.is_likely_tts_echo(transcription)
            if is_echo:
                self.logger.info(f"🛡️ Detected TTS echo, ignoring: {echo_reason}")
                return
            
            # Additional filtering for background noise and meaningless audio
            transcription_clean = transcription.lower().strip()
            
            # Filter out very short or meaningless transcriptions
            if len(transcription_clean) < 3:
                self.logger.info(f"🛡️ Ignoring very short transcription: '{transcription}'")
                return
            
            # Filter out common background noise transcriptions
            noise_patterns = [
                "um", "uh", "ah", "oh", "mm", "hmm", "hm", "eh", "er",
                "the", "a", "an", "and", "or", "but", "so", "well",
                "music", "sound", "audio", "noise", "background"
            ]
            
            if transcription_clean in noise_patterns:
                self.logger.info(f"🛡️ Ignoring background noise pattern: '{transcription}'")
                return
            
            # Filter out repetitive patterns (likely audio artifacts)
            words = transcription_clean.split()
            if len(words) >= 2:
                # Check for word repetition
                unique_words = set(words)
                if len(unique_words) == 1:  # All words are the same
                    self.logger.info(f"🛡️ Ignoring repetitive transcription: '{transcription}'")
                    return
            
            # Only process transcriptions that seem like actual meaningful speech
            # Require at least 5 characters and not just common filler words
            if len(transcription_clean) < 5:
                self.logger.info(f"🛡️ Ignoring too short transcription: '{transcription}'")
                return
            
            # Clean up old feedback prevention entries
            self.feedback_prevention.cleanup_old_entries()
            
            # Add user input to short-term memory
            if self.app_context.short_term_memory_service:
                self.app_context.short_term_memory_service.add_entry(
                    user_name=user_name,
                    content=transcription,
                    entry_type='user_input'
                )
                self.logger.debug(f"[ShortTermMemory] Stored user input: '{transcription[:50]}...'")
            
            # Get configured text channel
            text_channel = self.get_configured_text_channel()
            if not text_channel:
                self.logger.warning("⚠️  No configured text channel available for VAD responses")
                return
            
            # Send transcription to channel
            await text_channel.send(f"🎤 **{user_name}**: {transcription}")
            
            # Process with full LLM/RAG pipeline
            if self.llm_service:
                try:
                    self.logger.info(f"🧠 Processing with full LLM/RAG pipeline...")
                    
                    # Check if LLM-guided search mode is enabled
                    if getattr(self, 'llm_search_mode', False):
                        self.logger.info("🔍 Using LLM-guided search mode")
                        response = await self.llm_service.handle_user_text_query_with_llm_search(
                            user_text=transcription,
                            user_name=user_name
                        )
                        # Extract response text if it's a tuple
                        if isinstance(response, tuple):
                            response_text, metadata = response
                            self.logger.info(f"🔍 LLM search method: {metadata.get('method', 'unknown')}")
                            response = response_text
                    else:
                        # Use the existing LLM service's comprehensive text processing
                        response = await self.llm_service.handle_user_text_query(
                            user_text=transcription,
                            user_name=user_name
                        )
                    
                    if response and len(response.strip()) > 0:
                        # Strip think tags from LLM response
                        clean_response = self._strip_think_tags(response)
                        
                        # Send LLM response
                        await text_channel.send(f"🤖 **Danzar**: {clean_response}")
                        
                        # Add bot response to short-term memory (use clean response)
                        if self.app_context.short_term_memory_service:
                            self.app_context.short_term_memory_service.add_entry(
                                user_name=user_name,
                                content=clean_response,
                                entry_type='bot_response'
                            )
                            self.logger.debug(f"[ShortTermMemory] Stored bot response: '{clean_response[:50]}...'")
                        
                        # Generate TTS if available (use clean response)
                        if self.tts_service:
                            try:
                                self.logger.info("🔊 Generating TTS response...")
                                tts_text = self._strip_markdown_for_tts(clean_response)
                                
                                # ===== START FEEDBACK PREVENTION =====
                                self.feedback_prevention.start_tts_playback(tts_text)
                                
                                # Generate TTS audio asynchronously
                                loop = asyncio.get_event_loop()
                                tts_audio = await loop.run_in_executor(
                                    None,
                                    self.tts_service.generate_audio,
                                    tts_text
                                )
                                
                                if tts_audio:
                                    self.logger.info("✅ TTS audio generated successfully")
                                    await self._play_tts_audio_with_feedback_prevention(tts_audio)
                                else:
                                    self.logger.warning("⚠️ TTS generation failed")
                                    self.feedback_prevention.stop_tts_playback()
                                    
                            except Exception as tts_e:
                                self.logger.error(f"❌ TTS error: {tts_e}")
                                # Stop feedback prevention on error
                                self.feedback_prevention.stop_tts_playback()
                    else:
                        await text_channel.send("🤖 **DanzarAI**: I heard you, but I'm not sure how to respond to that.")
                        
                except Exception as e:
                    self.logger.error(f"❌ LLM processing error: {e}")
                    await text_channel.send("🤖 **DanzarAI**: Sorry, I had trouble processing that.")
            else:
                # Fallback response if LLM not available
                response = f"Hello {user_name}! I heard you say: '{transcription}'. My LLM service isn't available right now."
                await text_channel.send(f"🤖 **DanzarAI**: {response}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing VAD transcription: {e}")

    async def process_offline_voice_response(self, transcription: str, response: str, audio_file: str, user_name: str):
        """Process complete offline voice response (STT → LLM → TTS)."""
        try:
            self.logger.info(f"🎯 Processing offline voice response from {user_name}")
            
            # Get configured text channel
            text_channel = self.get_configured_text_channel()
            if not text_channel:
                self.logger.warning("⚠️  No configured text channel available for offline responses")
                return
            
            # Send transcription and response to channel
            await text_channel.send(f"🎤 **{user_name}**: {transcription}")
            await text_channel.send(f"🤖 **DanzarAI (Offline)**: {response}")
            
            # Play TTS audio in voice channel if available
            if audio_file:
                try:
                    # Find voice connection
                    voice_client = None
                    for guild in self.guilds:
                        if guild.voice_client:
                            voice_client = guild.voice_client
                            break
                    
                    if voice_client and voice_client.is_connected():
                        # Play the generated audio
                        audio_source = discord.FFmpegPCMAudio(audio_file)
                        voice_client.play(audio_source)
                        
                        # Wait for playback to finish
                        while voice_client.is_playing():
                            await asyncio.sleep(0.1)
                        
                        self.logger.info("🔊 Offline TTS audio played successfully")
                        
                        # Clean up temporary file
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                    else:
                        self.logger.warning("⚠️  No voice connection for TTS playback")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error playing offline TTS: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error processing offline voice response: {e}")
    
    async def _on_realtime_transcription(self, transcription: str, user_id: int):
        """Callback for real-time transcription events"""
        try:
            # Get user name from Discord
            user = self.get_user(user_id)
            user_name = user.display_name if user else f"User {user_id}"
            
            # Get configured text channel
            text_channel = self.get_configured_text_channel()
            if text_channel:
                await text_channel.send(f"🎤 **{user_name}**: {transcription}")
                
            self.logger.info(f"⚡ [RealTime] Transcription: '{transcription}' from {user_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Real-time transcription callback error: {e}")
    
    async def _on_realtime_response(self, response: str, user_id: int):
        """Callback for real-time response events"""
        try:
            # Get user name from Discord
            user = self.get_user(user_id)
            user_name = user.display_name if user else f"User {user_id}"
            
            # Get configured text channel
            text_channel = self.get_configured_text_channel()
            if text_channel:
                await text_channel.send(f"🤖 **Danzar**: {response}")
                
            self.logger.info(f"⚡ [RealTime] Response: '{response[:50]}...' to {user_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Real-time response callback error: {e}")

    async def _preprocess_discord_audio(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced preprocessing specifically for Discord's compressed audio."""
        try:
            # Basic validation
            if len(audio_data) == 0:
                return None
            
            # Calculate audio metrics
            audio_duration = len(audio_data) / 48000  # Discord uses 48kHz
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Audio preprocessing - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # More lenient quality checks for Discord
            if audio_max_volume < 0.005:  # Very low threshold
                self.logger.warning(f"🔇 Audio volume too low (max: {audio_max_volume:.4f})")
                return None
            
            if audio_duration < 0.3:  # Minimum duration
                self.logger.warning(f"🔇 Audio too short ({audio_duration:.2f}s)")
                return None
            
            # Enhanced Discord audio processing
            processed_audio = audio_data.copy()
            
            try:
                # Import scipy for advanced processing
                from scipy import signal
                
                # 1. Remove DC offset (common in Discord audio)
                processed_audio = processed_audio - np.mean(processed_audio)
                
                # 2. Gentler noise gate for Discord's background noise
                noise_threshold = audio_rms * 0.15  # Less aggressive threshold
                processed_audio = np.where(
                    np.abs(processed_audio) < noise_threshold,
                    processed_audio * 0.3,  # Reduce noise by 70% (less aggressive)
                    processed_audio
                )
                
                # 3. Pre-emphasis filter to counteract Discord's compression
                pre_emphasis = 0.97
                emphasized = np.append(processed_audio[0], processed_audio[1:] - pre_emphasis * processed_audio[:-1])
                
                # 4. Band-pass filter optimized for speech (300-3400 Hz)
                try:
                    sos = signal.butter(4, [300, 3400], 'bp', fs=48000, output='sos')
                    filtered_audio = signal.sosfilt(sos, emphasized)
                    processed_audio = np.asarray(filtered_audio, dtype=np.float32)
                except Exception as filter_error:
                    self.logger.warning(f"⚠️ Filter error, using emphasized audio: {filter_error}")
                    processed_audio = emphasized.astype(np.float32)
                
                # 5. Gentler dynamic range compression
                # Lighter compression to avoid artifacts
                threshold = 0.5  # Higher threshold
                ratio = 2.0      # Lower ratio (less compression)
                above_threshold = np.abs(processed_audio) > threshold
                compressed = processed_audio.copy()
                compressed[above_threshold] = np.sign(processed_audio[above_threshold]) * (
                    threshold + (np.abs(processed_audio[above_threshold]) - threshold) / ratio
                )
                processed_audio = compressed
                
                # 6. Normalize to optimal level for STT
                if np.max(np.abs(processed_audio)) > 0:
                    processed_audio = processed_audio * (0.7 / np.max(np.abs(processed_audio)))
                
                # 7. Convert sample rate to 16kHz for STT (if needed)
                if len(processed_audio) > 0:
                    # Simple downsampling from 48kHz to 16kHz
                    target_length = int(len(processed_audio) * 16000 / 48000)
                    if target_length > 0:
                        downsampled = np.interp(
                            np.linspace(0, len(processed_audio), target_length),
                            np.arange(len(processed_audio)),
                            processed_audio
                        )
                        processed_audio = downsampled.astype(np.float32)
                
                final_rms = np.sqrt(np.mean(np.square(processed_audio)))
                self.logger.info(f"🎵 Audio enhanced - Original RMS: {audio_rms:.4f}, Final RMS: {final_rms:.4f}, Gain: {final_rms/audio_rms:.2f}x")
                
            except ImportError:
                self.logger.warning("⚠️ scipy not available, using basic processing")
                # Basic processing without scipy
                processed_audio = audio_data - np.mean(audio_data)  # Remove DC offset
                if np.max(np.abs(processed_audio)) > 0:
                    processed_audio = processed_audio * (0.7 / np.max(np.abs(processed_audio)))
                
                # Simple downsampling to 16kHz
                target_length = int(len(processed_audio) * 16000 / 48000)
                if target_length > 0:
                    processed_audio = np.interp(
                        np.linspace(0, len(processed_audio), target_length),
                        np.arange(len(processed_audio)),
                        processed_audio
                    )
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"❌ Audio preprocessing error: {e}")
            return audio_data  # Return original if preprocessing fails

    async def process_speech_segment(self, audio_data: np.ndarray, text_channel: discord.TextChannel, user_name: str):
        """Process a complete speech segment with STT → LLM → TTS pipeline."""
        try:
            self.logger.info(f"🎤 Processing speech segment from {user_name}")
            self.logger.info(f"🎵 Audio segment details: shape={audio_data.shape}, max_vol={np.max(np.abs(audio_data)):.4f}")
            
            # Step 1: STT - Convert speech to text
            self.logger.info(f"🎵 Starting transcription for {user_name}...")
            transcription = await self.transcribe_audio(audio_data)
            self.logger.info(f"🎵 Transcription completed for {user_name}: {transcription}")
            
            if transcription and len(transcription.strip()) > 0:
                self.logger.info(f"📝 Transcription from {user_name}: {transcription}")
                
                # Send transcription to channel
                await text_channel.send(f"🎤 **{user_name}**: {transcription}")
                
                # Step 2: LLM - Process with LLM service (simplified to avoid timeout issues)
                if self.llm_service:
                    try:
                        self.logger.info(f"🧠 Processing with LLM...")
                        
                        # Use a simple fallback response to avoid complex async issues
                        response = f"Hello {user_name}! I heard you say: '{transcription}'. I'm DanzarAI and I'm ready to help with your gaming!"
                        
                        # Send text response to channel
                        await text_channel.send(f"🤖 **DanzarAI**: {response}")
                        
                        # Skip TTS for now to avoid additional complexity
                        self.logger.info("🔊 TTS temporarily disabled for stability")
                            
                    except Exception as e:
                        self.logger.error(f"❌ LLM processing error: {e}")
                        await text_channel.send("🤖 **DanzarAI**: Sorry, I had trouble processing that.")
                else:
                    # Fallback response if LLM not available
                    response = f"Hello {user_name}! I heard you say: '{transcription}'. I'm DanzarAI and I'm ready to help with your gaming!"
                    await text_channel.send(f"🤖 **DanzarAI**: {response}")
                    
            else:
                self.logger.info(f"🔇 No clear speech detected from {user_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing speech segment: {e}", exc_info=True)

    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data using available STT service."""
        try:
            # Enhanced Discord audio preprocessing
            processed_audio = await self._preprocess_discord_audio(audio_data)
            if processed_audio is None:
                return None
            
            # Check if LlamaCpp Qwen can handle audio directly
            llamacpp_config = self.app_context.global_settings.get('LLAMACPP_QWEN', {})
            if (llamacpp_config.get('enabled') and llamacpp_config.get('use_for_audio') and
                hasattr(self.app_context, 'llamacpp_qwen_service') and self.app_context.llamacpp_qwen_service):
                return await self._transcribe_with_llamacpp(processed_audio)
            
            # Fallback to Qwen2.5-Omni if enabled
            elif (hasattr(self, 'use_omni_for_audio') and self.use_omni_for_audio and
                hasattr(self.app_context, 'qwen_omni_service') and self.app_context.qwen_omni_service):
                return await self._transcribe_with_omni(processed_audio)
            
            # Use faster-whisper STT service
            if (hasattr(self.app_context, 'faster_whisper_stt_service') and 
                self.app_context.faster_whisper_stt_service):
                
                self.logger.info("🎯 Using faster-whisper for transcription...")
                result = self.app_context.faster_whisper_stt_service.transcribe_audio_data(processed_audio)
                
                if result and len(result.strip()) > 0:
                    self.logger.info(f"✅ faster-whisper transcription: '{result}'")
                    return result
                else:
                    self.logger.info("🔇 faster-whisper returned no result")
                    return None
            else:
                self.logger.error("❌ No STT service available")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None
    
    async def _transcribe_with_whisper(self, audio_data: np.ndarray) -> Optional[str]:
        """Fallback transcription using Whisper."""
        if not self.whisper_model:
            self.logger.error("❌ Whisper model not loaded")
            return None
            
        try:
            # Debug audio data
            self.logger.info(f"🎵 Audio data shape: {audio_data.shape}, duration: {len(audio_data)/16000:.2f}s, max: {np.max(np.abs(audio_data)):.4f}")
            
            # Enhanced audio validation
            audio_duration = len(audio_data) / 16000
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Audio stats - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # More lenient audio quality checks - focus on fixing the actual audio processing
            if audio_max_volume < 0.01:  # Lowered back down - the issue isn't volume
                self.logger.warning(f"🔇 Audio volume too low (max: {audio_max_volume:.4f}), skipping transcription")
                return None
            
            if audio_rms < 0.003:  # Lowered back down - need to process more audio to debug
                self.logger.warning(f"🔇 Audio RMS too low ({audio_rms:.4f}), likely background noise")
                return None
            
            # Log audio quality metrics for debugging
            signal_to_noise = audio_max_volume / (audio_rms + 1e-8)
            self.logger.info(f"🎵 Audio quality - SNR: {signal_to_noise:.2f}, Duration: {audio_duration:.2f}s")
            
            # Check if audio is long enough (very lenient now)
            if audio_duration < 0.2:  # Lowered to 0.2 seconds - very permissive
                self.logger.warning(f"🔇 Audio too short ({audio_duration:.2f}s), skipping transcription")
                return None
            
            # Normalize audio to prevent clipping and improve recognition
            if audio_max_volume > 0:
                # Normalize to 70% of max range to prevent clipping
                normalized_audio = audio_data * (0.7 / audio_max_volume)
                self.logger.info(f"🎵 Audio normalized from max {audio_max_volume:.4f} to {np.max(np.abs(normalized_audio)):.4f}")
            else:
                normalized_audio = audio_data
            
            # Aggressive Discord audio preprocessing (based on Discord Audio Fixer research)
            try:
                from scipy import signal
                
                # 1. Aggressive pre-emphasis to counteract Discord's audio compression
                pre_emphasis = 0.95  # More aggressive than standard 0.97
                emphasized_audio = np.append(normalized_audio[0], normalized_audio[1:] - pre_emphasis * normalized_audio[:-1])
                
                # 2. Remove DC offset that Discord often introduces
                emphasized_audio = emphasized_audio - np.mean(emphasized_audio)
                
                # 3. Wider band-pass filter for Discord's compressed audio (200-4000 Hz)
                sos_bp = signal.butter(3, [200, 4000], 'bp', fs=16000, output='sos')
                filtered_result = signal.sosfilt(sos_bp, emphasized_audio)
                filtered_audio = np.asarray(filtered_result, dtype=np.float32)
                
                # 4. Spectral subtraction to reduce Discord's background noise
                # Simple approach: reduce quiet sections more aggressively
                rms_threshold = np.sqrt(np.mean(np.square(filtered_audio))) * 0.1
                filtered_audio_abs = np.abs(filtered_audio)
                noise_reduced = np.where(
                    filtered_audio_abs < rms_threshold,
                    filtered_audio * 0.1,  # Reduce noise by 90%
                    filtered_audio
                )
                
                # 5. Adaptive gain control for Discord's variable levels
                # Calculate RMS in sliding windows
                window_size = 1600  # 100ms windows
                adaptive_audio = noise_reduced.copy()
                for i in range(0, len(noise_reduced) - window_size, window_size // 2):
                    window = np.array(noise_reduced[i:i + window_size])  # Ensure it's a numpy array
                    window_rms = np.sqrt(np.mean(np.square(window)))
                    if window_rms > 0.01:  # Only adjust if there's significant signal
                        target_rms = 0.15
                        gain = target_rms / window_rms
                        gain = float(np.clip(gain, 0.5, 3.0))  # Ensure gain is a scalar
                        adaptive_audio[i:i + window_size] = window * gain
                
                # 6. Final normalization optimized for Whisper
                if np.max(np.abs(adaptive_audio)) > 0:
                    # Normalize to 80% to leave headroom for Whisper
                    final_audio = adaptive_audio * (0.8 / np.max(np.abs(adaptive_audio)))
                else:
                    final_audio = adaptive_audio
                
                # 7. Add very light dithering to break up quantization
                dither = np.random.normal(0, 0.00005, len(final_audio))
                final_audio = final_audio + dither
                
                original_rms = np.sqrt(np.mean(np.square(normalized_audio)))
                final_rms = np.sqrt(np.mean(np.square(final_audio)))
                self.logger.info(f"🎵 Discord audio fix applied - Original RMS: {original_rms:.4f}, Final RMS: {final_rms:.4f}, Gain: {final_rms/original_rms:.2f}x")
                
            except ImportError:
                self.logger.warning("🔇 scipy not available, using basic processing")
                final_audio = normalized_audio
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                import scipy.io.wavfile as wavfile
                # Ensure audio is in the right format for Whisper (16-bit PCM)
                audio_int16 = np.clip(final_audio * 32767, -32767, 32767).astype(np.int16)
                wavfile.write(temp_file.name, 16000, audio_int16)
                temp_file_path = temp_file.name
            
            self.logger.info(f"🎵 Saved processed audio to {temp_file_path}, starting Whisper transcription...")
            
            # Run Whisper with ultra-conservative parameters to minimize Discord hallucinations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe( # type: ignore
                    temp_file_path,
                    language='en',  # Force English
                    task='transcribe',
                    temperature=0.0,  # Deterministic
                    best_of=1,  # Single beam to reduce hallucinations
                    beam_size=1,  # No beam search to prevent repetition
                    word_timestamps=False,  # Disable for speed
                    condition_on_previous_text=False,  # No context to prevent bias
                    initial_prompt="",  # Empty prompt to avoid hallucinations
                    suppress_tokens=[-1],  # Suppress hallucination tokens
                    logprob_threshold=-0.3,  # Even more aggressive filtering
                    no_speech_threshold=0.9,  # Very high threshold
                    compression_ratio_threshold=1.5  # Detect repetitive text more aggressively
                )
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if result and "text" in result:
                text = str(result["text"]).strip()
                # Log more detailed results
                segments = result.get("segments", [])
                avg_logprob = result.get("avg_logprob", "unknown")
                no_speech_prob = result.get("no_speech_prob", "unknown")
                
                self.logger.info(f"📝 Whisper result: '{text}'")
                self.logger.info(f"📊 Whisper stats - Segments: {len(segments)}, Avg LogProb: {avg_logprob}, No Speech Prob: {no_speech_prob}")
                
                # Check confidence metrics to reject hallucinations
                if isinstance(avg_logprob, (int, float)) and avg_logprob < -0.8:
                    self.logger.info(f"🚫 Rejected low confidence transcription (logprob: {avg_logprob})")
                    return None
                
                if isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.6:
                    self.logger.info(f"🚫 Rejected high no-speech probability ({no_speech_prob})")
                    return None
                
                # Enhanced hallucination filtering with detailed logging
                if text and len(text.strip()) > 0:
                    text_lower = text.lower().strip()
                    self.logger.info(f"🔍 Analyzing transcription: '{text}' (length: {len(text_lower)})")
                    
                    # Comprehensive Discord hallucination patterns (based on observed behavior)
                    obvious_hallucinations = [
                        "and a lot of people are watching this video",
                        "a lot of people are watching this video", 
                        "people are watching this video",
                        "watching this video",
                        "thank you for watching",
                        "thanks for watching",
                        "thank you so much for watching",
                        "subscribe",
                        "like and subscribe",
                        "thank you",
                        "you",
                        "booooo",
                        "oh",
                        "um", 
                        "uh",
                        "i'm sorry",
                        "sorry",
                        "1 2 3",
                        "one two three",
                        "testing",
                        "test",
                        "yes yes",
                        "no no",
                        "!!!!",  # Discord audio artifact
                        "!!!",   # Discord audio artifact
                        "????",  # Discord audio artifact
                        "...",   # Discord audio artifact
                        "---",   # Discord audio artifact
                        # Whisper hallucination patterns
                        "the human speech is not a human speech",
                        "human speech is not a human speech",
                        "this is clear human speech",
                        "you can't be a human being"
                    ]
                    
                    # Check for character-level repetition (AAAA pattern)
                    if len(text) > 10:
                        char_counts = {}
                        for char in text_lower:
                            if char.isalpha():  # Only count letters
                                char_counts[char] = char_counts.get(char, 0) + 1
                        
                        if char_counts:
                            max_char_count = max(char_counts.values())
                            char_repetition_ratio = max_char_count / len([c for c in text_lower if c.isalpha()])
                            
                            if char_repetition_ratio > 0.7:  # 70% of characters are the same
                                most_repeated_char = max(char_counts.keys(), key=lambda k: char_counts[k])
                                self.logger.info(f"🚫 Filtered out character repetition hallucination: '{most_repeated_char}' appears {max_char_count} times ({char_repetition_ratio:.1%})")
                                return None
                    
                    # Check for word-level repetitive patterns (major hallucination indicator)
                    words = text_lower.split()
                    if len(words) >= 2:  # Check even 2-word transcriptions
                        # Check for excessive repetition
                        word_counts = {}
                        for word in words:
                            word_counts[word] = word_counts.get(word, 0) + 1
                        
                        # Special case: exact word repetition (like "hello hello")
                        if len(words) == 2 and words[0] == words[1]:
                            self.logger.info(f"🚫 Filtered out exact word repetition: '{text}'")
                            return None
                        
                        # If any word appears more than 30% of the time, it's likely a hallucination
                        max_word_count = max(word_counts.values()) if word_counts else 0
                        repetition_ratio = max_word_count / len(words) if words else 0
                        
                        if repetition_ratio > 0.3:  # 30% threshold
                            most_repeated_word = max(word_counts.keys(), key=lambda k: word_counts[k])
                            self.logger.info(f"🚫 Filtered out repetitive hallucination: '{most_repeated_word}' appears {max_word_count}/{len(words)} times ({repetition_ratio:.1%})")
                            return None
                    
                    # Check for specific Discord audio hallucination patterns
                    discord_hallucinations = [
                        "sound of the sound",
                        "and the sound of",
                        "the sound of the",
                        "sound of sound",
                        "and sound of"
                    ]
                    
                    for pattern in discord_hallucinations:
                        if pattern in text_lower:
                            self.logger.info(f"🚫 Filtered out Discord audio hallucination pattern: '{pattern}'")
                            return None
                    

                    
                    # Check for exact matches only
                    if text_lower in obvious_hallucinations:
                        self.logger.info(f"🚫 Filtered out exact hallucination match: '{text}'")
                        return None
                    
                    # Check for partial matches only for very specific phrases
                    hallucination_patterns = [
                        "watching this video", 
                        "thank you for watching", 
                        "human speech is not", 
                        "the human speech",
                        "human speech is",
                        "speech is not",
                        "not a human speech"
                    ]
                    
                    for pattern in hallucination_patterns:
                        if pattern in text_lower:
                            self.logger.info(f"🚫 Filtered out partial hallucination match: '{text}' (contains '{pattern}')")
                            return None
                    
                    # Only filter extremely short responses (1-2 characters)
                    if len(text_lower) <= 2:
                        self.logger.info(f"🚫 Filtered out very short text: '{text}'")
                        return None
                    
                    # Accept everything else for now to debug
                    self.logger.info(f"✅ Accepted transcription: '{text}'")
                    return text
                else:
                    self.logger.info("🔇 Whisper returned empty text")
                    return None
            else:
                self.logger.warning("🔇 Whisper returned no result or malformed result")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Whisper transcription error: {e}", exc_info=True)
            return None

    async def _transcribe_with_omni(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Qwen2.5-Omni multimodal model."""
        try:
            self.logger.info("🎵 Using Qwen2.5-Omni for direct audio processing...")
            
            # Save audio to temporary file for Omni
            import tempfile
            import scipy.io.wavfile as wavfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                audio_int16 = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
                wavfile.write(temp_file.name, 16000, audio_int16)
                temp_file_path = temp_file.name
            
            self.logger.info(f"🎵 Saved audio to {temp_file_path}, processing with Qwen2.5-Omni...")
            
            # Use Qwen2.5-Omni to process audio directly
            response = await self.app_context.qwen_omni_service.generate_response(
                text="Please transcribe the speech in this audio accurately.",
                audio_path=temp_file_path
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if response and len(response.strip()) > 0:
                # Extract just the transcription from the response
                transcription = response.strip()
                
                # Remove any assistant commentary, keep just the transcription
                if "transcription:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif "says:" in transcription.lower():
                    transcription = transcription.split(":", 1)[1].strip()
                elif '"' in transcription:
                    # Extract quoted text if present
                    import re
                    quoted = re.findall(r'"([^"]*)"', transcription)
                    if quoted:
                        transcription = quoted[0]
                
                self.logger.info(f"✅ Qwen2.5-Omni transcription: '{transcription}'")
                return transcription
            else:
                self.logger.info("🔇 Qwen2.5-Omni returned no transcription")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Qwen2.5-Omni transcription error: {e}")
            return None

    async def _transcribe_with_llamacpp(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using LlamaCpp Qwen - DISABLED for GGUF."""
        # LlamaCpp GGUF doesn't support audio transcription, return None to fall back to Whisper
        self.logger.info("🎵 LlamaCpp audio transcription disabled - falling back to Whisper")
        return None

    async def recording_finished_callback(self, sink, channel: discord.TextChannel, *args):
        """Callback when recording is finished (for compatibility)."""
        self.logger.info("🎵 Recording session ended")

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and comprehensive reasoning content from LLM responses"""
        import re
        
        if not text:
            return text
        
        # Remove <think>...</think> tags and content (case insensitive, multiline)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove reasoning sections that start with common patterns
        reasoning_starters = [
            r'let me think.*?\.',
            r'hmm,.*?\.',
            r'okay,.*?\.',
            r'first,.*?\.',
            r'looking at.*?\.',
            r'based on.*?\.',
            r'from.*?results.*?\.',
            r'the user.*?asking.*?\.',
            r'this.*?question.*?\.',
            r'i need to.*?\.',
            r'i should.*?\.',
            r'let me.*?\.',
            r'i\'ll.*?\.',
            r'i can.*?\.',
            r'to answer.*?\.',
            r'for this.*?\.',
            r'regarding.*?\.',
            r'about.*?topic.*?\.',
            r'concerning.*?\.',
            r'with respect to.*?\.',
            r'as for.*?\.',
            r'in.*?to.*?question.*?\.'
        ]
        
        for pattern in reasoning_starters:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove standalone reasoning words/phrases at the start of lines
        reasoning_lines = [
            r'^hmm[,.]?\s*',
            r'^okay[,.]?\s*',
            r'^well[,.]?\s*',
            r'^so[,.]?\s*',
            r'^now[,.]?\s*',
            r'^first[,.]?\s*',
            r'^let me see[,.]?\s*',
            r'^let me think[,.]?\s*',
            r'^i need to[,.]?\s*',
            r'^i should[,.]?\s*'
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            original_line = line
            for pattern in reasoning_lines:
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
            
            # Only keep lines that have substantial content after cleaning
            if line.strip() and len(line.strip()) > 3:
                cleaned_lines.append(line)
            elif original_line.strip() and not any(re.match(pattern, original_line.strip(), re.IGNORECASE) for pattern in reasoning_lines):
                # Keep original line if it wasn't a reasoning line
                cleaned_lines.append(original_line)
        
        # Join lines and clean up extra whitespace
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n\s*\n', '\n\n', result)  # Remove extra blank lines
        result = result.strip()
        
        # If we cleaned everything away, try to extract the last meaningful sentence
        if not result and text:
            sentences = re.split(r'[.!?]+', text)
            for sentence in reversed(sentences):
                sentence = sentence.strip()
                if (sentence and len(sentence) > 15 and 
                    not any(re.search(pattern, sentence, re.IGNORECASE) for pattern in reasoning_starters)):
                    result = sentence + '.'
                    break
        
        return result

    async def _generate_and_play_voice_response(self, response_text: str, channel: discord.TextChannel, user_name: str):
        """Generate TTS audio and play it in the voice channel."""
        try:
            self.logger.info(f"🔊 Generating TTS for response to {user_name}")
            
            # Strip markdown formatting for TTS
            tts_text = self._strip_markdown_for_tts(response_text)
            
            # Generate TTS audio
            if not self.tts_service:
                self.logger.error("❌ TTS service not available")
                return
                
            loop = asyncio.get_event_loop()
            tts_audio = await loop.run_in_executor(
                None,
                self.tts_service.generate_audio,
                tts_text
            )
            
            if tts_audio:
                self.logger.info(f"✅ TTS audio generated ({len(tts_audio)} bytes)")
                
                # Find voice channel to play response
                voice_channel = None
                for guild in self.guilds:
                    if guild.id in self.connections:
                        voice_channel = self.connections[guild.id].channel
                        break
                
                if voice_channel:
                    try:
                        # Connect to voice channel for playback
                        voice_client = await voice_channel.connect(timeout=10.0)
                        
                        # Save TTS audio to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_file.write(tts_audio)
                            temp_file_path = temp_file.name
                        
                        # Play the audio
                        audio_source = discord.FFmpegPCMAudio(temp_file_path)
                        voice_client.play(audio_source)
                        
                        # Wait for playback to finish
                        while voice_client.is_playing():
                            await asyncio.sleep(0.1)
                        
                        # Disconnect and cleanup
                        await voice_client.disconnect()
                        os.unlink(temp_file_path)
                        
                        self.logger.info("🎵 Voice response played successfully")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Failed to play voice response: {e}")
                        await channel.send("🔊 Generated voice response but couldn't play it in voice channel.")
                else:
                    self.logger.warning(f"🔇 Could not find voice channel for playback")
                    await channel.send("🔊 Generated voice response but no voice channel available.")
            else:
                self.logger.error("❌ TTS generation failed")
                await channel.send("🔊 Sorry, I couldn't generate a voice response.")
                
        except Exception as e:
            self.logger.error(f"❌ Error in voice response generation: {e}")

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Remove Discord formatting for TTS playback"""
        import re
        # Remove Discord code blocks
        text = re.sub(r'```[a-zA-Z]*\n.*?\n```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove Discord formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Underline
        text = re.sub(r'~~([^~]+)~~', r'\1', text)      # Strikethrough
        
        # Remove URLs in markdown format
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        return text.strip()

    async def _play_tts_audio(self, tts_audio: bytes):
        """Play TTS audio in the connected voice channel"""
        try:
            # Find an active voice connection
            voice_client = None
            for guild in self.guilds:
                if guild.voice_client and guild.voice_client.is_connected():
                    voice_client = guild.voice_client
                    break
            
            if not voice_client:
                self.logger.warning("🔇 No voice connection available for TTS playback")
                return
            
            # Save TTS audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(tts_audio)
                temp_file_path = temp_file.name
            
            self.logger.info(f"🔊 Playing TTS audio in {voice_client.channel.name}")
            
            # Create audio source and play
            audio_source = discord.FFmpegPCMAudio(temp_file_path)
            voice_client.play(audio_source)
            
            # Wait for playback to finish
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # Don't fail if cleanup fails
            
            self.logger.info("✅ TTS audio playback completed")
            
        except Exception as e:
            self.logger.error(f"❌ TTS playback failed: {e}")

    async def _play_tts_audio_with_feedback_prevention(self, tts_audio: bytes):
        """Play TTS audio in the connected voice channel with feedback prevention"""
        try:
            # Check if Discord-only mode is enabled
            discord_only_mode = self.app_context.global_settings.get('TTS_SERVER', {}).get('discord_only_mode', False)
            
            # Find an active voice connection
            voice_client = None
            for guild in self.guilds:
                if guild.voice_client and guild.voice_client.is_connected():
                    voice_client = guild.voice_client
                    break
            
            if not voice_client:
                self.logger.warning("🔇 No voice connection available for TTS playback")
                # Stop feedback prevention since we can't play
                self.feedback_prevention.stop_tts_playback()
                return
            
            # Save TTS audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(tts_audio)
                temp_file_path = temp_file.name
            
            channel_name = getattr(voice_client.channel, 'name', 'Unknown Channel')
            
            if discord_only_mode:
                self.logger.info(f"🔊 Playing TTS audio in {channel_name} (Discord-only mode - local audio muted)")
            else:
                self.logger.info(f"🔊 Playing TTS audio in {channel_name} with feedback prevention")
            
            # Create audio source and play
            audio_source = discord.FFmpegPCMAudio(temp_file_path)
            voice_client.play(audio_source)
            
            # Wait for playback to finish
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            
            # Stop feedback prevention when playback is done
            self.feedback_prevention.stop_tts_playback()
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # Don't fail if cleanup fails
            
            if discord_only_mode:
                self.logger.info("✅ TTS audio playback completed (Discord-only mode)")
            else:
                self.logger.info("✅ TTS audio playback completed with feedback prevention")
            
        except Exception as e:
            self.logger.error(f"❌ TTS playback with feedback prevention failed: {e}")
            # Make sure to stop feedback prevention on error
            self.feedback_prevention.stop_tts_playback()

    async def process_transcription_queue(self):
        """Process transcriptions from the audio worker queue."""
        self.logger.info("🎯 Starting transcription queue processor")
        
        while True:
            try:
                # Check if virtual audio capture has transcriptions
                if (hasattr(self, 'virtual_audio') and 
                    self.virtual_audio and 
                    hasattr(self.virtual_audio, 'transcription_queue')):
                    
                    try:
                        # Get transcription from queue (non-blocking)
                        transcription_data = self.virtual_audio.transcription_queue.get_nowait()
                        
                        transcription = transcription_data['transcription']
                        user_name = transcription_data['user']
                        
                        self.logger.info(f"📥 Processing queued transcription: '{transcription}'")
                        
                        # ===== ENHANCED FEEDBACK PREVENTION CHECK =====
                        # Check if we should ignore this input due to TTS playback
                        should_ignore, ignore_reason = self.feedback_prevention.should_ignore_input()
                        if should_ignore:
                            self.logger.info(f"🛡️ Ignoring queued input due to feedback prevention: {ignore_reason}")
                            continue
                        
                        # Check if this transcription is likely an echo of our own TTS
                        is_echo, echo_reason = self.feedback_prevention.is_likely_tts_echo(transcription)
                        if is_echo:
                            self.logger.info(f"🛡️ Detected TTS echo in queue, ignoring: {echo_reason}")
                            continue
                        
                        # Additional filtering for background noise and meaningless audio
                        transcription_clean = transcription.lower().strip()
                        
                        # Filter out very short or meaningless transcriptions
                        if len(transcription_clean) < 3:
                            self.logger.info(f"🛡️ Ignoring very short queued transcription: '{transcription}'")
                            continue
                        
                        # Filter out common background noise transcriptions and TTS artifacts
                        # Split into single-word patterns and phrase patterns for better matching
                        single_word_noise = ["um", "uh", "ah", "oh", "mm", "hmm", "hm", "eh", "er"]
                        phrase_noise_patterns = [
                            "music", "sound", "audio", "noise", "background",
                            # TTS-specific patterns that often get picked up
                            "i'm having trouble processing", "please try again", "error processing",
                            "audio transcription failed", "i encountered an error", 
                            "that right now", "sorry to hear", "clarify what",
                            "processing that", "trouble processing", "try again"
                        ]
                        
                        # Check for noise patterns with improved logic
                        is_noise = False
                        transcription_words = transcription_clean.split()
                        
                        # Check if transcription is ONLY single-word noise (exact match)
                        if len(transcription_words) == 1 and transcription_clean in single_word_noise:
                            self.logger.info(f"🛡️ Ignoring single-word noise in queue: '{transcription}' (exact match: '{transcription_clean}')")
                            is_noise = True
                        
                        # Check for phrase patterns (substring match for longer phrases only)
                        if not is_noise:
                            for pattern in phrase_noise_patterns:
                                if pattern in transcription_clean:
                                    self.logger.info(f"🛡️ Ignoring TTS echo/noise pattern in queue: '{transcription}' (matched: '{pattern}')")
                                    is_noise = True
                                    break
                        
                        if is_noise:
                            continue
                        
                        # Filter out repetitive patterns (likely audio artifacts)
                        words = transcription_clean.split()
                        if len(words) >= 2:
                            # Check for word repetition
                            unique_words = set(words)
                            if len(unique_words) == 1:  # All words are the same
                                self.logger.info(f"🛡️ Ignoring repetitive queued transcription: '{transcription}'")
                                continue
                        
                        # Only process transcriptions that seem like actual meaningful speech
                        # Require at least 5 characters and not just common filler words
                        if len(transcription_clean) < 5:
                            self.logger.info(f"🛡️ Ignoring too short queued transcription: '{transcription}'")
                            continue
                        
                        # Clean up old feedback prevention entries
                        self.feedback_prevention.cleanup_old_entries()
                        
                        # Add user input to short-term memory
                        if self.app_context.short_term_memory_service:
                            self.app_context.short_term_memory_service.add_entry(
                                user_name=user_name,
                                content=transcription,
                                entry_type='user_input'
                            )
                            self.logger.debug(f"[ShortTermMemory] Stored user input from queue: '{transcription[:50]}...'")
                        
                        # Get configured text channel
                        text_channel = self.get_configured_text_channel()
                        if text_channel:
                            # Send transcription to channel
                            await text_channel.send(f"🎤 **{user_name}**: {transcription}")
                            self.logger.info(f"📤 Sent transcription to Discord: '{transcription}'")
                            
                            # Process with LLM
                            if self.llm_service:
                                try:
                                    self.logger.info("🧠 Processing with LLM...")
                                    
                                    # Check if LLM-guided search mode is enabled
                                    if getattr(self, 'llm_search_mode', False):
                                        self.logger.info("🔍 Using LLM-guided search mode")
                                        response = await self.llm_service.handle_user_text_query_with_llm_search(
                                            user_text=transcription,
                                            user_name=user_name
                                        )
                                        # Extract response text if it's a tuple
                                        if isinstance(response, tuple):
                                            response_text, metadata = response
                                            self.logger.info(f"🔍 LLM search method: {metadata.get('method', 'unknown')}")
                                            response = response_text
                                    else:
                                        response = await self.llm_service.handle_user_text_query(
                                            user_text=transcription,
                                            user_name=user_name
                                        )
                                    
                                    if response and len(response.strip()) > 0:
                                        # Strip think tags from LLM response
                                        clean_response = self._strip_think_tags(response)
                                        
                                        await text_channel.send(f"🤖 **DanzarAI**: {clean_response}")
                                        self.logger.info(f"🤖 Sent LLM response: '{clean_response[:100]}...'")
                                        
                                        # Add bot response to short-term memory (use clean response)
                                        if self.app_context.short_term_memory_service:
                                            self.app_context.short_term_memory_service.add_entry(
                                                user_name=user_name,
                                                content=clean_response,
                                                entry_type='bot_response'
                                            )
                                            self.logger.debug(f"[ShortTermMemory] Stored bot response from queue: '{clean_response[:50]}...'")
                                        
                                        # Generate and play TTS if available (use clean response)
                                        if self.tts_service:
                                            try:
                                                self.logger.info("🔊 Generating TTS for queue response...")
                                                tts_text = self._strip_markdown_for_tts(clean_response)
                                                
                                                # ===== START FEEDBACK PREVENTION =====
                                                self.feedback_prevention.start_tts_playback(tts_text)
                                                
                                                # Generate TTS audio asynchronously
                                                loop = asyncio.get_event_loop()
                                                tts_audio = await loop.run_in_executor(
                                                    None,
                                                    self.tts_service.generate_audio,
                                                    tts_text
                                                )
                                                
                                                if tts_audio:
                                                    self.logger.info("✅ TTS audio generated successfully")
                                                    await self._play_tts_audio_with_feedback_prevention(tts_audio)
                                                else:
                                                    self.logger.warning("⚠️ TTS generation failed")
                                                    self.feedback_prevention.stop_tts_playback()
                                                    
                                            except Exception as tts_e:
                                                self.logger.error(f"❌ TTS error in queue processor: {tts_e}")
                                                # Stop feedback prevention on error
                                                self.feedback_prevention.stop_tts_playback()
                                    else:
                                        await text_channel.send("🤖 **DanzarAI**: I heard you, but I'm not sure how to respond to that.")
                                        
                                except Exception as e:
                                    self.logger.error(f"❌ LLM processing error: {e}")
                                    await text_channel.send("🤖 **DanzarAI**: Sorry, I had trouble processing that.")
                            else:
                                # Fallback response
                                response = f"I heard you say: '{transcription}'. My LLM service isn't available right now."
                                await text_channel.send(f"🤖 **DanzarAI**: {response}")
                                self.logger.info("🤖 Sent fallback response (no LLM service)")
                        else:
                            self.logger.warning("⚠️  No configured text channel available for transcription responses")
                            
                    except queue.Empty:
                        # No transcriptions in queue, continue
                        pass
                    except Exception as e:
                        self.logger.error(f"❌ Error processing transcription from queue: {e}")
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in transcription queue processor: {e}")
                await asyncio.sleep(1.0)  # Longer sleep on error

    def _chunk_response_for_tts(self, response_text: str, max_chunk_length: int = 200) -> list[str]:
        """
        Intelligently chunk long responses into smaller segments for TTS.
        
        Args:
            response_text: Full response text to chunk
            max_chunk_length: Maximum characters per chunk
            
        Returns:
            List of text chunks optimized for TTS
        """
        try:
            # Strip markdown first
            clean_text = self._strip_markdown_for_tts(response_text)
            
            # If text is short enough, return as single chunk
            if len(clean_text) <= max_chunk_length:
                return [clean_text]
            
            self.logger.info(f"🔄 Chunking long response ({len(clean_text)} chars) for TTS")
            
            # Split by sentences first
            import re
            
            # Split on sentence boundaries
            sentence_endings = r'[.!?]+\s+'
            sentences = re.split(sentence_endings, clean_text)
            
            # Clean up sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # If adding this sentence would exceed limit, start new chunk
                if current_chunk and len(current_chunk + " " + sentence) > max_chunk_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Single sentence is too long, split it further
                        long_chunks = self._split_long_sentence(sentence, max_chunk_length)
                        chunks.extend(long_chunks)
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Filter out very short chunks (less than 10 characters)
            chunks = [chunk for chunk in chunks if len(chunk) >= 10]
            
            self.logger.info(f"🔄 Created {len(chunks)} TTS chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error chunking response for TTS: {e}")
            # Fallback: return original text as single chunk
            return [self._strip_markdown_for_tts(response_text)]

    def _split_long_sentence(self, sentence: str, max_length: int) -> list[str]:
        """
        Split a long sentence into smaller chunks at natural break points.
        
        Args:
            sentence: Long sentence to split
            max_length: Maximum length per chunk
            
        Returns:
            List of sentence chunks
        """
        if len(sentence) <= max_length:
            return [sentence]
        
        # Try to split at natural break points
        break_points = [', ', ' and ', ' or ', ' but ', ' so ', ' then ', ' when ', ' where ', ' which ']
        
        chunks = []
        remaining = sentence
        
        while len(remaining) > max_length:
            best_split = -1
            
            # Find the best break point within the limit
            for break_point in break_points:
                split_pos = remaining.rfind(break_point, 0, max_length)
                if split_pos > best_split:
                    best_split = split_pos + len(break_point)
            
            if best_split > 0:
                # Split at natural break point
                chunks.append(remaining[:best_split].strip())
                remaining = remaining[best_split:].strip()
            else:
                # No natural break point, split at word boundary
                words = remaining[:max_length].split()
                if len(words) > 1:
                    # Remove last word to avoid cutting mid-word
                    chunk = " ".join(words[:-1])
                    chunks.append(chunk)
                    remaining = remaining[len(chunk):].strip()
                else:
                    # Single long word, just split it
                    chunks.append(remaining[:max_length])
                    remaining = remaining[max_length:]
        
        # Add remaining text
        if remaining:
            chunks.append(remaining)
        
        return chunks

    async def _generate_and_play_chunked_tts(self, response_text: str, text_channel, user_name: str):
        """
        Generate and play TTS for long responses using streaming playback.
        Generates and plays chunks progressively for faster response times.
        
        Args:
            response_text: Full response text
            text_channel: Discord text channel for status updates
            user_name: User who triggered the response
        """
        try:
            if not self.tts_service:
                self.logger.warning("⚠️ TTS service not available for chunked playback")
                return
            
            # Check if streaming TTS is enabled
            use_streaming = self.app_context.global_settings.get('TTS_SERVER', {}).get('enable_streaming', True)
            
            if use_streaming and hasattr(self.tts_service, 'generate_audio_streaming'):
                # Use new streaming TTS approach
                self.logger.info("🎵 Using streaming TTS for progressive playback")
                
                # Start feedback prevention for the entire response
                self.feedback_prevention.start_tts_playback(response_text)
                
                # Start TTS queue processor for ordered playback
                await self._start_tts_queue_processor()
                
                # Create callback to handle each audio chunk as it's generated
                successful_chunks = 0
                
                def audio_chunk_callback(audio_data: bytes):
                    nonlocal successful_chunks
                    try:
                        # Add to TTS queue for ordered playback
                        asyncio.create_task(self._queue_tts_audio(audio_data))
                        successful_chunks += 1
                        self.logger.info(f"🎵 Streaming chunk queued for ordered playback ({len(audio_data)} bytes)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to queue streaming chunk: {e}")
                
                # Generate and stream audio chunks
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    None,
                    self.tts_service.generate_audio_streaming,
                    response_text,
                    audio_chunk_callback
                )
                
                # Stop feedback prevention after streaming completes
                self.feedback_prevention.stop_tts_playback()
                
                if success and successful_chunks > 0:
                    self.logger.info(f"✅ Streaming TTS completed: {successful_chunks} chunks streamed")
                else:
                    self.logger.error("❌ Streaming TTS failed")
                    
            else:
                # Fall back to traditional chunked approach
                self.logger.info("🔊 Using traditional chunked TTS")
                
                # Chunk the response
                chunks = self._chunk_response_for_tts(response_text, max_chunk_length=150)
                
                if len(chunks) == 1:
                    # Single chunk, use normal TTS processing
                    self.logger.info("🔊 Single chunk TTS generation...")
                    tts_text = chunks[0]
                    
                    # Start feedback prevention
                    self.feedback_prevention.start_tts_playback(tts_text)
                    
                    # Generate TTS audio
                    loop = asyncio.get_event_loop()
                    tts_audio = await loop.run_in_executor(
                        None,
                        self.tts_service.generate_audio,
                        tts_text
                    )
                    
                    if tts_audio:
                        self.logger.info("✅ Single chunk TTS audio generated")
                        await self._play_tts_audio_with_feedback_prevention(tts_audio)
                    else:
                        self.logger.warning("⚠️ Single chunk TTS generation failed")
                        self.feedback_prevention.stop_tts_playback()
                else:
                    # Multiple chunks, play sequentially
                    self.logger.info(f"🔊 Multi-chunk TTS generation ({len(chunks)} chunks)...")
                    
                    # Start feedback prevention for the entire sequence
                    full_text = " ".join(chunks)
                    self.feedback_prevention.start_tts_playback(full_text)
                    
                    successful_chunks = 0
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            self.logger.info(f"🔊 Generating TTS chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                            
                            # Generate TTS for this chunk
                            loop = asyncio.get_event_loop()
                            tts_audio = await loop.run_in_executor(
                                None,
                                self.tts_service.generate_audio,
                                chunk
                            )
                            
                            if tts_audio:
                                self.logger.info(f"✅ Chunk {i+1} TTS audio generated")
                                
                                # Play this chunk (without stopping feedback prevention)
                                await self._play_single_tts_chunk(tts_audio)
                                successful_chunks += 1
                                
                                # Brief pause between chunks for natural flow
                                if i < len(chunks) - 1:  # Don't pause after last chunk
                                    await asyncio.sleep(0.3)
                            else:
                                self.logger.warning(f"⚠️ Chunk {i+1} TTS generation failed")
                                
                        except Exception as e:
                            self.logger.error(f"❌ Error processing TTS chunk {i+1}: {e}")
                            continue
                    
                    # Stop feedback prevention after all chunks
                    self.feedback_prevention.stop_tts_playback()
                    
                    if successful_chunks > 0:
                        self.logger.info(f"✅ Multi-chunk TTS completed ({successful_chunks}/{len(chunks)} successful)")
                    else:
                        self.logger.error("❌ All TTS chunks failed")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in chunked TTS generation: {e}")
            # Make sure to stop feedback prevention on error
            self.feedback_prevention.stop_tts_playback()

    async def _play_single_tts_chunk(self, tts_audio: bytes):
        """
        Play a single TTS audio chunk without managing feedback prevention.
        
        Args:
            tts_audio: TTS audio bytes to play
        """
        try:
            # Check if Discord-only mode is enabled
            discord_only_mode = self.app_context.global_settings.get('TTS_SERVER', {}).get('discord_only_mode', False)
            
            # Find an active voice connection
            voice_client = None
            for guild in self.guilds:
                if guild.voice_client and guild.voice_client.is_connected():
                    voice_client = guild.voice_client
                    break
            
            if not voice_client:
                self.logger.warning("🔇 No voice connection available for chunk playback")
                return
            
            # Save TTS audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(tts_audio)
                temp_file_path = temp_file.name
            
            # Create audio source and play
            audio_source = discord.FFmpegPCMAudio(temp_file_path)
            voice_client.play(audio_source)
            
            # Wait for playback to finish
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # Don't fail if cleanup fails
            
            self.logger.debug("✅ TTS chunk playback completed")
            
        except Exception as e:
            self.logger.error(f"❌ TTS chunk playback failed: {e}")

    async def _start_tts_queue_processor(self):
        """Start the TTS queue processor to ensure ordered playback"""
        if self.tts_queue_active:
            return
        
        self.tts_queue_active = True
        self.tts_queue_task = asyncio.create_task(self._process_tts_queue())
        self.logger.info("🎵 TTS queue processor started")

    async def _stop_tts_queue_processor(self):
        """Stop the TTS queue processor"""
        self.tts_queue_active = False
        if self.tts_queue_task:
            self.tts_queue_task.cancel()
            try:
                await self.tts_queue_task
            except asyncio.CancelledError:
                pass
            self.tts_queue_task = None
        self.logger.info("🎵 TTS queue processor stopped")

    async def _process_tts_queue(self):
        """Process TTS audio chunks in order from the queue"""
        try:
            while self.tts_queue_active:
                try:
                    # Wait for next audio chunk with timeout
                    audio_data = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                    
                    # Play this chunk
                    await self._play_single_tts_chunk(audio_data)
                    
                    # Mark task as done
                    self.tts_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No audio in queue, continue waiting
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing TTS queue: {e}")
                    
        except asyncio.CancelledError:
            self.logger.info("🎵 TTS queue processor cancelled")
        except Exception as e:
            self.logger.error(f"❌ TTS queue processor error: {e}")

    async def _queue_tts_audio(self, audio_data: bytes):
        """Add audio data to the TTS queue for ordered playback"""
        try:
            await self.tts_queue.put(audio_data)
            self.logger.debug(f"🎵 Queued TTS audio chunk ({len(audio_data)} bytes)")
        except Exception as e:
            self.logger.error(f"❌ Failed to queue TTS audio: {e}")

# VADSink class removed - now using RawAudioReceiver for direct Opus packet capture

def monitor_hotkeys(app_context: AppContext):
    """Monitor for hotkey combinations and handle shutdown"""
    try:
        app_context.logger.info("Hotkey monitor started. Press Ctrl+D for quick shutdown.")
        
        def handle_shutdown():
            app_context.logger.info("🔥 HOTKEY TRIGGERED: Ctrl+D detected! Initiating quick shutdown...")
            app_context.shutdown_event.set()
        
        # Register Ctrl+D hotkey
        keyboard.add_hotkey('ctrl+d', handle_shutdown)
        
        # Keep the monitor running until shutdown
        while not app_context.shutdown_event.is_set():
            time.sleep(0.1)
            
    except Exception as e:
        app_context.logger.error(f"Hotkey monitor error: {e}")
    finally:
        try:
            keyboard.unhook_all()
            app_context.logger.info("Hotkey monitor stopped.")
        except:
            pass

def signal_handler(signum, frame):
    """Force exit signal handler"""
    print("\n🔥 Force exit signal received! Terminating immediately...")
    os._exit(1)

async def main():
    try:
        # Check for single instance
        if not app_lock.acquire():
            print("❌ Another instance of DanzarAI is already running!")
            print("   Please close the existing instance before starting a new one.")
            print("   If you believe this is an error, delete the 'danzar_voice.lock' file.")
            sys.exit(1)
        
        print("🔒 Single instance lock acquired successfully")
        
        # Set up signal handlers for force exit
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break
        
        parser = argparse.ArgumentParser(description="DanzarVLM - AI Game Commentary and Interaction Suite")
        parser.add_argument(
            "--profile",
            help="Name of the game profile to load from config/profiles/ (e.g., rimworld). If not set, will use profile from global_settings.yaml or 'generic_game'."
        )
        parser.add_argument(
            "--log-level",
            default=None, 
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level (overrides global_settings.yaml if specified)."
        )
        parser.add_argument(
            "--web-port",
            type=int,
            default=5000,
            help="Port for the web interface (default: 5000)"
        )
        parser.add_argument(
            "--virtual-audio",
            action="store_true",
            help="Enable virtual audio capture mode (requires sounddevice)"
        )
        parser.add_argument(
            "--audio-device",
            type=int,
            help="Audio device ID for virtual audio capture (use --list-devices to see options)"
        )
        parser.add_argument(
            "--list-devices",
            action="store_true",
            help="List available audio devices and exit"
        )
        args = parser.parse_args()

        # Handle device listing
        if args.list_devices:
            if not VIRTUAL_AUDIO_AVAILABLE or sd is None:
                print("❌ sounddevice not available - install with: pip install sounddevice")
                sys.exit(1)
            
            print("🎵 Available Audio Input Devices:")
            try:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    try:
                        # Handle different device info formats
                        max_channels = 0
                        device_name = f'Device {i}'
                        
                        if isinstance(device, dict):
                            max_channels = device.get('max_input_channels', 0)
                            device_name = device.get('name', f'Device {i}')
                        else:
                            # Try to access as attributes
                            try:
                                max_channels = getattr(device, 'max_input_channels', 0)
                                device_name = getattr(device, 'name', f'Device {i}')
                            except:
                                pass
                        
                        if max_channels > 0:
                            print(f"  {i}: {device_name} (channels: {max_channels})")
                            if any(keyword in str(device_name).lower() for keyword in 
                                  ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'stereo mix']):
                                print(f"      ⭐ VIRTUAL AUDIO DEVICE")
                    except Exception as e:
                        print(f"  {i}: Error reading device info - {e}")
            except Exception as e:
                print(f"❌ Error listing devices: {e}")
            sys.exit(0)

        # Load settings using core.config_loader
        settings = load_global_settings() or {}
        
        # Override virtual audio setting from command line
        if args.virtual_audio:
            settings['USE_VIRTUAL_AUDIO'] = True
            logger.info("🎵 Virtual audio mode enabled via command line")
        
        logger.info("🚀 Starting DanzarAI with Full Voice Integration (STT → LLM → TTS)...")

        # Create a game profile for Discord voice bot
        from core.game_profile import GameProfile
        discord_profile = GameProfile(
            game_name="discord_voice",
            vlm_model="qwen2.5:7b",
            system_prompt_commentary="You are Danzar, an upbeat and witty gaming assistant who's always ready to help players crush their goals in EverQuest (or any game). Speak casually, like a friendly raid leader—cheer people on, crack a clever joke now and then, and keep the energy high. When giving advice, be forward-thinking: mention upcoming expansions, meta strategies, or ways to optimize both platinum farming and experience gains. Use gamer lingo naturally, but explain anything arcane so newcomers feel included. Above all, stay encouraging—everyone levels up at their own pace, and you're here to make the journey fun and rewarding.",
            user_prompt_template_commentary="User said: {user_text}. Respond helpfully about gaming.",
            vlm_max_tokens=200,
            vlm_temperature=0.7,
            vlm_max_commentary_sentences=2,
            conversational_llm_model="qwen2.5:7b",
            system_prompt_chat="You are Danzar, an upbeat and witty gaming assistant who's always ready to help players crush their goals in EverQuest (or any game). Speak casually, like a friendly raid leader—cheer people on, crack a clever joke now and then, and keep the energy high. When giving advice, be forward-thinking: mention upcoming expansions, meta strategies, or ways to optimize both platinum farming and experience gains. Use gamer lingo naturally, but explain anything arcane so newcomers feel included. Above all, stay encouraging—everyone levels up at their own pace, and you're here to make the journey fun and rewarding."
        )
        app_context = AppContext(settings, discord_profile, logger)

        # Create the Discord bot with voice capabilities and full service integration
        bot = DanzarVoiceBot(settings, app_context)

        # Start the bot
        logger.info("🎤 Starting DanzarAI Voice Bot with LLM and TTS integration...")
        bot_token = settings.get('DISCORD_BOT_TOKEN', '')
        if not bot_token:
            logger.error("❌ DISCORD_BOT_TOKEN is missing from configuration!")
            return
        
        logger.info(f"🔑 Attempting to connect with token: {bot_token[:20]}...")
        
        # Use the proper async context manager pattern
        async with bot:
            try:
                await bot.start(bot_token)
            except discord.LoginFailure:
                logger.error("❌ Invalid Discord bot token")
            except discord.HTTPException as e:
                logger.error(f"❌ HTTP error connecting to Discord: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error starting bot: {e}", exc_info=True)
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}", exc_info=True)
        raise
    finally:
        # Ensure lock is released
        app_lock.release()
        logger.info("🔓 Single instance lock released")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Final cleanup
        app_lock.release()