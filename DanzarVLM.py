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
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from services.faster_whisper_stt_service import WhisperSTTService
from services.simple_voice_receiver import SimpleVoiceReceiver
# VAD Voice Receiver import
try:
    from services.vad_voice_receiver import VADVoiceReceiver
    VAD_VOICE_AVAILABLE = True
except ImportError:
    VAD_VOICE_AVAILABLE = False
    VADVoiceReceiver = None
from services.short_term_memory_service import ShortTermMemoryService

# MiniGPT-4 Video Service import - REMOVED
# MiniGPT-4 has been removed from this project
MINIGPT4_VIDEO_AVAILABLE = False
MiniGPT4VideoService = None

# Setup logging with Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/danzar_voice.log', mode='w', encoding='utf-8')  # Changed mode from 'a' to 'w' to reset log each run
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
        self.faster_whisper_stt_service: Optional[Any] = None
        self.simple_voice_receiver: Optional[SimpleVoiceReceiver] = None
        self.vad_voice_receiver: Optional[Any] = None
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
                
                # Word match - only for extremely high word overlap (85%+)
                if len(trans_words) >= 4 and len(tts_words) >= 4:  # Require longer phrases
                    matching_words = sum(1 for word in trans_words if word in tts_words)
                    match_ratio = matching_words / len(trans_words)
                    if match_ratio >= 0.85:  # Raised from 50% to 85% to prevent false positives
                        return True, f"WORD match ({match_ratio:.1%}) with recent TTS output #{i}"
        
        # Enhanced TTS echo patterns - very specific to avoid false positives
        echo_patterns = [
            # Exact TTS response patterns (very specific)
            "i heard you say", "i'm danzar", "danzarai",
            "gaming assistant", "how can i help",
            "that's interesting", "let me help you", "i understand",
            # EverQuest TTS-specific phrases (from actual TTS responses) - removed generic terms
            "first they mentioned", "user is confident", "prep expansions ready",
            "detailed spreadsheet for platinum", 
            "they want to crush goals", "looking at the tools provided",
            "there's info on necromancer", "they mentioned",
            # Common TTS response starters
            "looking to maximize their", "so they're probably",
            "the user is confident with their",
            # Very specific TTS artifacts that shouldn't be user input
            "boxing games are fantastic for more than just entertainment",
            "that's right boxing games are not just fun"
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
        # Reduced to only very specific system words that shouldn't be user input
        single_word_echoes = ["danzar", "danzarai"]  # Removed game terms that users might legitimately say
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
            
            for dev_id, dev_name in virtual_devices:
                if dev_id == working_device_id:
                    device_id = working_device_id
                    working_device_found = True
                    self.logger.info(f"✅ Selected audio device: {dev_name}")
                    self.logger.info(f"🎯 Using working VB-Audio device: {dev_name}")
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
    
    async def initialize_whisper(self, model_size: str = "large"):
        """Initialize Whisper model with GPU acceleration."""
        try:
            self.logger.info(f"🔧 Loading Whisper model '{model_size}' with GPU acceleration...")
            self.logger.info("💡 Available models: tiny, base, small, medium, large")
            self.logger.info("💡 Accuracy: tiny < base < small < medium < large")
            self.logger.info("💡 Speed: large < medium < small < base < tiny")
            
            # Get GPU configuration from settings
            whisper_gpu_config = self.app_context.global_settings.get('WHISPER_GPU_CONFIG', {})
            preferred_device = whisper_gpu_config.get('device', 'cuda:0')
            compute_type = whisper_gpu_config.get('compute_type', 'float16')
            
            self.logger.info(f"🎯 Preferred GPU device: {preferred_device}")
            self.logger.info(f"🎯 Compute type: {compute_type}")
            
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            if WHISPER_AVAILABLE and whisper:
                # Load model with GPU acceleration
                def load_whisper_with_gpu():
                    try:
                        # Try to use GPU acceleration
                        import torch
                        if torch.cuda.is_available():
                            # Check available GPUs and their memory
                            gpu_count = torch.cuda.device_count()
                            self.logger.info(f"🎯 Found {gpu_count} CUDA devices")
                            
                            # Try preferred device first
                            try:
                                device_id = int(preferred_device.split(':')[1]) if ':' in preferred_device else 0
                                if device_id < gpu_count:
                                    # Check memory on preferred device
                                    torch.cuda.set_device(device_id)
                                    memory_allocated = torch.cuda.memory_allocated(device_id)
                                    memory_reserved = torch.cuda.memory_reserved(device_id)
                                    memory_total = torch.cuda.get_device_properties(device_id).total_memory
                                    memory_free = memory_total - memory_reserved
                                    
                                    self.logger.info(f"🎯 GPU {device_id} memory: {memory_free/1024**3:.1f}GB free / {memory_total/1024**3:.1f}GB total")
                                    
                                    # If we have enough memory (at least 2GB free), use this device
                                    if memory_free > 2 * 1024**3:  # 2GB
                                        self.logger.info(f"✅ Using preferred GPU {device_id} for Whisper")
                                        return whisper.load_model(model_size, device=preferred_device)
                                    else:
                                        self.logger.warning(f"⚠️ GPU {device_id} has insufficient memory ({memory_free/1024**3:.1f}GB free)")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to use preferred device {preferred_device}: {e}")
                            
                            # Try to find a GPU with enough memory
                            for gpu_id in range(gpu_count):
                                try:
                                    torch.cuda.set_device(gpu_id)
                                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                                    memory_reserved = torch.cuda.memory_reserved(gpu_id)
                                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                                    memory_free = memory_total - memory_reserved
                                    
                                    self.logger.info(f"🎯 GPU {gpu_id} memory: {memory_free/1024**3:.1f}GB free / {memory_total/1024**3:.1f}GB total")
                                    
                                    if memory_free > 2 * 1024**3:  # 2GB free
                                        device_str = f"cuda:{gpu_id}"
                                        self.logger.info(f"✅ Using GPU {gpu_id} for Whisper (device: {device_str})")
                                        return whisper.load_model(model_size, device=device_str)
                                except Exception as e:
                                    self.logger.warning(f"⚠️ Failed to check GPU {gpu_id}: {e}")
                                    continue
                            
                            # If no GPU has enough memory, try CPU
                            self.logger.warning("⚠️ No GPU has sufficient memory, falling back to CPU")
                            return whisper.load_model(model_size, device="cpu")
                        else:
                            self.logger.warning("⚠️ CUDA not available, falling back to CPU")
                            return whisper.load_model(model_size, device="cpu")
                    except Exception as e:
                        self.logger.warning(f"⚠️ GPU loading failed, falling back to CPU: {e}")
                        return whisper.load_model(model_size, device="cpu")
                
                self.whisper_model = await loop.run_in_executor(None, load_whisper_with_gpu)
            else:
                raise Exception("Whisper not available")
                
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ Whisper model '{model_size}' loaded successfully in {load_time:.1f}s")
            
            # Show model info and device
            if hasattr(self.whisper_model, 'dims'):
                dims = self.whisper_model.dims
                self.logger.info(f"📊 Model info: {dims.n_audio_ctx} audio tokens, {dims.n_text_ctx} text tokens")
            
            # Check if model is on GPU
            if hasattr(self.whisper_model, 'model') and hasattr(self.whisper_model.model, 'device'):
                device_info = str(self.whisper_model.model.device)
                self.logger.info(f"🎯 Model loaded on device: {device_info}")
            
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
                    # Convert buffer to numpy array
                    speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer.clear()
                    self.is_speaking = False
                    self.logger.info(f"🎤 Speech ended (duration: {len(speech_audio) / self.sample_rate:.2f}s)")
                    
                    # Process with local Whisper model directly
                    try:
                        self.logger.info("🎯 Processing speech audio with local Whisper...")
                        
                        # Use local Whisper model directly in this thread
                        if hasattr(self, 'whisper_model') and self.whisper_model:
                            try:
                                # Run Whisper transcription directly in this thread
                                result = self.whisper_model.transcribe(speech_audio)
                                
                                if result and isinstance(result, dict) and 'text' in result:
                                    transcription = str(result['text']).strip()
                                    if transcription:
                                        self.logger.info(f"✅ Local Whisper transcription: '{transcription}'")
                                        # Put the transcription in the queue for Discord bot to process
                                        if hasattr(self, 'transcription_queue') and self.transcription_queue:
                                            self.transcription_queue.put(transcription)
                                        else:
                                            self.logger.warning("No transcription queue available to handle transcription.")
                                    else:
                                        self.logger.info("🔇 Empty transcription from local Whisper")
                                else:
                                    self.logger.info("🔇 No transcription from local Whisper")
                            except Exception as e:
                                self.logger.error(f"❌ Local Whisper error: {e}")
                        else:
                            self.logger.warning("⚠️ Local Whisper model not available")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Processing error: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing worker error: {e}")
                continue

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
                        # Qwen2.5-Omni service not available
                        self.logger.warning("⚠️ Qwen2.5-Omni service not available")
                        response = None
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
        # Initialize loop attribute for async operations
        self.loop = None
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
        self.faster_whisper_stt_service: Optional[Any] = None
        self.simple_voice_receiver: Optional[SimpleVoiceReceiver] = None
        self.vad_voice_receiver: Optional[Any] = None
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
            # Load environment variables for TTS configuration
            from dotenv import load_dotenv
            load_dotenv()
            
            # Initialize Azure TTS Service (replaces Chatterbox TTS)
            try:
                from services.tts_service_azure import AzureTTSService
                
                # Check if Azure TTS is configured in environment variables
                azure_tts_key = os.environ.get('AZURE_TTS_SUBSCRIPTION_KEY')
                if not azure_tts_key:
                    self.logger.warning("⚠️ Azure TTS subscription key not configured in environment")
                    self.logger.info("🔄 Falling back to default TTS service...")
                    from services.tts_service import TTSService
                    self.tts_service = TTSService(self.app_context)
                else:
                    # Update global settings with environment variables
                    self.app_context.global_settings['AZURE_TTS_SUBSCRIPTION_KEY'] = azure_tts_key
                    self.app_context.global_settings['AZURE_TTS_REGION'] = os.environ.get('AZURE_TTS_REGION', 'eastus')
                    self.app_context.global_settings['AZURE_TTS_VOICE'] = os.environ.get('AZURE_TTS_VOICE', 'en-US-AdamMultilingualNeural')
                    self.app_context.global_settings['AZURE_TTS_SPEECH_RATE'] = os.environ.get('AZURE_TTS_SPEECH_RATE', '+0%')
                    self.app_context.global_settings['AZURE_TTS_PITCH'] = os.environ.get('AZURE_TTS_PITCH', '+0%')
                    self.app_context.global_settings['AZURE_TTS_VOLUME'] = os.environ.get('AZURE_TTS_VOLUME', '+0%')
                    
                    self.tts_service = AzureTTSService(self.app_context)
                    if await self.tts_service.initialize():
                        self.logger.info("✅ Azure TTS Service initialized successfully")
                        self.logger.info(f"🎤 Voice: {self.tts_service.voice_name}")
                        self.logger.info(f"🌍 Region: {self.tts_service.region}")
                    else:
                        self.logger.error("❌ Azure TTS Service initialization failed")
                        self.logger.info("🔄 Falling back to default TTS service...")
                        from services.tts_service import TTSService
                        self.tts_service = TTSService(self.app_context)
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Azure TTS Service: {e}")
                self.logger.info("🔄 Falling back to default TTS service...")
                from services.tts_service import TTSService
                self.tts_service = TTSService(self.app_context)
            
            self.app_context.tts_service = self.tts_service
            self.logger.info("✅ TTS Service initialized")
            
            # Initialize Memory Service
            memory_service = MemoryService(self.app_context)
            # Note: MemoryService doesn't have initialize method, it's ready after construction
            self.app_context.memory_service = memory_service
            self.logger.info("✅ Memory Service initialized")
            
            # Initialize Qwen2.5-VL Integration Service (Primary VLM for OBS + Discord)
            try:
                from services.qwen_vl_integration_service import QwenVLIntegrationService
                self.app_context.qwen_vl_integration = QwenVLIntegrationService(self.app_context)
                if await self.app_context.qwen_vl_integration.initialize():
                    self.logger.info("✅ Qwen2.5-VL Integration Service initialized")
                    self.logger.info("🎯 Primary VLM: CUDA-enabled Qwen2.5-VL for OBS vision + Discord")
                    
                    # Get performance stats
                    stats = self.app_context.qwen_vl_integration.get_performance_stats()
                    self.logger.info(f"📊 Qwen2.5-VL Stats: CUDA: {stats['cuda_available']}, Fallback: {stats['transformers_fallback']}")
                else:
                    self.logger.error("❌ Qwen2.5-VL Integration Service initialization failed")
                    self.app_context.qwen_vl_integration = None
            except Exception as e:
                self.logger.error(f"❌ Qwen2.5-VL Integration Service error: {e}")
                self.app_context.qwen_vl_integration = None
            
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
            
            # Set model client in both places for compatibility
            self.app_context.model_client = model_client
            self.model_client = model_client  # This fixes the streaming service
            self.logger.info("✅ Model Client initialized")
            
            # Initialize RAG Service (Qdrant connection)
            try:
                from services.llamacpp_rag_service import LlamaCppRAGService
                rag_service = LlamaCppRAGService(self.app_context.global_settings)
                self.app_context.rag_service_instance = rag_service
                self.logger.info("✅ Llama.cpp + Qdrant RAG Service initialized")
            except Exception as e:
                self.logger.error(f"❌ RAG Service initialization failed: {e}")
                rag_service = None
            
            # Initialize faster-whisper STT Service with maximum accuracy
            self.app_context.faster_whisper_stt_service = WhisperSTTService(
                self.app_context, 
                model_size="medium"  # Changed from 'large' to 'medium' for better performance
                # Removed device parameter as it's not supported by WhisperSTTService
            )
            if await self.app_context.faster_whisper_stt_service.initialize():
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
            if VAD_VOICE_AVAILABLE and VADVoiceReceiver:
                try:
                    self.app_context.vad_voice_receiver = VADVoiceReceiver(
                        self.app_context,
                        speech_callback=self.process_vad_transcription
                    )
                    if await self.app_context.vad_voice_receiver.initialize():
                        self.logger.info("✅ VAD Voice Receiver initialized")
                    else:
                        self.logger.error("❌ VAD Voice Receiver initialization failed")
                except Exception as e:
                    self.logger.error(f"❌ VAD Voice Receiver error: {e}")
                    self.app_context.vad_voice_receiver = None
            else:
                self.logger.info("ℹ️ VAD Voice Receiver not available (missing dependencies)")
                self.app_context.vad_voice_receiver = None
            
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
            
            # Initialize Streaming LLM Service for real-time responses
            try:
                from services.streaming_llm_service import StreamingLLMService
                self.streaming_llm_service = StreamingLLMService(
                    app_context=self.app_context,
                    audio_service=None,  # Not needed for voice bot
                    rag_service=rag_service,    # Use the initialized RAG service
                    model_client=model_client
                )
                self.app_context.streaming_llm_service = self.streaming_llm_service
                self.logger.info("✅ Streaming LLM Service initialized")
            except Exception as e:
                self.logger.error(f"❌ Streaming LLM Service initialization failed: {e}")
                # Fallback to regular LLM service
                self.streaming_llm_service = None
            
            # Initialize regular LLM Service as fallback
            self.llm_service = LLMService(
                app_context=self.app_context,
                audio_service=None,  # Not needed for voice bot
                rag_service=rag_service,    # Use the initialized RAG service
                model_client=model_client
            )
            # Note: LLMService doesn't have initialize method, it's ready after construction
            self.app_context.llm_service = self.llm_service
            self.logger.info("✅ LLM Service initialized")
            
            # Initialize Real-Time Streaming LLM Service for immediate voice responses
            try:
                from services.real_time_streaming_llm import RealTimeStreamingLLMService
                self.real_time_streaming_service = RealTimeStreamingLLMService(
                    app_context=self.app_context,
                    model_client=model_client,
                    tts_service=self.tts_service
                )
                if await self.real_time_streaming_service.initialize():
                    self.logger.info("✅ Real-Time Streaming LLM Service initialized")
                    self.logger.info("🎵 Voice responses will stream in real-time")
                else:
                    self.logger.error("❌ Real-Time Streaming LLM Service initialization failed")
                    self.real_time_streaming_service = None
            except Exception as e:
                self.logger.error(f"❌ Real-Time Streaming LLM Service error: {e}")
                self.real_time_streaming_service = None
            
            # Initialize Short-Term Memory Service
            self.app_context.short_term_memory_service = ShortTermMemoryService(self.app_context)
            self.logger.info("✅ Short-Term Memory Service initialized")
            
            # Initialize Hybrid Vision Service (replaces MiniGPT-4 Video Service)
            try:
                from services.hybrid_vision_service import HybridVisionService
                self.app_context.hybrid_vision_service = HybridVisionService(self.app_context)
                if await self.app_context.hybrid_vision_service.initialize():
                    self.logger.info("✅ Hybrid Vision Service initialized")
                    self.logger.info("✅ Using Phi-4 for fast video analysis")
                    
                    # Get service status
                    status = self.app_context.hybrid_vision_service.get_status()
                    self.logger.info(f"📊 Hybrid Service Status: Phi-4: {status['phi4_available']}")
                else:
                    self.logger.error("❌ Hybrid Vision Service initialization failed")
                    self.app_context.hybrid_vision_service = None
            except ImportError:
                self.logger.warning("ℹ️ Hybrid Vision Service not available - falling back to basic video analysis")
                self.app_context.hybrid_vision_service = None
            except Exception as e:
                self.logger.error(f"❌ Hybrid Vision Service error: {e}")
                self.app_context.hybrid_vision_service = None
            
            # Initialize Conversational AI Service for turn-taking and game awareness
            try:
                from services.conversational_ai_service import ConversationalAIService
                self.app_context.conversational_ai_service = ConversationalAIService(self.app_context)
                if await self.app_context.conversational_ai_service.initialize():
                    self.logger.info("✅ Conversational AI Service initialized")
                    self.logger.info("🎯 Features: Turn-taking, game awareness, BLIP/CLIP integration")
                else:
                    self.logger.error("❌ Conversational AI Service initialization failed")
                    self.app_context.conversational_ai_service = None
            except Exception as e:
                self.logger.error(f"❌ Conversational AI Service error: {e}")
                self.app_context.conversational_ai_service = None
            
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

                # Check if already connected to this channel
                if hasattr(self, 'voice_clients') and self.voice_clients:
                    for vc in self.voice_clients:
                        if vc.channel and vc.channel.id == channel.id:
                            if vc.is_connected():
                                await ctx.send(f"✅ **Already connected to {channel.name}**\n"
                                             f"🎤 **Windows Audio Capture Active**\n"
                                             f"💬 **Ready to process voice input!**")
                                self.logger.info(f"✅ Already connected to {channel.name}")
                                return
                            else:
                                # Disconnect stale connection
                                try:
                                    await vc.disconnect(force=True)
                                    await asyncio.sleep(1)
                                except:
                                    pass

                # Connect with py-cord's native VoiceClient with retry logic
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        # Disconnect any existing voice connections first
                        if hasattr(self, 'voice_clients') and self.voice_clients:
                            for vc in self.voice_clients:
                                if vc.is_connected():
                                    await vc.disconnect(force=True)
                                    await asyncio.sleep(1)
                        
                        voice_client = await channel.connect(timeout=30.0)
                        self.logger.info(f"Successfully connected to {channel.name} with py-cord VoiceClient (attempt {attempt + 1})")
                        break
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Voice connection attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise e
                
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
                    self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors
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
                    try:
                        await voice_client.disconnect()
                    except:
                        pass

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
                    self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors
                
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
                    self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors
                    # Initialize Whisper with configured model size
                    model_size = self.app_context.global_settings.get('WHISPER_MODEL_SIZE', 'medium')
                    await self.virtual_audio.initialize_whisper(model_size)
                
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
                        new_stt_service = WhisperSTTService(
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

        # Video Analysis Commands with FPS Control
        @self.command(name='video')
        async def video_command(ctx, action: str = "help", *, args: str = ""):
            """Main video analysis command with subcommands"""
            self.logger.info(f"📺 !video {action} used by {ctx.author.name}")
            
            if action.lower() == "help":
                embed = discord.Embed(
                    title="📺 Video Analysis Commands",
                    color=discord.Color.blue()
                )
                embed.add_field(
                    name="🎯 Analysis",
                    value="`!video analyze [prompt]` - Analyze current screen\n"
                          "`!video quick` - Quick game state analysis\n"
                          "`!video detailed` - Detailed gameplay analysis",
                    inline=False
                )
                embed.add_field(
                    name="🎮 Live Monitoring (FPS Optimized)",
                    value="`!video start [seconds] [fps]` - Start live monitoring\n"
                          "`!video stop` - Stop live monitoring\n"
                          "`!video status` - Check monitoring status",
                    inline=False
                )
                embed.add_field(
                    name="🎬 Qwen2.5-VL Streaming (NEW!)",
                    value="`!video stream start` - Start Qwen2.5-VL analysis every 30s\n"
                          "`!video stream stop` - Stop streaming analysis\n"
                          "`!video stream status` - Check streaming status",
                    inline=False
                )
                embed.add_field(
                    name="🚀 Qwen2.5-VL Features",
                    value="• **30-second intervals** - Perfect for gaming commentary\n"
                          "• **1920x1080 resolution** - High-quality capture\n"
                          "• **CUDA acceleration** - Fast 4-bit model (~940MB VRAM)\n"
                          "• **Gaming focus** - Optimized for game analysis",
                    inline=False
                )
                embed.add_field(
                    name="🔧 Settings",
                    value="`!video settings` - Show current settings\n"
                          "`!video quality [low/medium/high]` - Set quality\n"
                          "`!video fps [0.033/0.05/0.1]` - Set monitoring FPS",
                    inline=False
                )
                embed.add_field(
                    name="💡 FPS Guide",
                    value="**0.033** = 30s intervals (ultra-safe)\n"
                          "**0.05** = 20s intervals (recommended)\n"
                          "**0.1** = 10s intervals (aggressive)",
                    inline=False
                )
                await ctx.send(embed=embed)
                
            elif action.lower() == "analyze":
                prompt = args if args else "What's happening in this game right now?"
                await self._video_analyze(ctx, prompt, use_fast_model=False)  # Use detailed model
                
            elif action.lower() == "quick":
                await self._video_analyze(ctx, "Quick analysis: What's the current game state?", use_fast_model=True)  # Use fast model
                
            elif action.lower() == "detailed":
                await self._video_analyze(ctx, "Provide detailed analysis of the current gameplay, including UI elements, characters, and situation.", use_fast_model=False)  # Use detailed model
                
            elif action.lower() == "start":
                try:
                    parts = args.split() if args else []
                    duration = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 60
                    fps = float(parts[1]) if len(parts) > 1 else 0.05  # Default to 20-second intervals
                    await self._video_start_monitoring(ctx, duration, fps)
                except ValueError:
                    await ctx.send("❌ Invalid parameters. Use: `!video start [seconds] [fps]`")
                    
            elif action.lower() == "stop":
                await self._video_stop_monitoring(ctx)
                
            elif action.lower() == "status":
                await self._video_status(ctx)
                
            elif action.lower() == "settings":
                await self._video_settings(ctx)
                
            elif action.lower() == "quality":
                if args.lower() in ["low", "medium", "high"]:
                    await self._video_quality(ctx, args.lower())
                else:
                    await ctx.send("❌ Invalid quality. Use: `!video quality [low/medium/high]`")
                    
            elif action.lower() == "fps":
                try:
                    fps = float(args) if args else 0.05
                    if fps <= 0 or fps > 2.0:
                        await ctx.send("❌ FPS must be between 0.001 and 2.0")
                        return
                    interval = 1.0 / fps
                    await ctx.send(f"✅ **FPS set to {fps}** (analysis every {interval:.1f} seconds)")
                except ValueError:
                    await ctx.send("❌ Invalid FPS value. Use: `!video fps [0.033/0.05/0.1]`")
                    
            elif action.lower() == "stream":
                # Handle streaming video commands (batch analysis)
                subaction = args.split()[0] if args else "help"
                await self._video_stream_command(ctx, subaction)
                    
            else:
                await ctx.send("❌ Unknown video command. Use `!video help` for available commands.")

        # Qwen2.5-VL Vision Analysis Commands
        @self.command(name='qwen')
        async def qwen_command(ctx, action: str = "help", *, args: str = ""):
            """Qwen2.5-VL vision analysis commands with CUDA acceleration"""
            self.logger.info(f"🎯 !qwen {action} used by {ctx.author.name}")
            
            if action.lower() == "help":
                embed = discord.Embed(
                    title="🎯 Qwen2.5-VL Vision Analysis",
                    description="CUDA-accelerated vision analysis for gaming commentary",
                    color=discord.Color.purple()
                )
                embed.add_field(
                    name="🔍 Analysis",
                    value="`!qwen analyze [prompt]` - Analyze current OBS frame\n"
                          "`!qwen quick` - Quick gaming scene analysis\n"
                          "`!qwen detailed` - Detailed gameplay analysis\n"
                          "`!qwen commentary` - Generate live commentary",
                    inline=False
                )
                embed.add_field(
                    name="🎮 Gaming Focus",
                    value="`!qwen game` - What game is being played?\n"
                          "`!qwen action` - What's happening right now?\n"
                          "`!qwen ui` - Describe UI elements and interface\n"
                          "`!qwen scene` - Describe the game environment",
                    inline=False
                )
                embed.add_field(
                    name="📊 Status",
                    value="`!qwen status` - Show performance stats\n"
                          "`!qwen history` - Show recent analyses",
                    inline=False
                )
                embed.add_field(
                    name="⚡ Performance",
                    value="• **CUDA Acceleration** - GPU-powered analysis\n"
                          "• **Fast Response** - ~2-5 seconds per analysis\n"
                          "• **High Quality** - Detailed gaming insights",
                    inline=False
                )
                await ctx.send(embed=embed)
                
            elif action.lower() == "analyze":
                prompt = args if args else "Analyze this gaming screenshot. What's happening and what should the player do?"
                await self._qwen_analyze(ctx, prompt)
                
            elif action.lower() == "quick":
                await self._qwen_analyze(ctx, "Quick analysis: What's the current game state and what's happening?")
                
            elif action.lower() == "detailed":
                await self._qwen_analyze(ctx, "Provide detailed analysis of this gaming scene. Describe the game, characters, UI elements, current situation, and what actions the player might take.")
                
            elif action.lower() == "commentary":
                await self._qwen_commentary(ctx)
                
            elif action.lower() == "game":
                await self._qwen_analyze(ctx, "What video game is being played in this screenshot? Identify the game and describe what type of game it is.")
                
            elif action.lower() == "action":
                await self._qwen_analyze(ctx, "What action or activity is happening in this game right now? Describe what the player is doing or what's occurring.")
                
            elif action.lower() == "ui":
                await self._qwen_analyze(ctx, "Describe the UI elements visible in this game screenshot: health bars, menus, buttons, inventory, or any interface components.")
                
            elif action.lower() == "scene":
                await self._qwen_analyze(ctx, "Describe the game environment or location shown in this screenshot. What type of area is this?")
                
            elif action.lower() == "status":
                await self._qwen_status(ctx)
                
            elif action.lower() == "history":
                await self._qwen_history(ctx)
                
            else:
                await ctx.send("❌ Unknown Qwen command. Use `!qwen help` for available commands.")

        @self.command(name='conversation')
        async def conversation_command(ctx, action: str = "status", *, args: str = ""):
            """Manage conversation settings and turn-taking"""
            try:
                if not hasattr(self.app_context, 'conversational_ai_service') or not self.app_context.conversational_ai_service:
                    await ctx.send("❌ **Conversational AI Service not available**")
                    return
                
                service = self.app_context.conversational_ai_service
                
                if action == "status":
                    status = service.get_conversation_status()
                    embed = discord.Embed(
                        title="💬 Conversation Status",
                        color=discord.Color.blue()
                    )
                    embed.add_field(name="State", value=status["state"], inline=True)
                    embed.add_field(name="Current Speaker", value=status["current_speaker"] or "None", inline=True)
                    embed.add_field(name="History Length", value=status["conversation_length"], inline=True)
                    embed.add_field(name="Recent Events", value=status["recent_events"], inline=True)
                    embed.add_field(name="Vision Models", value="✅ Loaded" if status["vision_models_loaded"] else "❌ Not Available", inline=True)
                    embed.add_field(name="Game Type", value=service.current_game, inline=True)
                    
                    await ctx.send(embed=embed)
                
                elif action == "clear":
                    service.clear_conversation_history()
                    await ctx.send("✅ **Conversation history cleared**")
                
                elif action == "game":
                    if args:
                        service.set_game_type(args)
                        await ctx.send(f"🎮 **Game type set to: {args}**")
                    else:
                        await ctx.send(f"🎮 **Current game type: {service.current_game}**")
                
                elif action == "events":
                    recent_events = [e for e in service.game_events 
                                   if time.time() - e.timestamp < 300]  # Last 5 minutes
                    if recent_events:
                        embed = discord.Embed(
                            title="🎮 Recent Game Events",
                            color=discord.Color.green()
                        )
                        for i, event in enumerate(recent_events[-5:], 1):  # Show last 5
                            embed.add_field(
                                name=f"Event {i}",
                                value=f"**{event.event_type}** ({event.confidence:.2f})\n{event.description}",
                                inline=False
                            )
                        await ctx.send(embed=embed)
                    else:
                        await ctx.send("📝 **No recent game events detected**")
                
                else:
                    await ctx.send("""💬 **Conversation Commands:**
- `!conversation status` - Show conversation status
- `!conversation clear` - Clear conversation history
- `!conversation game <type>` - Set game type (everquest, generic, etc.)
- `!conversation events` - Show recent game events""")
                    
            except Exception as e:
                self.logger.error(f"❌ Conversation command error: {e}")
                await ctx.send("❌ **Error processing conversation command**")

    async def _video_analyze(self, ctx, prompt: str, use_fast_model: bool = False):
        """Analyze current screen with given prompt using Qwen2.5-VL"""
        try:
            # Check if Qwen2.5-VL integration is available
            if not hasattr(self.app_context, 'qwen_vl_integration') or not self.app_context.qwen_vl_integration:
                await ctx.send("❌ **Qwen2.5-VL Integration not available**\n"
                              "Make sure the CUDA server is running and the service is initialized")
                return
            
            model_type = "⚡ Fast Analysis" if use_fast_model else "🧠 Detailed Analysis"
            await ctx.send(f"📺 **Analyzing current screen with {model_type}...** 🔍")
            
            # Capture screenshot from OBS
            frame = await self.app_context.qwen_vl_integration.capture_obs_frame()
            if not frame:
                await ctx.send("❌ **Failed to capture screenshot**\nMake sure OBS Studio is running")
                return
            
            # Analyze with Qwen2.5-VL
            start_time = time.time()
            analysis = await self.app_context.qwen_vl_integration.analyze_obs_frame(frame, prompt)
            analysis_time = time.time() - start_time
            
            if analysis:
                embed = discord.Embed(
                    title=f"📺 Screen Analysis ({analysis_time:.1f}s)",
                    description=analysis[:2000] + "..." if len(analysis) > 2000 else analysis,
                    color=discord.Color.green()
                )
                embed.add_field(
                    name="🎯 Model",
                    value="Qwen2.5-VL (CUDA)",
                    inline=True
                )
                embed.add_field(
                    name="⚡ Speed",
                    value=f"{analysis_time:.1f}s",
                    inline=True
                )
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ **Analysis failed**\nPlease try again or check the CUDA server status")
                
        except Exception as e:
            self.logger.error(f"❌ Video analysis error: {e}")
            await ctx.send(f"❌ **Analysis error**: {str(e)}")

    async def _video_start_monitoring(self, ctx, duration: int, fps: float = 0.05):
        """Start live video monitoring with configurable FPS using Hybrid Vision Service"""
        try:
            # Check if Hybrid Vision Service is available
            if not hasattr(self.app_context, 'hybrid_vision_service') or not self.app_context.hybrid_vision_service:
                await ctx.send("❌ **Hybrid Vision Service not available**\n"
                              "Make sure the vision services are properly set up and OBS Studio is running")
                return
            
            # Check if already monitoring
            if hasattr(self, '_video_monitoring_task') and self._video_monitoring_task and not self._video_monitoring_task.done():
                await ctx.send("⚠️ **Video monitoring already active!** Use `!video stop` first.")
                return
            
            interval = 1.0 / fps
            await ctx.send(f"🎬 **Starting live video monitoring with Hybrid Vision Service**\n"
                          f"⏱️ Duration: {duration} seconds\n"
                          f"🎯 FPS: {fps} (analysis every {interval:.1f}s)")
            
            # Start monitoring task
            self._video_monitoring_task = asyncio.create_task(
                self._video_monitoring_loop(ctx, duration, fps)
            )
            
        except Exception as e:
            self.logger.error(f"Video monitoring start error: {e}")
            await ctx.send(f"❌ **Failed to start monitoring**: {str(e)}")

    async def _video_monitoring_loop(self, ctx, duration: int, fps: float = 0.05):
        """Live video monitoring loop with configurable FPS"""
        try:
            start_time = asyncio.get_event_loop().time()
            end_time = start_time + duration
            analysis_count = 0
            
            # Calculate sleep interval from FPS (frames per second)
            sleep_interval = 1.0 / fps
            
            while asyncio.get_event_loop().time() < end_time:
                try:
                    # Analyze current frame
                    result = await self.obs_service.analyze_current_frame(
                        "Brief analysis: What's happening now in the game?"
                    )
                    
                    if result and result.get('success'):
                        analysis_count += 1
                        analysis = result.get('analysis', 'No analysis')
                        timing = result.get('timing', {})
                        
                        # Send brief update
                        embed = discord.Embed(
                            title=f"🎮 Live Analysis #{analysis_count} ({fps} FPS)",
                            description=analysis[:500] + ("..." if len(analysis) > 500 else ""),
                            color=discord.Color.blue()
                        )
                        
                        if timing:
                            embed.add_field(
                                name="⏱️ Timing",
                                value=f"Analysis: {timing.get('total_time', 0):.1f}s\nNext in: {sleep_interval:.1f}s",
                                inline=True
                            )
                        
                        await ctx.send(embed=embed)
                    
                    # Wait based on FPS setting
                    await asyncio.sleep(sleep_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(5)  # Wait before retry
            
            # Monitoring completed
            await ctx.send(f"✅ **Live monitoring completed!** Analyzed {analysis_count} frames in {duration} seconds at {fps} FPS.")
            
        except asyncio.CancelledError:
            await ctx.send("🛑 **Live monitoring stopped.**")
        except Exception as e:
            self.logger.error(f"Video monitoring loop error: {e}")
            await ctx.send(f"❌ **Monitoring error**: {str(e)}")

    async def _video_stop_monitoring(self, ctx):
        """Stop live video monitoring"""
        try:
            if hasattr(self, '_video_monitoring_task') and self._video_monitoring_task and not self._video_monitoring_task.done():
                self._video_monitoring_task.cancel()
                await ctx.send("🛑 **Video monitoring stopped.**")
            else:
                await ctx.send("ℹ️ **No active video monitoring to stop.**")
                
        except Exception as e:
            self.logger.error(f"Video stop error: {e}")
            await ctx.send(f"❌ **Failed to stop monitoring**: {str(e)}")

    async def _video_status(self, ctx):
        """Show video monitoring status"""
        try:
            # Initialize OBS service if not available
            if not hasattr(self, 'obs_service') or not self.obs_service:
                from services.obs_video_service import OBSVideoService
                self.obs_service = OBSVideoService(self.app_context)
            
            # Get OBS info
            obs_info = self.obs_service.get_obs_info()
            
            embed = discord.Embed(
                title="📺 Video Analysis Status",
                color=discord.Color.blue()
            )
            
            # OBS Connection
            if obs_info.get('connected', False):
                embed.add_field(
                    name="📺 OBS Connection",
                    value=f"✅ Connected to OBS {obs_info.get('obs_version', 'Unknown')}",
                    inline=False
                )
                embed.add_field(
                    name="🎬 Current Scene",
                    value=obs_info.get('current_scene', 'Unknown'),
                    inline=True
                )
            else:
                embed.add_field(
                    name="📺 OBS Connection",
                    value="❌ Not connected to OBS",
                    inline=False
                )
            
            # Monitoring Status
            monitoring_active = (hasattr(self, '_video_monitoring_task') and 
                               self._video_monitoring_task and 
                               not self._video_monitoring_task.done())
            
            embed.add_field(
                name="🎮 Live Monitoring",
                value="🟢 Active" if monitoring_active else "🔴 Inactive",
                inline=True
            )
            
            # Video Settings
            video_settings = self.app_context.global_settings.get('VIDEO_ANALYSIS', {})
            embed.add_field(
                name="⚙️ Settings",
                value=f"Quality: {video_settings.get('quality', 'medium')}\n"
                      f"Enabled: {'✅' if video_settings.get('enabled', True) else '❌'}",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Video status error: {e}")
            await ctx.send(f"❌ **Status check failed**: {str(e)}")

    async def _video_settings(self, ctx):
        """Show current video settings"""
        try:
            video_settings = self.app_context.global_settings.get('VIDEO_ANALYSIS', {})
            
            embed = discord.Embed(
                title="⚙️ Video Analysis Settings",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="🎯 Analysis",
                value=f"Enabled: {'✅' if video_settings.get('enabled', True) else '❌'}\n"
                      f"Gaming Mode: {'✅' if video_settings.get('gaming_mode', True) else '❌'}\n"
                      f"Save Debug Frames: {'✅' if video_settings.get('save_debug_frames', True) else '❌'}",
                inline=False
            )
            
            embed.add_field(
                name="🔧 Performance (FPS Optimized)",
                value=f"Quality: {video_settings.get('quality', 'medium')}\n"
                      f"Recommended FPS: 0.05 (20s intervals)\n"
                      f"Ultra-Safe FPS: 0.033 (30s intervals)",
                inline=False
            )
            
            embed.add_field(
                name="📺 OBS Integration",
                value=f"Host: {self.app_context.global_settings.get('OBS_HOST', 'localhost')}\n"
                      f"Port: {self.app_context.global_settings.get('OBS_PORT', 4455)}\n"
                      f"Password: {'Set' if self.app_context.global_settings.get('OBS_PASSWORD') else 'None'}",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Video settings error: {e}")
            await ctx.send(f"❌ **Settings display failed**: {str(e)}")

    async def _video_quality(self, ctx, quality: str):
        """Set video quality"""
        try:
            # Update settings (this would ideally save to config file)
            if not hasattr(self.app_context, 'global_settings'):
                await ctx.send("❌ **Settings not available**")
                return
            
            if 'VIDEO_ANALYSIS' not in self.app_context.global_settings:
                self.app_context.global_settings['VIDEO_ANALYSIS'] = {}
            
            self.app_context.global_settings['VIDEO_ANALYSIS']['quality'] = quality
            
            # Quality settings mapping
            quality_settings = {
                'low': {'capture_quality': 0.5, 'recommended_fps': 0.033},
                'medium': {'capture_quality': 0.8, 'recommended_fps': 0.05},
                'high': {'capture_quality': 1.0, 'recommended_fps': 0.1}
            }
            
            settings = quality_settings.get(quality, quality_settings['medium'])
            self.app_context.global_settings['VIDEO_ANALYSIS'].update(settings)
            
            fps = settings['recommended_fps']
            interval = 1.0 / fps
            
            await ctx.send(f"✅ **Video quality set to {quality}**\n"
                         f"📊 Capture Quality: {settings['capture_quality']}\n"
                         f"🎯 Recommended FPS: {fps} ({interval:.1f}s intervals)")
            
        except Exception as e:
            self.logger.error(f"Video quality error: {e}")
            await ctx.send(f"❌ **Quality setting failed**: {str(e)}")

    async def _video_stream_command(self, ctx, subaction: str):
        """Handle streaming video commands with Qwen2.5-VL 1-minute collection periods"""
        try:
            if subaction.lower() == "start":
                await self._video_stream_start(ctx)
            elif subaction.lower() == "stop":
                await self._video_stream_stop(ctx)
            elif subaction.lower() == "status":
                await self._video_stream_status(ctx)
            else:
                embed = discord.Embed(
                    title="🎬 Qwen2.5-VL Video Streaming",
                    description="CUDA-accelerated vision analysis with 1-minute collection periods",
                    color=discord.Color.purple()
                )
                embed.add_field(
                    name="📸 Collection & Analysis",
                    value="`!video stream start` - Start 1-minute collection periods\n"
                          "`!video stream stop` - Stop streaming analysis\n"
                          "`!video stream status` - Check streaming status",
                    inline=False
                )
                embed.add_field(
                    name="🚀 How It Works",
                    value="• **1-minute collection** - Captures frames every 10s\n"
                          "• **Individual analysis** - Each frame analyzed separately\n"
                          "• **Comprehensive summary** - All analyses sent to Qwen2.5-VL\n"
                          "• **Gaming commentary** - Professional-style commentary",
                    inline=False
                )
                embed.add_field(
                    name="⚡ Performance",
                    value="• Frame analysis: ~4-8 seconds per frame\n"
                          "• Collection period: 60 seconds (6 frames)\n"
                          "• Comprehensive analysis: ~10-15 seconds\n"
                          "• GPU VRAM usage: ~940MB",
                    inline=False
                )
                await ctx.send(embed=embed)
                
        except Exception as e:
            self.logger.error(f"Video stream command error: {e}")
            await ctx.send(f"❌ **Stream command error**: {str(e)}")

    async def _video_stream_start(self, ctx):
        """Start Qwen2.5-VL streaming with 1-minute collection periods"""
        try:
            # Check if already streaming
            if hasattr(self, '_qwen_streaming_service') and self._qwen_streaming_service and self._qwen_streaming_service.is_active():
                await ctx.send("⚠️ **Qwen2.5-VL streaming already active!** Use `!video stream stop` first.")
                return
            
            # Initialize Qwen2.5-VL streaming service
            from services.qwen_streaming_service import QwenStreamingService
            self._qwen_streaming_service = QwenStreamingService(self.app_context)
            
            # Initialize the streaming service
            if not await self._qwen_streaming_service.initialize():
                await ctx.send("❌ **Failed to initialize Qwen2.5-VL streaming service.**\n"
                              "Make sure the CUDA server is running on port 8083.")
                return
            
            # Set up Discord callback for results
            async def discord_callback(analysis_result: str, frame_count: int, duration: float):
                """Send Qwen2.5-VL comprehensive analysis results to Discord with game event detection"""
                try:
                    embed = discord.Embed(
                        title=f"🎯 Qwen2.5-VL Comprehensive Analysis ({frame_count} frames, {duration:.1f}s)",
                        description=analysis_result[:1500] + ("..." if len(analysis_result) > 1500 else ""),
                        color=discord.Color.green()
                    )
                    
                    embed.add_field(
                        name="📊 Analysis Details",
                        value=f"Frames Analyzed: {frame_count}\nCollection Time: {duration:.1f}s\nModel: Qwen2.5-VL 4-bit",
                        inline=True
                    )
                    
                    embed.add_field(
                        name="⏱️ Next Collection",
                        value="Starting new 1-minute collection period...",
                        inline=True
                    )
                    
                    await ctx.send(embed=embed)
                    
                    # Check for game events using conversational AI service
                    if (hasattr(self.app_context, 'conversational_ai_service') and 
                        self.app_context.conversational_ai_service):
                        
                        service = self.app_context.conversational_ai_service
                        
                        # Analyze text result for notable game events
                        analysis_lower = analysis_result.lower()
                        event_keywords = service.game_event_keywords.get(service.current_game, 
                                                                       service.game_event_keywords["generic"])
                        
                        detected_events = []
                        for keyword in event_keywords:
                            if keyword in analysis_lower:
                                # Calculate confidence based on keyword presence
                                confidence = 0.5  # Base confidence
                                keyword_count = analysis_lower.count(keyword)
                                confidence += min(keyword_count * 0.2, 0.3)
                                
                                if keyword in service.urgent_keywords:
                                    confidence += 0.2
                                
                                detected_events.append((keyword, confidence))
                        
                        if detected_events:
                            # Get highest confidence event
                            best_event = max(detected_events, key=lambda x: x[1])
                            event_type, confidence = best_event
                            
                            if confidence > 0.6:  # Only comment on high-confidence events
                                # Generate commentary
                                commentary = await service.generate_game_commentary(
                                    service.GameEvent(
                                        event_type=event_type,
                                        description=f"Detected {event_type} in game analysis",
                                        confidence=confidence,
                                        timestamp=time.time(),
                                        context=analysis_result
                                    )
                                )
                                
                                if commentary:
                                    # Send commentary to Discord
                                    commentary_embed = discord.Embed(
                                        title="🎯 Game Event Detected!",
                                        description=commentary,
                                        color=discord.Color.red()
                                    )
                                    commentary_embed.add_field(name="Event Type", value=event_type, inline=True)
                                    commentary_embed.add_field(name="Confidence", value=f"{confidence:.2f}", inline=True)
                                    
                                    await ctx.send(embed=commentary_embed)
                                    
                                    # Generate TTS for commentary
                                    if hasattr(self, 'tts_service') and self.tts_service:
                                        try:
                                            tts_audio = self.tts_service.generate_audio(commentary)
                                            if tts_audio:
                                                await self._queue_tts_audio(tts_audio)
                                                self.logger.info(f"🎵 TTS generated for game commentary ({len(tts_audio)} bytes)")
                                        except Exception as e:
                                            self.logger.error(f"❌ TTS generation error for commentary: {e}")
                    
                except Exception as e:
                    self.logger.error(f"Discord callback error: {e}")
                    await ctx.send(f"⚠️ Analysis completed but failed to display: {str(e)}")
            
            # Start streaming with Discord callback
            success = await self._qwen_streaming_service.start_streaming(
                analysis_callback=discord_callback
            )
            
            if success:
                embed = discord.Embed(
                    title="🎬 Qwen2.5-VL Streaming Started!",
                    description="Now collecting frames for 1 minute, then providing comprehensive gaming commentary",
                    color=discord.Color.green()
                )
                
                embed.add_field(
                    name="📸 Collection Settings",
                    value="• Resolution: 1920x1080\n• Quality: 95% JPEG\n• Frame Interval: 10 seconds\n• Collection Period: 60 seconds",
                    inline=True
                )
                
                embed.add_field(
                    name="🧠 Analysis Process",
                    value="• Individual frame analysis\n• Comprehensive summary generation\n• Professional gaming commentary\n• CUDA acceleration",
                    inline=True
                )
                
                embed.add_field(
                    name="⏱️ Performance",
                    value="• ~6 frames per collection period\n• ~4-8 seconds per frame analysis\n• ~10-15 seconds for comprehensive analysis\n• GPU VRAM: ~940MB",
                    inline=False
                )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ **Failed to start Qwen2.5-VL streaming.**\n"
                              "Check OBS connection and CUDA server availability.")
                
        except Exception as e:
            self.logger.error(f"Video stream start error: {e}")
            await ctx.send(f"❌ **Failed to start streaming**: {str(e)}")

    async def _video_stream_stop(self, ctx):
        """Stop Qwen2.5-VL streaming"""
        try:
            if hasattr(self, '_qwen_streaming_service') and self._qwen_streaming_service:
                was_active = self._qwen_streaming_service.is_active()
                await self._qwen_streaming_service.stop_streaming()
                
                if was_active:
                    status = self._qwen_streaming_service.get_status()
                    embed = discord.Embed(
                        title="🛑 Qwen2.5-VL Streaming Stopped",
                        color=discord.Color.red()
                    )
                    
                    embed.add_field(
                        name="📊 Session Summary",
                        value=f"Total Frames: {status.get('frame_count', 0)}\n"
                              f"Successful Analyses: {status.get('successful_analyses', 0)}\n"
                              f"Failed Analyses: {status.get('failed_analyses', 0)}\n"
                              f"Comprehensive Analyses: {status.get('comprehensive_analyses', 0)}",
                        inline=True
                    )
                    
                    if status.get('avg_analysis_time'):
                        embed.add_field(
                            name="⏱️ Performance",
                            value=f"Avg Frame Analysis: {status.get('avg_analysis_time', 0):.2f}s\n"
                                  f"Collection Duration: {status.get('collection_duration', 60)}s\n"
                                  f"Frame Interval: {status.get('frame_interval', 10)}s",
                            inline=True
                        )
                    
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("ℹ️ **No active Qwen2.5-VL streaming to stop.**")
            else:
                await ctx.send("ℹ️ **No Qwen2.5-VL streaming service initialized.**")
                
        except Exception as e:
            self.logger.error(f"Video stream stop error: {e}")
            await ctx.send(f"❌ **Failed to stop streaming**: {str(e)}")

    async def _video_stream_status(self, ctx):
        """Show Qwen2.5-VL streaming status"""
        try:
            embed = discord.Embed(
                title="🎬 Qwen2.5-VL Streaming Status",
                color=discord.Color.blue()
            )
            
            # Check if streaming service exists and is active
            if hasattr(self, '_qwen_streaming_service') and self._qwen_streaming_service:
                is_active = self._qwen_streaming_service.is_active()
                status = self._qwen_streaming_service.get_status()
                
                embed.add_field(
                    name="🎬 Streaming Status",
                    value="🟢 **ACTIVE**" if is_active else "🔴 **INACTIVE**",
                    inline=True
                )
                
                if is_active and status:
                    embed.add_field(
                        name="📊 Current Session",
                        value=f"Total Frames: {status.get('frame_count', 0)}\n"
                              f"Successful: {status.get('successful_analyses', 0)}\n"
                              f"Failed: {status.get('failed_analyses', 0)}\n"
                              f"Comprehensive: {status.get('comprehensive_analyses', 0)}",
                        inline=True
                    )
                    
                    if status.get('running_time'):
                        embed.add_field(
                            name="⏱️ Session Time",
                            value=f"Running: {status.get('running_time', 0):.1f}s\n"
                                  f"Collection: {status.get('collection_duration', 60)}s\n"
                                  f"Interval: {status.get('frame_interval', 10)}s",
                            inline=True
                        )
                    
                    if status.get('avg_analysis_time'):
                        embed.add_field(
                            name="⚡ Performance",
                            value=f"Avg Frame Analysis: {status.get('avg_analysis_time', 0):.2f}s\n"
                                  f"Last Analysis: {status.get('last_analysis_time', 0):.2f}s",
                            inline=False
                        )
            else:
                embed.add_field(
                    name="🎬 Streaming Status",
                    value="🔴 **NOT INITIALIZED**",
                    inline=True
                )
            
            # OBS Connection Status
            if hasattr(self, 'obs_service') and self.obs_service:
                obs_info = self.obs_service.get_obs_info()
                embed.add_field(
                    name="📺 OBS Connection",
                    value="✅ Connected" if obs_info.get('connected', False) else "❌ Disconnected",
                    inline=True
                )
            else:
                embed.add_field(
                    name="📺 OBS Connection",
                    value="❌ Not initialized",
                    inline=True
                )
            
            # CUDA Server Status
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:8083/health", timeout=5) as response:
                        cuda_status = "✅ Available" if response.status == 200 else "❌ Unavailable"
            except:
                cuda_status = "❌ Unavailable"
            
            embed.add_field(
                name="🚀 CUDA Server",
                value=cuda_status,
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Video stream status error: {e}")
            await ctx.send(f"❌ **Failed to get streaming status**: {str(e)}")

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
        # Set the event loop for async operations
        self.loop = asyncio.get_event_loop()
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
                model_size = self.settings.get('WHISPER_MODEL_SIZE', 'large')
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
                    model_size = self.settings.get('WHISPER_MODEL_SIZE', 'large')
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
        
        # Start Discord connection monitoring
        await self._start_connection_monitor()
        
        # Initialize virtual audio capture after voice connection
        await self._initialize_virtual_audio_capture()

    async def _auto_join_voice_channel(self):
        """Automatically join the configured voice channel with improved stability."""
        try:
            # Get voice channel ID from settings
            voice_channel_id = self.app_context.global_settings.get('DISCORD_VOICE_CHANNEL_ID')
            if not voice_channel_id:
                self.logger.warning("⚠️ No voice channel ID configured")
                return False
            
            # Find the voice channel
            voice_channel = self.get_channel(int(voice_channel_id))
            if not voice_channel:
                self.logger.error(f"❌ Voice channel {voice_channel_id} not found")
                return False
            
            # Check if already connected
            if self.voice_clients:
                for vc in self.voice_clients:
                    if vc.channel and vc.channel.id == voice_channel_id:
                        if vc.is_connected():
                            self.logger.info(f"Already connected to voice channel: {voice_channel.name}")
                            return True
                        else:
                            # Disconnect from broken connection
                            try:
                                await vc.disconnect(force=True)
                                await asyncio.sleep(2)
                            except:
                                pass
            
            # Check for existing voice connections in the target channel
            existing_connections = []
            for member in voice_channel.members:
                if member.voice and member.voice.channel:
                    existing_connections.append({
                        'user_name': member.display_name,
                        'user_id': member.id,
                        'session_id': getattr(member.voice, 'session_id', 'unknown')
                    })
            
            if existing_connections:
                self.logger.info(f"Found {len(existing_connections)} existing voice connections in {voice_channel.name}:")
                for conn in existing_connections:
                    self.logger.info(f"  - {conn['user_name']} (ID: {conn['user_id']}, Session: {conn['session_id']})")
            
            # Attempt connection with improved error handling
            max_retries = 5
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Attempting to join voice channel: {voice_channel.name} (Attempt {attempt + 1}/{max_retries})")
                    
                    # Disconnect from any existing connections first
                    for vc in self.voice_clients:
                        try:
                            await vc.disconnect(force=True)
                            await asyncio.sleep(1)
                        except:
                            pass
                    
                    # Connect with minimal parameters and proper timeout
                    voice_client = await asyncio.wait_for(
                        voice_channel.connect(timeout=30.0),
                        timeout=35.0
                    )
                    
                    # Wait for connection to stabilize
                    await asyncio.sleep(3)
                    
                    if voice_client and voice_client.is_connected():
                        self.logger.info(f"✅ Successfully connected to voice channel: {voice_channel.name}")
                        return True
                    else:
                        self.logger.warning(f"Voice client connected but not ready (Attempt {attempt + 1})")
                        
                except asyncio.TimeoutError:
                    self.logger.error(f"❌ Voice connection timeout (Attempt {attempt + 1})")
                except discord.ClientException as e:
                    if "Already connected" in str(e):
                        self.logger.warning("⚠️ Already connected, disconnecting first...")
                        for vc in self.voice_clients:
                            await vc.disconnect(force=True)
                        await asyncio.sleep(3)
                        continue
                    else:
                        self.logger.error(f"❌ ClientException (Attempt {attempt + 1}): {e}")
                except discord.HTTPException as e:
                    if e.status == 4006:
                        self.logger.warning(f"⚠️ WebSocket 4006 error (Attempt {attempt + 1}) - Session expired")
                        # For 4006 errors, wait longer and try to refresh the connection
                        await asyncio.sleep(10)
                        
                        # Try to refresh the Discord connection
                        try:
                            await self.close()
                            await asyncio.sleep(5)
                            await self.start(self.app_context.global_settings.get('DISCORD_BOT_TOKEN'))
                            await asyncio.sleep(3)
                        except Exception as refresh_error:
                            self.logger.error(f"Failed to refresh Discord connection: {refresh_error}")
                    else:
                        self.logger.error(f"❌ HTTPException (Attempt {attempt + 1}): {e}")
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error (Attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 30)  # Exponential backoff with cap
                    self.logger.info(f"⏳ Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
            
            self.logger.error(f"Failed to connect to voice channel after {max_retries} attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in _auto_join_voice_channel: {e}")
            return False

    async def _check_discord_connection_health(self):
        """Monitor Discord connection health and reconnect if needed."""
        try:
            # Check if we have any voice clients
            if not self.voice_clients:
                self.logger.warning("⚠️ No voice clients found, attempting to reconnect...")
                await self._auto_join_voice_channel()
                return
            
            # Check each voice client
            for vc in self.voice_clients:
                if not vc.is_connected():
                    self.logger.warning(f"⚠️ Voice client disconnected from {vc.channel.name if vc.channel else 'Unknown'}")
                    try:
                        await vc.disconnect(force=True)
                    except:
                        pass
                    continue
                
                # Check websocket health
                try:
                    # Check if websocket exists and is connected
                    if hasattr(vc, 'ws') and vc.ws:
                        # Check the actual websocket connection
                        if hasattr(vc.ws, 'ws') and vc.ws.ws:
                            # Check if the websocket is closed
                            if hasattr(vc.ws.ws, 'closed') and vc.ws.ws.closed:
                                self.logger.warning(f"⚠️ Voice WebSocket closed for {vc.channel.name if vc.channel else 'Unknown'}")
                                try:
                                    await vc.disconnect(force=True)
                                except:
                                    pass
                                continue
                        else:
                            self.logger.warning(f"⚠️ Voice WebSocket not available for {vc.channel.name if vc.channel else 'Unknown'}")
                            try:
                                await vc.disconnect(force=True)
                            except:
                                pass
                            continue
                    else:
                        self.logger.warning(f"⚠️ No WebSocket found for {vc.channel.name if vc.channel else 'Unknown'}")
                        try:
                            await vc.disconnect(force=True)
                        except:
                            pass
                        continue
                        
                except Exception as e:
                    self.logger.error(f"❌ Error checking WebSocket health: {e}")
                    try:
                        await vc.disconnect(force=True)
                    except:
                        pass
                    continue
                
                # If we get here, the connection is healthy
                self.logger.debug(f"✅ Voice connection healthy for {vc.channel.name if vc.channel else 'Unknown'}")
                
        except Exception as e:
            self.logger.error(f"❌ Error in connection health check: {e}")

    async def _start_connection_monitor(self):
        """Start Discord connection monitoring"""
        try:
            self.logger.info("🔍 Starting Discord connection monitor...")
            
            async def monitor_loop():
                while not self.is_closed():
                    try:
                        await self._check_discord_connection_health()
                        await asyncio.sleep(30)  # Check every 30 seconds
                    except Exception as e:
                        self.logger.error(f"Connection monitor error: {e}")
                        await asyncio.sleep(60)  # Wait longer on error
            
            # Start monitoring in background
            asyncio.create_task(monitor_loop())
            self.logger.info("✅ Discord connection monitor started")
            
        except Exception as e:
            self.logger.error(f"Failed to start connection monitor: {e}")

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
                        transcription = self.virtual_audio.transcription_queue.get_nowait()
                        
                        # Use a default user name since we don't have user info in the string-only approach
                        user_name = "VirtualAudio"
                        
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
                            
                            # Process with Streaming LLM for real-time responses
                            use_streaming = self.app_context.global_settings.get('STREAMING_RESPONSE', {}).get('enabled', True)
                            
                            if use_streaming:
                                try:
                                    await self._process_streaming_response(transcription, user_name, text_channel)
                                except Exception as e:
                                    self.logger.error(f"❌ Streaming LLM processing error: {e}")
                                    # Fallback to regular LLM
                                    await self._process_regular_response(transcription, user_name, text_channel)
                            else:
                                # Use regular LLM processing
                                await self._process_regular_response(transcription, user_name, text_channel)
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
            
            # Import AzureTTSService for isinstance check
            try:
                from services.tts_service_azure import AzureTTSService
            except ImportError:
                AzureTTSService = None
            
            # Check if streaming TTS is enabled
            use_streaming = self.app_context.global_settings.get('TTS_SERVER', {}).get('enable_streaming', True)
            
            if use_streaming and hasattr(self.tts_service, 'generate_audio_streaming'):
                # Use new streaming TTS approach
                self.logger.info("🎵 Using streaming TTS for progressive playback")
                
                self.feedback_prevention.start_tts_playback(response_text)
                await self._start_tts_queue_processor()
                successful_chunks = 0
                def audio_chunk_callback(audio_data: bytes):
                    nonlocal successful_chunks
                    try:
                        if hasattr(self, 'tts_queue') and self.tts_queue:
                            try:
                                self.tts_queue.put_nowait(audio_data)
                                successful_chunks += 1
                                self.logger.info(f"🎵 Streaming chunk queued for ordered playback ({len(audio_data)} bytes)")
                            except asyncio.QueueFull:
                                self.logger.warning("⚠️ TTS queue full, dropping chunk")
                        else:
                            self.logger.warning("⚠️ TTS queue not available")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to queue streaming chunk: {e}")
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    None,
                    self.tts_service.generate_audio_streaming,
                    response_text,
                    audio_chunk_callback
                )
                self.feedback_prevention.stop_tts_playback()
                if success and successful_chunks > 0:
                    self.logger.info(f"✅ Streaming TTS completed: {successful_chunks} chunks streamed")
                else:
                    self.logger.error("❌ Streaming TTS failed")
            else:
                self.logger.info("🔊 Using traditional chunked TTS")
                chunks = self._chunk_response_for_tts(response_text, max_chunk_length=150)
                if len(chunks) == 1:
                    self.logger.info("🔊 Single chunk TTS generation...")
                    tts_text = chunks[0]
                    self.feedback_prevention.start_tts_playback(tts_text)
                    tts_audio = None
                    if AzureTTSService and isinstance(self.tts_service, AzureTTSService):
                        tts_audio = await self.tts_service.synthesize_speech(tts_text)
                    else:
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
                    self.logger.info(f"🔊 Multi-chunk TTS generation ({len(chunks)} chunks)...")
                    full_text = " ".join(chunks)
                    self.feedback_prevention.start_tts_playback(full_text)
                    successful_chunks = 0
                    if AzureTTSService and isinstance(self.tts_service, AzureTTSService):
                        audio_chunks = await self.tts_service.synthesize_speech_chunked(full_text, max_chunk_length=150)
                        for i, tts_audio in enumerate(audio_chunks):
                            try:
                                if tts_audio:
                                    self.logger.info(f"✅ Chunk {i+1} TTS audio generated (Azure)")
                                    await self._play_single_tts_chunk(tts_audio)
                                    successful_chunks += 1
                                    if i < len(audio_chunks) - 1:
                                        await asyncio.sleep(0.3)
                                else:
                                    self.logger.warning(f"⚠️ Chunk {i+1} TTS generation failed (Azure)")
                            except Exception as e:
                                self.logger.error(f"❌ Error processing TTS chunk {i+1} (Azure): {e}")
                                continue
                    else:
                        for i, chunk in enumerate(chunks):
                            try:
                                self.logger.info(f"🔊 Generating TTS chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                                loop = asyncio.get_event_loop()
                                tts_audio = await loop.run_in_executor(
                                    None,
                                    self.tts_service.generate_audio,
                                    chunk
                                )
                                if tts_audio:
                                    self.logger.info(f"✅ Chunk {i+1} TTS audio generated")
                                    await self._play_single_tts_chunk(tts_audio)
                                    successful_chunks += 1
                                    if i < len(chunks) - 1:
                                        await asyncio.sleep(0.3)
                                else:
                                    self.logger.warning(f"⚠️ Chunk {i+1} TTS generation failed")
                            except Exception as e:
                                self.logger.error(f"❌ Error processing TTS chunk {i+1}: {e}")
                                continue
                    self.feedback_prevention.stop_tts_playback()
                    if successful_chunks > 0:
                        self.logger.info(f"✅ Multi-chunk TTS completed ({successful_chunks}/{len(chunks)} successful)")
                    else:
                        self.logger.error("❌ All TTS chunks failed")
        except Exception as e:
            self.logger.error(f"❌ Error in chunked TTS generation: {e}")
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

    async def _process_streaming_response(self, transcription: str, user_name: str, text_channel):
        """Process user input with real-time streaming response generation and immediate TTS"""
        try:
            self.logger.info("🎵 Processing with Real-Time Streaming LLM for immediate responses...")
            
            # Initialize real-time streaming service if not already done
            if not hasattr(self, 'real_time_streaming_service'):
                self.logger.warning("⚠️ Real-Time Streaming Service not available, falling back to regular processing")
                await self._process_regular_response(transcription, user_name, text_channel)
                return
            
            # Start TTS queue processor for ordered playback
            await self._start_tts_queue_processor()
            
            # Create callbacks for real-time streaming
            response_chunks = []
            current_response = ""
            
            def text_callback(chunk_text: str):
                """Callback for text chunks - update Discord in real-time"""
                nonlocal current_response
                current_response += chunk_text
                response_chunks.append(chunk_text)
                self.logger.debug(f"🎵 Text chunk received: '{chunk_text[:30]}...'")
            
            async def tts_callback(chunk_text: str):
                """Callback for TTS chunks - generate audio immediately"""
                try:
                    if chunk_text.strip():
                        self.logger.info(f"🎵 Generating TTS for chunk: '{chunk_text[:50]}...'")
                        
                        # Check if it's Azure TTS
                        try:
                            from services.tts_service_azure import AzureTTSService
                            if isinstance(self.tts_service, AzureTTSService):
                                tts_audio = await self.tts_service.synthesize_speech(chunk_text)
                            else:
                                tts_audio = self.tts_service.generate_audio(chunk_text)
                        except ImportError:
                            tts_audio = self.tts_service.generate_audio(chunk_text)
                        
                        if tts_audio:
                            await self._queue_tts_audio(tts_audio)
                            self.logger.info(f"🎵 ✅ TTS generated and queued for chunk ({len(tts_audio)} bytes)")
                        else:
                            self.logger.warning(f"🎵 ⚠️ Failed to generate TTS for chunk")
                except Exception as e:
                    self.logger.error(f"🎵 ❌ Error in TTS callback: {e}")
            
            def progress_callback(chunk_text: str, is_final: bool):
                """Callback for progress updates"""
                if is_final:
                    self.logger.info("🎵 Streaming response completed")
                else:
                    self.logger.debug(f"🎵 Progress: {len(response_chunks)} chunks processed")
            
            # Calculate estimated duration for feedback prevention
            estimated_words = len(transcription.split()) * 2  # Rough estimate
            estimated_duration = max(3.0, estimated_words / 2.5)  # ~2.5 words per second
            
            # Start feedback prevention
            self.feedback_prevention.start_tts_playback("Streaming response", estimated_duration=estimated_duration)
            self.logger.info(f"🛡️ Started feedback prevention for {estimated_duration:.1f}s")
            
            # Generate streaming response with real-time callbacks
            full_response = await self.real_time_streaming_service.handle_user_text_query_streaming(
                user_text=transcription,
                user_name=user_name,
                text_callback=text_callback,
                tts_callback=tts_callback,
                progress_callback=progress_callback
            )
            
            # Send complete response to Discord if we have chunks
            if response_chunks:
                clean_response = self._strip_think_tags(full_response)
                await text_channel.send(f"🤖 **DanzarAI**: {clean_response}")
                self.logger.info(f"🤖 Sent streaming response: '{clean_response[:100]}...'")
                
                # Add to short-term memory
                if self.app_context.short_term_memory_service:
                    self.app_context.short_term_memory_service.add_entry(
                        user_name=user_name,
                        content=clean_response,
                        entry_type='bot_response'
                    )
            else:
                await text_channel.send("🤖 **DanzarAI**: I heard you, but I'm not sure how to respond to that.")
                # Stop feedback prevention if no response
                self.feedback_prevention.stop_tts_playback()
            
            self.logger.info(f"🎵 ✅ Real-time streaming completed with {len(response_chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Error in real-time streaming response processing: {e}")
            self.feedback_prevention.stop_tts_playback()
            # Fallback to regular processing
            await self._process_regular_response(transcription, user_name, text_channel)

    async def _process_regular_response(self, transcription: str, user_name: str, text_channel):
        """Process user input with regular LLM and traditional TTS"""
        try:
            if self.llm_service:
                self.logger.info("🧠 Processing with regular LLM...")
                
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
                    
                    # Add bot response to short-term memory
                    if self.app_context.short_term_memory_service:
                        self.app_context.short_term_memory_service.add_entry(
                            user_name=user_name,
                            content=clean_response,
                            entry_type='bot_response'
                        )
                    
                    # Generate and play TTS using chunked approach
                    await self._generate_and_play_chunked_tts(clean_response, text_channel, user_name)
                    
                else:
                    await text_channel.send("🤖 **DanzarAI**: I heard you, but I'm not sure how to respond to that.")
                    
            else:
                # Fallback response
                response = f"I heard you say: '{transcription}'. My LLM service isn't available right now."
                await text_channel.send(f"🤖 **DanzarAI**: {response}")
                self.logger.info("🤖 Sent fallback response (no LLM service)")
                
        except Exception as e:
            self.logger.error(f"❌ Error in regular response processing: {e}")
            await text_channel.send("🤖 **DanzarAI**: Sorry, I had trouble processing that.")

    async def _qwen_analyze(self, ctx, prompt: str):
        """Analyze current OBS frame with Qwen2.5-VL"""
        try:
            # Check if Qwen2.5-VL integration is available
            if not hasattr(self.app_context, 'qwen_vl_integration') or not self.app_context.qwen_vl_integration:
                await ctx.send("❌ **Qwen2.5-VL Integration not available**\n"
                              "Make sure the CUDA server is running and the service is initialized")
                return
            
            await ctx.send("🎯 **Analyzing with Qwen2.5-VL (CUDA)...** 🔍")
            
            # Get OBS frame from the integration service
            try:
                from services.obs_video_service import OBSVideoService
                obs_service = OBSVideoService(self.app_context)
                await obs_service.initialize()
                
                frame = obs_service.capture_obs_screenshot()
                if frame is None:
                    await ctx.send("❌ **Failed to capture OBS frame**\nMake sure OBS Studio is running with NDI source")
                    return
                
            except Exception as e:
                self.logger.error(f"OBS capture error: {e}")
                await ctx.send("❌ **Failed to capture OBS frame**\nCheck OBS connection and source configuration")
                return
            
            # Analyze with Qwen2.5-VL
            start_time = time.time()
            analysis = await self.app_context.qwen_vl_integration.analyze_obs_frame(frame, prompt)
            analysis_time = time.time() - start_time
            
            if analysis:
                embed = discord.Embed(
                    title="🎯 Qwen2.5-VL Analysis Result",
                    description=analysis,
                    color=discord.Color.purple()
                )
                
                embed.add_field(
                    name="⚡ Performance",
                    value=f"⏱️ {analysis_time:.2f}s | 🎯 CUDA Accelerated",
                    inline=True
                )
                
                embed.add_field(
                    name="💡 Tip",
                    value="Use `!qwen quick` for fast analysis or `!qwen detailed` for comprehensive analysis",
                    inline=False
                )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ **Qwen2.5-VL analysis failed**\nCheck if the CUDA server is running properly")
                
        except Exception as e:
            self.logger.error(f"Qwen analysis error: {e}")
            await ctx.send(f"❌ **Qwen2.5-VL analysis error**: {str(e)}")
    
    async def _qwen_commentary(self, ctx):
        """Generate live gaming commentary with Qwen2.5-VL"""
        try:
            if not hasattr(self.app_context, 'qwen_vl_integration') or not self.app_context.qwen_vl_integration:
                await ctx.send("❌ **Qwen2.5-VL Integration not available**")
                return
            
            await ctx.send("🎮 **Generating live gaming commentary...** 🎯")
            
            # Get OBS frame
            try:
                from services.obs_video_service import OBSVideoService
                obs_service = OBSVideoService(self.app_context)
                await obs_service.initialize()
                
                frame = obs_service.capture_obs_screenshot()
                if frame is None:
                    await ctx.send("❌ **Failed to capture OBS frame**")
                    return
                
            except Exception as e:
                self.logger.error(f"OBS capture error: {e}")
                await ctx.send("❌ **Failed to capture OBS frame**")
                return
            
            # Generate commentary
            commentary = await self.app_context.qwen_vl_integration.generate_gaming_commentary(frame)
            
            if commentary:
                embed = discord.Embed(
                    title="🎮 Live Gaming Commentary",
                    description=commentary,
                    color=discord.Color.green()
                )
                
                embed.add_field(
                    name="🎯 Qwen2.5-VL",
                    value="Real-time gaming insights with CUDA acceleration",
                    inline=False
                )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ **Commentary generation failed**")
                
        except Exception as e:
            self.logger.error(f"Qwen commentary error: {e}")
            await ctx.send(f"❌ **Commentary error**: {str(e)}")
    
    async def _qwen_status(self, ctx):
        """Show Qwen2.5-VL performance statistics"""
        try:
            if not hasattr(self.app_context, 'qwen_vl_integration') or not self.app_context.qwen_vl_integration:
                await ctx.send("❌ **Qwen2.5-VL Integration not available**")
                return
            
            stats = self.app_context.qwen_vl_integration.get_performance_stats()
            
            embed = discord.Embed(
                title="📊 Qwen2.5-VL Performance Stats",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="🎯 Analysis Stats",
                value=f"✅ **Successful**: {stats['successful_analyses']}\n"
                      f"❌ **Failed**: {stats['failed_analyses']}\n"
                      f"📈 **Success Rate**: {stats['success_rate']:.1f}%",
                inline=True
            )
            
            embed.add_field(
                name="⏱️ Performance",
                value=f"⚡ **Avg Time**: {stats['average_time']:.2f}s\n"
                      f"🎯 **CUDA**: {'✅' if stats['cuda_available'] else '❌'}\n"
                      f"🔄 **Fallback**: {'✅' if stats['transformers_fallback'] else '❌'}",
                inline=True
            )
            
            if stats['recent_times']:
                recent_avg = sum(stats['recent_times']) / len(stats['recent_times'])
                embed.add_field(
                    name="📈 Recent Performance",
                    value=f"🕒 **Last 5 Avg**: {recent_avg:.2f}s\n"
                          f"📊 **Recent Times**: {', '.join([f'{t:.1f}s' for t in stats['recent_times'][-3:]])}",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Qwen status error: {e}")
            await ctx.send(f"❌ **Status error**: {str(e)}")
    
    async def _qwen_history(self, ctx):
        """Show recent Qwen2.5-VL analysis history"""
        try:
            if not hasattr(self.app_context, 'qwen_vl_integration') or not self.app_context.qwen_vl_integration:
                await ctx.send("❌ **Qwen2.5-VL Integration not available**")
                return
            
            history = self.app_context.qwen_vl_integration.get_commentary_history(limit=5)
            
            if not history:
                await ctx.send("📝 **No recent analyses found**")
                return
            
            embed = discord.Embed(
                title="📝 Recent Qwen2.5-VL Analyses",
                color=discord.Color.green()
            )
            
            for i, entry in enumerate(reversed(history), 1):
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                commentary = entry['commentary'][:100] + "..." if len(entry['commentary']) > 100 else entry['commentary']
                
                embed.add_field(
                    name=f"📊 Analysis #{i} ({timestamp})",
                    value=commentary,
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Qwen history error: {e}")
            await ctx.send(f"❌ **History error**: {str(e)}")

    async def process_simple_discord_audio(self, audio_data: bytes):
        """Process audio data from Discord voice channel"""
        try:
            # Convert audio data to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process with VAD to detect speech
            is_speech, is_speech_end = self.vad.process_audio_chunk(audio_data)
            
            if is_speech:
                self.logger.debug("🎤 Speech detected in Discord audio")
                
                # Get speech audio when speech ends
                if is_speech_end:
                    speech_audio = self.vad.get_speech_audio()
                    if speech_audio is not None:
                        # Transcribe the speech
                        transcription = await self.transcribe_audio(speech_audio)
                        if transcription and len(transcription.strip()) > 0:
                            self.logger.info(f"🎯 Discord transcription: '{transcription}'")
                            
                            # Process the transcription
                            await self._process_transcription(transcription, "Discord User")
                            
        except Exception as e:
            self.logger.error(f"Error processing Discord audio: {e}")

    async def _process_transcription(self, transcription: str, user_name: str):
        """Process transcription with conversational AI service for turn-taking"""
        try:
            # Use conversational AI service if available
            if (hasattr(self.app_context, 'conversational_ai_service') and 
                self.app_context.conversational_ai_service):
                
                service = self.app_context.conversational_ai_service
                
                # Get current game context from video analysis if available
                game_context = None
                if hasattr(self.app_context, 'qwen_vl_integration') and self.app_context.qwen_vl_integration:
                    try:
                        # Get recent game context from video analysis
                        game_context = f"Playing {service.current_game}"
                    except:
                        pass
                
                # Process with conversational AI service
                response = await service.process_user_message(
                    user_id="discord_user",  # We don't have user ID in this context
                    user_name=user_name,
                    message=transcription,
                    game_context=game_context
                )
                
                if response:
                    # Update conversation state
                    service.conversation_state = service.ConversationState.SPEAKING
                    
                    # Send response to Discord
                    text_channel = self.get_configured_text_channel()
                    if text_channel:
                        await text_channel.send(f"🤖 **Danzar**: {response}")
                    
                    # Generate TTS if available
                    if hasattr(self, 'tts_service') and self.tts_service:
                        try:
                            tts_audio = self.tts_service.generate_audio(response)
                            if tts_audio:
                                await self._queue_tts_audio(tts_audio)
                                self.logger.info(f"🎵 TTS generated for conversational response ({len(tts_audio)} bytes)")
                        except Exception as e:
                            self.logger.error(f"❌ TTS generation error: {e}")
                    
                    # Reset conversation state after speaking
                    await asyncio.sleep(1)  # Brief pause
                    service.conversation_state = service.ConversationState.IDLE
                else:
                    self.logger.info(f"⏳ User {user_name} must wait for turn")
            else:
                # Fallback to regular processing
                text_channel = self.get_configured_text_channel()
                if text_channel:
                    await self._process_regular_response(transcription, user_name, text_channel)
                else:
                    self.logger.warning("No text channel configured for responses")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing transcription with conversational AI: {e}")
            # Fallback to regular processing
            text_channel = self.get_configured_text_channel()
            if text_channel:
                await self._process_regular_response(transcription, user_name, text_channel)

    def process_virtual_audio_sync(self, audio_data: np.ndarray):
        """Process virtual audio data synchronously (called from audio callback thread)"""
        try:
            # This method is called from the audio callback thread
            # We need to schedule the async processing in the main event loop
            if hasattr(self, 'loop') and self.loop:
                # Schedule the async processing
                asyncio.run_coroutine_threadsafe(
                    self._process_virtual_audio_async(audio_data), 
                    self.loop
                )
        except Exception as e:
            self.logger.error(f"Error in process_virtual_audio_sync: {e}")

    async def _process_virtual_audio_async(self, audio_data: np.ndarray):
        """Process virtual audio data asynchronously (runs in main event loop)"""
        try:
            # Transcribe the audio
            if hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service:
                # Use faster-whisper for transcription
                transcription = await self.app_context.faster_whisper_stt_service.transcribe_audio_data(audio_data)
            else:
                # Fallback to basic transcription
                transcription = "Audio received but no STT service available"
            
            if transcription and len(transcription.strip()) > 0:
                self.logger.info(f"🎤 Virtual audio transcription: '{transcription}'")
                
                # Process the transcription
                await self._process_transcription(transcription, "Virtual Audio User")
                
        except Exception as e:
            self.logger.error(f"Error in _process_virtual_audio_async: {e}")

    def process_vad_transcription(self, transcription: str):
        """Process VAD transcription (called from VAD service)"""
        try:
            if hasattr(self, 'loop') and self.loop:
                # Schedule the async processing
                asyncio.run_coroutine_threadsafe(
                    self._process_transcription(transcription, "VAD User"), 
                    self.loop
                )
        except Exception as e:
            self.logger.error(f"Error in process_vad_transcription: {e}")

    def process_offline_voice_response(self, response: str):
        """Process offline voice response (called from offline VAD service)"""
        try:
            if hasattr(self, 'loop') and self.loop:
                # Schedule the async processing
                asyncio.run_coroutine_threadsafe(
                    self._process_offline_response_async(response), 
                    self.loop
                )
        except Exception as e:
            self.logger.error(f"Error in process_offline_voice_response: {e}")

    async def _process_offline_response_async(self, response: str):
        """Process offline response asynchronously"""
        try:
            if response and len(response.strip()) > 0:
                self.logger.info(f"🤖 Offline response: '{response[:100]}...'")
                
                # Send to text channel if available
                text_channel = self.get_configured_text_channel()
                if text_channel:
                    await text_channel.send(f"🤖 **DanzarAI**: {response}")
                
                # Generate TTS if available
                if hasattr(self, 'tts_service') and self.tts_service:
                    try:
                        tts_audio = self.tts_service.generate_audio(response)
                        if tts_audio:
                            await self._queue_tts_audio(tts_audio)
                    except Exception as e:
                        self.logger.error(f"TTS generation failed: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in _process_offline_response_async: {e}")

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Strip markdown formatting for TTS processing"""
        import re
        # Remove Discord markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'__(.*?)__', r'\1', text)      # Underline
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove emojis
        text = re.sub(r'<:[^>]+>', '', text)
        text = re.sub(r'<a:[^>]+>', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _strip_think_tags(self, text: str) -> str:
        """Strip think tags from LLM responses"""
        import re
        # Remove <think>...</think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove [THINK]...[/THINK] tags
        text = re.sub(r'\[THINK\].*?\[/THINK\]', '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data using available STT services"""
        try:
            # Try faster-whisper first
            if hasattr(self.app_context, 'faster_whisper_stt_service') and self.app_context.faster_whisper_stt_service:
                return await self.app_context.faster_whisper_stt_service.transcribe_audio_data(audio_data)
            
            # Fallback to other STT services if available
            if hasattr(self.app_context, 'qwen_omni_service') and self.app_context.qwen_omni_service:
                # Use Qwen2.5-Omni for transcription
                return await self._transcribe_with_omni(audio_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in transcribe_audio: {e}")
            return None

    async def _initialize_virtual_audio_capture(self):
        """Initialize virtual audio capture for VB-Audio Virtual Cable."""
        try:
            self.logger.info("🎤 Initializing virtual audio capture...")
            
            # Initialize virtual audio if not already done
            if not self.virtual_audio:
                self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors
                self.logger.info("✅ Virtual audio capture object created")
            
            # Get list of virtual audio devices
            virtual_devices = self.virtual_audio.list_audio_devices()
            self.logger.info(f"Found {len(virtual_devices)} virtual audio devices")
            
            if virtual_devices:
                # Use the first VB-Audio device
                selected_device_id, selected_device_name = virtual_devices[0]
                self.logger.info(f"🎯 Using virtual audio device: {selected_device_name} (ID: {selected_device_id})")
                
                # Select the input device
                if self.virtual_audio.select_input_device(selected_device_id):
                    self.logger.info(f"✅ Selected virtual audio device: {selected_device_name}")
                    
                    # Initialize Whisper model with configured size from settings
                    model_size = self.app_context.global_settings.get('WHISPER_MODEL_SIZE', 'medium')
                    if await self.virtual_audio.initialize_whisper(model_size):
                        self.logger.info("✅ Whisper model initialized successfully")
                        
                        # Start recording
                        if self.virtual_audio.start_recording():
                            self.logger.info("✅ Virtual audio capture started successfully")
                            return True
                        else:
                            self.logger.error("❌ Failed to start virtual audio recording")
                            return False
                    else:
                        self.logger.error("❌ Failed to initialize Whisper model")
                        return False
                else:
                    self.logger.error(f"❌ Failed to select virtual audio device: {selected_device_name}")
                    return False
            else:
                self.logger.warning("⚠️ No VB-Audio Virtual Cable devices found")
                self.logger.info("💡 Install VB-Audio Virtual Cable for virtual audio capture")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize virtual audio capture: {e}")
            return False

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
            system_prompt_chat="You are Danzar, an upbeat and witty gaming assistant who's always ready to help players crush their goals in EverQuest (or any game). Speak casually, like a friendly raid leader—cheer people on, crack a clever joke now and then, and keep the energy high. When giving advice, be forward-thinking: mention upcoming expansions, meta strategies, or ways to optimize both platinum farming and experience gains. Use gamer lingo naturally, but explain anything arcane so newcomers feel included. Above all, stay encouraging—everyone levels up at their own pace, and you're here to make the journey fun and rewarding.\n\nIMPORTANT: You are receiving voice transcriptions that may contain errors from speech-to-text processing. If a word or phrase seems out of context, unclear, or doesn't make sense, use your best judgment to interpret what the user likely meant based on the conversation context. Common STT errors include:\n- Homophones (e.g., 'there' vs 'their', 'to' vs 'too')\n- Similar-sounding words (e.g., 'game' vs 'gain', 'quest' vs 'test')\n- Missing or extra words\n- Punctuation errors\n- Background noise interpreted as words\n\nIf you're unsure about a transcription, you can ask for clarification, but try to respond naturally to what you believe the user intended to say."
        )
        app_context = AppContext(settings, discord_profile, logger)

        # Create the Discord bot with voice capabilities and full service integration
        bot = DanzarVoiceBot(settings, app_context)
        
        # Store bot instance in app context for services to access
        app_context.discord_bot_runner_instance = bot

        # Start the bot
        logger.info("🎤 Starting DanzarAI Voice Bot with LLM and TTS integration...")
                # Load Discord bot token from environment variable
        from dotenv import load_dotenv
        load_dotenv()
        
        bot_token = os.environ.get('DISCORD_BOT_TOKEN')
        if not bot_token:
            logger.error("❌ DISCORD_BOT_TOKEN environment variable not set!")
            return
        
        # Load Discord voice channel ID from environment variable
        voice_channel_id = os.environ.get('DISCORD_VOICE_CHANNEL_ID')
        if voice_channel_id:
            # Override the voice channel ID in settings
            settings['DISCORD_VOICE_CHANNEL_ID'] = voice_channel_id
            logger.info(f"🎯 Voice channel ID loaded from environment: {voice_channel_id}")
        else:
            logger.warning("⚠️ DISCORD_VOICE_CHANNEL_ID not set in environment, using config file value")
        
        # Load Discord text channel ID from environment variable
        text_channel_id = os.environ.get('DISCORD_TEXT_CHANNEL_ID')
        if text_channel_id:
            # Override the text channel ID in settings
            settings['DISCORD_TEXT_CHANNEL_ID'] = text_channel_id
            logger.info(f"📝 Text channel ID loaded from environment: {text_channel_id}")
        else:
            logger.warning("⚠️ DISCORD_TEXT_CHANNEL_ID not set in environment, using config file value")
        
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
    asyncio.run(main())

