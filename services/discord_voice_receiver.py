#!/usr/bin/env python3
"""
Discord Voice Receiver Service for DanzarAI
Uses native Discord.py 2.5+ voice receiving for pristine audio quality
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
import numpy as np
from collections import deque
import threading

try:
    import discord
    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False
    discord = None


class DiscordVoiceReceiver:
    """Native Discord.py voice receiver with Opus decoding for crystal-clear audio."""
    
    def __init__(self, app_context, speech_callback: Optional[Callable] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        # Audio settings (Discord provides PCM data)
        self.sample_rate = 48000  # Discord uses 48kHz
        self.channels = 2         # Stereo (we'll convert to mono)
        self.user_buffers: Dict[int, deque] = {}
        self.user_speech_state: Dict[int, dict] = {}
        
        # Enhanced user tracking with username mapping
        self.user_names: Dict[int, str] = {}  # Map user_id to display_name
        self.user_cache: Dict[int, Any] = {}  # Cache Discord user objects
        
        # Buffer settings for speech accumulation
        self.min_speech_duration = 1.0   # Minimum 1 second of speech
        self.max_speech_duration = 10.0  # Maximum 10 seconds before processing
        self.silence_timeout = 2.0       # Process after 2 seconds of silence
        
        self.logger.info("[DiscordVoiceReceiver] Initializing native Discord.py voice capture...")
        
    def initialize(self) -> bool:
        """Initialize the voice receiver."""
        if not VOICE_RECV_AVAILABLE:
            self.logger.error("❌ Discord.py not available")
            return False
            
        try:
            self.logger.info("✅ Native Discord voice receiver initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Discord voice receiver: {e}")
            return False
    
    def update_user_info(self, user_id: int, user_object: Any) -> str:
        """Update user information and return display name."""
        try:
            if user_object and hasattr(user_object, 'id'):
                # Store user object for future reference
                self.user_cache[user_id] = user_object
                
                # Get display name (prefer display_name over name)
                if hasattr(user_object, 'display_name') and user_object.display_name:
                    display_name = user_object.display_name
                elif hasattr(user_object, 'name') and user_object.name:
                    display_name = user_object.name
                else:
                    display_name = f"User{user_id}"
                
                # Update user mapping
                self.user_names[user_id] = display_name
                self.logger.debug(f"[DiscordVoiceReceiver] Updated user {user_id} -> {display_name}")
                return display_name
            else:
                # Fallback for when user object is not available
                display_name = self.user_names.get(user_id, f"User{user_id}")
                return display_name
                
        except Exception as e:
            self.logger.error(f"[DiscordVoiceReceiver] Error updating user info for {user_id}: {e}")
            return self.user_names.get(user_id, f"User{user_id}")
    
    def get_user_display_name(self, user_id: int) -> str:
        """Get display name for a user ID."""
        return self.user_names.get(user_id, f"User{user_id}")
    
    async def handle_silence_timeout(self):
        """Background task to handle silence timeouts."""
        while True:
            try:
                current_time = time.time()
                
                # Check each user for silence timeout
                for user_id, user_state in list(self.user_speech_state.items()):
                    if (user_state['is_speaking'] and 
                        current_time - user_state['last_packet_time'] > self.silence_timeout and
                        user_state['total_duration'] >= self.min_speech_duration):
                        
                        # Use actual username instead of generic name
                        display_name = self.get_user_display_name(user_id)
                        await self._process_user_speech(user_id, display_name, "silence timeout")
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"❌ Silence timeout handler error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_user_speech(self, user_id: int, display_name: str, reason: str):
        """Process accumulated speech audio for a user."""
        try:
            # Get accumulated audio
            speech_audio = np.array(list(self.user_buffers[user_id]))
            audio_duration = len(speech_audio) / self.sample_rate
            max_volume = np.max(np.abs(speech_audio))
            
            self.logger.info(f"🎤 Processing {display_name}: {audio_duration:.2f}s, max_vol: {max_volume:.4f} - {reason}")
            
            # Clear buffer and reset state
            self.user_buffers[user_id].clear()
            self.user_speech_state[user_id].update({
                'speech_start_time': None,
                'total_duration': 0.0,
                'is_speaking': False
            })
            
            # Call the speech callback if provided
            if self.speech_callback:
                try:
                    await self.speech_callback(speech_audio, display_name)
                    self.logger.info(f"✅ Processed direct Discord audio for {display_name}")
                except Exception as e:
                    self.logger.error(f"❌ Speech callback error for {display_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing speech for {display_name}: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.user_buffers.clear()
        self.user_speech_state.clear()
        self.user_names.clear()
        self.user_cache.clear()
        
        self.logger.info("🧹 Discord voice receiver cleaned up")


class NativeDiscordSink:
    """Native Discord.py sink for direct voice packet processing."""
    
    def __init__(self, voice_receiver: DiscordVoiceReceiver):
        self.voice_receiver = voice_receiver
        self.logger = voice_receiver.logger
        
    def write(self, data, user):
        """Process incoming voice data using native Discord.py format."""
        try:
            # Handle case where user is passed as ID instead of object
            if isinstance(user, int):
                user_id = user
                display_name = self.voice_receiver.get_user_display_name(user_id)
            else:
                if user.bot:
                    return  # Ignore bot audio
                user_id = user.id
                # Update user info and get display name
                display_name = self.voice_receiver.update_user_info(user_id, user)
            
            # Initialize user state if needed
            if user_id not in self.voice_receiver.user_buffers:
                self.voice_receiver.user_buffers[user_id] = deque()
                self.voice_receiver.user_speech_state[user_id] = {
                    'last_packet_time': time.time(),
                    'speech_start_time': None,
                    'total_duration': 0.0,
                    'is_speaking': False
                }
            
            current_time = time.time()
            user_state = self.voice_receiver.user_speech_state[user_id]
            
            # Convert raw audio data to numpy array
            try:
                # Discord.py provides raw PCM data directly
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to user buffer
                self.voice_receiver.user_buffers[user_id].extend(audio_np)
                
                # Update timing
                packet_duration = len(audio_np) / self.voice_receiver.sample_rate
                user_state['last_packet_time'] = current_time
                user_state['total_duration'] += packet_duration
                
                # Check if we should start/continue speech detection
                if not user_state['is_speaking']:
                    user_state['is_speaking'] = True
                    user_state['speech_start_time'] = current_time
                    self.logger.info(f"🎤 Started capturing voice from {display_name}")
                
                # Log audio quality metrics
                audio_max = np.max(np.abs(audio_np))
                audio_rms = np.sqrt(np.mean(np.square(audio_np)))
                
                if int(current_time) % 2 == 0:  # Log every 2 seconds
                    self.logger.info(f"🎵 Native Discord audio from {display_name}: max={audio_max:.4f}, rms={audio_rms:.4f}, duration={user_state['total_duration']:.2f}s")
                
                # Check if we should process the accumulated audio
                should_process = False
                process_reason = ""
                
                # Condition 1: Minimum duration reached and recent silence
                if (user_state['total_duration'] >= self.voice_receiver.min_speech_duration and 
                    current_time - user_state['last_packet_time'] > self.voice_receiver.silence_timeout):
                    should_process = True
                    process_reason = f"silence timeout ({user_state['total_duration']:.2f}s)"
                
                # Condition 2: Maximum duration reached
                elif user_state['total_duration'] >= self.voice_receiver.max_speech_duration:
                    should_process = True
                    process_reason = f"max duration ({user_state['total_duration']:.2f}s)"
                
                # Process if conditions are met
                if should_process and len(self.voice_receiver.user_buffers[user_id]) > 0:
                    # Schedule async processing
                    asyncio.create_task(
                        self.voice_receiver._process_user_speech(user_id, display_name, process_reason)
                    )
                
            except Exception as decode_error:
                self.logger.error(f"❌ Audio decode error for {display_name}: {decode_error}")
                
        except Exception as e:
            self.logger.error(f"❌ Native Discord sink error: {e}")
    
    def cleanup(self):
        """Clean up the sink."""
        self.logger.info("🧹 NativeDiscordSink cleaned up") 