#!/usr/bin/env python3
"""
Simple Voice Receiver for DanzarAI
Uses Discord.py's standard voice receiving capabilities for better compatibility
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict
import numpy as np
from collections import deque

try:
    import discord
    import discord.sinks
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None


class SimpleVoiceReceiver:
    """Simple voice receiver that uses Discord.py's built-in voice receiving."""
    
    def __init__(self, app_context, speech_callback: Optional[Callable] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        # User audio buffers for accumulating speech
        self.user_buffers: Dict[int, deque] = {}
        self.user_speech_state: Dict[int, dict] = {}
        
        # Buffer settings
        self.min_speech_duration = 1.0   # Minimum 1 second
        self.max_speech_duration = 10.0  # Maximum 10 seconds
        self.silence_timeout = 2.0       # Process after 2 seconds of silence
        
        self.logger.info("[SimpleVoiceReceiver] Initializing simple Discord voice receiver...")
        
    def initialize(self) -> bool:
        """Initialize the voice receiver."""
        if not DISCORD_AVAILABLE:
            self.logger.error("❌ Discord.py not available")
            return False
            
        self.logger.info("✅ Simple voice receiver initialized")
        return True
    
    async def start_listening(self, voice_client):
        """Start listening for voice data using Discord.py's built-in capabilities."""
        try:
            self.logger.info("🎯 Starting simple voice listening...")
            
            # Create a simple sink that accumulates audio
            sink = SimpleSink(self)
            
            # Check if Discord.py supports recording
            if hasattr(voice_client, 'start_recording'):
                voice_client.start_recording(sink, self._recording_finished)
                self.logger.info("✅ Started recording with Discord.py's built-in sink")
            else:
                self.logger.error("❌ Discord.py does not support voice recording")
                return False
            
            # Start silence timeout handler
            asyncio.create_task(self._handle_silence_timeouts())
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start voice listening: {e}")
            return False
    
    async def _handle_silence_timeouts(self):
        """Handle silence timeouts for all users."""
        while True:
            try:
                current_time = time.time()
                
                # Check each user for silence timeout
                for user_id, user_state in list(self.user_speech_state.items()):
                    if (user_state.get('is_speaking', False) and 
                        current_time - user_state.get('last_audio_time', 0) > self.silence_timeout and
                        user_state.get('total_duration', 0) >= self.min_speech_duration):
                        
                        await self._process_user_speech(user_id, "silence timeout")
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"❌ Silence timeout handler error: {e}")
                await asyncio.sleep(1.0)
    
    def process_audio_data(self, user_id: int, audio_data: bytes, display_name: str):
        """Process incoming audio data from a user."""
        try:
            # Initialize user state if needed
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = deque()
                self.user_speech_state[user_id] = {
                    'last_audio_time': time.time(),
                    'total_duration': 0.0,
                    'is_speaking': False,
                    'display_name': display_name
                }
            
            current_time = time.time()
            user_state = self.user_speech_state[user_id]
            
            # Convert audio data to numpy array
            try:
                # Discord typically provides 16-bit signed PCM
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to user buffer
                self.user_buffers[user_id].extend(audio_np)
                
                # Update timing
                audio_duration = len(audio_np) / 48000  # Discord uses 48kHz
                user_state['last_audio_time'] = current_time
                user_state['total_duration'] += audio_duration
                user_state['display_name'] = display_name
                
                # Start speaking if not already
                if not user_state['is_speaking']:
                    user_state['is_speaking'] = True
                    self.logger.info(f"🎤 Started capturing audio from {display_name}")
                
                # Check if we should process
                should_process = False
                process_reason = ""
                
                # Process if we have enough duration
                if user_state['total_duration'] >= self.max_speech_duration:
                    should_process = True
                    process_reason = f"max duration ({user_state['total_duration']:.2f}s)"
                
                # Process if conditions are met
                if should_process and len(self.user_buffers[user_id]) > 0:
                    # Schedule processing in the event loop safely
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._process_user_speech(user_id, process_reason))
                    except RuntimeError:
                        # No running loop, schedule for later
                        self.logger.debug(f"🔄 No event loop available, scheduling processing for {display_name}")
                
            except Exception as audio_error:
                self.logger.error(f"❌ Audio processing error for {display_name}: {audio_error}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing audio data: {e}")
    
    async def _process_user_speech(self, user_id: int, reason: str):
        """Process accumulated speech audio for a user."""
        try:
            if user_id not in self.user_buffers or len(self.user_buffers[user_id]) == 0:
                return
            
            # Get accumulated audio
            speech_audio = np.array(list(self.user_buffers[user_id]))
            audio_duration = len(speech_audio) / 48000  # 48kHz
            max_volume = np.max(np.abs(speech_audio))
            
            user_state = self.user_speech_state[user_id]
            display_name = user_state.get('display_name', f'User{user_id}')
            
            self.logger.info(f"🎤 Processing audio from {display_name}: {audio_duration:.2f}s, max_vol: {max_volume:.4f} - {reason}")
            
            # Clear buffer and reset state
            self.user_buffers[user_id].clear()
            user_state.update({
                'total_duration': 0.0,
                'is_speaking': False
            })
            
            # Call the speech callback if provided
            if self.speech_callback and audio_duration >= self.min_speech_duration:
                try:
                    await self.speech_callback(speech_audio, display_name)
                    self.logger.info(f"✅ Processed audio for {display_name}")
                except Exception as e:
                    self.logger.error(f"❌ Speech callback error for {display_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing speech for User{user_id}: {e}")
    
    async def _recording_finished(self, sink, *args):
        """Callback when recording is finished."""
        self.logger.info("🎵 Recording session ended")
    
    def cleanup(self):
        """Clean up resources."""
        self.user_buffers.clear()
        self.user_speech_state.clear()
        self.logger.info("🧹 Simple voice receiver cleaned up")


# Define SimpleSink based on available Discord features
if DISCORD_AVAILABLE and hasattr(discord, 'sinks'):
    class SimpleSink(discord.sinks.Sink):
        """Simple sink for Pycord voice receiving."""
        
        def __init__(self, voice_receiver: SimpleVoiceReceiver):
            super().__init__()
            self.voice_receiver = voice_receiver
            self.logger = voice_receiver.logger
            
        def write(self, data, user):
            """Called by Pycord when audio data is received."""
            try:
                # Handle both user objects and user IDs
                if isinstance(user, int):
                    # user is actually a user_id (int)
                    user_id = user
                    display_name = f'User{user_id}'
                    is_bot = False  # Assume not a bot if we only have ID
                elif hasattr(user, 'id'):
                    # user is a User object
                    user_id = user.id
                    display_name = getattr(user, 'display_name', getattr(user, 'name', f'User{user_id}'))
                    is_bot = getattr(user, 'bot', False)
                else:
                    # Unknown user format, skip
                    self.logger.debug(f"🔍 Unknown user format: {type(user)} - {user}")
                    return
                
                # Skip bot audio
                if is_bot:
                    return
                
                if data and user_id:
                    # Process the audio data
                    self.voice_receiver.process_audio_data(user_id, data, display_name)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in sink write: {e}")
        
        def cleanup(self):
            """Clean up the sink."""
            pass
else:
    class SimpleSink:
        """Fallback sink when Pycord sinks not available."""
        
        def __init__(self, voice_receiver: SimpleVoiceReceiver):
            self.voice_receiver = voice_receiver
            self.logger = voice_receiver.logger
            
        def write(self, data, user):
            """Called when audio data is received."""
            try:
                # Handle both user objects and user IDs
                if isinstance(user, int):
                    # user is actually a user_id (int)
                    user_id = user
                    display_name = f'User{user_id}'
                    is_bot = False  # Assume not a bot if we only have ID
                elif hasattr(user, 'id'):
                    # user is a User object
                    user_id = user.id
                    display_name = getattr(user, 'display_name', getattr(user, 'name', f'User{user_id}'))
                    is_bot = getattr(user, 'bot', False)
                else:
                    # Unknown user format, skip
                    self.logger.debug(f"🔍 Unknown user format: {type(user)} - {user}")
                    return
                
                # Skip bot audio
                if is_bot:
                    return
                
                if data and user_id:
                    # Process the audio data
                    self.voice_receiver.process_audio_data(user_id, data, display_name)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in sink write: {e}")
        
        def cleanup(self):
            """Clean up the sink."""
            pass 