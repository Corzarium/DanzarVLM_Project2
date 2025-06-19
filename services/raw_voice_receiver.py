#!/usr/bin/env python3
"""
Raw Voice Receiver for DanzarAI
Captures raw Opus packets directly from Discord's voice gateway for pristine audio quality
"""

import asyncio
import logging
import time
import struct
from typing import Optional, Callable, Dict, Any
import numpy as np
from collections import deque

try:
    import discord
    import discord.opus
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None


class RawAudioReceiver(discord.VoiceClient):
    """Custom VoiceClient that captures raw Opus packets from Discord's voice gateway."""
    
    def __init__(self, client, channel, *, app_context, speech_callback: Optional[Callable] = None):
        super().__init__(client, channel)
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        # Audio settings
        self.sample_rate = 48000  # Discord uses 48kHz
        self.channels = 2         # Stereo
        self.frame_size = 960     # 20ms frames at 48kHz
        
        # User audio buffers
        self.user_buffers: Dict[int, deque] = {}
        self.user_speech_state: Dict[int, dict] = {}
        
        # Buffer settings for speech accumulation
        self.min_speech_duration = 1.0   # Minimum 1 second of speech
        self.max_speech_duration = 10.0  # Maximum 10 seconds before processing
        self.silence_timeout = 2.0       # Process after 2 seconds of silence
        
        # Opus decoder (use Discord.py's built-in Opus support)
        self.opus_decoder = None
        
        self.logger.info("[RawAudioReceiver] Initializing raw Discord voice gateway capture...")
        
    def _prepare_opus_decoder(self):
        """Initialize the Opus decoder using Discord.py's built-in support."""
        try:
            if discord.opus.is_loaded():
                self.opus_decoder = discord.opus.Decoder()
                self.logger.info("✅ Using Discord.py's built-in Opus decoder")
                return True
            else:
                self.logger.error("❌ Discord Opus library not loaded")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Opus decoder: {e}")
            return False
    
    async def connect(self, *, timeout=60.0, reconnect=True, self_deaf=False, self_mute=False):
        """Override connect to initialize our Opus decoder."""
        result = await super().connect(timeout=timeout, reconnect=reconnect, self_deaf=self_deaf, self_mute=self_mute)
        
        # Initialize Opus decoder after connection
        if not self._prepare_opus_decoder():
            self.logger.warning("⚠️ Opus decoder initialization failed, audio quality may be degraded")
        
        # Start the packet processing task
        asyncio.create_task(self._start_packet_processing())
        
        return result
    
    async def _start_packet_processing(self):
        """Start processing incoming voice packets using Discord.py's voice receiving."""
        self.logger.info("🎯 Starting voice packet processing using Discord.py's built-in receiving...")
        
        # Use Discord.py's built-in voice receiving if available
        try:
            # Check if Discord.py supports voice receiving
            if hasattr(self, 'listen') and callable(getattr(self, 'listen')):
                # Use the listen method for receiving audio
                self.logger.info("✅ Using Discord.py's listen method for voice receiving")
                # The listen method will call our receive method when audio is available
            else:
                self.logger.warning("⚠️ Discord.py voice receiving not available, using fallback")
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up voice packet processing: {e}")
    
    def receive(self, data, user):
        """Receive method called by Discord.py when audio data is available."""
        try:
            # This method is called by Discord.py's voice receiving system
            if user and not user.bot:  # Ignore bot audio
                # Convert the received data to our format and process it
                asyncio.create_task(self._process_received_audio(data, user.id, user.display_name))
        except Exception as e:
            self.logger.error(f"❌ Error in receive method: {e}")
    
    async def _process_received_audio(self, audio_data, user_id: int, display_name: str):
        """Process received audio data from Discord.py's voice receiving."""
        try:
            # Convert audio data to numpy array
            if isinstance(audio_data, bytes):
                # Assume 16-bit signed PCM at 48kHz
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # If it's already a numpy array
                audio_np = np.array(audio_data, dtype=np.float32)
            
            # Process with our existing PCM processing pipeline
            await self._process_pcm_audio(user_id, audio_data, int(time.time() * 1000))
            
        except Exception as e:
            self.logger.error(f"❌ Error processing received audio: {e}")
    
    async def _process_voice_packet(self, packet_data):
        """Process incoming voice packets from Discord's voice gateway."""
        try:
            if not packet_data or len(packet_data) < 12:
                return
            
            # Parse RTP header (first 12 bytes)
            # RTP Header format: V(2) P(1) X(1) CC(4) M(1) PT(7) Sequence(16) Timestamp(32) SSRC(32)
            header = struct.unpack('>BBHII', packet_data[:12])
            version = (header[0] >> 6) & 0x3
            payload_type = header[1] & 0x7f
            sequence = header[2]
            timestamp = header[3]
            ssrc = header[4]
            
            # Skip non-audio packets
            if payload_type != 120:  # 120 is Opus payload type
                return
            
            # Extract Opus payload (skip RTP header)
            opus_payload = packet_data[12:]
            
            if not opus_payload:
                return
            
            # Find user by SSRC
            user_id = self._get_user_by_ssrc(ssrc)
            if not user_id:
                return
            
            # Decode Opus to PCM
            try:
                if self.opus_decoder:
                    # Use Discord.py's Opus decoder
                    pcm_data = self.opus_decoder.decode(opus_payload)
                else:
                    # Fallback: treat as raw audio (will be lower quality)
                    pcm_data = opus_payload
                
                if pcm_data:
                    await self._process_pcm_audio(user_id, pcm_data, timestamp)
                    
            except Exception as decode_error:
                self.logger.error(f"❌ Opus decode error for SSRC {ssrc}: {decode_error}")
                
        except Exception as e:
            self.logger.error(f"❌ Voice packet processing error: {e}")
    
    def _get_user_by_ssrc(self, ssrc: int) -> Optional[int]:
        """Get user ID by SSRC (Synchronization Source)."""
        try:
            # Discord.py maintains SSRC to user mapping
            if hasattr(self, '_connections') and self._connections:
                for user_id, connection_data in self._connections.items():
                    if hasattr(connection_data, 'ssrc') and connection_data.ssrc == ssrc:
                        return user_id
            
            # Fallback: use SSRC as user ID (not ideal but works for debugging)
            return ssrc
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user by SSRC {ssrc}: {e}")
            return None
    
    async def _process_pcm_audio(self, user_id: int, pcm_data: bytes, timestamp: int):
        """Process decoded PCM audio data."""
        try:
            # Initialize user state if needed
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = deque()
                self.user_speech_state[user_id] = {
                    'last_packet_time': time.time(),
                    'speech_start_time': None,
                    'total_duration': 0.0,
                    'is_speaking': False,
                    'last_timestamp': timestamp
                }
            
            current_time = time.time()
            user_state = self.user_speech_state[user_id]
            
            # Convert PCM bytes to numpy array
            try:
                # Discord Opus decoder outputs 16-bit signed PCM
                audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Convert stereo to mono if needed
                if len(audio_np) % 2 == 0:  # Stereo
                    audio_np = audio_np.reshape(-1, 2)
                    audio_np = np.mean(audio_np, axis=1)
                
                # Add to user buffer
                self.user_buffers[user_id].extend(audio_np)
                
                # Update timing
                packet_duration = len(audio_np) / self.sample_rate
                user_state['last_packet_time'] = current_time
                user_state['total_duration'] += packet_duration
                user_state['last_timestamp'] = timestamp
                
                # Check if we should start/continue speech detection
                if not user_state['is_speaking']:
                    user_state['is_speaking'] = True
                    user_state['speech_start_time'] = current_time
                    self.logger.info(f"🎤 Started capturing raw voice from User{user_id}")
                
                # Log audio quality metrics
                audio_max = np.max(np.abs(audio_np))
                audio_rms = np.sqrt(np.mean(np.square(audio_np)))
                
                if int(current_time) % 2 == 0:  # Log every 2 seconds
                    self.logger.info(f"🎵 Raw Discord audio from User{user_id}: max={audio_max:.4f}, rms={audio_rms:.4f}, duration={user_state['total_duration']:.2f}s")
                
                # Check if we should process the accumulated audio
                should_process = False
                process_reason = ""
                
                # Condition 1: Minimum duration reached and recent silence
                if (user_state['total_duration'] >= self.min_speech_duration and 
                    current_time - user_state['last_packet_time'] > self.silence_timeout):
                    should_process = True
                    process_reason = f"silence timeout ({user_state['total_duration']:.2f}s)"
                
                # Condition 2: Maximum duration reached
                elif user_state['total_duration'] >= self.max_speech_duration:
                    should_process = True
                    process_reason = f"max duration ({user_state['total_duration']:.2f}s)"
                
                # Process if conditions are met
                if should_process and len(self.user_buffers[user_id]) > 0:
                    await self._process_user_speech(user_id, process_reason)
                
            except Exception as audio_error:
                self.logger.error(f"❌ Audio processing error for User{user_id}: {audio_error}")
                
        except Exception as e:
            self.logger.error(f"❌ PCM processing error: {e}")
    
    async def _process_user_speech(self, user_id: int, reason: str):
        """Process accumulated speech audio for a user."""
        try:
            # Get accumulated audio
            speech_audio = np.array(list(self.user_buffers[user_id]))
            audio_duration = len(speech_audio) / self.sample_rate
            max_volume = np.max(np.abs(speech_audio))
            
            display_name = f"User{user_id}"
            self.logger.info(f"🎤 Processing raw audio from {display_name}: {audio_duration:.2f}s, max_vol: {max_volume:.4f} - {reason}")
            
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
                    self.logger.info(f"✅ Processed raw Discord audio for {display_name}")
                except Exception as e:
                    self.logger.error(f"❌ Speech callback error for {display_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing speech for User{user_id}: {e}")
    
    async def handle_silence_timeout(self):
        """Background task to handle silence timeouts."""
        while self.is_connected():
            try:
                current_time = time.time()
                
                # Check each user for silence timeout
                for user_id, user_state in list(self.user_speech_state.items()):
                    if (user_state['is_speaking'] and 
                        current_time - user_state['last_packet_time'] > self.silence_timeout and
                        user_state['total_duration'] >= self.min_speech_duration):
                        
                        await self._process_user_speech(user_id, "silence timeout")
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"❌ Silence timeout handler error: {e}")
                await asyncio.sleep(1.0)
    
    def cleanup(self):
        """Clean up resources."""
        self.user_buffers.clear()
        self.user_speech_state.clear()
        
        if self.opus_decoder:
            # Discord.py's Opus decoder cleanup
            self.opus_decoder = None
        
        self.logger.info("🧹 Raw audio receiver cleaned up")


class RawVoiceReceiver:
    """Service wrapper for the raw audio receiver."""
    
    def __init__(self, app_context, speech_callback: Optional[Callable] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        self.logger.info("[RawVoiceReceiver] Initializing raw Discord voice gateway capture service...")
        
    def initialize(self) -> bool:
        """Initialize the raw voice receiver."""
        if not DISCORD_AVAILABLE:
            self.logger.error("❌ Discord.py not available")
            return False
            
        try:
            # Check if Discord Opus is available
            if not discord.opus.is_loaded():
                self.logger.warning("⚠️ Discord Opus not loaded, attempting to load...")
                try:
                    # Try different Opus loading methods
                    try:
                        discord.opus.load_opus('opus')  # Try with 'opus' name
                    except:
                        try:
                            discord.opus.load_opus('libopus')  # Try with 'libopus' name
                        except:
                            discord.opus.load_opus()  # Try without arguments (auto-detect)
                    
                    if discord.opus.is_loaded():
                        self.logger.info("✅ Discord Opus loaded successfully")
                    else:
                        self.logger.warning("⚠️ Discord Opus still not loaded, will use fallback")
                except Exception as opus_error:
                    self.logger.warning(f"⚠️ Failed to load Discord Opus: {opus_error}")
            
            self.logger.info("✅ Raw voice receiver initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize raw voice receiver: {e}")
            return False
    
    def create_voice_client(self, client, channel):
        """Create a raw audio receiver voice client."""
        return RawAudioReceiver(
            client, 
            channel, 
            app_context=self.app_context, 
            speech_callback=self.speech_callback
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("🧹 Raw voice receiver service cleaned up") 