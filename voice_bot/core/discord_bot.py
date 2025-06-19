"""
Main Discord bot client for voice-enabled conversation.
"""
import logging
import discord
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..services.vad_service import VADService
from ..services.stt_service import STTService
from ..services.llm_service import LLMService
from ..services.tts_service import TTSService
from ..services.memory_service import MemoryService

logger = logging.getLogger(__name__)

class VoiceBot:
    """Voice-enabled Discord bot."""
    
    def __init__(self, settings: dict):
        """Initialize voice bot with settings."""
        self.settings = settings
        
        # Initialize Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        self.client = discord.Client(intents=intents)
        
        # Initialize services
        self.vad_service = VADService(settings["voice"])
        self.stt_service = STTService(settings["stt"])
        self.llm_service = LLMService(settings["llm"])
        self.tts_service = TTSService(settings["tts"])
        self.memory_service = MemoryService(settings["memory"])
        
        # Bot state
        self.voice_client: Optional[discord.VoiceClient] = None
        self.current_channel: Optional[discord.VoiceChannel] = None
        self.is_listening = False
        self.is_speaking = False
        self.last_activity = datetime.now()
        self.inactivity_timeout = timedelta(minutes=5)
        
        # Audio processing
        self.audio_buffer: List[bytes] = []
        self.current_speaker: Optional[discord.Member] = None
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup Discord event handlers."""
        
        @self.client.event
        async def on_ready():
            """Handle bot ready event."""
            logger.info(f"Bot logged in as {self.client.user}")
            
            # Join target voice channel if specified
            if self.settings["discord"]["VOICE_CHANNEL_ID"]:
                channel = self.client.get_channel(self.settings["discord"]["VOICE_CHANNEL_ID"])
                if channel and isinstance(channel, discord.VoiceChannel):
                    await self.join_voice_channel(channel)
        
        @self.client.event
        async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
            """Handle voice state updates."""
            # Check if bot was disconnected
            if member.id == self.client.user.id and before.channel and not after.channel:
                self.voice_client = None
                self.current_channel = None
                self.is_listening = False
                self.is_speaking = False
                logger.info("Bot disconnected from voice channel")
            
            # Check if we should leave due to inactivity
            if self.current_channel and len(self.current_channel.members) <= 1:
                await self.handle_inactivity()
        
        @self.client.event
        async def on_message(message: discord.Message):
            """Handle text messages."""
            if message.author.bot:
                return
            
            # Handle commands
            if message.content.startswith(self.settings["discord"]["COMMAND_PREFIX"]):
                await self.handle_command(message)
    
    async def join_voice_channel(self, channel: discord.VoiceChannel):
        """
        Join a voice channel.
        
        Args:
            channel: Voice channel to join
        """
        try:
            if self.voice_client and self.voice_client.is_connected():
                await self.voice_client.disconnect()
            
            self.voice_client = await channel.connect()
            self.current_channel = channel
            logger.info(f"Joined voice channel: {channel.name}")
            
            # Start listening
            self.is_listening = True
            asyncio.create_task(self.listen_for_speech())
            
        except Exception as e:
            logger.error(f"Error joining voice channel: {e}")
            raise
    
    async def listen_for_speech(self):
        """Listen for speech in the voice channel."""
        if not self.voice_client or not self.voice_client.is_connected():
            return
        
        try:
            while self.is_listening:
                # Get audio data
                audio_data = await self.voice_client.receive_audio()
                
                # Process with VAD
                is_speaking, audio = await self.vad_service.process_audio(audio_data)
                
                if is_speaking:
                    self.last_activity = datetime.now()
                    self.audio_buffer.append(audio)
                elif self.audio_buffer:
                    # Process complete utterance
                    await self.process_utterance()
                
                # Check for inactivity
                if datetime.now() - self.last_activity > self.inactivity_timeout:
                    await self.handle_inactivity()
                    break
                
        except Exception as e:
            logger.error(f"Error in speech listening: {e}")
            self.is_listening = False
    
    async def process_utterance(self):
        """Process a complete utterance."""
        try:
            # Get transcribed text
            text = await self.stt_service.transcribe_stream(self.audio_buffer)
            if not text:
                return
            
            # Get context and memory
            context = self.memory_service.get_stm_context()
            memory_results = await self.memory_service.search_memory(text)
            
            # Generate response
            response = await self.llm_service.generate_response(
                text,
                context=context,
                memory_results=memory_results
            )
            
            # Update memory
            self.memory_service.add_to_stm(text, response)
            
            # Generate and play speech
            await self.speak_response(response)
            
            # Clear audio buffer
            self.audio_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
            self.audio_buffer.clear()
    
    async def speak_response(self, text: str):
        """
        Generate and play speech response.
        
        Args:
            text: Text to speak
        """
        try:
            self.is_speaking = True
            
            # Generate speech
            audio_file = await self.tts_service.generate_speech(text)
            if not audio_file:
                return
            
            # Play audio
            if self.voice_client and self.voice_client.is_connected():
                self.voice_client.play(
                    discord.FFmpegPCMAudio(str(audio_file)),
                    after=lambda e: self._after_speech(e, audio_file)
                )
            
        except Exception as e:
            logger.error(f"Error speaking response: {e}")
            self.is_speaking = False
    
    def _after_speech(self, error: Optional[Exception], audio_file: Path):
        """Handle completion of speech playback."""
        try:
            # Clean up audio file
            if audio_file.exists():
                audio_file.unlink()
            
            self.is_speaking = False
            
            if error:
                logger.error(f"Error during speech playback: {error}")
            
        except Exception as e:
            logger.error(f"Error in speech completion handler: {e}")
    
    async def handle_inactivity(self):
        """Handle bot inactivity."""
        try:
            if not self.voice_client or not self.voice_client.is_connected():
                return
            
            # Generate summary
            messages = [
                {"role": "user", "content": entry["user_message"]}
                for entry in self.memory_service.short_term_memory
            ]
            
            summary = await self.llm_service.summarize_conversation(messages)
            topic = await self.llm_service.extract_topic(messages)
            
            # Store in long-term memory
            await self.memory_service.store_conversation(summary, topic)
            
            # Clear short-term memory
            self.memory_service.clear_stm()
            
            # Disconnect
            await self.voice_client.disconnect()
            self.voice_client = None
            self.current_channel = None
            self.is_listening = False
            
            logger.info("Bot disconnected due to inactivity")
            
        except Exception as e:
            logger.error(f"Error handling inactivity: {e}")
    
    async def handle_command(self, message: discord.Message):
        """
        Handle bot commands.
        
        Args:
            message: Command message
        """
        try:
            command = message.content[1:].lower().split()[0]
            
            if command == "join":
                if message.author.voice:
                    await self.join_voice_channel(message.author.voice.channel)
                    await message.channel.send("✅ Joined voice channel")
                else:
                    await message.channel.send("❌ You must be in a voice channel")
            
            elif command == "leave":
                if self.voice_client and self.voice_client.is_connected():
                    await self.voice_client.disconnect()
                    await message.channel.send("✅ Left voice channel")
                else:
                    await message.channel.send("❌ Not in a voice channel")
            
            elif command == "voices":
                voices = await self.tts_service.get_available_voices()
                if voices:
                    voice_list = "\n".join(f"- {v}" for v in voices)
                    await message.channel.send(f"Available voices:\n{voice_list}")
                else:
                    await message.channel.send("❌ Failed to get available voices")
            
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            await message.channel.send("❌ An error occurred while processing the command")
    
    def run(self):
        """Run the bot."""
        try:
            self.client.run(self.settings["discord"]["TOKEN"])
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise 