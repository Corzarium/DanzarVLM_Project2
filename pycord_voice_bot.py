#!/usr/bin/env python3
"""
DanzarAI Py-cord Voice Bot - Real Voice Chat with STT
Uses py-cord's built-in voice recording capabilities
"""

import asyncio
import logging
import sys
import os
import tempfile
import wave
from datetime import datetime
from typing import Optional

import discord
import whisper
import numpy as np

# Add project directory to path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_loader import ConfigLoader
from services.tts_service import TTSService
from services.memory_service import MemoryService
from services.llm_service import LLMService
from models.models import GameProfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pycord_voice_bot.log', 'a', 'utf-8')
    ]
)
logger = logging.getLogger(__name__)

class VoiceAppContext:
    """Application context for py-cord voice bot"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.global_settings = config
        self.logger = logger
        
        # Initialize core attributes for DanzarAI services
        self.queues = {}
        self.events = {'is_in_conversation': asyncio.Event()}
        self.profiles = {}
        self.active_profile_change_subscribers = []
        
        # Initialize profile
        self.active_profile = GameProfile(
            name="voice_chat",
            game_title="Voice Chat",
            ai_identity="I am DanzarAI, your helpful gaming assistant with voice capabilities.",
            memory_enabled=True,
            tts_enabled=True
        )
        
        # Service instances
        self.tts_service_instance: Optional[TTSService] = None
        self.memory_service_instance: Optional[MemoryService] = None
        self.llm_service_instance: Optional[LLMService] = None
        
        logger.info("[VoiceAppContext] Initialized for py-cord voice chat")

class VoiceChatBot(discord.Bot):
    """Py-cord bot with built-in voice recording and STT"""
    
    def __init__(self, app_context: VoiceAppContext):
        super().__init__(
            command_prefix=app_context.global_settings.get('DISCORD_COMMAND_PREFIX', '!'),
            intents=discord.Intents.all()
        )
        
        self.app_context = app_context
        self.logger = app_context.logger
        self.whisper_model = None
        self.connections = {}  # Voice connections cache
        
    async def on_ready(self):
        """Called when bot is ready"""
        self.logger.info(f"[VoiceChatBot] Bot ready! Logged in as {self.user}")
        
        # Initialize services
        await self.initialize_services()
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model("base.en")
            self.logger.info("[VoiceChatBot] Whisper STT model loaded")
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] Failed to load Whisper: {e}")
    
    async def initialize_services(self):
        """Initialize DanzarAI services"""
        try:
            # Initialize TTS Service
            self.app_context.tts_service_instance = TTSService(self.app_context)
            await self.app_context.tts_service_instance.initialize()
            self.logger.info("[VoiceChatBot] TTS service initialized")
            
            # Initialize Memory Service
            self.app_context.memory_service_instance = MemoryService(self.app_context)
            await self.app_context.memory_service_instance.initialize()
            self.logger.info("[VoiceChatBot] Memory service initialized")
            
            # Initialize LLM Service
            self.app_context.llm_service_instance = LLMService(self.app_context)
            await self.app_context.llm_service_instance.initialize()
            self.logger.info("[VoiceChatBot] LLM service initialized")
            
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] Service initialization failed: {e}")

    @discord.slash_command(name="join_voice", description="Join voice channel and start listening")
    async def join_voice(self, ctx):
        """Join voice channel with recording capability"""
        if not ctx.author.voice:
            await ctx.respond("❌ You need to be in a voice channel first!")
            return
        
        channel = ctx.author.voice.channel
        
        try:
            # Connect to voice channel
            vc = await channel.connect()
            self.connections[ctx.guild.id] = vc
            
            # Start recording with built-in py-cord functionality
            vc.start_recording(
                discord.sinks.WaveSink(),  # Built-in WAV sink
                self.recording_finished,   # Callback when recording stops
                ctx.channel,              # Pass channel for responses
                ctx.author                # Pass user info
            )
            
            await ctx.respond(f"🎤 **Connected to {channel.name}** and listening for voice input!")
            self.logger.info(f"[VoiceChatBot] Started recording in {channel.name}")
            
        except Exception as e:
            await ctx.respond(f"❌ Failed to join voice channel: {e}")
            self.logger.error(f"[VoiceChatBot] Failed to join voice: {e}")

    @discord.slash_command(name="leave_voice", description="Leave voice channel and stop recording")
    async def leave_voice(self, ctx):
        """Stop recording and leave voice channel"""
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            vc.stop_recording()  # This triggers the callback
            del self.connections[ctx.guild.id]
            await ctx.respond("👋 **Left voice channel** and stopped recording")
        else:
            await ctx.respond("❌ I'm not currently in a voice channel")

    @discord.slash_command(name="voice_status", description="Check voice recording status")
    async def voice_status(self, ctx):
        """Check current voice status"""
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            status = "🎤 **Recording**" if vc.recording else "⏸️ **Connected but not recording**"
            await ctx.respond(f"{status} in {vc.channel.name}")
        else:
            await ctx.respond("❌ **Not connected** to any voice channel")

    async def recording_finished(self, sink, channel, *args):
        """Callback when recording finishes - process STT here"""
        try:
            self.logger.info(f"[VoiceChatBot] Recording finished, processing {len(sink.audio_data)} users")
            
            # Process each user's audio
            for user_id, audio in sink.audio_data.items():
                user = self.get_user(user_id)
                if user and not user.bot:
                    # Process STT for this user
                    await self.process_user_audio(user, audio, channel)
            
            # Disconnect from voice
            await sink.vc.disconnect()
            
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] Recording processing failed: {e}")

    async def process_user_audio(self, user: discord.User, audio, channel):
        """Process individual user's audio through STT"""
        try:
            if not self.whisper_model:
                self.logger.warning("[VoiceChatBot] Whisper model not loaded")
                return
            
            # Get audio file path
            audio_file = audio.file
            self.logger.info(f"[VoiceChatBot] Processing audio from {user.display_name}: {audio_file}")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_file)
            transcribed_text = result.get('text', '').strip()
            
            if transcribed_text and len(transcribed_text) > 3:
                self.logger.info(f"[VoiceChatBot] STT: '{transcribed_text}' from {user.display_name}")
                
                # Send transcription to channel
                await channel.send(f"🎤 **{user.display_name}**: {transcribed_text}")
                
                # Process with LLM and respond
                await self.handle_voice_message(user, transcribed_text, channel)
            else:
                self.logger.debug(f"[VoiceChatBot] No meaningful speech detected from {user.display_name}")
                
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] STT processing failed for {user.display_name}: {e}")

    async def handle_voice_message(self, user: discord.User, message: str, channel):
        """Handle transcribed voice message with LLM and TTS response"""
        try:
            # Get LLM response
            llm_service = self.app_context.llm_service_instance
            if not llm_service:
                await channel.send("❌ LLM service not available")
                return
            
            response = await llm_service.generate_response(
                prompt=message,
                conversation_context=f"Voice message from {user.display_name}: {message}"
            )
            
            if response:
                self.logger.info(f"[VoiceChatBot] AI Response: {response}")
                
                # Send text response
                await channel.send(f"🤖 **DanzarAI**: {response}")
                
                # Generate and play TTS if in voice channel
                await self.play_tts_response(response, channel.guild.id)
                
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] Voice message handling failed: {e}")
            await channel.send(f"❌ Error processing voice message: {e}")

    async def play_tts_response(self, text: str, guild_id: int):
        """Generate TTS and play in voice channel"""
        try:
            tts_service = self.app_context.tts_service_instance
            if not tts_service or guild_id not in self.connections:
                return
            
            # Generate TTS audio
            audio_file = await tts_service.generate_audio(text)
            if audio_file and os.path.exists(audio_file):
                vc = self.connections[guild_id]
                
                # Play TTS response
                if not vc.is_playing():
                    audio_source = discord.FFmpegPCMAudio(audio_file)
                    vc.play(audio_source)
                    self.logger.info("[VoiceChatBot] Playing TTS response")
                else:
                    self.logger.warning("[VoiceChatBot] Already playing audio, skipping TTS")
                    
        except Exception as e:
            self.logger.error(f"[VoiceChatBot] TTS playback failed: {e}")

    # Text commands for testing
    @discord.slash_command(name="chat", description="Chat with DanzarAI via text")
    async def chat(self, ctx, *, message: str):
        """Text-based chat command"""
        try:
            llm_service = self.app_context.llm_service_instance
            if not llm_service:
                await ctx.respond("❌ LLM service not available")
                return
            
            response = await llm_service.generate_response(
                prompt=message,
                conversation_context=f"Text message from {ctx.author.display_name}: {message}"
            )
            
            if response:
                await ctx.respond(f"🤖 **DanzarAI**: {response}")
            else:
                await ctx.respond("❌ No response generated")
                
        except Exception as e:
            await ctx.respond(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_global_settings()
        
        # Validate required settings
        if not config.get('DISCORD_BOT_TOKEN'):
            logger.error("DISCORD_BOT_TOKEN is required")
            return
        
        # Create app context and bot
        app_context = VoiceAppContext(config, logger)
        bot = VoiceChatBot(app_context)
        
        logger.info("🎤 Starting DanzarAI Py-cord Voice Bot...")
        await bot.start(config['DISCORD_BOT_TOKEN'])
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
    finally:
        if 'bot' in locals():
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 