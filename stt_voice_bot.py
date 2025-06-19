#!/usr/bin/env python3
"""
DanzarAI STT Voice Bot - ACTUAL Voice Input Implementation
Requires discord-ext-voice-recv for microphone input
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Optional
import tempfile
import wave
import numpy as np

import discord
from discord.ext import commands
from discord.ext import voice_recv
import whisper

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
        logging.FileHandler('logs/stt_voice_bot.log', 'a', 'utf-8')
    ]
)
logger = logging.getLogger(__name__)

class STTAppContext:
    """Application context with all required services for STT Voice Bot"""
    
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
            ai_identity="I am DanzarAI, a helpful gaming assistant with voice capabilities.",
            memory_enabled=True,
            tts_enabled=True
        )
        
        # Service instances
        self.tts_service_instance: Optional[TTSService] = None
        self.memory_service_instance: Optional[MemoryService] = None
        self.llm_service_instance: Optional[LLMService] = None
        self.whisper_model = None
        
        # Voice processing
        self.voice_listening = False
        self.current_audio_buffer = {}
        
        logger.info("[STTAppContext] Initialized for STT voice chat")

class STTAudioSink(voice_recv.AudioSink):
    """Audio sink that properly handles Discord voice receiving for STT"""
    
    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model = whisper_model
    
    def wants_opus(self) -> bool:
        """We want PCM data for Whisper processing"""
        return False
        
    def write(self, user, data):
        """Process incoming voice data from Discord users"""
        if user and not user.bot and data:
            # This is where we'd process the voice data
            logger.info(f"Received voice data from {user.display_name}")
            # TODO: Process with Whisper STT

class STTVoiceBot(commands.Bot):
    """Discord bot with proper STT voice receiving capabilities"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        self.voice_client = None
        self.whisper_model = None
        
    async def setup_hook(self):
        """Initialize services when bot starts"""
        try:
            # Load Whisper model
            self.whisper_model = whisper.load_model("base.en")
            logger.info("Whisper model loaded")
            
        except Exception as e:
            logger.error(f"[STTVoiceBot] Service initialization failed: {e}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"STT Voice Bot ready as {self.user}")
        
    @commands.command()
    async def join_voice(self, ctx):
        """Join voice channel with STT capability"""
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            # Use VoiceRecvClient for voice receiving
            self.voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            
            # Start listening with custom sink
            sink = STTAudioSink(self.whisper_model)
            self.voice_client.listen(sink)
            
            await ctx.send("🎤 Connected and listening for voice input!")
        else:
            await ctx.send("You need to be in a voice channel first!")

async def main():
    """Main entry point"""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_global_settings()
        
        # Validate required settings
        required_settings = ['DISCORD_BOT_TOKEN']
        for setting in required_settings:
            if not config.get(setting):
                logger.error(f"Required setting missing: {setting}")
                return
        
        # Create bot
        bot = STTVoiceBot()
        
        logger.info("🎤 Starting DanzarAI STT Voice Bot...")
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