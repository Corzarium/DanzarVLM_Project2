#!/usr/bin/env python3
"""
Simple Py-cord Voice Recording Test
Tests basic voice recording functionality with py-cord
"""

import asyncio
import logging
import os
import discord
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVoiceBot(discord.Bot):
    """Simple py-cord bot to test voice recording"""
    
    def __init__(self):
        super().__init__(intents=discord.Intents.all())
        self.connections = {}
        self.whisper_model = None
        
    async def on_ready(self):
        logger.info(f"Bot ready as {self.user}")
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model("base.en")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")

    @discord.slash_command(name="join", description="Join voice channel and start recording")
    async def join_voice(self, ctx):
        """Join voice channel with recording"""
        if not ctx.author.voice:
            await ctx.respond("❌ You need to be in a voice channel!")
            return
        
        channel = ctx.author.voice.channel
        
        try:
            # Connect to voice channel
            vc = await channel.connect()
            self.connections[ctx.guild.id] = vc
            
            # Start recording with py-cord's built-in functionality
            vc.start_recording(
                discord.sinks.WaveSink(),  # Built-in WAV sink
                self.recording_finished,   # Callback
                ctx.channel               # Pass channel for responses
            )
            
            await ctx.respond(f"🎤 **Connected to {channel.name}** and recording!")
            logger.info(f"Started recording in {channel.name}")
            
        except Exception as e:
            await ctx.respond(f"❌ Failed to join: {e}")
            logger.error(f"Join failed: {e}")

    @discord.slash_command(name="stop", description="Stop recording and leave")
    async def stop_recording(self, ctx):
        """Stop recording and leave voice channel"""
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            vc.stop_recording()  # This triggers the callback
            del self.connections[ctx.guild.id]
            await ctx.respond("👋 **Stopped recording and left voice channel**")
        else:
            await ctx.respond("❌ Not currently recording")

    async def recording_finished(self, sink, channel, *args):
        """Process recorded audio when recording stops"""
        try:
            logger.info(f"Recording finished! Processing {len(sink.audio_data)} users")
            
            # Process each user's audio
            for user_id, audio in sink.audio_data.items():
                user = self.get_user(user_id)
                if user and not user.bot:
                    await self.process_audio(user, audio, channel)
            
            # Disconnect
            await sink.vc.disconnect()
            await channel.send("✅ **Recording processing complete!**")
            
        except Exception as e:
            logger.error(f"Recording processing failed: {e}")
            await channel.send(f"❌ Processing error: {e}")

    async def process_audio(self, user, audio, channel):
        """Process individual user's audio with STT"""
        try:
            if not self.whisper_model:
                logger.warning("Whisper model not available")
                return
            
            # Get the audio file path from py-cord's sink
            audio_file = audio.file
            logger.info(f"Processing audio file: {audio_file}")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_file)
            text = result.get('text', '').strip()
            
            if text and len(text) > 3:
                logger.info(f"STT Result: '{text}' from {user.display_name}")
                await channel.send(f"🎤 **{user.display_name}**: {text}")
            else:
                logger.info(f"No meaningful speech from {user.display_name}")
                
        except Exception as e:
            logger.error(f"STT processing failed: {e}")
            await channel.send(f"❌ STT error for {user.display_name}: {e}")

async def main():
    """Main function"""
    # Get bot token from environment or config
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        # Try to load from config file
        try:
            import yaml
            with open('config/global_settings.yaml', 'r') as f:
                config = yaml.safe_load(f)
                token = config.get('DISCORD_BOT_TOKEN')
        except:
            pass
    
    if not token:
        logger.error("DISCORD_BOT_TOKEN not found! Set environment variable or config file.")
        return
    
    bot = SimpleVoiceBot()
    logger.info("🎤 Starting Simple Py-cord Voice Test Bot...")
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 