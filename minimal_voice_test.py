#!/usr/bin/env python3
"""
Minimal DanzarAI Voice Test - Avoiding Asyncio Loop Conflicts
"""

import asyncio
import logging
import os
import discord
from discord.ext import commands
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load DanzarAI settings
from core.config_loader import load_global_settings

class MinimalVoiceBot(commands.Bot):
    def __init__(self):
        # Setup Discord intents for voice
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(command_prefix="!", intents=intents, help_command=None)
        
        self.whisper_model = None
        self.voice_connections = {}
        
    async def setup_hook(self):
        """Setup hook called after login but before connecting to gateway"""
        logger.info("🔄 Loading Whisper model in setup_hook...")
        try:
            # Load Whisper in the setup hook to avoid blocking
            self.whisper_model = whisper.load_model("tiny")
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
    
    async def on_ready(self):
        """Bot ready event"""
        logger.info(f"🎤 Minimal Voice Bot ready as {self.user}")
        logger.info("🎤 Available commands: !join, !leave, !test")

# Create bot instance
bot = MinimalVoiceBot()

@bot.command(name='test')
async def test_command(ctx):
    """Test bot functionality"""
    logger.info(f"🧪 !test used by {ctx.author.name}")
    
    # Check voice connection status
    voice_status = "Not connected"
    try:
        voice_client = ctx.guild.voice_client
        if voice_client:
            if voice_client.is_connected():
                voice_status = f"Connected to {voice_client.channel.name}"
            else:
                voice_status = "Connection failed"
    except Exception as e:
        voice_status = f"Error: {str(e)}"
    
    # Check Whisper status
    whisper_status = "✅ Loaded" if bot.whisper_model else "❌ Not loaded"
    
    embed = discord.Embed(
        title="🎤 Minimal DanzarAI Voice Test",
        color=discord.Color.green()
    )
    embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
    embed.add_field(name="🎤 Whisper STT", value=whisper_status, inline=True)
    embed.add_field(name="📡 Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="🔊 Voice Status", value=voice_status, inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='join')
async def join_command(ctx):
    """Join voice channel and start recording"""
    logger.info(f"📞 !join used by {ctx.author.name}")
    
    try:
        # Check if user is in voice channel
        if not ctx.author.voice:
            await ctx.send("❌ **You need to be in a voice channel first!**")
            return
        
        channel = ctx.author.voice.channel
        logger.info(f"🎯 Target channel: {channel.name}")
        
        # Check if bot is already connected
        if ctx.guild.voice_client:
            if ctx.guild.voice_client.channel == channel:
                await ctx.send(f"✅ **Already connected to {channel.name}!**")
                return
            else:
                # Move to new channel
                logger.info(f"🔄 Moving to {channel.name}")
                await ctx.guild.voice_client.move_to(channel)
                await ctx.send(f"🔄 **Moved to {channel.name}!**")
                return
        
        # Connect to voice channel
        await ctx.send(f"📞 **Connecting to {channel.name}...**")
        logger.info(f"🔗 Attempting connection to {channel.name}")
        
        # Use a simple connection approach
        voice_client = await channel.connect(timeout=10.0, reconnect=True)
        
        if voice_client and voice_client.is_connected():
            logger.info(f"✅ Successfully connected to {channel.name}")
            
            # Start recording with minimal setup
            try:
                sink = discord.sinks.WaveSink()
                voice_client.start_recording(
                    sink,
                    recording_finished,
                    ctx
                )
                
                await ctx.send(f"🎤 **Connected and recording in {channel.name}!**\n"
                              f"💬 **Speak now - I'm listening!**\n"
                              f"🛑 **Use `!leave` when done.**")
                logger.info(f"🎙️ Recording started in {channel.name}")
                
            except Exception as recording_error:
                logger.error(f"❌ Recording setup failed: {recording_error}")
                await ctx.send(f"❌ **Connected but recording failed:** {str(recording_error)}")
        else:
            await ctx.send("❌ **Failed to connect to voice channel.**")
            logger.error("❌ Voice connection failed")
            
    except asyncio.TimeoutError:
        logger.error("❌ Connection timeout")
        await ctx.send("❌ **Connection timeout - try again.**")
    except Exception as e:
        logger.error(f"❌ Join failed: {e}")
        await ctx.send(f"❌ **Failed to join voice channel:** {str(e)}")

@bot.command(name='leave')
async def leave_command(ctx):
    """Stop recording and leave voice channel"""
    logger.info(f"🛑 !leave used by {ctx.author.name}")
    
    try:
        voice_client = ctx.guild.voice_client
        
        if not voice_client:
            await ctx.send("❌ **Not connected to any voice channel.**")
            return
        
        # Stop recording if active
        try:
            voice_client.stop_recording()
            logger.info("🛑 Recording stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping recording: {e}")
        
        # Disconnect
        await voice_client.disconnect()
        await ctx.send("👋 **Disconnected from voice channel.**")
        logger.info("✅ Successfully disconnected from voice")
        
    except Exception as e:
        logger.error(f"❌ Leave failed: {e}")
        await ctx.send(f"❌ **Failed to leave:** {str(e)}")

async def recording_finished(sink, ctx):
    """Handle recorded audio"""
    logger.info("🎵 Processing recorded audio...")
    
    try:
        # Process each user's audio
        for user_id, audio in sink.audio_data.items():
            user = bot.get_user(user_id)
            if not user or user.bot:
                continue
                
            # Save audio file
            filename = f"recorded_audio_{user_id}.wav"
            with open(filename, "wb") as f:
                f.write(audio.file.getvalue())
            
            # Check file size
            file_size = os.path.getsize(filename)
            logger.info(f"📁 Audio file size: {file_size} bytes")
            
            if file_size < 1000:  # Less than 1KB
                await ctx.send(f"⚠️ **{user.display_name}**: Audio too short to process")
                os.remove(filename)
                continue
            
            # Transcribe with Whisper
            if bot.whisper_model:
                try:
                    logger.info(f"🔄 Transcribing audio from {user.display_name}...")
                    result = bot.whisper_model.transcribe(filename)
                    text = str(result["text"]).strip()
                    
                    if text:
                        await ctx.send(f"🗣️ **{user.display_name}**: {text}")
                        logger.info(f"✅ Transcribed: {text}")
                    else:
                        await ctx.send(f"🤔 **{user.display_name}**: No speech detected")
                        
                except Exception as e:
                    logger.error(f"❌ Transcription failed: {e}")
                    await ctx.send(f"❌ **Failed to transcribe {user.display_name}'s audio**")
            else:
                await ctx.send("❌ **Whisper model not loaded**")
            
            # Clean up audio file
            try:
                os.remove(filename)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Callback failed: {e}")
        await ctx.send(f"❌ **Error processing audio:** {str(e)}")

async def main():
    """Main function with proper asyncio handling"""
    logger.info("🚀 Starting Minimal DanzarAI Voice Test...")
    
    # Get Discord token from DanzarAI config
    settings = load_global_settings()
    if not settings:
        logger.error("❌ Failed to load DanzarAI settings!")
        return
    
    token = settings.get('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("❌ DISCORD_BOT_TOKEN not found in DanzarAI settings!")
        return
    
    try:
        # Use the proper asyncio approach
        async with bot:
            await bot.start(token)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")

if __name__ == "__main__":
    # Use asyncio.run for proper event loop management
    asyncio.run(main()) 