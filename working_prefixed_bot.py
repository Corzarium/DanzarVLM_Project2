#!/usr/bin/env python3
"""
Working DanzarAI Prefixed Commands Bot - Simplified and Fixed
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

# Enable required intents
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

# Create bot instance
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Global variables
connections = {}
whisper_model = None

@bot.event
async def on_ready():
    """Bot ready event"""
    global whisper_model
    logger.info(f"🎤 Bot ready as {bot.user}")
    
    # Load Whisper model
    try:
        whisper_model = whisper.load_model("base.en")
        logger.info("✅ Whisper model loaded")
    except Exception as e:
        logger.error(f"❌ Whisper load failed: {e}")
    
    logger.info("🎤 DanzarAI Voice Bot is ready! Available commands:")
    logger.info("   !test - Test if bot is working")
    logger.info("   !join - Join voice channel and start recording")
    logger.info("   !stop - Stop recording and leave voice")
    logger.info("   !help - Show this help message")

@bot.command(name="test")
async def test_command(ctx):
    """Test if the bot is working"""
    logger.info(f"🧪 !test used by {ctx.author.display_name}")
    
    embed = discord.Embed(
        title="🤖 DanzarAI Voice Bot Status",
        color=discord.Color.green()
    )
    embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
    embed.add_field(name="🎤 Whisper STT", value="Loaded" if whisper_model else "Not loaded", inline=True)
    embed.add_field(name="📡 Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="🔊 Voice Status", value="Recording" if ctx.guild.id in connections else "Not connected", inline=True)
    embed.add_field(name="🎯 Command Prefix", value="`!`", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="join")
async def join_voice(ctx):
    """Join voice channel and start recording"""
    logger.info(f"📞 !join used by {ctx.author.display_name}")
    
    # Check if user is in voice channel
    if not ctx.author.voice:
        await ctx.send("❌ **You need to be in a voice channel first!**")
        return
    
    # Check if already connected
    if ctx.guild.id in connections:
        await ctx.send("⚠️ **Already recording!** Use `!stop` first.")
        return
    
    channel = ctx.author.voice.channel
    
    try:
        msg = await ctx.send(f"🔄 **Connecting to {channel.name}...**")
        
        # Connect to voice channel
        vc = await channel.connect()
        connections[ctx.guild.id] = vc
        
        # Start recording
        vc.start_recording(
            discord.sinks.WaveSink(),
            recording_finished,
            ctx.channel
        )
        
        await msg.edit(content=f"🎤 **Recording in {channel.name}!**\n"
                             f"💬 **Speak now** - I'll transcribe your voice\n"
                             f"🛑 Use `!stop` when done")
        
        logger.info(f"✅ Recording started in {channel.name}")
        
    except Exception as e:
        await ctx.send(f"❌ **Failed to join:** {e}")
        logger.error(f"❌ Join failed: {e}")

@bot.command(name="stop")
async def stop_recording(ctx):
    """Stop recording and leave voice channel"""
    logger.info(f"🛑 !stop used by {ctx.author.display_name}")
    
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.send("👋 **Stopped recording and left voice channel**")
        logger.info("✅ Recording stopped")
    else:
        await ctx.send("❌ **Not currently recording**")

@bot.command(name="help")
async def help_command(ctx):
    """Show help message"""
    embed = discord.Embed(
        title="🎤 DanzarAI Voice Bot Commands",
        description="Voice-to-text bot with Speech Recognition",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📋 Available Commands",
        value="""
        `!test` - Test bot functionality
        `!join` - Join voice channel and start recording
        `!stop` - Stop recording and leave voice
        `!help` - Show this help message
        """,
        inline=False
    )
    
    embed.add_field(
        name="🎯 How to Use",
        value="""
        1. Join a voice channel
        2. Type `!join` to start recording
        3. Speak - I'll transcribe your voice to text
        4. Type `!stop` when done
        """,
        inline=False
    )
    
    embed.add_field(
        name="🔧 Features",
        value="• Real-time Speech-to-Text\n• Whisper AI transcription\n• Multi-user support",
        inline=False
    )
    
    await ctx.send(embed=embed)

async def recording_finished(sink, channel, *args):
    """Process recorded audio"""
    try:
        logger.info(f"🎵 Processing {len(sink.audio_data)} audio files")
        
        if not sink.audio_data:
            await channel.send("⚠️ **No audio recorded**")
            await sink.vc.disconnect()
            return
        
        # Process each user's audio
        for user_id, audio in sink.audio_data.items():
            user = bot.get_user(user_id)
            if user and not user.bot:
                await process_audio(user, audio, channel)
        
        await sink.vc.disconnect()
        await channel.send("✅ **Recording processing complete!**")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        await channel.send(f"❌ **Error:** {e}")

async def process_audio(user, audio, channel):
    """Process Speech-to-Text"""
    try:
        if not whisper_model:
            await channel.send("❌ **Whisper not loaded**")
            return
        
        audio_file = audio.file
        logger.info(f"🎤 Transcribing {user.display_name}'s audio")
        
        # Check file exists and has content
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            logger.info(f"⚠️ No meaningful audio from {user.display_name}")
            return
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_file)
        text = result.get('text', '').strip()
        
        if text and len(text) > 3:
            logger.info(f"✅ STT: '{text}' from {user.display_name}")
            
            # Send transcription with nice formatting
            embed = discord.Embed(
                title="🎤 Voice Transcription",
                description=f"**{user.display_name}**: {text}",
                color=discord.Color.green()
            )
            embed.set_thumbnail(url=user.display_avatar.url)
            await channel.send(embed=embed)
        else:
            logger.info(f"⚠️ No speech detected from {user.display_name}")
            
    except Exception as e:
        logger.error(f"❌ STT failed for {user.display_name}: {e}")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"❌ **Unknown command.** Use `!help` for available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ **Missing required argument.** Use `!help` for usage.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"❌ **An error occurred:** {error}")

async def main():
    """Main function"""
    # Get configuration
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        try:
            import yaml
            with open('config/global_settings.yaml', 'r') as f:
                config = yaml.safe_load(f)
                token = config.get('DISCORD_BOT_TOKEN')
        except:
            pass
    
    if not token:
        logger.error("❌ DISCORD_BOT_TOKEN required!")
        return
    
    logger.info("🚀 Starting DanzarAI Prefixed Commands Bot...")
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 