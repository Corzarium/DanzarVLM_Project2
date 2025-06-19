#!/usr/bin/env python3
"""
Simple DanzarAI Voice Bot - No Threading, Direct Approach
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
    
    # Load Whisper model (tiny for speed)
    try:
        logger.info("🔄 Loading Whisper tiny model...")
        whisper_model = whisper.load_model("tiny")
        logger.info("✅ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper: {e}")
    
    logger.info("🎤 DanzarAI Voice Bot is ready! Available commands:")
    logger.info("   !test - Test if bot is working")
    logger.info("   !join - Join voice channel and start recording")
    logger.info("   !stop - Stop recording and leave voice")
    logger.info("   !help - Show this help message")

@bot.command(name='test')
async def test_command(ctx):
    """Test bot functionality"""
    logger.info(f"🧪 !test used by {ctx.author.name}")
    
    # Check voice connection status
    voice_status = "Not connected"
    try:
        if ctx.guild.voice_client:
            if ctx.guild.voice_client.is_connected():
                voice_status = f"Connected to {ctx.guild.voice_client.channel.name}"
                # Check if recording
                if hasattr(ctx.guild.voice_client, '_recording') and ctx.guild.voice_client._recording:
                    voice_status += " (Recording)"
            else:
                voice_status = "Connection failed"
    except Exception as e:
        voice_status = f"Error checking voice status: {str(e)}"
        logger.error(f"Voice status check error: {e}")
    
    # Check Whisper status
    whisper_status = "✅ Loaded" if whisper_model else "❌ Not loaded"
    
    embed = discord.Embed(
        title="🤖 DanzarAI Voice Bot Status",
        color=discord.Color.green()
    )
    embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
    embed.add_field(name="🎤 Whisper STT", value=whisper_status, inline=True)
    embed.add_field(name="📡 Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="🔊 Voice Status", value=voice_status, inline=False)
    embed.add_field(name="🎯 Command Prefix", value="!", inline=True)
    
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
        
        # Check if bot is already connected
        if ctx.guild.voice_client:
            if ctx.guild.voice_client.channel == channel:
                await ctx.send(f"✅ **Already connected to {channel.name}!**")
                return
            else:
                # Move to new channel
                await ctx.guild.voice_client.move_to(channel)
                await ctx.send(f"🔄 **Moved to {channel.name}!**")
                return
        
        # Connect to voice channel
        await ctx.send(f"📞 **Connecting to {channel.name}...**")
        voice_client = await channel.connect()
        
        if voice_client and voice_client.is_connected():
            # Start recording
            sink = discord.sinks.WaveSink()
            voice_client.start_recording(
                sink,
                finished_callback,
                ctx
            )
            
            await ctx.send(f"🎤 **Connected and recording in {channel.name}!**\n"
                          f"💬 **Speak now - I'm listening!**\n"
                          f"🛑 **Use `!stop` when done.**")
            logger.info(f"✅ Successfully connected to {channel.name}")
        else:
            await ctx.send("❌ **Failed to connect to voice channel.**")
            logger.error("Failed to establish voice connection")
            
    except Exception as e:
        logger.error(f"❌ Join failed: {e}")
        await ctx.send(f"❌ **Failed to join voice channel:** {str(e)}")

@bot.command(name='stop')
async def stop_command(ctx):
    """Stop recording and leave voice channel"""
    logger.info(f"🛑 !stop used by {ctx.author.name}")
    
    try:
        voice_client = ctx.guild.voice_client
        
        if not voice_client:
            await ctx.send("❌ **Not connected to any voice channel.**")
            return
        
        # Check if recording and stop it
        try:
            if hasattr(voice_client, 'stop_recording'):
                voice_client.stop_recording()
                await ctx.send("🛑 **Stopped recording.**")
            else:
                await ctx.send("🛑 **Recording stopped (if any).**")
        except Exception as recording_error:
            logger.warning(f"Error stopping recording: {recording_error}")
            await ctx.send("🛑 **Recording stopped (with warnings).**")
        
        await voice_client.disconnect()
        await ctx.send("👋 **Disconnected from voice channel.**")
        logger.info("✅ Successfully disconnected from voice")
        
    except Exception as e:
        logger.error(f"❌ Stop failed: {e}")
        await ctx.send(f"❌ **Failed to stop:** {str(e)}")

@bot.command(name='help')
async def help_command(ctx):
    """Show help message"""
    embed = discord.Embed(
        title="🎤 DanzarAI Voice Bot Commands",
        description="Voice-to-text Discord bot with Whisper STT",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📋 **Available Commands**",
        value=(
            "`!test` - Test bot status and connectivity\n"
            "`!join` - Join your voice channel and start recording\n"
            "`!stop` - Stop recording and leave voice channel\n"
            "`!help` - Show this help message"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🎯 **How to Use**",
        value=(
            "1️⃣ Join a voice channel\n"
            "2️⃣ Type `!join` to start recording\n"
            "3️⃣ Speak into your microphone\n"
            "4️⃣ Type `!stop` when finished"
        ),
        inline=False
    )
    
    embed.set_footer(text="DanzarAI • Powered by Whisper STT")
    await ctx.send(embed=embed)

async def finished_callback(sink, ctx):
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
            if whisper_model:
                try:
                    logger.info(f"🔄 Transcribing audio from {user.display_name}...")
                    result = whisper_model.transcribe(filename)
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
    logger.info("🚀 Starting DanzarAI Voice Bot with Py-cord Integration...")
    
    # Get Discord token from DanzarAI config
    from core.config_loader import load_global_settings
    settings = load_global_settings()
    if not settings:
        logger.error("❌ Failed to load DanzarAI settings!")
        return
    
    token = settings.get('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("❌ DISCORD_BOT_TOKEN not found in DanzarAI settings!")
        return
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 