#!/usr/bin/env python3
"""
DanzarAI Prefixed Commands Bot - Using !commands with py-cord
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

class DanzarVoiceBot(commands.Bot):
    """Py-cord bot with prefixed commands (!commands)"""
    
    def __init__(self, command_prefix="!", guild_id=None):
        # Enable message content intent for prefixed commands
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(
            command_prefix=command_prefix,
            intents=intents,
            help_command=None  # We'll create our own help
        )
        
        self.connections = {}
        self.whisper_model = None
        self.guild_id = guild_id
        
    async def on_ready(self):
        logger.info(f"🎤 Bot ready as {self.user}")
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model("base.en")
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"❌ Whisper load failed: {e}")
        
        logger.info("🎤 DanzarAI Voice Bot is ready! Available commands:")
        logger.info("   !test - Test if bot is working")
        logger.info("   !join - Join voice channel and start recording")
        logger.info("   !stop - Stop recording and leave voice")
        logger.info("   !help - Show this help message")

    @commands.command(name="test")
    async def test_command(self, ctx):
        """Test if the bot is working"""
        logger.info(f"🧪 !test used by {ctx.author.display_name}")
        
        embed = discord.Embed(
            title="🤖 DanzarAI Voice Bot Status",
            color=discord.Color.green()
        )
        embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
        embed.add_field(name="🎤 Whisper STT", value="Loaded" if self.whisper_model else "Not loaded", inline=True)
        embed.add_field(name="📡 Latency", value=f"{round(self.latency * 1000)}ms", inline=True)
        embed.add_field(name="🔊 Voice Status", value="Recording" if ctx.guild.id in self.connections else "Not connected", inline=True)
        embed.add_field(name="🎯 Command Prefix", value=f"`{self.command_prefix}`", inline=True)
        
        await ctx.send(embed=embed)

    @commands.command(name="join")
    async def join_voice(self, ctx):
        """Join voice channel and start recording"""
        logger.info(f"📞 !join used by {ctx.author.display_name}")
        
        # Check if user is in voice channel
        if not ctx.author.voice:
            await ctx.send("❌ **You need to be in a voice channel first!**")
            return
        
        # Check if already connected
        if ctx.guild.id in self.connections:
            await ctx.send("⚠️ **Already recording!** Use `!stop` first.")
            return
        
        channel = ctx.author.voice.channel
        
        try:
            msg = await ctx.send(f"🔄 **Connecting to {channel.name}...**")
            
            # Connect to voice channel
            vc = await channel.connect()
            self.connections[ctx.guild.id] = vc
            
            # Start recording
            vc.start_recording(
                discord.sinks.WaveSink(),
                self.recording_finished,
                ctx.channel
            )
            
            await msg.edit(content=f"🎤 **Recording in {channel.name}!**\n"
                                 f"💬 **Speak now** - I'll transcribe your voice\n"
                                 f"🛑 Use `!stop` when done")
            
            logger.info(f"✅ Recording started in {channel.name}")
            
        except Exception as e:
            await ctx.send(f"❌ **Failed to join:** {e}")
            logger.error(f"❌ Join failed: {e}")

    @commands.command(name="stop")
    async def stop_recording(self, ctx):
        """Stop recording and leave voice channel"""
        logger.info(f"🛑 !stop used by {ctx.author.display_name}")
        
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            vc.stop_recording()
            del self.connections[ctx.guild.id]
            await ctx.send("👋 **Stopped recording and left voice channel**")
            logger.info("✅ Recording stopped")
        else:
            await ctx.send("❌ **Not currently recording**")

    @commands.command(name="help")
    async def help_command(self, ctx):
        """Show help message"""
        embed = discord.Embed(
            title="🎤 DanzarAI Voice Bot Commands",
            description="Voice-to-text bot with Speech Recognition",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="📋 Available Commands",
            value=f"""
            `{self.command_prefix}test` - Test bot functionality
            `{self.command_prefix}join` - Join voice channel and start recording
            `{self.command_prefix}stop` - Stop recording and leave voice
            `{self.command_prefix}help` - Show this help message
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

    async def recording_finished(self, sink, channel, *args):
        """Process recorded audio"""
        try:
            logger.info(f"🎵 Processing {len(sink.audio_data)} audio files")
            
            if not sink.audio_data:
                await channel.send("⚠️ **No audio recorded**")
                await sink.vc.disconnect()
                return
            
            # Process each user's audio
            for user_id, audio in sink.audio_data.items():
                user = self.get_user(user_id)
                if user and not user.bot:
                    await self.process_audio(user, audio, channel)
            
            await sink.vc.disconnect()
            await channel.send("✅ **Recording processing complete!**")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            await channel.send(f"❌ **Error:** {e}")

    async def process_audio(self, user, audio, channel):
        """Process Speech-to-Text"""
        try:
            if not self.whisper_model:
                await channel.send("❌ **Whisper not loaded**")
                return
            
            audio_file = audio.file
            logger.info(f"🎤 Transcribing {user.display_name}'s audio")
            
            # Check file exists and has content
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
                logger.info(f"⚠️ No meaningful audio from {user.display_name}")
                return
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_file)
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

    async def on_command_error(self, context, exception):
        """Handle command errors"""
        if isinstance(exception, commands.CommandNotFound):
            await context.send(f"❌ **Unknown command.** Use `{self.command_prefix}help` for available commands.")
        elif isinstance(exception, commands.MissingRequiredArgument):
            await context.send(f"❌ **Missing required argument.** Use `{self.command_prefix}help` for usage.")
        else:
            logger.error(f"Command error: {exception}")
            await context.send(f"❌ **An error occurred:** {exception}")

async def main():
    """Main function"""
    # Get configuration
    token = os.getenv('DISCORD_BOT_TOKEN')
    guild_id = os.getenv('DISCORD_GUILD_ID')
    
    if not token:
        try:
            import yaml
            with open('config/global_settings.yaml', 'r') as f:
                config = yaml.safe_load(f)
                token = config.get('DISCORD_BOT_TOKEN')
                guild_id = guild_id or config.get('DISCORD_GUILD_ID')
        except:
            pass
    
    if not token:
        logger.error("❌ DISCORD_BOT_TOKEN required!")
        return
    
    # Convert guild_id to int if provided
    if guild_id:
        try:
            guild_id = int(guild_id)
        except:
            guild_id = None
    
    # Create and run bot
    bot = DanzarVoiceBot(command_prefix="!", guild_id=guild_id)
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