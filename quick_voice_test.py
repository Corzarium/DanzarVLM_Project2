"""
Quick Voice Test Bot for DanzarAI
Minimal implementation to test TTS functionality with Discord
"""

import discord
from discord.ext import commands
import asyncio
import logging
import tempfile
import os
from typing import Optional
import time
import queue
import threading

from core.config_loader import load_global_settings
from core.game_profile import GameProfile


class QuickAppContext:
    """Minimal AppContext for testing"""
    def __init__(self, global_settings: dict, logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.logger = logger_instance
        
        # Simple profile for testing
        self.active_profile = GameProfile(
            game_name="discord_voice_test",
            system_prompt_commentary="You are DanzarAI, a helpful gaming assistant.",
            user_prompt_template_commentary="",
            vlm_model="test",
            vlm_max_tokens=100,
            vlm_temperature=0.7
        )
        
        # Required events and queues
        self.shutdown_event = threading.Event()
        self.tts_is_playing = threading.Event()
        self.tts_is_playing.clear()
        
        # Queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.tts_queue = queue.Queue(maxsize=20)
        self.text_message_queue = queue.Queue(maxsize=20)
        
        # Service instances
        self.tts_service_instance = None
        self.llm_service_instance = None
        self.memory_service_instance = None
        
        self.logger.info("[QuickAppContext] Initialized for testing")


class QuickVoiceBot(commands.Bot):
    """Quick test bot for TTS functionality"""
    
    def __init__(self):
        # Load configuration
        self.config = load_global_settings() or {}
        
        # Setup Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        # Initialize bot
        command_prefix = self.config.get('DISCORD_COMMAND_PREFIX', '!')
        super().__init__(command_prefix=command_prefix, intents=intents)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create app context
        self.app_context = QuickAppContext(self.config, self.logger)
        
        self.logger.info("[QuickVoiceBot] Bot initialized")
    
    async def on_ready(self):
        """Called when bot is ready and connected."""
        self.logger.info(f'[QuickVoiceBot] Bot ready! Logged in as {self.user}')
        
        # Initialize TTS service
        try:
            from services.tts_service import TTSService
            self.app_context.tts_service_instance = TTSService(self.app_context)
            self.logger.info("[QuickVoiceBot] TTS service initialized")
        except Exception as e:
            self.logger.error(f"[QuickVoiceBot] TTS service init failed: {e}")
        
        # Log commands after everything is initialized
        self.logger.info(f'[QuickVoiceBot] Available commands: {[cmd.name for cmd in self.commands]}')
        
        # Send a ready message to the configured text channel if available
        if self.config.get('DISCORD_TEXT_CHANNEL_ID'):
            try:
                channel = self.get_channel(self.config.get('DISCORD_TEXT_CHANNEL_ID'))
                if channel:
                    await channel.send("🎤 **DanzarAI Quick Voice Bot is ready!**\nUse `!join` to connect to voice, then `!say <message>` to test TTS")
            except Exception as e:
                self.logger.warning(f"Could not send ready message: {e}")

# Define commands outside the class to ensure proper registration
async def join_command(self, ctx):
        """Join voice channel."""
        try:
            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.send("❌ You need to be in a voice channel first!")
                return
            
            channel = ctx.author.voice.channel
            
            if ctx.voice_client:
                if ctx.voice_client.channel == channel:
                    await ctx.send(f"ℹ️ Already in: **{channel.name}**")
                    return
                else:
                    await ctx.voice_client.move_to(channel)
                    await ctx.send(f"✅ Moved to: **{channel.name}**")
            else:
                await channel.connect()
                await ctx.send(f"✅ Joined: **{channel.name}**")
            
            await ctx.send("🎤 Ready for voice commands!")
            await ctx.send("💬 Try: `!say Hello world`")
            
        except Exception as e:
            self.logger.error(f"Join error: {e}", exc_info=True)
            await ctx.send(f"❌ Join failed: {e}")
    
    @commands.command(name='leave')
    async def leave_command(self, ctx):
        """Leave voice channel."""
        try:
            if not ctx.voice_client:
                await ctx.send("ℹ️ Not in any voice channel!")
                return
            
            channel_name = ctx.voice_client.channel.name
            await ctx.voice_client.disconnect()
            await ctx.send(f"✅ Left: **{channel_name}**")
            
        except Exception as e:
            await ctx.send(f"❌ Leave error: {e}")
    
    @commands.command(name='say')
    async def say_command(self, ctx, *, message: str):
        """Test TTS with given message."""
        try:
            if not ctx.voice_client:
                await ctx.send("❌ Join a voice channel first with `!join`")
                return
            
            if not self.app_context.tts_service_instance:
                await ctx.send("❌ TTS service not available")
                return
            
            await ctx.send(f"🗣️ Saying: *{message}*")
            
            # Generate TTS
            try:
                tts_audio = self.app_context.tts_service_instance.generate_audio(message)
                if not tts_audio:
                    await ctx.send("❌ TTS generation failed")
                    return
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(tts_audio)
                    temp_file_path = temp_file.name
                
                # Play through Discord
                source = discord.FFmpegPCMAudio(temp_file_path)
                
                def cleanup_after(error):
                    if error:
                        self.logger.error(f"Playback error: {error}")
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                
                ctx.voice_client.play(source, after=cleanup_after)
                await ctx.send("✅ Playing TTS audio!")
                
            except Exception as tts_error:
                await ctx.send(f"❌ TTS error: {tts_error}")
                self.logger.error(f"TTS error: {tts_error}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"Say command error: {e}", exc_info=True)
            await ctx.send(f"❌ Command error: {e}")
    
    @commands.command(name='test')
    async def test_command(self, ctx):
        """Test basic functionality."""
        await ctx.send("✅ Quick voice test bot is working!")
        
        # Check TTS service
        if self.app_context.tts_service_instance:
            await ctx.send("✅ TTS service is available")
        else:
            await ctx.send("❌ TTS service not available")
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Show status."""
        embed = discord.Embed(title="🎤 Quick Voice Test Status", color=0x00ff00)
        
        embed.add_field(
            name="Voice", 
            value="✅ Connected" if ctx.voice_client else "❌ Not connected",
            inline=True
        )
        embed.add_field(
            name="TTS", 
            value="✅ Ready" if self.app_context.tts_service_instance else "❌ Not ready",
            inline=True
        )
        
        embed.add_field(
            name="Commands",
            value="`!join` - Join voice\n`!leave` - Leave voice\n`!say <text>` - Test TTS",
            inline=False
        )
        
        await ctx.send(embed=embed)


async def main():
    """Main entry point."""
    try:
        config = load_global_settings()
        if not config:
            print("❌ Failed to load configuration")
            return
        
        token = config.get('DISCORD_BOT_TOKEN')
        if not token:
            print("❌ DISCORD_BOT_TOKEN not found")
            return
        
        print("🚀 Starting Quick Voice Test Bot...")
        bot = QuickVoiceBot()
        
        try:
            await bot.start(token)
        except KeyboardInterrupt:
            print("Bot stopped")
        finally:
            await bot.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}") 