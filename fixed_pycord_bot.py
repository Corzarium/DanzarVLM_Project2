#!/usr/bin/env python3
"""
Fixed Py-cord Voice Bot with Proper Slash Command Syncing
"""

import asyncio
import logging
import os
import discord
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedVoiceBot(discord.Bot):
    """Py-cord bot with proper slash command syncing"""
    
    def __init__(self, guild_id=None):
        super().__init__(
            intents=discord.Intents.all(),
            debug_guilds=[guild_id] if guild_id else None  # Register commands to specific guild for instant sync
        )
        self.connections = {}
        self.whisper_model = None
        self.guild_id = guild_id
        
    async def on_ready(self):
        logger.info(f"🎤 Bot ready as {self.user}")
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model("base.en")
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
        
        # Sync commands if guild specified (py-cord method)
        if self.guild_id:
            try:
                guild = self.get_guild(self.guild_id)
                if guild:
                    await self.sync_commands(guild_ids=[self.guild_id])
                    logger.info(f"✅ Synced commands to guild {guild.name}")
                else:
                    logger.warning(f"⚠️ Guild {self.guild_id} not found")
            except Exception as e:
                logger.error(f"❌ Failed to sync commands: {e}")
        else:
            # Global sync (takes up to 1 hour)
            try:
                await self.sync_commands()
                logger.info("✅ Synced commands globally (may take up to 1 hour to appear)")
            except Exception as e:
                logger.error(f"❌ Failed to sync commands globally: {e}")
        
        logger.info("🎤 Voice bot is ready! Available commands:")
        logger.info("   /join - Join voice channel and start recording")
        logger.info("   /stop - Stop recording and leave")
        logger.info("   /test - Test basic functionality")

    @discord.slash_command(
        name="join", 
        description="Join voice channel and start recording your voice"
    )
    async def join_voice(self, ctx: discord.ApplicationContext):
        """Join voice channel with recording"""
        logger.info(f"📞 /join command used by {ctx.author.display_name}")
        
        if not ctx.author.voice:
            await ctx.respond("❌ **You need to be in a voice channel first!**")
            return
        
        channel = ctx.author.voice.channel
        
        try:
            # Check if already connected
            if ctx.guild.id in self.connections:
                await ctx.respond("⚠️ **Already connected to a voice channel!** Use `/stop` first.")
                return
            
            await ctx.respond(f"🔄 **Connecting to {channel.name}...**")
            
            # Connect to voice channel
            vc = await channel.connect()
            self.connections[ctx.guild.id] = vc
            
            # Start recording with py-cord's built-in functionality
            vc.start_recording(
                discord.sinks.WaveSink(),  # Built-in WAV sink
                self.recording_finished,   # Callback
                ctx.channel,              # Pass channel for responses
                ctx.author                # Pass user info
            )
            
            await ctx.edit(content=f"🎤 **Connected to {channel.name}** and recording!\n"
                                 f"💬 **Speak now** - your voice will be transcribed\n"
                                 f"🛑 Use `/stop` when finished")
            
            logger.info(f"✅ Started recording in {channel.name}")
            
        except Exception as e:
            await ctx.edit(content=f"❌ **Failed to join voice channel:** {e}")
            logger.error(f"❌ Join failed: {e}")

    @discord.slash_command(
        name="stop", 
        description="Stop recording and leave voice channel"
    )
    async def stop_recording(self, ctx: discord.ApplicationContext):
        """Stop recording and leave voice channel"""
        logger.info(f"🛑 /stop command used by {ctx.author.display_name}")
        
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            vc.stop_recording()  # This triggers the callback
            del self.connections[ctx.guild.id]
            await ctx.respond("👋 **Stopped recording and left voice channel**")
            logger.info("✅ Stopped recording and disconnected")
        else:
            await ctx.respond("❌ **Not currently recording in any voice channel**")

    @discord.slash_command(
        name="test", 
        description="Test if the bot is working properly"
    )
    async def test_command(self, ctx: discord.ApplicationContext):
        """Test command to verify bot functionality"""
        logger.info(f"🧪 /test command used by {ctx.author.display_name}")
        
        status_msg = "🤖 **DanzarAI Voice Bot Status:**\n"
        status_msg += f"✅ **Bot Online:** {self.user.display_name}\n"
        status_msg += f"✅ **Whisper STT:** {'Loaded' if self.whisper_model else 'Not loaded'}\n"
        status_msg += f"🎤 **Voice Status:** {'Recording' if ctx.guild.id in self.connections else 'Not connected'}\n"
        status_msg += f"📡 **Latency:** {round(self.latency * 1000)}ms\n"
        status_msg += f"\n**Available Commands:**\n"
        status_msg += f"• `/join` - Start voice recording\n"
        status_msg += f"• `/stop` - Stop voice recording\n"
        status_msg += f"• `/test` - Show this status"
        
        await ctx.respond(status_msg)

    async def recording_finished(self, sink, channel, *args):
        """Process recorded audio when recording stops"""
        try:
            logger.info(f"🎵 Recording finished! Processing {len(sink.audio_data)} users")
            
            if not sink.audio_data:
                await channel.send("⚠️ **No audio recorded** - try speaking louder or check your microphone")
                await sink.vc.disconnect()
                return
            
            # Process each user's audio
            processed_users = []
            for user_id, audio in sink.audio_data.items():
                user = self.get_user(user_id)
                if user and not user.bot:
                    result = await self.process_audio(user, audio, channel)
                    if result:
                        processed_users.append(user.display_name)
            
            # Disconnect
            await sink.vc.disconnect()
            
            if processed_users:
                await channel.send(f"✅ **Recording processing complete!** Processed audio from: {', '.join(processed_users)}")
            else:
                await channel.send("⚠️ **No speech detected** - try speaking more clearly")
            
        except Exception as e:
            logger.error(f"❌ Recording processing failed: {e}")
            await channel.send(f"❌ **Processing error:** {e}")

    async def process_audio(self, user, audio, channel):
        """Process individual user's audio with STT"""
        try:
            if not self.whisper_model:
                logger.warning("⚠️ Whisper model not available")
                await channel.send("❌ **STT not available** - Whisper model not loaded")
                return False
            
            # Get the audio file path from py-cord's sink
            audio_file = audio.file
            logger.info(f"🎵 Processing audio file: {audio_file}")
            
            # Check if audio file exists and has content
            if not os.path.exists(audio_file):
                logger.warning(f"⚠️ Audio file not found: {audio_file}")
                return False
            
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:  # Less than 1KB probably means no audio
                logger.info(f"⚠️ Audio file too small ({file_size} bytes) from {user.display_name}")
                return False
            
            # Transcribe with Whisper
            logger.info(f"🎤 Transcribing audio from {user.display_name} ({file_size} bytes)")
            result = self.whisper_model.transcribe(audio_file)
            text = result.get('text', '').strip()
            
            if text and len(text) > 3:
                logger.info(f"✅ STT Result: '{text}' from {user.display_name}")
                await channel.send(f"🎤 **{user.display_name}**: {text}")
                return True
            else:
                logger.info(f"⚠️ No meaningful speech from {user.display_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ STT processing failed for {user.display_name}: {e}")
            await channel.send(f"❌ **STT error for {user.display_name}:** {e}")
            return False

async def main():
    """Main function with proper configuration"""
    # Get bot token and guild ID
    token = os.getenv('DISCORD_BOT_TOKEN')
    guild_id = os.getenv('DISCORD_GUILD_ID')
    
    if not token:
        # Try to load from config file
        try:
            import yaml
            with open('config/global_settings.yaml', 'r') as f:
                config = yaml.safe_load(f)
                token = config.get('DISCORD_BOT_TOKEN')
                guild_id = guild_id or config.get('DISCORD_GUILD_ID')
        except Exception as e:
            logger.warning(f"⚠️ Could not load config file: {e}")
    
    if not token:
        logger.error("❌ DISCORD_BOT_TOKEN not found! Set environment variable or config file.")
        return
    
    # Convert guild_id to int if it's a string
    if guild_id and isinstance(guild_id, str):
        try:
            guild_id = int(guild_id)
        except ValueError:
            logger.warning(f"⚠️ Invalid guild ID: {guild_id}")
            guild_id = None
    
    bot = FixedVoiceBot(guild_id=guild_id)
    logger.info("🚀 Starting Fixed Py-cord Voice Bot...")
    
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