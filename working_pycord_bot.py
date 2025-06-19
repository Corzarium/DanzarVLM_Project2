#!/usr/bin/env python3
"""
Working Py-cord Voice Bot - Simplified with Proper Command Syncing
"""

import asyncio
import logging
import os
import discord
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingVoiceBot(discord.Bot):
    """Simplified py-cord bot with working slash commands"""
    
    def __init__(self, guild_id=None):
        super().__init__(
            intents=discord.Intents.all(),
            debug_guilds=[guild_id] if guild_id else None
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
        
        logger.info("🎤 Bot is ready! Try these commands:")
        logger.info("   /test - Test if commands work")
        logger.info("   /join - Join voice and record")
        logger.info("   /stop - Stop recording")

    @discord.slash_command(name="test", description="Test if the bot is working")
    async def test_command(self, ctx):
        """Simple test command"""
        logger.info(f"🧪 /test used by {ctx.author.display_name}")
        
        embed = discord.Embed(
            title="🤖 DanzarAI Voice Bot Status",
            color=discord.Color.green()
        )
        embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
        embed.add_field(name="🎤 Whisper STT", value="Loaded" if self.whisper_model else "Not loaded", inline=True)
        embed.add_field(name="📡 Latency", value=f"{round(self.latency * 1000)}ms", inline=True)
        embed.add_field(name="🔊 Voice Status", value="Recording" if ctx.guild.id in self.connections else "Not connected", inline=True)
        
        await ctx.respond(embed=embed)

    @discord.slash_command(name="join", description="Join voice channel and start recording")
    async def join_voice(self, ctx):
        """Join voice channel with recording"""
        logger.info(f"📞 /join used by {ctx.author.display_name}")
        
        # Check if user is in voice channel
        if not ctx.author.voice:
            await ctx.respond("❌ **You need to be in a voice channel first!**")
            return
        
        # Check if already connected
        if ctx.guild.id in self.connections:
            await ctx.respond("⚠️ **Already recording!** Use `/stop` first.")
            return
        
        channel = ctx.author.voice.channel
        
        try:
            await ctx.respond(f"🔄 **Connecting to {channel.name}...**")
            
            # Connect to voice channel
            vc = await channel.connect()
            self.connections[ctx.guild.id] = vc
            
            # Start recording
            vc.start_recording(
                discord.sinks.WaveSink(),
                self.recording_finished,
                ctx.channel
            )
            
            await ctx.edit(content=f"🎤 **Recording in {channel.name}!**\n"
                                 f"💬 **Speak now** - I'll transcribe your voice\n"
                                 f"🛑 Use `/stop` when done")
            
            logger.info(f"✅ Recording started in {channel.name}")
            
        except Exception as e:
            await ctx.edit(content=f"❌ **Failed to join:** {e}")
            logger.error(f"❌ Join failed: {e}")

    @discord.slash_command(name="stop", description="Stop recording and leave voice")
    async def stop_recording(self, ctx):
        """Stop recording"""
        logger.info(f"🛑 /stop used by {ctx.author.display_name}")
        
        if ctx.guild.id in self.connections:
            vc = self.connections[ctx.guild.id]
            vc.stop_recording()
            del self.connections[ctx.guild.id]
            await ctx.respond("👋 **Stopped recording and left voice channel**")
            logger.info("✅ Recording stopped")
        else:
            await ctx.respond("❌ **Not currently recording**")

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
        """Process STT"""
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
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_file)
            text = result.get('text', '').strip()
            
            if text and len(text) > 3:
                logger.info(f"✅ STT: '{text}' from {user.display_name}")
                await channel.send(f"🎤 **{user.display_name}**: {text}")
            else:
                logger.info(f"⚠️ No speech detected from {user.display_name}")
                
        except Exception as e:
            logger.error(f"❌ STT failed for {user.display_name}: {e}")

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
    
    # Convert guild_id to int
    if guild_id:
        try:
            guild_id = int(guild_id)
        except:
            guild_id = None
    
    # Create and run bot
    bot = WorkingVoiceBot(guild_id=guild_id)
    logger.info("🚀 Starting Working Py-cord Voice Bot...")
    
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