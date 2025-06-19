"""
Py-cord Voice Patch for DanzarAI Discord Bot
Replaces discord-ext-voice-recv with py-cord's built-in voice recording
Integrates with existing DanzarAI services (TTS, LLM, Memory)
"""

import discord
from discord.ext import commands
import asyncio
import logging
import os
import tempfile
import whisper
from typing import Optional, Dict, Any
import io

from services.tts_service import TTSService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from core.config_loader import load_global_settings


class PycordVoiceBot(commands.Bot):
    """
    Enhanced Discord Bot with Py-cord Voice Recording
    Integrates with existing DanzarAI services
    """
    
    def __init__(self, 
                 command_prefix: str,
                 intents: discord.Intents,
                 tts_service: TTSService,
                 llm_service: LLMService,
                 memory_service: MemoryService):
        super().__init__(command_prefix=command_prefix, intents=intents)
        
        self.logger = logging.getLogger(__name__)
        self.tts_service = tts_service
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.settings = load_global_settings() or {}
        
        # Voice recording state
        self.voice_client: Optional[discord.VoiceClient] = None
        self.is_recording = False
        self.whisper_model = None
        
        # Load Whisper model
        self._load_whisper_model()
        
        self.logger.info("[PycordVoiceBot] Initialized with py-cord voice recording")
    
    def _load_whisper_model(self):
        """Load Whisper model for STT processing"""
        try:
            model_size = self.settings.get('WHISPER_MODEL_SIZE', 'base.en')
            self.whisper_model = whisper.load_model(model_size)
            self.logger.info(f"[PycordVoiceBot] Whisper model '{model_size}' loaded")
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] Failed to load Whisper: {e}")
    
    async def on_ready(self):
        """Bot ready event"""
        self.logger.info(f"[PycordVoiceBot] Bot ready as {self.user}")
        self.logger.info("[PycordVoiceBot] Available commands: !join, !leave, !status, !chat")
    
    @commands.command(name='join')
    async def join_voice(self, ctx):
        """Join voice channel and start recording"""
        try:
            if not ctx.author.voice:
                await ctx.send("❌ **You need to be in a voice channel first!**")
                return
            
            channel = ctx.author.voice.channel
            
            # Check if already connected
            if self.voice_client:
                if self.voice_client.channel == channel:
                    await ctx.send(f"✅ **Already connected to {channel.name}!**")
                    return
                else:
                    await self.voice_client.move_to(channel)
                    await ctx.send(f"🔄 **Moved to {channel.name}!**")
                    return
            
            # Connect to voice channel
            await ctx.send(f"📞 **Connecting to {channel.name}...**")
            self.voice_client = await channel.connect()
            
            if self.voice_client and self.voice_client.is_connected():
                # Start recording with py-cord's WaveSink
                sink = discord.sinks.WaveSink()
                self.voice_client.start_recording(
                    sink,
                    self._recording_finished_callback,
                    ctx
                )
                
                self.is_recording = True
                await ctx.send(f"🎤 **Connected and recording in {channel.name}!**\n"
                              f"💬 **Speak now - I'm listening with Whisper STT!**\n"
                              f"🛑 **Use `!leave` when done.**")
                self.logger.info(f"[PycordVoiceBot] Started recording in {channel.name}")
            else:
                await ctx.send("❌ **Failed to connect to voice channel.**")
                
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] Join failed: {e}")
            await ctx.send(f"❌ **Failed to join voice channel:** {str(e)}")
    
    @commands.command(name='leave')
    async def leave_voice(self, ctx):
        """Stop recording and leave voice channel"""
        try:
            if not self.voice_client:
                await ctx.send("❌ **Not connected to any voice channel.**")
                return
            
            if self.is_recording:
                self.voice_client.stop_recording()
                self.is_recording = False
                await ctx.send("🛑 **Stopped recording.**")
            
            await self.voice_client.disconnect()
            self.voice_client = None
            await ctx.send("👋 **Disconnected from voice channel.**")
            self.logger.info("[PycordVoiceBot] Disconnected from voice")
            
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] Leave failed: {e}")
            await ctx.send(f"❌ **Failed to leave:** {str(e)}")
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Show bot status"""
        # Voice connection status
        voice_status = "Not connected"
        if self.voice_client:
            if self.voice_client.is_connected():
                voice_status = f"Connected to {self.voice_client.channel.name}"
                if self.is_recording:
                    voice_status += " (Recording)"
            else:
                voice_status = "Connection failed"
        
        # Service status
        whisper_status = "✅ Loaded" if self.whisper_model else "❌ Not loaded"
        tts_status = "✅ Available" if self.tts_service else "❌ Not available"
        llm_status = "✅ Available" if self.llm_service else "❌ Not available"
        
        embed = discord.Embed(
            title="🎤 DanzarAI Voice Bot Status",
            color=discord.Color.green()
        )
        embed.add_field(name="✅ Bot Status", value="Online and Working!", inline=False)
        embed.add_field(name="🎤 Whisper STT", value=whisper_status, inline=True)
        embed.add_field(name="🔊 TTS Service", value=tts_status, inline=True)
        embed.add_field(name="🧠 LLM Service", value=llm_status, inline=True)
        embed.add_field(name="📡 Latency", value=f"{round(self.latency * 1000)}ms", inline=True)
        embed.add_field(name="🔊 Voice Status", value=voice_status, inline=False)
        embed.add_field(name="🎯 Command Prefix", value=self.command_prefix, inline=True)
        
        await ctx.send(embed=embed)
    
    @commands.command(name='chat')
    async def chat_command(self, ctx, *, message: str):
        """Chat with DanzarAI LLM and get TTS response"""
        try:
            if not self.llm_service:
                await ctx.send("❌ **LLM service not available**")
                return
            
            # Show typing indicator
            async with ctx.typing():
                # Get LLM response using existing service
                response = await self.llm_service.handle_user_text_query(message, ctx.author.display_name)
                
                if response and response.strip():
                    # Clean response (remove think tags)
                    clean_response = self._strip_think_tags(response)
                    
                    # Send text response
                    if len(clean_response) > 1900:
                        # Split long responses
                        chunks = [clean_response[i:i+1900] for i in range(0, len(clean_response), 1900)]
                        for i, chunk in enumerate(chunks):
                            await ctx.send(f"**[Part {i+1}/{len(chunks)}]**\n{chunk}")
                    else:
                        await ctx.send(f"**DanzarAI:** {clean_response}")
                    
                    # If in voice channel, also play TTS
                    if self.voice_client and self.tts_service:
                        try:
                            await self._play_tts_response(clean_response)
                        except Exception as tts_error:
                            self.logger.warning(f"TTS playback failed: {tts_error}")
                            await ctx.send("🔇 *(TTS failed, but text response above)*")
                else:
                    await ctx.send("🤔 **I couldn't generate a response. Please try again.**")
                    
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] Chat command error: {e}")
            await ctx.send("❌ **Error processing your message.**")
    
    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM responses"""
        if not text:
            return text
        
        import re
        clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = clean_text.strip()
        
        if not clean_text and text.strip():
            clean_text = "I'm thinking about that... let me get back to you."
        
        return clean_text
    
    async def _play_tts_response(self, text: str):
        """Play TTS response through Discord voice"""
        try:
            if not self.voice_client or not self.voice_client.is_connected():
                return
            
            # Generate TTS audio using existing service
            tts_audio = self.tts_service.generate_audio(text)
            if not tts_audio:
                self.logger.warning("[PycordVoiceBot] No TTS audio generated")
                return
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(tts_audio)
                temp_file_path = temp_file.name
            
            # Play through Discord voice
            source = discord.FFmpegPCMAudio(temp_file_path)
            
            def after_playback(error):
                if error:
                    self.logger.error(f"TTS playback error: {error}")
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
            self.voice_client.play(source, after=after_playback)
            self.logger.info("[PycordVoiceBot] TTS playback started")
            
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] TTS playback error: {e}")
    
    async def _recording_finished_callback(self, sink, ctx):
        """Handle recorded audio when recording stops"""
        self.logger.info("[PycordVoiceBot] Processing recorded audio...")
        
        try:
            # Process each user's audio
            for user_id, audio in sink.audio_data.items():
                user = self.get_user(user_id)
                if not user or user.bot:
                    continue
                
                # Save audio file
                filename = f"recorded_audio_{user_id}.wav"
                with open(filename, "wb") as f:
                    f.write(audio.file.getvalue())
                
                # Check file size
                file_size = os.path.getsize(filename)
                self.logger.info(f"[PycordVoiceBot] Audio file size: {file_size} bytes")
                
                if file_size < 1000:  # Less than 1KB
                    await ctx.send(f"⚠️ **{user.display_name}**: Audio too short to process")
                    os.remove(filename)
                    continue
                
                # Transcribe with Whisper
                if self.whisper_model:
                    try:
                        self.logger.info(f"[PycordVoiceBot] Transcribing audio from {user.display_name}...")
                        result = self.whisper_model.transcribe(filename)
                        text = str(result["text"]).strip()
                        
                        if text:
                            await ctx.send(f"🗣️ **{user.display_name}**: {text}")
                            self.logger.info(f"[PycordVoiceBot] Transcribed: {text}")
                            
                            # Process with LLM if available
                            if self.llm_service:
                                try:
                                    response = await self.llm_service.handle_user_text_query(text, user.display_name)
                                    if response and response.strip():
                                        clean_response = self._strip_think_tags(response)
                                        await ctx.send(f"🤖 **DanzarAI**: {clean_response}")
                                        
                                        # Play TTS response
                                        if self.voice_client:
                                            await self._play_tts_response(clean_response)
                                except Exception as llm_error:
                                    self.logger.error(f"LLM processing error: {llm_error}")
                        else:
                            await ctx.send(f"🤔 **{user.display_name}**: No speech detected")
                            
                    except Exception as e:
                        self.logger.error(f"[PycordVoiceBot] Transcription failed: {e}")
                        await ctx.send(f"❌ **Failed to transcribe {user.display_name}'s audio**")
                else:
                    await ctx.send("❌ **Whisper model not loaded**")
                
                # Clean up audio file
                try:
                    os.remove(filename)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"[PycordVoiceBot] Recording callback failed: {e}")
            await ctx.send(f"❌ **Error processing audio:** {str(e)}")


async def create_pycord_voice_bot(app_context) -> PycordVoiceBot:
    """
    Factory function to create a PycordVoiceBot with DanzarAI services
    Integrates with existing AppContext and services
    """
    # Setup Discord intents for voice
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True
    
    # Get command prefix from settings
    command_prefix = app_context.global_settings.get('DISCORD_COMMAND_PREFIX', '!')
    
    # Create bot with existing services
    bot = PycordVoiceBot(
        command_prefix=command_prefix,
        intents=intents,
        tts_service=app_context.tts_service_instance,
        llm_service=app_context.llm_service_instance,
        memory_service=app_context.memory_service_instance
    )
    
    return bot


if __name__ == "__main__":
    """
    Standalone test runner for the py-cord voice patch
    """
    import asyncio
    from core.config_loader import load_global_settings
    
    async def main():
        # Load settings
        settings = load_global_settings()
        if not settings:
            print("❌ Failed to load settings")
            return
        
        # Get Discord token
        token = settings.get('DISCORD_BOT_TOKEN')
        if not token:
            print("❌ DISCORD_BOT_TOKEN not found in settings")
            return
        
        # Create mock services for testing
        class MockService:
            def __init__(self, name):
                self.name = name
            
            async def handle_user_text_query(self, text, user):
                return f"Mock {self.name} response to: {text}"
            
            def generate_audio(self, text):
                return b"mock_audio_data"
        
        # Setup Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        # Create bot with mock services
        bot = PycordVoiceBot(
            command_prefix='!',
            intents=intents,
            tts_service=MockService("TTS"),
            llm_service=MockService("LLM"),
            memory_service=MockService("Memory")
        )
        
        try:
            print("🚀 Starting Py-cord Voice Bot...")
            await bot.start(token)
        except KeyboardInterrupt:
            print("👋 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")
        finally:
            await bot.close()
    
    asyncio.run(main()) 