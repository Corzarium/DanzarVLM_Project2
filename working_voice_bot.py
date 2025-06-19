"""
Working Voice Bot for DanzarAI
Simple implementation that actually registers commands properly
"""

import discord
from discord.ext import commands, voice_recv
import asyncio
import logging
import tempfile
import os
from typing import Optional
import time
import queue
import threading
import io
import whisper
import wave
import webrtcvad

from core.config_loader import load_global_settings
from core.game_profile import GameProfile


class WorkingAppContext:
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
        
        # Additional events needed by LLMService
        self.is_in_conversation = threading.Event()
        self.is_in_conversation.clear()
        
        # Queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.tts_queue = queue.Queue(maxsize=20)
        self.text_message_queue = queue.Queue(maxsize=20)
        
        # Service instances
        self.tts_service_instance = None
        self.llm_service_instance = None
        self.memory_service_instance = None
        
        # Required by LLMService
        self.active_profile_change_subscribers = []
        
        # Voice conversation state
        self.voice_listening = False
        self.audio_buffer = []
        
        # VAD for voice activity detection
        self.vad = webrtcvad.Vad(2)  # 0-3, 3 is most aggressive
        
        self.logger.info("[WorkingAppContext] Initialized for testing")


# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
config = load_global_settings() or {}

# Setup Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

# Create bot instance
bot = commands.Bot(command_prefix=config.get('DISCORD_COMMAND_PREFIX', '!'), intents=intents)

# Audio sink for capturing voice with proper discord-ext-voice-recv implementation
class VoiceAudioSink(voice_recv.AudioSink):
    def __init__(self, app_context):
        self.app_context = app_context
        self.audio_data = {}
    
    def wants_opus(self) -> bool:
        """Return False to get PCM audio instead of Opus."""
        return False
        
    def write(self, user, data):
        """Called when audio data is received from Discord."""
        if not user or user.bot:
            return
            
        if not self.app_context.voice_listening:
            return
            
        user_id = user.id
        if user_id not in self.audio_data:
            self.audio_data[user_id] = []
        
        # Append PCM audio data (20ms chunks at 48kHz stereo)
        if data and hasattr(data, 'pcm') and data.pcm:
            self.audio_data[user_id].append(data.pcm)
            logger.debug(f"[VoiceAudioSink] Received {len(data.pcm)} bytes from {user.display_name}")
    
    def cleanup(self):
        """Clean up audio data."""
        self.audio_data.clear()

# Create app context
app_context = WorkingAppContext(config, logger)


@bot.event
async def on_ready():
    """Called when bot is ready and connected."""
    logger.info(f'[WorkingVoiceBot] Bot ready! Logged in as {bot.user}')
    
    # Initialize services
    try:
        # Initialize TTS service
        from services.tts_service import TTSService
        app_context.tts_service_instance = TTSService(app_context)
        logger.info("[WorkingVoiceBot] TTS service initialized")
        
        # Initialize Memory service
        from services.memory_service import MemoryService
        app_context.memory_service_instance = MemoryService(app_context)
        logger.info("[WorkingVoiceBot] Memory service initialized")
        
        # Initialize LLM service
        from services.llm_service import LLMService
        app_context.llm_service_instance = LLMService(app_context, None)  # No audio service needed
        logger.info("[WorkingVoiceBot] LLM service initialized")
        
        # Initialize Whisper for STT
        whisper_model_size = config.get('WHISPER_MODEL_SIZE', 'base.en')
        app_context.whisper_model = whisper.load_model(whisper_model_size)
        logger.info(f"[WorkingVoiceBot] Whisper model '{whisper_model_size}' loaded")
        
    except Exception as e:
        logger.error(f"[WorkingVoiceBot] Service initialization failed: {e}")
    
    # Log available commands
    logger.info(f'[WorkingVoiceBot] Available commands: {[cmd.name for cmd in bot.commands]}')
    
    # Send ready message to configured text channel
    text_channel_id = config.get('DISCORD_TEXT_CHANNEL_ID')
    if text_channel_id:
        try:
            channel = bot.get_channel(int(text_channel_id))
            if channel and hasattr(channel, 'send'):
                await channel.send("🎤 **DanzarAI Voice Bot is ready!**\n`!join` for voice chat, `!say <text>` to test TTS, `!listen` to start voice conversation")
        except Exception as e:
            logger.warning(f"Could not send ready message: {e}")


@bot.command(name='join')
async def join_voice(ctx):
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
        await ctx.send("💬 Try: `!say Hello from DanzarAI`")
        
    except Exception as e:
        logger.error(f"Join error: {e}", exc_info=True)
        await ctx.send(f"❌ Join failed: {e}")


@bot.command(name='leave')
async def leave_voice(ctx):
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


@bot.command(name='say')
async def say_text(ctx, *, message: str):
    """Test TTS with given message."""
    try:
        if not ctx.voice_client:
            await ctx.send("❌ Join a voice channel first with `!join`")
            return
        
        if not app_context.tts_service_instance:
            await ctx.send("❌ TTS service not available")
            return
        
        await ctx.send(f"🗣️ Saying: *{message}*")
        await play_tts_message(ctx, message)
        
    except Exception as e:
        logger.error(f"Say command error: {e}", exc_info=True)
        await ctx.send(f"❌ Command error: {e}")


@bot.command(name='chat')
async def chat_with_llm(ctx, *, message: str):
    """Chat with DanzarAI LLM and get voice response."""
    try:
        if not ctx.voice_client:
            await ctx.send("❌ Join a voice channel first with `!join`")
            return
        
        if not app_context.llm_service_instance:
            await ctx.send("❌ LLM service not available")
            return
        
        await ctx.send(f"🤔 Processing: *{message}*")
        
        # Get LLM response using existing service
        try:
            async with ctx.typing():
                response = await app_context.llm_service_instance.handle_user_text_query(message, ctx.author.display_name)
                
                if not response:
                    await ctx.send("🤔 I couldn't generate a response. Please try again.")
                    return
                
                # Clean response (remove think tags)
                clean_response = strip_think_tags(response)
                
                # Send text response
                await ctx.send(f"**DanzarAI:** {clean_response}")
                
                # Play TTS response
                await play_tts_message(ctx, clean_response)
                
        except Exception as llm_error:
            await ctx.send(f"❌ LLM error: {llm_error}")
            logger.error(f"LLM error: {llm_error}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Chat command error: {e}", exc_info=True)
        await ctx.send(f"❌ Command error: {e}")


async def play_tts_message(ctx, message: str):
    """Helper function to generate and play TTS audio."""
    try:
        # Generate TTS
        tts_audio = app_context.tts_service_instance.generate_audio(message)
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
                logger.error(f"Playback error: {error}")
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        ctx.voice_client.play(source, after=cleanup_after)
        
    except Exception as e:
        logger.error(f"TTS playback error: {e}", exc_info=True)
        await ctx.send(f"❌ TTS error: {e}")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from LLM responses."""
    if not text:
        return text
    
    import re
    clean_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = clean_text.strip()
    
    if not clean_text and text.strip():
        clean_text = "I'm thinking about that... let me get back to you."
    
    return clean_text


async def voice_processing_loop(ctx, audio_sink):
    """Main voice processing loop for speech-to-text."""
    logger.info("[WorkingVoiceBot] Starting voice processing loop")
    
    while app_context.voice_listening:
        try:
            await asyncio.sleep(2)  # Check every 2 seconds for new audio
            
            if not ctx.voice_client or not ctx.voice_client.is_connected():
                break
            
            # Process audio data from each user
            for user_id, audio_chunks in audio_sink.audio_data.items():
                if not audio_chunks:
                    continue
                
                # Get user info
                user = bot.get_user(user_id)
                if not user or user.bot:
                    continue
                
                # Combine audio chunks
                combined_audio = b''.join(audio_chunks)
                audio_sink.audio_data[user_id] = []  # Clear processed chunks
                
                if len(combined_audio) < 3200:  # Skip very short audio (< 0.1 seconds)
                    continue
                
                logger.info(f"[WorkingVoiceBot] Processing audio from {user.display_name} ({len(combined_audio)} bytes)")
                
                # Transcribe audio using Whisper
                transcript = await transcribe_voice_audio(combined_audio)
                if not transcript or len(transcript.strip()) < 3:
                    continue
                
                logger.info(f"[WorkingVoiceBot] Transcribed from {user.display_name}: {transcript}")
                
                # Check for stop command
                if any(phrase in transcript.lower() for phrase in ["stop listening", "stop voice", "end conversation"]):
                    app_context.voice_listening = False
                    await ctx.send(f"🔇 **{user.display_name} ended voice conversation**")
                    break
                
                # Process with LLM and respond
                await process_voice_message(ctx, transcript, user.display_name)
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}", exc_info=True)
            await asyncio.sleep(1)
    
    # Stop recording when loop ends
    if ctx.voice_client and ctx.voice_client.recording:
        ctx.voice_client.stop_recording()
    
    logger.info("[WorkingVoiceBot] Voice processing loop ended")


async def transcribe_voice_audio(audio_data: bytes) -> str:
    """Transcribe audio data using Whisper."""
    try:
        if not app_context.whisper_model or not audio_data:
            return ""
        
        # Save audio to temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Discord audio is 48kHz stereo, 16-bit PCM
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(audio_data)
            
            temp_file_path = temp_file.name
        
        # Transcribe using Whisper
        result = app_context.whisper_model.transcribe(temp_file_path)
        transcript = result.get('text', '').strip()
        
        # Cleanup temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        return transcript
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


async def process_voice_message(ctx, transcript: str, user_name: str):
    """Process transcribed voice message through LLM and respond with TTS."""
    try:
        # Send transcription to chat for visibility
        await ctx.send(f"👤 **{user_name}:** {transcript}")
        
        # Process through LLM (same as !chat command)
        if not app_context.llm_service_instance:
            await ctx.send("❌ LLM service not available")
            return
        
        # Show typing indicator
        async with ctx.typing():
            response = await app_context.llm_service_instance.handle_user_text_query(transcript, user_name)
            
            if not response:
                await ctx.send("🤔 I couldn't generate a response. Please try again.")
                return
            
            # Clean response
            clean_response = strip_think_tags(response)
            
            # Send text response
            await ctx.send(f"🤖 **DanzarAI:** {clean_response}")
            
            # Play TTS response
            await play_tts_message(ctx, clean_response)
        
    except Exception as e:
        logger.error(f"Voice message processing error: {e}", exc_info=True)
        await ctx.send(f"❌ Error processing voice message: {e}")


@bot.command(name='test')
async def test_bot(ctx):
    """Test basic functionality."""
    await ctx.send("✅ Working voice bot is active!")
    
    # Check TTS service
    if app_context.tts_service_instance:
        await ctx.send("✅ TTS service is available")
    else:
        await ctx.send("❌ TTS service not available")


@bot.command(name='listen')
async def start_listening(ctx):
    """Show voice conversation information and current capabilities."""
    if not ctx.voice_client:
        await ctx.send("❌ Join a voice channel first with `!join`")
        return
    
    embed = discord.Embed(title="🎤 DanzarAI Voice Conversation", color=0x3498db)
    
    embed.add_field(
        name="✅ Current Capability",
        value="**Text-to-Voice Conversation**\nType your message and get AI voice response!",
        inline=False
    )
    
    embed.add_field(
        name="🎯 How to Use",
        value="Use `!chat <your message>` to talk with DanzarAI\n• AI processes with LLM + RAG memory\n• Responds with natural voice (TTS)",
        inline=False
    )
    
    embed.add_field(
        name="💬 Examples",
        value="`!chat Hello DanzarAI, how are you?`\n`!chat What are good gaming strategies?`\n`!chat What do you remember about me?`",
        inline=False
    )
    
    embed.add_field(
        name="🚧 Voice Input (Speech-to-Text)",
        value="**Limited by Discord.py library**\n• Standard discord.py doesn't support voice receiving\n• For full voice input, external solutions needed",
        inline=False
    )
    
    embed.add_field(
        name="🔧 Voice Input Alternatives",
        value="• Use external STT + `!chat` command\n• Switch to py-cord library (more complex)\n• Use push-to-talk with external transcription",
        inline=False
    )
    
    embed.set_footer(text="DanzarAI - Advanced AI with voice output ready!")
    
    await ctx.send(embed=embed)

@bot.command(name='stop')
async def stop_listening(ctx):
    """Stop voice conversation mode."""
    if ctx.voice_client and ctx.voice_client.recording:
        ctx.voice_client.stop_recording()
    
    app_context.voice_listening = False
    
    await ctx.send("🔇 **Voice conversation stopped**\nUse `!listen` to start again or `!chat <message>` for text-to-voice")

@bot.command(name='status')
async def show_status(ctx):
    """Show status."""
    embed = discord.Embed(title="🎤 DanzarAI Voice Bot Status", color=0x00ff00)
    
    # Voice connection
    if ctx.voice_client:
        embed.add_field(name="Voice", value=f"✅ Connected to **{ctx.voice_client.channel.name}**", inline=False)
    else:
        embed.add_field(name="Voice", value="❌ Not connected", inline=False)
    
    # Services status  
    services = []
    services.append(f"TTS: {'✅' if app_context.tts_service_instance else '❌'}")
    services.append(f"LLM: {'✅' if app_context.llm_service_instance else '❌'}")
    services.append(f"Memory: {'✅' if app_context.memory_service_instance else '❌'}")
    services.append(f"Whisper: {'✅' if hasattr(app_context, 'whisper_model') and app_context.whisper_model else '❌'}")
    
    embed.add_field(name="Services", value="\n".join(services), inline=False)
    
    # Commands
    embed.add_field(
        name="Voice Commands",
        value="`!join` - Connect to voice\n`!chat <message>` - **AI conversation with voice**\n`!listen` - Show voice conversation info\n`!say <text>` - Test TTS\n`!leave` - Disconnect",
        inline=False
    )
    
    embed.set_footer(text="DanzarAI Working Voice Bot")
    await ctx.send(embed=embed)


async def main():
    """Main entry point."""
    try:
        token = config.get('DISCORD_BOT_TOKEN')
        if not token:
            print("❌ DISCORD_BOT_TOKEN not found")
            return
        
        print("🚀 Starting DanzarAI Working Voice Bot...")
        await bot.start(token)
        
    except KeyboardInterrupt:
        print("Bot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await bot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}") 