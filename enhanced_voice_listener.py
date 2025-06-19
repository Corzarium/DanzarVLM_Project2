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
import audioop
import webrtcvad

from core.config_loader import load_global_settings
from core.game_profile import GameProfile

# Global app context
app_context = None

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAppContext:
    """Enhanced app context for voice bot functionality."""
    def __init__(self, config, profile):
        self.global_settings = config
        self.default_profile = profile
        self.logger = logger
        
        # Initialize queues and events (needed by services)
        self.transcription_queue = queue.Queue()
        self.memory_queue = queue.Queue()
        self.audio_out_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        
        # Service instances
        self.tts_service_instance = None
        self.memory_service_instance = None
        self.llm_service_instance = None
        self.whisper_model = None
        
        # Voice conversation state
        self.is_listening = False
        self.conversation_active = False
        
        # Audio processing
        self.vad = webrtcvad.Vad()
        self.vad.set_aggressiveness(2)  # 0-3, 3 is most aggressive
        self.audio_buffer = []
        self.silence_threshold = 1.0  # seconds of silence to end speech

# Enhanced VoiceClient that can receive audio
class VoiceReceiver(discord.VoiceClient):
    def __init__(self, client, channel):
        super().__init__(client, channel)
        self.audio_data = {}
        self.listening = False
    
    def listen(self, sink):
        """Start listening for voice input."""
        if not self.listening:
            self.listening = True
            self.start_recording(sink)

# Audio sink to capture voice
class AudioSink(voice_recv.AudioSink):
    def __init__(self):
        self.audio_data = {}
        
    def write(self, data, user):
        """Called when audio data is received."""
        if user.id not in self.audio_data:
            self.audio_data[user.id] = []
        
        # Convert opus to PCM if needed
        if hasattr(data, 'packet') and data.packet:
            # Decode opus packet to PCM
            try:
                pcm_audio = data.packet.decode()
                self.audio_data[user.id].append(pcm_audio)
            except:
                pass
    
    def cleanup(self):
        """Clean up audio data."""
        self.audio_data.clear()

# Setup bot
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    global app_context
    logger.info(f'[EnhancedVoiceBot] Bot ready! Logged in as {bot.user}')
    
    # Load configuration
    config = load_global_settings() or {}
    profile = GameProfile.load_default_profile()
    
    # Create app context
    app_context = VoiceAppContext(config, profile)
    
    # Initialize services
    try:
        # Initialize TTS service
        from services.tts_service import TTSService
        app_context.tts_service_instance = TTSService(app_context)
        logger.info("[EnhancedVoiceBot] TTS service initialized")
        
        # Initialize Memory service  
        from services.memory_service import MemoryService
        app_context.memory_service_instance = MemoryService(app_context)
        logger.info("[EnhancedVoiceBot] Memory service initialized")
        
        # Initialize LLM service
        from services.llm_service import LLMService
        app_context.llm_service_instance = LLMService(app_context, None)
        logger.info("[EnhancedVoiceBot] LLM service initialized")
        
        # Initialize Whisper for STT
        whisper_model_size = config.get('WHISPER_MODEL_SIZE', 'base.en')
        app_context.whisper_model = whisper.load_model(whisper_model_size)
        logger.info(f"[EnhancedVoiceBot] Whisper model '{whisper_model_size}' loaded")
        
    except Exception as e:
        logger.error(f"[EnhancedVoiceBot] Service initialization failed: {e}")
    
    # Send ready message
    text_channel_id = config.get('DISCORD_TEXT_CHANNEL_ID')
    if text_channel_id:
        try:
            channel = bot.get_channel(int(text_channel_id))
            if channel and hasattr(channel, 'send'):
                await channel.send("🎤 **DanzarAI Enhanced Voice Bot is ready!**\n`!join` - Connect to voice\n`!chat <text>` - Text to voice chat\n`!listen` - Start voice conversation\n`!stop` - Stop voice mode")
        except Exception as e:
            logger.warning(f"Could not send ready message: {e}")

@bot.command(name='join')
async def join_voice(ctx):
    """Join the voice channel."""
    if ctx.author.voice and ctx.author.voice.channel:
        channel = ctx.author.voice.channel
        if ctx.voice_client is None:
            try:
                voice_client = await channel.connect()
                await ctx.send(f"✅ Connected to **{channel.name}**")
                await ctx.send("🎤 **Enhanced Voice Chat Ready!**\n• `!chat <message>` - Text to voice chat\n• `!listen` - Start voice conversation")
            except Exception as e:
                await ctx.send(f"❌ Failed to connect: {e}")
        else:
            await ctx.send(f"ℹ️ Already connected to **{ctx.voice_client.channel.name}**")
    else:
        await ctx.send("❌ You need to be in a voice channel first!")

@bot.command(name='leave')
async def leave_voice(ctx):
    """Leave the voice channel."""
    if ctx.voice_client:
        channel_name = ctx.voice_client.channel.name
        if hasattr(app_context, 'conversation_active'):
            app_context.conversation_active = False
            app_context.is_listening = False
        await ctx.voice_client.disconnect()
        await ctx.send(f"✅ Disconnected from **{channel_name}**")
    else:
        await ctx.send("ℹ️ Not connected to any voice channel!")

@bot.command(name='chat')
async def chat_command(ctx, *, message: str):
    """Chat with DanzarAI and get voice response."""
    try:
        if not ctx.voice_client:
            await ctx.send("❌ Join a voice channel first with `!join`")
            return
        
        if not app_context.llm_service_instance:
            await ctx.send("❌ LLM service not available")
            return
        
        await ctx.send(f"🤔 Processing: *{message}*")
        
        # Get LLM response
        async with ctx.typing():
            response = await get_llm_response(message, ctx.author.display_name)
            
            if not response:
                await ctx.send("🤔 I couldn't generate a response. Please try again.")
                return
            
            # Send text response
            await ctx.send(f"**DanzarAI:** {response}")
            
            # Play TTS response
            await play_tts_response(ctx, response)
        
    except Exception as e:
        logger.error(f"Chat command error: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")

@bot.command(name='listen')
async def start_voice_listening(ctx):
    """Start listening for voice input."""
    if not ctx.voice_client:
        await ctx.send("❌ Join a voice channel first with `!join`")
        return
    
    if app_context.conversation_active:
        await ctx.send("ℹ️ Voice listening already active! Use `!stop` to end")
        return
    
    try:
        # Create audio sink
        sink = AudioSink()
        
        # Start recording
        ctx.voice_client.start_recording(sink, lambda e: logger.error(f"Recording error: {e}") if e else None)
        
        app_context.conversation_active = True
        await ctx.send("🎤 **Voice listening started!**\n• Speak naturally to chat with DanzarAI\n• Use `!stop` to end voice conversation")
        
        # Start voice processing loop
        asyncio.create_task(voice_processing_loop(ctx, sink))
        
    except Exception as e:
        logger.error(f"Voice listening error: {e}")
        await ctx.send(f"❌ Failed to start voice listening: {e}")

@bot.command(name='stop')
async def stop_voice_listening(ctx):
    """Stop voice listening."""
    if ctx.voice_client and ctx.voice_client.recording:
        ctx.voice_client.stop_recording()
    
    app_context.conversation_active = False
    app_context.is_listening = False
    await ctx.send("🔇 Voice listening stopped")

async def voice_processing_loop(ctx, sink):
    """Process voice input continuously."""
    logger.info("[EnhancedVoiceBot] Starting voice processing loop")
    
    while app_context.conversation_active:
        try:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            if not ctx.voice_client or not ctx.voice_client.is_connected():
                break
            
            # Check for new audio data
            for user_id, audio_chunks in sink.audio_data.items():
                if not audio_chunks:
                    continue
                
                # Get user
                user = bot.get_user(user_id)
                if not user or user.bot:
                    continue
                
                # Process audio chunks
                audio_data = b''.join(audio_chunks)
                sink.audio_data[user_id] = []  # Clear processed chunks
                
                if len(audio_data) < 3200:  # Skip very short audio
                    continue
                
                # Transcribe audio
                transcript = await transcribe_audio(audio_data)
                if not transcript or len(transcript.strip()) < 3:
                    continue
                
                logger.info(f"[EnhancedVoiceBot] {user.display_name}: {transcript}")
                
                # Get LLM response
                response = await get_llm_response(transcript, user.display_name)
                if response:
                    # Send conversation to chat
                    await ctx.send(f"👤 **{user.display_name}:** {transcript}\n🤖 **DanzarAI:** {response}")
                    
                    # Play voice response
                    await play_tts_response(ctx, response)
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}", exc_info=True)
            await asyncio.sleep(1)
    
    logger.info("[EnhancedVoiceBot] Voice processing loop ended")

async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Whisper."""
    try:
        if not app_context.whisper_model or not audio_data:
            return ""
        
        # Convert audio data to proper format for Whisper
        # Discord audio is 48kHz stereo, Whisper expects 16kHz mono
        try:
            # Save raw audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Create WAV header for Discord audio (48kHz, 16-bit, stereo)
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(2)  # Stereo
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(48000)  # 48kHz
                    wav_file.writeframes(audio_data)
                
                temp_file_path = temp_file.name
            
            # Transcribe with Whisper
            result = app_context.whisper_model.transcribe(temp_file_path)
            transcript = result.get('text', '').strip()
            
            # Cleanup
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            return transcript
            
        except Exception as audio_error:
            logger.error(f"Audio processing error: {audio_error}")
            return ""
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

async def get_llm_response(message: str, user_name: str) -> str:
    """Get response from LLM service."""
    try:
        if not app_context.llm_service_instance:
            return "Sorry, the AI service is not available right now."
        
        # Use existing LLM service with Agentic RAG
        response = await app_context.llm_service_instance.handle_user_text_query(message, user_name)
        
        if not response:
            return "I'm having trouble generating a response right now."
        
        # Clean response (remove think tags)
        clean_response = strip_think_tags(response)
        return clean_response
        
    except Exception as e:
        logger.error(f"LLM response error: {e}")
        return "Sorry, I encountered an error processing your request."

async def play_tts_response(ctx, message: str):
    """Generate and play TTS response."""
    try:
        if not app_context.tts_service_instance:
            logger.error("TTS service not available")
            return
        
        # Generate TTS audio
        tts_audio = app_context.tts_service_instance.generate_audio(message)
        if not tts_audio:
            logger.error("TTS generation failed")
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
        
        if ctx.voice_client and not ctx.voice_client.is_playing():
            ctx.voice_client.play(source, after=cleanup_after)
        
    except Exception as e:
        logger.error(f"TTS playback error: {e}")

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

@bot.command(name='status')
async def status_command(ctx):
    """Show bot status."""
    embed = discord.Embed(title="🎤 DanzarAI Enhanced Voice Bot Status", color=0x00ff00)
    
    # Voice connection status
    if ctx.voice_client:
        recording_status = "🎙️ Recording" if ctx.voice_client.recording else "⏸️ Not recording"
        embed.add_field(name="Voice", value=f"✅ Connected to {ctx.voice_client.channel.name}\n{recording_status}", inline=False)
    else:
        embed.add_field(name="Voice", value="❌ Not connected", inline=False)
    
    # Service status
    services_status = []
    if app_context:
        services_status.append(f"TTS: {'✅' if app_context.tts_service_instance else '❌'}")
        services_status.append(f"LLM: {'✅' if app_context.llm_service_instance else '❌'}")
        services_status.append(f"Memory: {'✅' if app_context.memory_service_instance else '❌'}")
        services_status.append(f"Whisper: {'✅' if app_context.whisper_model else '❌'}")
    
    embed.add_field(name="Services", value="\n".join(services_status), inline=False)
    
    # Conversation status
    if app_context:
        conv_status = "🎤 Active" if app_context.conversation_active else "⏸️ Inactive"
        embed.add_field(name="Voice Conversation", value=conv_status, inline=False)
    
    embed.set_footer(text="DanzarAI Enhanced Voice Chat System with Discord Audio Capture")
    await ctx.send(embed=embed)

if __name__ == "__main__":
    # Load configuration
    config = load_global_settings() or {}
    token = config.get('DISCORD_BOT_TOKEN', '')
    
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not found in configuration!")
        exit(1)
    
    print(f"Starting DanzarAI Enhanced Voice Bot...")
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Bot startup error: {e}") 