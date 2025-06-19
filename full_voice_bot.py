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
import io
import whisper
import wave
import audioop

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

# Setup bot
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    global app_context
    logger.info(f'[VoiceBot] Bot ready! Logged in as {bot.user}')
    
    # Load configuration
    config = load_global_settings() or {}
    
    # Create simple profile for voice chat
    profile = GameProfile(
        game_name="voice_chat", 
        system_prompt_commentary="You are DanzarAI, a helpful voice assistant.",
        user_prompt_template_commentary="",
        vlm_model="voice",
        vlm_max_tokens=200,
        vlm_temperature=0.7
    )
    
    # Create app context
    app_context = VoiceAppContext(config, profile)
    
    # Initialize services
    try:
        # Initialize TTS service
        from services.tts_service import TTSService
        app_context.tts_service_instance = TTSService(app_context)
        logger.info("[VoiceBot] TTS service initialized")
        
        # Initialize Memory service  
        from services.memory_service import MemoryService
        app_context.memory_service_instance = MemoryService(app_context)
        logger.info("[VoiceBot] Memory service initialized")
        
        # Initialize LLM service
        from services.llm_service import LLMService
        app_context.llm_service_instance = LLMService(app_context, None)
        logger.info("[VoiceBot] LLM service initialized")
        
        # Initialize Whisper for STT
        whisper_model_size = config.get('WHISPER_MODEL_SIZE', 'base.en')
        app_context.whisper_model = whisper.load_model(whisper_model_size)
        logger.info(f"[VoiceBot] Whisper model '{whisper_model_size}' loaded")
        
    except Exception as e:
        logger.error(f"[VoiceBot] Service initialization failed: {e}")
    
    # Send ready message
    text_channel_id = config.get('DISCORD_TEXT_CHANNEL_ID')
    if text_channel_id:
        try:
            channel = bot.get_channel(int(text_channel_id))
            if channel and hasattr(channel, 'send'):
                await channel.send("🎤 **DanzarAI Full Voice Bot is ready!**\n`!join` - Connect to voice\n`!chat <text>` - Text to voice chat\n`!voice` - Start voice conversation\n`!stop` - Stop voice mode")
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
                await ctx.send("🎤 Ready for voice chat! Use `!voice` to start voice conversation or `!chat <message>` for text-to-speech")
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

@bot.command(name='voice')
async def start_voice_conversation(ctx):
    """Start voice conversation mode."""
    if not ctx.voice_client:
        await ctx.send("❌ Join a voice channel first with `!join`")
        return
    
    if app_context.conversation_active:
        await ctx.send("ℹ️ Voice conversation already active! Say 'stop listening' or use `!stop`")
        return
    
    app_context.conversation_active = True
    await ctx.send("🎤 **Voice conversation started!**\n• Speak naturally to chat with DanzarAI\n• Say 'stop listening' or use `!stop` to end\n• I'll respond with voice")
    
    # Start voice listening loop
    asyncio.create_task(voice_conversation_loop(ctx))

@bot.command(name='stop')
async def stop_voice_conversation(ctx):
    """Stop voice conversation mode."""
    app_context.conversation_active = False
    app_context.is_listening = False
    await ctx.send("🔇 Voice conversation stopped")

async def voice_conversation_loop(ctx):
    """Main voice conversation loop."""
    logger.info("[VoiceBot] Starting voice conversation loop")
    
    while app_context.conversation_active:
        try:
            if not ctx.voice_client or not ctx.voice_client.is_connected():
                break
            
            # Listen for voice input
            audio_data = await listen_for_voice(ctx)
            if not audio_data:
                await asyncio.sleep(0.1)
                continue
            
            # Transcribe speech
            transcript = await transcribe_audio(audio_data)
            if not transcript or len(transcript.strip()) < 3:
                continue
            
            # Check for stop command
            if "stop listening" in transcript.lower():
                app_context.conversation_active = False
                await ctx.send("🔇 Voice conversation stopped")
                break
            
            logger.info(f"[VoiceBot] Transcribed: {transcript}")
            
            # Get LLM response
            response = await get_llm_response(transcript, ctx.author.display_name)
            if response:
                # Send text for visibility
                await ctx.send(f"👤 **You:** {transcript}\n🤖 **DanzarAI:** {response}")
                
                # Play voice response
                await play_tts_response(ctx, response)
            
        except Exception as e:
            logger.error(f"Voice conversation error: {e}", exc_info=True)
            await asyncio.sleep(1)
    
    logger.info("[VoiceBot] Voice conversation loop ended")

async def listen_for_voice(ctx) -> Optional[bytes]:
    """Listen for voice input from Discord."""
    # This is a simplified placeholder - Discord voice receiving is complex
    # For full implementation, you'd use discord-ext-voice-recv
    # For now, return None to indicate no audio captured
    await asyncio.sleep(0.1)
    return None

async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Whisper."""
    try:
        if not app_context.whisper_model or not audio_data:
            return ""
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
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
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

async def get_llm_response(message: str, user_name: str) -> str:
    """Get response from LLM service."""
    try:
        if not app_context.llm_service_instance:
            return "Sorry, the AI service is not available right now."
        
        # Use existing LLM service
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
    embed = discord.Embed(title="🎤 DanzarAI Voice Bot Status", color=0x00ff00)
    
    # Voice connection status
    if ctx.voice_client:
        embed.add_field(name="Voice", value=f"✅ Connected to {ctx.voice_client.channel.name}", inline=False)
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
    
    embed.set_footer(text="DanzarAI Full Voice Chat System")
    await ctx.send(embed=embed)

if __name__ == "__main__":
    # Load configuration
    config = load_global_settings() or {}
    token = config.get('DISCORD_BOT_TOKEN', '')
    
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not found in configuration!")
        exit(1)
    
    print(f"Starting DanzarAI Full Voice Bot...")
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Bot startup error: {e}") 