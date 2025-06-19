#!/usr/bin/env python3
"""
DanzarVLM - Virtual Audio Cable Version
Captures audio from virtual audio cables (VB-Cable, Windows Sound Mixer, etc.)
and processes it through STT → LLM → TTS pipeline
"""

import asyncio
import argparse
import logging
import os
import sys
import signal
import time
import threading
import tempfile
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any, List
import keyboard
import whisper
import discord
from discord.ext import commands
import torch
from collections import deque
import queue
import json

# Core imports
from core.config_loader import load_global_settings
from core.game_profile import GameProfile

# Service imports  
from services.tts_service import TTSService
from services.memory_service import MemoryService
from services.model_client import ModelClient
from services.llm_service import LLMService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/danzar_virtual_audio.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("DanzarVLM_VirtualAudio")

# Suppress noisy loggers
logging.getLogger('discord').setLevel(logging.WARNING)

class AppContext:
    """Application context for managing shared resources and services."""
    
    def __init__(self, global_settings: dict, active_profile: GameProfile, logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger_instance
        self.shutdown_event = threading.Event()
        
        # Service instances
        self.tts_service: Optional[TTSService] = None
        self.memory_service: Optional[MemoryService] = None
        self.model_client: Optional[ModelClient] = None
        self.llm_service: Optional[LLMService] = None
        
        # Voice processing components
        self.whisper_model = None
        self.vad_model = None
        
        logger_instance.info("[AppContext] Initialized for Virtual Audio.")

    def get_service(self, service_name: str):
        """Get a service instance by name."""
        return getattr(self, f"{service_name}_service", None)

class VirtualAudioCapture:
    """Captures audio from virtual audio cables and processes with VAD + STT."""
    
    def __init__(self, app_context: AppContext, callback_func=None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.callback_func = callback_func
        
        # Audio settings
        self.sample_rate = 44100  # Higher quality for better recognition
        self.channels = 2  # Stereo
        self.chunk_size = 4096  # Larger buffer for stability
        self.dtype = np.float32
        
        # VAD settings
        self.vad_model = None
        self.speech_threshold = 0.3
        self.silence_threshold = 0.2
        self.min_speech_duration = 0.5
        self.max_silence_duration = 2.0
        
        # State tracking
        self.is_recording = False
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = deque()
        
        # Audio device
        self.input_device = None
        self.stream = None
        
        # Processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
    def list_audio_devices(self):
        """List all available audio input devices."""
        self.logger.info("🎵 Available Audio Input Devices:")
        devices = sd.query_devices()
        
        virtual_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Input device
                device_name = device['name']
                self.logger.info(f"  {i}: {device_name} (channels: {device['max_input_channels']})")
                
                # Look for virtual audio cables
                if any(keyword in device_name.lower() for keyword in 
                      ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'stereo mix', 'wave out mix', 'what u hear']):
                    virtual_devices.append((i, device_name))
                    self.logger.info(f"      ⭐ VIRTUAL AUDIO DEVICE DETECTED")
        
        if virtual_devices:
            self.logger.info("🎯 Recommended Virtual Audio Devices:")
            for device_id, device_name in virtual_devices:
                self.logger.info(f"  Device {device_id}: {device_name}")
        else:
            self.logger.warning("⚠️  No virtual audio devices detected. Install VB-Cable or enable Stereo Mix.")
            
        return virtual_devices
    
    def select_input_device(self, device_id: Optional[int] = None):
        """Select audio input device."""
        if device_id is None:
            # Auto-detect virtual audio device
            virtual_devices = self.list_audio_devices()
            if virtual_devices:
                device_id = virtual_devices[0][0]  # Use first virtual device
                self.logger.info(f"🎯 Auto-selected virtual audio device: {device_id}")
            else:
                # Fall back to default device
                device_id = sd.default.device[0]
                self.logger.warning(f"⚠️  Using default input device: {device_id}")
        
        try:
            device_info = sd.query_devices(device_id, 'input')
            self.input_device = device_id
            self.logger.info(f"✅ Selected audio device: {device_info['name']}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to select device {device_id}: {e}")
            return False
    
    async def initialize_vad(self):
        """Initialize Voice Activity Detection."""
        try:
            self.logger.info("🔧 Loading Silero VAD model...")
            loop = asyncio.get_event_loop()
            self.vad_model, _ = await loop.run_in_executor(
                None,
                lambda: torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
            )
            self.logger.info("✅ VAD model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ VAD initialization failed: {e}")
            return False
    
    def process_vad(self, audio_chunk: np.ndarray) -> tuple[bool, bool]:
        """Process audio chunk with VAD."""
        if not self.vad_model:
            return False, False
            
        try:
            # Convert to 16kHz mono for VAD
            if len(audio_chunk.shape) > 1:
                audio_mono = np.mean(audio_chunk, axis=1)
            else:
                audio_mono = audio_chunk
            
            # Resample to 16kHz
            target_length = int(len(audio_mono) * 16000 / self.sample_rate)
            audio_16k = np.interp(
                np.linspace(0, len(audio_mono), target_length),
                np.arange(len(audio_mono)),
                audio_mono
            )
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_16k).float()
            
            # Get VAD probability
            speech_prob = self.vad_model(audio_tensor, 16000).item()
            
            current_time = time.time()
            is_speech = speech_prob > self.speech_threshold
            speech_ended = False
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.logger.info(f"🎤 Speech started (prob: {speech_prob:.3f})")
                
                self.last_speech_time = current_time
                
            elif self.is_speaking and self.last_speech_time is not None:
                silence_duration = current_time - self.last_speech_time
                
                if silence_duration > self.max_silence_duration and self.speech_start_time is not None:
                    speech_duration = current_time - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        speech_ended = True
                        self.logger.info(f"🎤 Speech ended (duration: {speech_duration:.2f}s)")
                    
                    self.is_speaking = False
                    self.speech_start_time = None
                    self.last_speech_time = None
            
            return is_speech, speech_ended
            
        except Exception as e:
            self.logger.error(f"❌ VAD processing error: {e}")
            return False, False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            self.logger.warning(f"Audio stream status: {status}")
            
        # Add audio to queue for processing
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            self.logger.warning("Audio queue full, dropping frames")
    
    def processing_worker(self):
        """Worker thread for processing audio."""
        self.logger.info("🎯 Audio processing worker started")
        
        while self.is_recording:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Process with VAD
                is_speech, speech_ended = self.process_vad(audio_chunk)
                
                # Add to buffer if speech is detected
                if is_speech or self.is_speaking:
                    self.audio_buffer.extend(audio_chunk)
                
                # Process accumulated audio if speech ended
                if speech_ended and len(self.audio_buffer) > 0:
                    accumulated_audio = np.array(list(self.audio_buffer))
                    self.audio_buffer.clear()
                    
                    # Schedule STT processing
                    if self.callback_func:
                        asyncio.run_coroutine_threadsafe(
                            self.callback_func(accumulated_audio),
                            asyncio.get_event_loop()
                        )
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing error: {e}")
        
        self.logger.info("🎯 Audio processing worker stopped")
    
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            self.logger.warning("Already recording")
            return False
            
        if not self.input_device:
            self.logger.error("No input device selected")
            return False
            
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.start()
            
            self.logger.info(f"🎙️ Started recording from device {self.input_device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join()
            self.processing_thread = None
        
        self.logger.info("🛑 Recording stopped")

class DiscordTTSBot(commands.Bot):
    """Discord bot for TTS output only."""
    
    def __init__(self, settings: dict, app_context: AppContext):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None
        )
        
        self.settings = settings
        self.app_context = app_context
        self.logger = app_context.logger
        
        # TTS service
        self.tts_service = None
        
        # Voice connections
        self.voice_connections: Dict[int, discord.VoiceClient] = {}
        
        self.add_commands()
    
    def add_commands(self):
        """Add Discord commands."""
        
        @self.command(name='connect')
        async def connect_command(ctx):
            """Connect to voice channel for TTS output."""
            if not ctx.author.voice:
                await ctx.send("❌ You need to be in a voice channel!")
                return
            
            channel = ctx.author.voice.channel
            
            if ctx.guild.id in self.voice_connections:
                await ctx.send(f"✅ Already connected to voice channel!")
                return
            
            try:
                vc = await channel.connect()
                self.voice_connections[ctx.guild.id] = vc
                await ctx.send(f"🔊 Connected to **{channel.name}** for TTS output!")
                self.logger.info(f"Connected to {channel.name} for TTS")
            except Exception as e:
                await ctx.send(f"❌ Failed to connect: {e}")
        
        @self.command(name='disconnect')
        async def disconnect_command(ctx):
            """Disconnect from voice channel."""
            if ctx.guild.id in self.voice_connections:
                vc = self.voice_connections[ctx.guild.id]
                await vc.disconnect()
                del self.voice_connections[ctx.guild.id]
                await ctx.send("👋 Disconnected from voice channel")
                self.logger.info("Disconnected from voice channel")
            else:
                await ctx.send("❌ Not connected to any voice channel")
        
        @self.command(name='say')
        async def say_command(ctx, *, text):
            """Test TTS with given text."""
            if ctx.guild.id not in self.voice_connections:
                await ctx.send("❌ Not connected to voice channel. Use `!connect` first.")
                return
            
            await self.speak_text(text, ctx.channel)
    
    async def speak_text(self, text: str, channel=None):
        """Generate and play TTS audio."""
        try:
            if not self.tts_service:
                self.logger.error("TTS service not available")
                return
            
            # Generate TTS audio
            loop = asyncio.get_event_loop()
            tts_audio = await loop.run_in_executor(
                None,
                self.tts_service.generate_audio,
                text
            )
            
            if not tts_audio:
                self.logger.error("TTS generation failed")
                return
            
            # Play in all connected voice channels
            for guild_id, vc in self.voice_connections.items():
                if vc.is_connected():
                    try:
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_file.write(tts_audio)
                            temp_file_path = temp_file.name
                        
                        # Play audio
                        audio_source = discord.FFmpegPCMAudio(temp_file_path)
                        vc.play(audio_source)
                        
                        # Wait for playback to finish
                        while vc.is_playing():
                            await asyncio.sleep(0.1)
                        
                        # Cleanup
                        os.unlink(temp_file_path)
                        
                        self.logger.info(f"🔊 Played TTS: {text[:50]}")
                        
                    except Exception as e:
                        self.logger.error(f"TTS playback error: {e}")
            
            # Also send text to channel
            if channel:
                await channel.send(f"🤖 **DanzarAI**: {text}")
                
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
    
    async def on_ready(self):
        """Called when bot is ready."""
        self.logger.info(f"Discord TTS Bot ready as {self.user}")
        
        # Initialize TTS service
        self.tts_service = TTSService(self.app_context)
        self.logger.info("TTS service initialized")

class DanzarVirtualAudio:
    """Main application class."""
    
    def __init__(self, settings: dict):
        self.settings = settings
        
        # Create game profile
        discord_profile = GameProfile(
            game_name="virtual_audio",
            vlm_model="qwen2.5:7b",
            system_prompt_commentary="You are DanzarAI, a helpful gaming assistant with voice capabilities.",
            user_prompt_template_commentary="User said: {user_text}. Respond helpfully about gaming.",
            vlm_max_tokens=200,
            vlm_temperature=0.7,
            vlm_max_commentary_sentences=2,
            conversational_llm_model="qwen2.5:7b",
            system_prompt_chat="You are DanzarAI, a helpful gaming assistant. Keep responses conversational and brief."
        )
        
        self.app_context = AppContext(settings, discord_profile, logger)
        
        # Components
        self.audio_capture = None
        self.discord_bot = None
        self.whisper_model = None
        
    async def initialize_services(self):
        """Initialize all services."""
        try:
            # Initialize TTS Service
            self.app_context.tts_service = TTSService(self.app_context)
            logger.info("✅ TTS Service initialized")
            
            # Initialize Memory Service
            self.app_context.memory_service = MemoryService(self.app_context)
            logger.info("✅ Memory Service initialized")
            
            # Initialize Model Client
            self.app_context.model_client = ModelClient(app_context=self.app_context)
            logger.info("✅ Model Client initialized")
            
            # Initialize LLM Service
            self.app_context.llm_service = LLMService(
                app_context=self.app_context,
                audio_service=None,
                rag_service=None,
                model_client=self.app_context.model_client
            )
            logger.info("✅ LLM Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
    
    async def initialize_whisper(self):
        """Initialize Whisper model."""
        try:
            logger.info("🔧 Loading Whisper model...")
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None,
                whisper.load_model,
                "base"  # Balance between speed and accuracy
            )
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"❌ Whisper initialization failed: {e}")
    
    async def process_audio_segment(self, audio_data: np.ndarray):
        """Process audio segment through STT → LLM → TTS pipeline."""
        try:
            logger.info(f"🎵 Processing audio segment: {audio_data.shape}")
            
            # Step 1: STT with Whisper
            transcription = await self.transcribe_audio(audio_data)
            
            if not transcription or len(transcription.strip()) == 0:
                logger.info("🔇 No clear speech detected")
                return
            
            logger.info(f"📝 Transcription: {transcription}")
            
            # Step 2: LLM Processing
            if self.app_context.llm_service:
                try:
                    # Simple response for now
                    response = f"I heard you say: '{transcription}'. How can I help you with your gaming?"
                    logger.info(f"🧠 LLM Response: {response}")
                    
                    # Step 3: TTS Output via Discord Bot
                    if self.discord_bot:
                        await self.discord_bot.speak_text(response)
                    
                except Exception as e:
                    logger.error(f"❌ LLM processing error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return None
        
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=1)
            else:
                audio_mono = audio_data
            
            # Normalize audio
            if np.max(np.abs(audio_mono)) > 0:
                audio_mono = audio_mono / np.max(np.abs(audio_mono)) * 0.8
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                import scipy.io.wavfile as wavfile
                audio_int16 = np.clip(audio_mono * 32767, -32767, 32767).astype(np.int16)
                wavfile.write(temp_file.name, 44100, audio_int16)
                temp_file_path = temp_file.name
            
            # Transcribe with Whisper
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    temp_file_path,
                    language='en',
                    temperature=0.0,
                    best_of=1,
                    beam_size=1,
                    condition_on_previous_text=False
                )
            )
            
            # Cleanup
            os.unlink(temp_file_path)
            
            if result and "text" in result:
                text = str(result["text"]).strip()
                return text if len(text) > 2 else None
            
        except Exception as e:
            logger.error(f"❌ Whisper transcription error: {e}")
            
        return None
    
    async def run(self, device_id: Optional[int] = None):
        """Main run method."""
        try:
            logger.info("🚀 Starting DanzarAI Virtual Audio System...")
            
            # Initialize services
            await self.initialize_services()
            await self.initialize_whisper()
            
            # Set up audio capture
            self.audio_capture = VirtualAudioCapture(
                self.app_context,
                callback_func=self.process_audio_segment
            )
            
            # Initialize VAD
            await self.audio_capture.initialize_vad()
            
            # Select audio device
            if not self.audio_capture.select_input_device(device_id):
                logger.error("Failed to select audio device")
                return
            
            # Set up Discord bot
            self.discord_bot = DiscordTTSBot(self.settings, self.app_context)
            
            # Start Discord bot in background
            bot_token = self.settings.get('DISCORD_BOT_TOKEN', '')
            if not bot_token:
                logger.error("DISCORD_BOT_TOKEN missing from configuration")
                return
            
            # Start bot
            bot_task = asyncio.create_task(self.discord_bot.start(bot_token))
            
            # Wait for bot to be ready
            await asyncio.sleep(3)
            
            # Start audio capture
            if not self.audio_capture.start_recording():
                logger.error("Failed to start audio recording")
                return
            
            logger.info("✅ DanzarAI Virtual Audio System is running!")
            logger.info("📝 Instructions:")
            logger.info("   1. Set your game/application audio output to VB-Cable Input")
            logger.info("   2. Use Discord bot command !connect to join voice channel")
            logger.info("   3. Speak into your game/application - DanzarAI will respond via Discord!")
            logger.info("   4. Press Ctrl+C to stop")
            
            # Keep running until interrupted
            try:
                await bot_task
            except KeyboardInterrupt:
                logger.info("🛑 Stopping DanzarAI...")
            
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            # Cleanup
            if self.audio_capture:
                self.audio_capture.stop_recording()
            if self.discord_bot:
                await self.discord_bot.close()

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DanzarVLM - Virtual Audio Cable Version")
    parser.add_argument(
        "--device",
        type=int,
        help="Audio input device ID (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("🎵 Available Audio Input Devices:")
        print("=" * 60)
        
        devices = sd.query_devices()
        virtual_devices = []
        
        for i, device in enumerate(devices):
            try:
                name = str(device.get('name', 'Unknown'))
                max_in = int(device.get('max_input_channels', 0))
                
                if max_in > 0:  # Only show input devices
                    print(f"  {i:2d}: {name}")
                    print(f"      Input channels: {max_in}")
                    
                    # Check for virtual audio keywords
                    name_lower = name.lower()
                    virtual_keywords = ['cable', 'virtual', 'vb-audio', 'voicemeeter', 
                                      'stereo mix', 'wave out mix', 'what u hear']
                    
                    if any(keyword in name_lower for keyword in virtual_keywords):
                        virtual_devices.append((i, name))
                        print(f"      ⭐ VIRTUAL AUDIO DEVICE DETECTED")
                    
                    print()
            except Exception as e:
                print(f"  {i:2d}: Error reading device info: {e}")
        
        if virtual_devices:
            print("🎯 Recommended devices for DanzarVLM:")
            print("=" * 60)
            for device_id, device_name in virtual_devices:
                print(f"  Device {device_id}: {device_name}")
            print("\nTo use with DanzarVLM:")
            print(f"  python DanzarVLM_VirtualAudio.py --device {virtual_devices[0][0]}")
        else:
            print("⚠️  No virtual audio devices found!")
            print("Install VB-Cable or enable Windows Stereo Mix")
        return
    
    # Load settings
    settings = load_global_settings() or {}
    
    # Create and run application
    app = DanzarVirtualAudio(settings)
    await app.run(device_id=args.device)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1) 