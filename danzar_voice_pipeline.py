#!/usr/bin/env python3
"""
Danzar Voice Pipeline - Real-time Multimodal Voice Analysis for Discord

This module provides a complete real-time voice analysis pipeline that:
1. Captures live audio from Discord voice channels
2. Performs parallel analysis: STT, emotion recognition, laughter detection
3. Correlates results by timestamp and sends to VLM
4. Synthesizes and plays back responses via TTS

Author: DanzarAI Team
License: MIT
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import discord
import numpy as np
import aiohttp
from discord.ext import commands
import whisper
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float


@dataclass
class AnalysisResult:
    """Container for analysis results from different models"""
    timestamp: float
    text: Optional[str] = None
    emotion: Optional[Tuple[str, float]] = None  # (emotion_label, confidence)
    laughter_detected: bool = False
    laughter_confidence: float = 0.0


class LaughterDetector:
    """Laughter detection using jrgillick's model"""
    
    def __init__(self, model_path: str = "jrgillick/laughter-detection"):
        """Initialize laughter detection model"""
        try:
            # Load the laughter detection model
            self.model = torch.hub.load('jrgillick/laughter-detection', 'laughter_detection')
            self.model.eval()
            logger.info("✅ Laughter detection model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load laughter detection model: {e}")
            self.model = None
    
    async def detect_laughter(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """Detect laughter in audio chunk"""
        if not self.model:
            return False, 0.0
        
        try:
            # Convert audio to tensor
            audio_tensor = torch.from_numpy(audio_chunk.data).float()
            
            # Ensure correct shape and sample rate
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                prediction = self.model(audio_tensor)
                confidence = torch.sigmoid(prediction).item()
                is_laughter = confidence > 0.5
            
            return is_laughter, confidence
            
        except Exception as e:
            logger.warning(f"Laughter detection failed: {e}")
            return False, 0.0


class EmotionRecognizer:
    """Emotion recognition using SpeechBrain"""
    
    def __init__(self, model_path: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        """Initialize emotion recognition model"""
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_path,
                savedir="models/emotion-recognition"
            )
            logger.info("✅ Emotion recognition model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load emotion recognition model: {e}")
            self.classifier = None
    
    async def recognize_emotion(self, audio_chunk: AudioChunk) -> Optional[Tuple[str, float]]:
        """Recognize emotion in audio chunk"""
        if not self.classifier:
            return None
        
        try:
            # Save audio chunk temporarily
            temp_path = f"temp_emotion_{int(time.time() * 1000)}.wav"
            torchaudio.save(temp_path, torch.from_numpy(audio_chunk.data), audio_chunk.sample_rate)
            
            # Run inference
            with torch.no_grad():
                out_prob, score, index, text_lab = self.classifier.classify_file(temp_path)
                emotion_label = text_lab[0]
                confidence = out_prob[0].max().item()
            
            # Clean up temp file
            import os
            os.remove(temp_path)
            
            return emotion_label, confidence
            
        except Exception as e:
            logger.warning(f"Emotion recognition failed: {e}")
            return None


class WhisperTranscriber:
    """Speech-to-text using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper model"""
        try:
            self.model = whisper.load_model(model_size)
            logger.info(f"✅ Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            self.model = None
    
    async def transcribe(self, audio_chunk: AudioChunk) -> Optional[str]:
        """Transcribe audio chunk to text"""
        if not self.model:
            return None
        
        try:
            # Run transcription
            result = self.model.transcribe(audio_chunk.data)
            text = result["text"].strip()
            
            return text if text else None
            
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return None


class VLMClient:
    """Client for local Qwen2.5-VL API"""
    
    def __init__(self, api_endpoint: str = "http://localhost:8083/chat/completions"):
        """Initialize VLM client"""
        self.api_endpoint = api_endpoint
        self.session = None
        logger.info(f"🎯 VLM client initialized for endpoint: {api_endpoint}")
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from VLM"""
        await self._ensure_session()
        
        try:
            payload = {
                "model": "qwen2.5-vl-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Danzar, an upbeat and witty gaming assistant. Analyze the user's voice input including their emotions and reactions, then respond appropriately and engagingly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            async with self.session.post(self.api_endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"VLM API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"VLM request failed: {e}")
            return None
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class TTSClient:
    """Client for Chatterbox TTS"""
    
    def __init__(self, tts_endpoint: str = "http://localhost:8055/tts"):
        """Initialize TTS client"""
        self.tts_endpoint = tts_endpoint
        self.session = None
        logger.info(f"🎵 TTS client initialized for endpoint: {tts_endpoint}")
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Synthesize speech from text"""
        await self._ensure_session()
        
        try:
            payload = {
                "text": text,
                "voice": "default",
                "speed": 1.0
            }
            
            async with self.session.post(self.tts_endpoint, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"TTS API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"TTS request failed: {e}")
            return None
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class DanzarVoicePipeline(commands.Bot):
    """Main voice analysis pipeline for Discord"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the voice pipeline"""
        super().__init__(
            command_prefix=config.get('command_prefix', '!'),
            intents=discord.Intents.default()
        )
        
        # Configuration
        self.config = config
        self.voice_channel_id = config.get('voice_channel_id')
        self.text_channel_id = config.get('text_channel_id')
        
        # Analysis components
        self.whisper = WhisperTranscriber(config.get('whisper_model', 'base'))
        self.emotion_recognizer = EmotionRecognizer()
        self.laughter_detector = LaughterDetector()
        
        # API clients
        self.vlm_client = VLMClient(config.get('vlm_endpoint'))
        self.tts_client = TTSClient(config.get('tts_endpoint'))
        
        # Audio processing
        self.audio_queue = asyncio.Queue()
        self.analysis_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
        # State management
        self.is_processing = False
        self.voice_client = None
        self.audio_buffer = deque(maxlen=100)  # Keep last 100 chunks
        
        # Analysis results buffer
        self.analysis_buffer = deque(maxlen=50)
        
        logger.info("🚀 Danzar Voice Pipeline initialized")
    
    async def setup_hook(self):
        """Setup bot hooks"""
        logger.info("🔧 Setting up bot hooks...")
    
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"✅ {self.user} is online and ready!")
        
        # Auto-join voice channel if specified
        if self.voice_channel_id:
            await self.auto_join_voice_channel()
        
        # Start processing loops
        asyncio.create_task(self.audio_processing_loop())
        asyncio.create_task(self.analysis_processing_loop())
        asyncio.create_task(self.response_processing_loop())
    
    async def auto_join_voice_channel(self):
        """Automatically join the specified voice channel"""
        try:
            channel = self.get_channel(self.voice_channel_id)
            if channel and isinstance(channel, discord.VoiceChannel):
                self.voice_client = await channel.connect()
                logger.info(f"🎤 Connected to voice channel: {channel.name}")
                
                # Start audio capture
                self.voice_client.listen(discord.sinks.WaveSink())
                asyncio.create_task(self.audio_capture_loop())
                
            else:
                logger.error(f"❌ Voice channel {self.voice_channel_id} not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to join voice channel: {e}")
    
    async def audio_capture_loop(self):
        """Capture audio from Discord voice client"""
        if not self.voice_client:
            return
        
        try:
            while self.voice_client.is_connected():
                # Get audio data from voice client
                if hasattr(self.voice_client, 'sink') and self.voice_client.sink:
                    audio_data = self.voice_client.sink.get_audio_data()
                    if audio_data:
                        # Convert to numpy array
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Create audio chunk
                        chunk = AudioChunk(
                            data=audio_np,
                            timestamp=time.time(),
                            sample_rate=48000,  # Discord default
                            duration=len(audio_np) / 48000
                        )
                        
                        # Add to queue
                        await self.audio_queue.put(chunk)
                        self.audio_buffer.append(chunk)
                
                await asyncio.sleep(0.1)  # 100ms intervals
                
        except Exception as e:
            logger.error(f"❌ Audio capture error: {e}")
    
    async def audio_processing_loop(self):
        """Process audio chunks through all analysis pipelines"""
        logger.info("🎵 Starting audio processing loop...")
        
        while not self.is_closed():
            try:
                # Get audio chunk
                chunk = await self.audio_queue.get()
                
                # Run all analyses in parallel
                tasks = [
                    self.whisper.transcribe(chunk),
                    self.emotion_recognizer.recognize_emotion(chunk),
                    self.laughter_detector.detect_laughter(chunk)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Create analysis result
                analysis = AnalysisResult(timestamp=chunk.timestamp)
                
                # Process results
                if isinstance(results[0], str) and results[0]:
                    analysis.text = results[0]
                
                if isinstance(results[1], tuple):
                    analysis.emotion = results[1]
                
                if isinstance(results[2], tuple):
                    analysis.laughter_detected, analysis.laughter_confidence = results[2]
                
                # Add to analysis queue
                await self.analysis_queue.put(analysis)
                self.analysis_buffer.append(analysis)
                
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def analysis_processing_loop(self):
        """Process analysis results and generate VLM prompts"""
        logger.info("🧠 Starting analysis processing loop...")
        
        while not self.is_closed():
            try:
                # Get analysis result
                analysis = await self.analysis_queue.get()
                
                # Build multimodal prompt
                prompt = self._build_multimodal_prompt(analysis)
                
                if prompt:
                    # Send to VLM
                    response = await self.vlm_client.generate_response(prompt)
                    
                    if response:
                        # Add to response queue
                        await self.response_queue.put(response)
                
            except Exception as e:
                logger.error(f"❌ Analysis processing error: {e}")
                await asyncio.sleep(0.1)
    
    def _build_multimodal_prompt(self, analysis: AnalysisResult) -> Optional[str]:
        """Build multimodal prompt from analysis results"""
        try:
            prompt_parts = []
            
            # Add timestamp
            timestamp_str = f"[{analysis.timestamp:.1f}s]"
            prompt_parts.append(timestamp_str)
            
            # Add text if available
            if analysis.text:
                prompt_parts.append(f'TEXT: "{analysis.text}"')
            
            # Add emotion if available
            if analysis.emotion:
                emotion_label, confidence = analysis.emotion
                prompt_parts.append(f'[EMOTION: {emotion_label} ({confidence:.2f})]')
            
            # Add laughter if detected
            if analysis.laughter_detected:
                prompt_parts.append(f'[LAUGHTER detected at {analysis.timestamp:.1f}s (confidence: {analysis.laughter_confidence:.2f})]')
            
            # Add context from recent analyses
            recent_context = self._get_recent_context()
            if recent_context:
                prompt_parts.append(f'[CONTEXT: {recent_context}]')
            
            return " ".join(prompt_parts) if prompt_parts else None
            
        except Exception as e:
            logger.error(f"❌ Prompt building error: {e}")
            return None
    
    def _get_recent_context(self) -> str:
        """Get context from recent analyses"""
        try:
            if len(self.analysis_buffer) < 2:
                return ""
            
            # Get last few analyses
            recent = list(self.analysis_buffer)[-3:]
            
            context_parts = []
            for analysis in recent:
                if analysis.text:
                    context_parts.append(analysis.text)
                if analysis.emotion:
                    context_parts.append(f"({analysis.emotion[0]})")
            
            return " | ".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"❌ Context building error: {e}")
            return ""
    
    async def response_processing_loop(self):
        """Process VLM responses and synthesize TTS"""
        logger.info("🎵 Starting response processing loop...")
        
        while not self.is_closed():
            try:
                # Get response
                response_text = await self.response_queue.get()
                
                # Synthesize speech
                audio_data = await self.tts_client.synthesize_speech(response_text)
                
                if audio_data and self.voice_client and self.voice_client.is_connected():
                    # Play audio
                    await self._play_audio(audio_data)
                    
                    # Send text to channel if available
                    if self.text_channel_id:
                        channel = self.get_channel(self.text_channel_id)
                        if channel:
                            await channel.send(f"🤖 **Danzar**: {response_text}")
                
            except Exception as e:
                logger.error(f"❌ Response processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio through voice client"""
        try:
            if not self.voice_client or not self.voice_client.is_connected():
                return
            
            # Create audio source from bytes
            import io
            audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_data))
            
            # Play audio
            self.voice_client.play(audio_source)
            
            logger.info("🎵 Playing TTS response")
            
        except Exception as e:
            logger.error(f"❌ Audio playback error: {e}")
    
    async def close(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Danzar Voice Pipeline...")
        
        # Close API clients
        await self.vlm_client.close()
        await self.tts_client.close()
        
        # Disconnect from voice
        if self.voice_client:
            await self.voice_client.disconnect()
        
        await super().close()


# Command handlers
@commands.command(name='join')
async def join_command(ctx, channel: discord.VoiceChannel = None):
    """Join a voice channel"""
    if not channel:
        channel = ctx.author.voice.channel if ctx.author.voice else None
    
    if channel:
        await channel.connect()
        await ctx.send(f"✅ Joined {channel.name}")
    else:
        await ctx.send("❌ Please specify a voice channel or join one first")


@commands.command(name='leave')
async def leave_command(ctx):
    """Leave the voice channel"""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Left the voice channel")
    else:
        await ctx.send("❌ Not connected to a voice channel")


@commands.command(name='status')
async def status_command(ctx):
    """Show pipeline status"""
    embed = discord.Embed(title="🎤 Danzar Voice Pipeline Status", color=0x00ff00)
    
    # Connection status
    if ctx.voice_client and ctx.voice_client.is_connected():
        embed.add_field(name="Voice Connection", value="✅ Connected", inline=True)
        embed.add_field(name="Channel", value=ctx.voice_client.channel.name, inline=True)
    else:
        embed.add_field(name="Voice Connection", value="❌ Disconnected", inline=True)
    
    # Queue status
    embed.add_field(name="Audio Queue", value=f"{ctx.bot.audio_queue.qsize()} chunks", inline=True)
    embed.add_field(name="Analysis Queue", value=f"{ctx.bot.analysis_queue.qsize()} results", inline=True)
    embed.add_field(name="Response Queue", value=f"{ctx.bot.response_queue.qsize()} responses", inline=True)
    
    await ctx.send(embed=embed)


def main():
    """Main entry point"""
    # Configuration
    config = {
        'command_prefix': '!',
        'voice_channel_id': None,  # Set to your voice channel ID
        'text_channel_id': None,   # Set to your text channel ID
        'whisper_model': 'base',
        'vlm_endpoint': 'http://localhost:8083/chat/completions',
        'tts_endpoint': 'http://localhost:8055/tts'
    }
    
    # Load environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        logger.error("❌ DISCORD_BOT_TOKEN environment variable not set!")
        return
    
    # Override config with environment variables
    if os.getenv('DISCORD_VOICE_CHANNEL_ID'):
        config['voice_channel_id'] = int(os.getenv('DISCORD_VOICE_CHANNEL_ID'))
    if os.getenv('DISCORD_TEXT_CHANNEL_ID'):
        config['text_channel_id'] = int(os.getenv('DISCORD_TEXT_CHANNEL_ID'))
    
    # Create and run bot
    bot = DanzarVoicePipeline(config)
    
    # Add commands
    bot.add_command(join_command)
    bot.add_command(leave_command)
    bot.add_command(status_command)
    
    try:
        bot.run(bot_token)
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")


if __name__ == "__main__":
    main() 