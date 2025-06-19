# offline_vad_voice_receiver.py
"""
100% Offline DanzarAI Voice Receiver
Uses discord-ext-voice-recv + Vosk STT + Local LLM + Silero TTS
"""

import asyncio
import logging
import threading
import time
import json
import tempfile
import os
from typing import Optional, Callable
import numpy as np

# Discord voice receiving
import discord
try:
    from discord.ext import voice_recv
    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False
    voice_recv = None

# Audio processing
import audioop
import webrtcvad

# Offline STT
import vosk

# Offline LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Offline TTS
import silero

class OfflineVADVoiceSink:
    """
    100% Offline Voice Sink using:
    - discord-ext-voice-recv for raw Opus frames
    - WebRTC VAD for utterance detection
    - Vosk for offline speech-to-text
    - Local Transformers LLM for response generation
    - Silero TTS for offline text-to-speech
    """
    
    def __init__(self, vosk_model_path: str, llm_model_name: str = "microsoft/DialoGPT-medium", 
                 speech_callback: Optional[Callable] = None, logger: Optional[logging.Logger] = None):
        if not VOICE_RECV_AVAILABLE:
            raise ImportError("discord-ext-voice-recv is required for offline voice processing")
        
        # Initialize as AudioSink if available
        # Note: OfflineVADVoiceSink doesn't inherit from AudioSink directly
        
        self.logger = logger or logging.getLogger(__name__)
        self.speech_callback = speech_callback
        
        # Initialize offline components
        self._init_vad()
        self._init_vosk(vosk_model_path)
        self._init_llm(llm_model_name)
        self._init_tts()
        
        # Audio processing state
        self._resample_state = None
        self._user_buffers = {}  # user_id -> bytearray
        self._user_names = {}    # user_id -> display_name
        self._lock = threading.Lock()
        
        self.logger.info("✅ Offline VAD Voice Sink initialized")

    def wants_opus(self) -> bool:
        """Return False to receive PCM data instead of Opus."""
        return False

    def write(self, user, data):
        """
        Process incoming audio frames.
        Called by discord-ext-voice-recv for each ~20ms frame.
        """
        try:
            # Store user info
            self._user_names[user.id] = user.display_name
            
            # Process PCM audio data
            self._process_pcm_frame(user.id, data.pcm)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing audio frame: {e}")

    def cleanup(self):
        """Cleanup when sink is finished."""
        try:
            self._user_buffers.clear()
            self._user_names.clear()
            self.logger.info("✅ Offline VAD Voice Sink cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up sink: {e}")

    def _init_vad(self):
        """Initialize WebRTC VAD."""
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.logger.info("✅ WebRTC VAD initialized (offline)")

    def _init_vosk(self, model_path: str):
        """Initialize Vosk offline STT."""
        self.vosk_model = vosk.Model(model_path)
        self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
        self.logger.info(f"✅ Vosk offline STT initialized: {model_path}")

    def _init_llm(self, model_name: str):
        """Initialize local LLM."""
        try:
            self.logger.info(f"🔧 Loading local LLM: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info(f"✅ Local LLM loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load LLM: {e}")
            self.llm_model = None
            self.tokenizer = None

    def _init_tts(self):
        """Initialize Silero TTS."""
        try:
            self.logger.info("🔧 Loading Silero TTS...")
            # Load Silero TTS model
            self.tts_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='v3_en'
            )
            self.logger.info("✅ Silero TTS loaded (offline)")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Silero TTS: {e}")
            self.tts_model = None

    def _process_pcm_frame(self, user_id: int, pcm_data: bytes):
        """Process PCM audio data through VAD and STT pipeline."""
        try:
            # 1) Convert PCM to 48kHz stereo
            pcm_48k_stereo = pcm_data
            
            # 2) Convert 48kHz stereo → 16kHz mono
            mono = audioop.tomono(pcm_48k_stereo, 2, 1, 1)
            pcm_16k, self._resample_state = audioop.ratecv(
                mono, 2, 1, 48000, 16000, self._resample_state
            )
            
            # 3) Process with VAD
            self._process_vad_frame(user_id, pcm_16k)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing PCM frame: {e}")

    def _process_vad_frame(self, user_id: int, pcm_16k: bytes):
        """Process 16kHz mono PCM through VAD."""
        with self._lock:
            buffer = self._user_buffers.get(user_id, bytearray())
            
            # VAD requires 30ms frames (480 samples * 2 bytes = 960 bytes at 16kHz)
            frame_size = 960
            
            for i in range(0, len(pcm_16k), frame_size):
                frame = pcm_16k[i:i+frame_size]
                if len(frame) < frame_size:
                    continue
                
                try:
                    is_speech = self.vad.is_speech(frame, 16000)
                    
                    if is_speech:
                        buffer.extend(frame)
                    else:
                        # Silence detected - finalize utterance if we have speech
                        if buffer:
                            self._finalize_utterance(user_id, bytes(buffer))
                            buffer = bytearray()
                            
                except Exception as e:
                    self.logger.error(f"❌ VAD processing error: {e}")
            
            self._user_buffers[user_id] = buffer

    def _finalize_utterance(self, user_id: int, speech_data: bytes):
        """Process complete utterance through offline STT → LLM → TTS."""
        try:
            user_name = self._user_names.get(user_id, f"User{user_id}")
            
            # 1) Offline STT with Vosk
            transcription = self._transcribe_with_vosk(speech_data)
            if not transcription:
                return
                
            self.logger.info(f"🎤 Offline transcription: '{transcription}' from {user_name}")
            
            # 2) Generate response with local LLM
            response = self._generate_llm_response(transcription)
            if not response:
                return
                
            self.logger.info(f"🧠 Local LLM response: '{response}'")
            
            # 3) Generate speech with Silero TTS
            audio_file = self._generate_tts_audio(response)
            if not audio_file:
                return
                
            # 4) Call speech callback with full pipeline result
            if self.speech_callback:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.speech_callback(
                        transcription, response, audio_file, user_name
                    ))
                except RuntimeError:
                    asyncio.run(self.speech_callback(
                        transcription, response, audio_file, user_name
                    ))
                    
        except Exception as e:
            self.logger.error(f"❌ Error finalizing utterance: {e}")

    def _transcribe_with_vosk(self, speech_data: bytes) -> Optional[str]:
        """Transcribe speech using offline Vosk."""
        try:
            if self.vosk_recognizer.AcceptWaveform(speech_data):
                result = json.loads(self.vosk_recognizer.Result())
                return result.get("text", "").strip()
            else:
                result = json.loads(self.vosk_recognizer.FinalResult())
                return result.get("text", "").strip()
        except Exception as e:
            self.logger.error(f"❌ Vosk transcription error: {e}")
            return None

    def _generate_llm_response(self, text: str) -> Optional[str]:
        """Generate response using local LLM."""
        if not self.llm_model or not self.tokenizer:
            return f"I heard you say: '{text}'. I'm DanzarAI running 100% offline!"
            
        try:
            # Prepare input for conversational model
            prompt = f"User: {text}\nDanzarAI:"
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "DanzarAI:" in response:
                response = response.split("DanzarAI:")[-1].strip()
            
            return response[:200]  # Limit response length
            
        except Exception as e:
            self.logger.error(f"❌ LLM generation error: {e}")
            return f"I heard you say: '{text}'. I'm DanzarAI running 100% offline!"

    def _generate_tts_audio(self, text: str) -> Optional[str]:
        """Generate speech using offline Silero TTS."""
        if not self.tts_model:
            return None
            
        try:
            # Generate audio with Silero
            audio = self.tts_model.apply_tts(
                text=text,
                speaker='en_0',  # English speaker
                sample_rate=48000
            )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # Convert tensor to numpy and save as WAV
            import scipy.io.wavfile as wavfile
            audio_np = audio.numpy()
            wavfile.write(temp_file.name, 48000, audio_np)
            
            temp_file.close()
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"❌ TTS generation error: {e}")
            return None


class OfflineVADVoiceReceiver:
    """
    100% Offline VAD Voice Receiver using discord-ext-voice-recv.
    """
    
    def __init__(self, app_context, speech_callback: Optional[Callable] = None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.speech_callback = speech_callback
        
        # Configuration
        self.vosk_model_path = "models/vosk-model-small-en-us-0.15"
        self.llm_model_name = "microsoft/DialoGPT-medium"  # Lightweight conversational model
        
        # Audio sink
        self.sink = None
        
        self.logger.info("✅ Offline VAD Voice Receiver initialized")

    async def initialize(self) -> bool:
        """Initialize the offline voice receiver."""
        try:
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize offline voice receiver: {e}")
            return False

    async def start_listening(self, voice_client) -> bool:
        """Start listening with 100% offline processing."""
        try:
            if not VOICE_RECV_AVAILABLE:
                self.logger.error("❌ discord-ext-voice-recv not available")
                return False
                
            self.logger.info("🎯 Starting 100% offline voice processing...")
            
            # Create offline VAD sink
            self.sink = OfflineVADVoiceSink(
                vosk_model_path=self.vosk_model_path,
                llm_model_name=self.llm_model_name,
                speech_callback=self.speech_callback,
                logger=self.logger
            )
            
            # Start listening with discord-ext-voice-recv
            # Check if voice_client has listen method (VoiceRecvClient)
            if hasattr(voice_client, 'listen'):
                voice_client.listen(self.sink)
                self.logger.info("✅ Started 100% offline voice processing (Vosk + Local LLM + Silero TTS)")
                return True
            else:
                self.logger.error("❌ Voice client does not support listening (VoiceRecvClient required)")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start offline listening: {e}")
            return False

    async def stop_listening(self, voice_client) -> bool:
        """Stop listening and cleanup."""
        try:
            if voice_client and voice_client.is_listening():
                voice_client.stop_listening()
            
            self.sink = None
            self.logger.info("✅ Offline voice receiver stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping offline voice receiver: {e}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.sink = None
            self.logger.info("✅ Offline voice receiver cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up offline voice receiver: {e}") 