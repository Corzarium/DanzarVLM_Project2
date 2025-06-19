import os
import logging
import pyttsx3
import threading
import io
import wave
from typing import Optional

class FastTTSService:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.FastTTSService")
        self.config = self.ctx.global_settings.get('TTS_SERVER', {})
        
        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()
        
        # Configure voice settings
        self.speech_rate = self.config.get('speech_rate', 200)  # Words per minute
        self.speech_volume = self.config.get('speech_volume', 0.9)  # 0.0 to 1.0
        
        # Apply settings
        self.engine.setProperty('rate', self.speech_rate)
        self.engine.setProperty('volume', self.speech_volume)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            female_voice = None
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'eva' in voice.name.lower():
                    female_voice = voice.id
                    break
            
            if female_voice:
                self.engine.setProperty('voice', female_voice)
                self.logger.info(f"[FastTTS] Using voice: {female_voice}")
            else:
                self.logger.info(f"[FastTTS] Using default voice: {voices[0].name}")
        
        self.logger.info(f"[FastTTS] Initialized with rate={self.speech_rate}, volume={self.speech_volume}")

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio using Windows SAPI - extremely fast"""
        try:
            self.logger.debug(f"[FastTTS] Generating audio for {len(text)} characters")
            
            # Save to temporary WAV file
            temp_file = "temp_tts.wav"
            
            # Use a lock to ensure thread safety
            with threading.Lock():
                self.engine.save_to_file(text, temp_file)
                self.engine.runAndWait()
            
            # Read the generated file
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                os.remove(temp_file)
                
                self.logger.debug(f"[FastTTS] Generated {len(audio_data)} bytes of audio")
                return audio_data
            else:
                self.logger.error("[FastTTS] No audio file was generated")
                return None
                
        except Exception as e:
            self.logger.error(f"[FastTTS] Audio generation failed: {e}")
            return None

    def generate_audio_stream(self, text: str) -> Optional[bytes]:
        """Alternative method for streaming audio generation"""
        try:
            # For immediate playback without file I/O
            self.engine.say(text)
            self.engine.runAndWait()
            return b"immediate_playback"  # Signal for immediate playback
        except Exception as e:
            self.logger.error(f"[FastTTS] Stream generation failed: {e}")
            return None 