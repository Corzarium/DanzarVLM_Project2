import os
import logging
import requests
from typing import Optional

class OpenAITTSService:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.OpenAITTSService")
        self.config = self.ctx.global_settings.get('TTS_SERVER', {})
        
        # OpenAI API configuration
        self.api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.endpoint = "https://api.openai.com/v1/audio/speech"
        self.timeout = self.config.get('timeout', 10)  # Much faster timeout
        
        # Voice settings
        self.voice = self.config.get('voice', 'alloy')  # alloy, echo, fable, onyx, nova, shimmer
        self.model = self.config.get('model', 'tts-1')  # tts-1 or tts-1-hd
        self.speed = self.config.get('speed', 1.0)  # 0.25 to 4.0
        
        if not self.api_key:
            self.logger.error("[OpenAI TTS] No API key found. Set OPENAI_API_KEY environment variable.")
        
        self.logger.info(f"[OpenAI TTS] Initialized with voice={self.voice}, model={self.model}, speed={self.speed}")

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio using OpenAI TTS - very fast and high quality"""
        if not self.api_key:
            self.logger.error("[OpenAI TTS] No API key available")
            return None
            
        try:
            self.logger.debug(f"[OpenAI TTS] Generating audio for {len(text)} characters")
            
            payload = {
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "speed": self.speed,
                "response_format": "wav"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            self.logger.debug(f"[OpenAI TTS] Generated {len(response.content)} bytes of audio")
            return response.content
            
        except requests.exceptions.Timeout:
            self.logger.error(f"[OpenAI TTS] Request timed out after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[OpenAI TTS] API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[OpenAI TTS] Audio generation failed: {e}")
            return None 