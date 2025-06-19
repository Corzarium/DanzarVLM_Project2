"""
Text-to-Speech (TTS) service using Chatterbox.
"""
import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using Chatterbox."""
    
    def __init__(self, settings: dict):
        """Initialize TTS service with settings."""
        self.settings = settings
        self.engine = settings["ENGINE"]
        self.voice = settings["VOICE"]
        self.sample_rate = settings["SAMPLE_RATE"]
        self.speaker_id = settings["SPEAKER_ID"]
        
        # Create temp directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_bot_tts"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def generate_speech(self, text: str) -> Optional[Path]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Optional[Path]: Path to generated audio file or None if generation fails
        """
        try:
            # Create temporary file for audio
            temp_file = self.temp_dir / f"tts_{hash(text)}_{self.speaker_id}.wav"
            
            if self.engine == "chatterbox":
                await self._generate_chatterbox_speech(text, temp_file)
            else:
                raise ValueError(f"Unsupported TTS engine: {self.engine}")
            
            if temp_file.exists():
                logger.debug(f"Generated speech file: {temp_file}")
                return temp_file
            return None
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None
    
    async def _generate_chatterbox_speech(self, text: str, output_file: Path):
        """
        Generate speech using Chatterbox TTS.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
        """
        try:
            # Prepare request data
            request_data = {
                "text": text,
                "voice": self.voice,
                "sample_rate": self.sample_rate,
                "speaker_id": self.speaker_id
            }
            
            # Make request to Chatterbox
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://chatterbox:8055/tts",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Chatterbox API error: {error_text}")
                    
                    # Save audio data to file
                    audio_data = await response.read()
                    output_file.write_bytes(audio_data)
            
        except Exception as e:
            logger.error(f"Error generating Chatterbox speech: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            for file in self.temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices.
        
        Returns:
            Dict[str, Any]: Dictionary of available voices
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://chatterbox:8055/voices") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Chatterbox API error: {error_text}")
                    
                    return await response.json()
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return {} 