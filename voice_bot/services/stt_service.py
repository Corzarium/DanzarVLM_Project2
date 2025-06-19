"""
Speech-to-Text (STT) service using Whisper.
"""
import asyncio
import logging
import numpy as np
import whisper
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using Whisper."""
    
    def __init__(self, settings: dict):
        """Initialize STT service with settings."""
        self.settings = settings
        self.model_path = Path(settings["MODEL_PATH"])
        self.language = settings["LANGUAGE"]
        self.compute_type = settings["COMPUTE_TYPE"]
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() and self.compute_type == "float16" else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(
                str(self.model_path),
                device=self.device
            )
            logger.info(f"Loaded Whisper model from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize audio
            audio_float32 = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_float32,
                language=self.language,
                fp16=(self.compute_type == "float16" and self.device == "cuda")
            )
            
            # Extract text from result
            if isinstance(result, dict) and "text" in result:
                text = result["text"].strip()
                if text:
                    logger.debug(f"Transcribed text: {text}")
                    return text
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    async def transcribe_stream(self, audio_stream: List[bytes]) -> Optional[str]:
        """
        Transcribe a stream of audio data.
        
        Args:
            audio_stream: List of audio data chunks in bytes
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        try:
            # Combine all audio chunks
            combined_audio = b"".join(audio_stream)
            
            # Transcribe combined audio
            return await self.transcribe_audio(combined_audio)
            
        except Exception as e:
            logger.error(f"Error transcribing audio stream: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "language": self.language,
            "compute_type": self.compute_type,
            "device": self.device,
            "model_size": self.model.dims
        } 