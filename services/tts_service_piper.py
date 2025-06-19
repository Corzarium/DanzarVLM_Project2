#!/usr/bin/env python3
"""
Piper TTS Service
High-performance local TTS using the working Piper installation
"""

import asyncio
import io
import logging
import os
import re
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, Union
import subprocess

try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

class PiperTTSService:
    """Piper TTS service for high-quality neural text-to-speech"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings.get('TTS_SERVER', {})
        
        # Configuration
        self.model_name = self.config.get('model', 'en_US-lessac-medium')
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.timeout = self.config.get('timeout', 10)
        
        # State
        self.piper_instance = None
        self.model_path = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_initialized = False
        
        self.logger.info(f"[PiperTTS] Initializing with model: {self.model_name}")
    
    async def initialize(self) -> bool:
        """Initialize the Piper TTS service"""
        try:
            if not PIPER_AVAILABLE:
                self.logger.error("[PiperTTS] Piper package not available")
                return False
            
            # Download model if needed
            if not await self._ensure_model_available():
                self.logger.error("[PiperTTS] Failed to ensure model availability")
                return False
            
            self.is_initialized = True
            self.logger.info("[PiperTTS] Successfully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"[PiperTTS] Initialization failed: {e}")
            return False
    
    async def _ensure_model_available(self) -> bool:
        """Ensure the TTS model is available locally"""
        try:
            # Create models directory
            models_dir = Path("models/piper")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if model exists
            model_file = models_dir / f"{self.model_name}.onnx"
            config_file = models_dir / f"{self.model_name}.onnx.json"
            
            if model_file.exists() and config_file.exists():
                self.model_path = str(model_file)
                self.logger.info(f"[PiperTTS] Model found: {self.model_path}")
                return True
            
            # Download model using piper command
            self.logger.info(f"[PiperTTS] Downloading model: {self.model_name}")
            
            # Use a simple test generation to trigger model download
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                cmd = [
                    sys.executable, "-m", "piper", 
                    "--model", self.model_name,
                    "--download-dir", str(models_dir),
                    "--output-file", temp_file.name
                ]
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._run_download_command,
                    cmd, "Hello"  # Simple test text
                )
            
            # Check if model files were downloaded
            if model_file.exists() and config_file.exists():
                self.model_path = str(model_file)
                self.logger.info(f"[PiperTTS] Model downloaded: {self.model_path}")
                return True
            else:
                self.logger.error(f"[PiperTTS] Failed to download model: {self.model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"[PiperTTS] Error ensuring model availability: {e}")
            return False
    
    def _run_download_command(self, cmd: list, test_text: str) -> bool:
        """Run model download command"""
        try:
            # Run piper with test text to trigger model download
            result = subprocess.run(
                cmd,
                input=test_text,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Return True if command succeeded (model should be downloaded)
            if result.returncode == 0:
                self.logger.info("[PiperTTS] Model download completed successfully")
                return True
            else:
                self.logger.error(f"[PiperTTS] Download failed: {result.stderr}")
                return False
            
        except subprocess.TimeoutExpired:
            self.logger.warning("[PiperTTS] Model download timed out")
            return False
        except Exception as e:
            self.logger.error(f"[PiperTTS] Download command failed: {e}")
            return False
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS by removing problematic characters like emojis"""
        if not text:
            return ""
        
        # Remove emojis and other Unicode symbols that Piper can't handle
        # This regex matches most emoji characters
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", 
            flags=re.UNICODE
        )
        
        # Remove emojis
        cleaned_text = emoji_pattern.sub('', text)
        
        # Replace multiple spaces with single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove any remaining non-ASCII characters that might cause issues
        cleaned_text = ''.join(char for char in cleaned_text if ord(char) < 128 or char.isspace())
        
        return cleaned_text.strip()

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio from text synchronously"""
        if not self.is_initialized:
            self.logger.warning("[PiperTTS] Service not initialized")
            return None
        
        if not text or not text.strip():
            self.logger.warning("[PiperTTS] Empty text provided")
            return None
        
        try:
            start_time = time.time()
            
            # Clean text to remove emojis and problematic characters
            cleaned_text = self._clean_text_for_tts(text.strip())
            
            if not cleaned_text:
                self.logger.warning("[PiperTTS] Text became empty after cleaning")
                return None
            
            # Use piper CLI for generation (most reliable method)
            audio_data = self._generate_with_cli(cleaned_text)
            
            generation_time = time.time() - start_time
            
            if audio_data:
                self.logger.info(
                    f"[PiperTTS] Generated {len(audio_data)} bytes in {generation_time:.2f}s"
                )
                return audio_data
            else:
                self.logger.error("[PiperTTS] Failed to generate audio")
                return None
                
        except Exception as e:
            self.logger.error(f"[PiperTTS] Audio generation failed: {e}")
            return None
    
    def _generate_with_cli(self, text: str) -> Optional[bytes]:
        """Generate audio using Piper CLI"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Use piper CLI to generate audio with full model path
                cmd = [
                    sys.executable, "-m", "piper",
                    "--model", self.model_path,
                    "--output-file", temp_path
                ]
                
                # Run piper with text input
                result = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0 and os.path.exists(temp_path):
                    # Read generated audio
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    if len(audio_data) > 1000:  # Sanity check
                        return audio_data
                    else:
                        self.logger.warning(f"[PiperTTS] Generated audio too small: {len(audio_data)} bytes")
                        return None
                else:
                    self.logger.error(f"[PiperTTS] CLI generation failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass
                    
        except subprocess.TimeoutExpired:
            self.logger.error(f"[PiperTTS] Generation timed out after {self.timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"[PiperTTS] CLI generation error: {e}")
            return None
    
    async def generate_audio_async(self, text: str) -> Optional[bytes]:
        """Generate audio asynchronously"""
        if not text or not text.strip():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.generate_audio,
                text.strip()
            )
        except Exception as e:
            self.logger.error(f"[PiperTTS] Async generation failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'provider': 'piper',
            'model': self.model_name,
            'initialized': self.is_initialized,
            'model_path': self.model_path,
            'sample_rate': self.sample_rate,
            'available': PIPER_AVAILABLE
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=False)
            self.logger.info("[PiperTTS] Cleanup completed")
        except Exception as e:
            self.logger.error(f"[PiperTTS] Cleanup error: {e}") 