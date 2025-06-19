#!/usr/bin/env python3
"""
High-Performance Piper TTS Service
Replaces Chatterbox TTS with GPU-accelerated local inference
"""

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import json

try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("Warning: piper-tts not installed. Install with: pip install piper-tts")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not installed. Install with: pip install soundfile")


class PiperTTSService:
    """High-performance Piper TTS service with GPU acceleration"""
    
    def __init__(self, app_context=None):
        self.app_context = app_context
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_name = "en_US-lessac-medium"
        self.model_path = None
        self.device = "cuda:1"  # 4070 Super
        self.sample_rate = 22050
        self.channels = 1
        
        # Performance settings
        self.max_workers = 2
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.generation_timeout = 10.0  # Much faster than Chatterbox
        
        # Model cache
        self.tts_engine = None
        self.model_loaded = False
        
        # Statistics
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "avg_generation_time": 0.0,
            "total_generation_time": 0.0
        }
        
        self.logger.info("[PiperTTS] Initializing high-performance TTS service")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Piper TTS model with GPU acceleration"""
        try:
            # Create models directory
            models_dir = Path("models/piper")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model if needed
            self.model_path = self._ensure_model_available(models_dir)
            
            if PIPER_AVAILABLE and self.model_path:
                self._load_piper_engine()
            else:
                self.logger.warning("[PiperTTS] Using CLI fallback mode")
                
        except Exception as e:
            self.logger.error(f"[PiperTTS] Model initialization failed: {e}")
            self.model_loaded = False

    def _ensure_model_available(self, models_dir: Path) -> Optional[Path]:
        """Download Piper model if not available"""
        model_file = models_dir / f"{self.model_name}.onnx"
        config_file = models_dir / f"{self.model_name}.onnx.json"
        
        if model_file.exists() and config_file.exists():
            self.logger.info(f"[PiperTTS] Model already available: {model_file}")
            return model_file
        
        try:
            self.logger.info(f"[PiperTTS] Downloading model: {self.model_name}")
            
            # Download model and config
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
            
            for file_name in [f"{self.model_name}.onnx", f"{self.model_name}.onnx.json"]:
                url = f"{base_url}/{file_name}"
                file_path = models_dir / file_name
                
                self.logger.info(f"[PiperTTS] Downloading {file_name}...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"[PiperTTS] Downloaded {file_name} ({file_path.stat().st_size} bytes)")
            
            return model_file
            
        except Exception as e:
            self.logger.error(f"[PiperTTS] Model download failed: {e}")
            return None

    def _load_piper_engine(self):
        """Load Piper TTS engine with GPU acceleration"""
        try:
            if not self.model_path or not self.model_path.exists():
                raise Exception("Model file not found")
            
            # Load Piper TTS engine
            from piper import PiperVoice
            
            self.tts_engine = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.model_path).replace('.onnx', '.onnx.json'),
                use_cuda=True
            )
            
            self.model_loaded = True
            self.logger.info(f"[PiperTTS] Engine loaded successfully with GPU acceleration")
            
        except Exception as e:
            self.logger.error(f"[PiperTTS] Engine loading failed: {e}")
            self.model_loaded = False

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate TTS audio (synchronous interface for compatibility)"""
        try:
            # Run async method in thread pool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.generate_audio_async(text))
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"[PiperTTS] Sync generation failed: {e}")
            return None

    async def generate_audio_async(self, text: str) -> Optional[bytes]:
        """Generate TTS audio asynchronously"""
        if not text or not text.strip():
            return None
        
        start_time = time.time()
        self.stats["requests"] += 1
        
        try:
            # Clean text for TTS
            clean_text = self._clean_text_for_tts(text)
            
            # Generate audio using thread pool
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_audio_sync,
                clean_text
            )
            
            if audio_bytes:
                generation_time = time.time() - start_time
                self.stats["successes"] += 1
                self.stats["total_generation_time"] += generation_time
                self.stats["avg_generation_time"] = self.stats["total_generation_time"] / self.stats["successes"]
                
                self.logger.info(f"[PiperTTS] Generated {len(audio_bytes)} bytes in {generation_time:.2f}s")
                return audio_bytes
            else:
                self.stats["failures"] += 1
                return None
                
        except asyncio.TimeoutError:
            self.logger.warning(f"[PiperTTS] Generation timeout after {self.generation_timeout}s")
            self.stats["failures"] += 1
            return None
        except Exception as e:
            self.logger.error(f"[PiperTTS] Generation failed: {e}")
            self.stats["failures"] += 1
            return None

    def _generate_audio_sync(self, text: str) -> Optional[bytes]:
        """Generate audio synchronously (runs in thread pool)"""
        try:
            if self.model_loaded and self.tts_engine:
                return self._generate_with_piper(text)
            else:
                return self._generate_with_cli(text)
        except Exception as e:
            self.logger.error(f"[PiperTTS] Sync generation error: {e}")
            return None

    def _generate_with_piper(self, text: str) -> Optional[bytes]:
        """Generate audio using Piper Python API"""
        try:
            # Generate audio samples
            audio_samples = []
            for audio_chunk in self.tts_engine.synthesize_stream(text):
                audio_samples.extend(audio_chunk)
            
            if not audio_samples:
                return None
            
            # Convert to WAV bytes
            return self._samples_to_wav_bytes(audio_samples)
            
        except Exception as e:
            self.logger.error(f"[PiperTTS] Piper API generation failed: {e}")
            return None

    def _generate_with_cli(self, text: str) -> Optional[bytes]:
        """Generate audio using Piper CLI (fallback)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run Piper CLI
            cmd = [
                "piper",
                "--model", str(self.model_path) if self.model_path else self.model_name,
                "--output_file", temp_path
            ]
            
            process = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=self.generation_timeout,
                check=True
            )
            
            # Read generated audio
            if os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(temp_path)
                return audio_bytes
            else:
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("[PiperTTS] CLI generation timeout")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[PiperTTS] CLI generation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[PiperTTS] CLI generation error: {e}")
            return None
        finally:
            # Cleanup temp file
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

    def _samples_to_wav_bytes(self, samples) -> bytes:
        """Convert audio samples to WAV bytes"""
        try:
            import numpy as np
            
            # Convert to numpy array
            audio_array = np.array(samples, dtype=np.float32)
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            
            if SOUNDFILE_AVAILABLE:
                # Use soundfile for better quality
                sf.write(buffer, audio_array, self.sample_rate, format='WAV')
            else:
                # Fallback to wave module
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    
                    # Convert float32 to int16
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
            
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"[PiperTTS] WAV conversion failed: {e}")
            return None

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS processing"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Limit length (Piper works best with shorter texts)
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS service statistics"""
        return {
            **self.stats,
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "success_rate": self.stats["successes"] / max(self.stats["requests"], 1) * 100
        }

    def health_check(self) -> bool:
        """Check if TTS service is healthy"""
        try:
            test_audio = self.generate_audio("Test")
            return test_audio is not None and len(test_audio) > 1000
        except Exception:
            return False

    def shutdown(self):
        """Shutdown TTS service"""
        self.logger.info("[PiperTTS] Shutting down TTS service")
        if self.executor:
            self.executor.shutdown(wait=True)


# Compatibility wrapper for existing code
class TTSService(PiperTTSService):
    """Compatibility wrapper to replace existing TTS service"""
    pass


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python piper_tts_service.py <text>")
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    
    # Test TTS service
    tts = PiperTTSService()
    
    print(f"Generating TTS for: {text}")
    start_time = time.time()
    
    audio_bytes = tts.generate_audio(text)
    
    generation_time = time.time() - start_time
    
    if audio_bytes:
        print(f"✅ Generated {len(audio_bytes)} bytes in {generation_time:.2f}s")
        
        # Save to file for testing
        with open("test_output.wav", "wb") as f:
            f.write(audio_bytes)
        print("💾 Saved to test_output.wav")
        
        # Show stats
        stats = tts.get_stats()
        print(f"📊 Stats: {stats}")
    else:
        print("❌ TTS generation failed")
    
    tts.shutdown() 