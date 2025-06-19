#!/usr/bin/env python3
"""
Whisper Audio Test - Standalone test for WhisperAudioCapture
Tests the new Whisper-only audio capture without Discord integration.
"""

import asyncio
import logging
import time
import numpy as np
import queue
import threading
import tempfile
import os
from typing import Optional

# Audio capture
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

# Whisper STT
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhisperAudioTest")

class SimpleAppContext:
    """Minimal app context for testing."""
    def __init__(self):
        self.logger = logger

class WhisperAudioCapture:
    """Captures audio from virtual audio cables and processes with Whisper STT only."""
    
    def __init__(self, app_context, callback_func=None):
        self.app_context = app_context
        self.logger = app_context.logger
        self.callback_func = callback_func
        
        # Audio settings optimized for Whisper
        self.sample_rate = 16000  # Whisper's native sample rate
        self.channels = 1  # Mono for Whisper
        self.chunk_size = 1024  # Smaller chunks for real-time processing
        self.dtype = np.float32
        
        # Speech detection settings (simple level-based)
        self.speech_threshold = 0.01  # RMS threshold for speech detection
        self.min_speech_duration = 1.0  # Minimum speech duration in seconds
        self.max_silence_duration = 2.0  # Maximum silence before processing
        
        # State tracking
        self.is_recording = False
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
        
        # Audio device
        self.input_device = None
        self.stream = None
        
        # Processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
        # Whisper model
        self.whisper_model = None
        
    def list_audio_devices(self):
        """List all available audio input devices."""
        if not SOUNDDEVICE_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available - install with: pip install sounddevice")
            return []
            
        self.logger.info("🎵 Available Audio Input Devices:")
        devices = sd.query_devices()
        
        virtual_devices = []
        for i, device in enumerate(devices):
            try:
                if isinstance(device, dict):
                    max_channels = device.get('max_input_channels', 0)
                    device_name = device.get('name', f'Device {i}')
                else:
                    max_channels = getattr(device, 'max_input_channels', 0)
                    device_name = getattr(device, 'name', f'Device {i}')
                
                if max_channels > 0:  # Input device
                    self.logger.info(f"  {i}: {device_name} (channels: {max_channels})")
                    
                    # Look for virtual audio cables
                    if any(keyword in device_name.lower() for keyword in 
                          ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'stereo mix', 'wave out mix', 'what u hear']):
                        virtual_devices.append((i, device_name))
                        self.logger.info(f"      ⭐ VIRTUAL AUDIO DEVICE DETECTED")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not process device {i}: {e}")
                continue
        
        if virtual_devices:
            self.logger.info("🎯 Recommended Virtual Audio Devices:")
            for device_id, device_name in virtual_devices:
                self.logger.info(f"  Device {device_id}: {device_name}")
        else:
            self.logger.warning("⚠️  No virtual audio devices detected. Install VB-Cable or enable Stereo Mix.")
            
        return virtual_devices
    
    def select_input_device(self, device_id: Optional[int] = None):
        """Select audio input device."""
        if not SOUNDDEVICE_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available")
            return False
            
        if device_id is None:
            # Auto-detect virtual audio device
            virtual_devices = self.list_audio_devices()
            if virtual_devices:
                device_id = virtual_devices[0][0]  # Use first virtual device
                self.logger.info(f"🎯 Auto-selected virtual audio device: {device_id}")
            else:
                # Fall back to default device
                try:
                    device_id = sd.default.device[0]
                    self.logger.warning(f"⚠️  Using default input device: {device_id}")
                except Exception:
                    device_id = 0  # Fallback to device 0
                    self.logger.warning("⚠️  Using device 0 as fallback")
        
        try:
            device_info = sd.query_devices(device_id, 'input')
            self.input_device = device_id
            device_name = device_info.get('name', f'Device {device_id}') if isinstance(device_info, dict) else str(device_info)
            self.logger.info(f"✅ Selected audio device: {device_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to select device {device_id}: {e}")
            return False
    
    async def initialize_whisper(self, model_size: str = "base"):
        """Initialize Whisper model."""
        try:
            self.logger.info(f"🔧 Loading Whisper model '{model_size}'...")
            self.logger.info("💡 Available models: tiny, base, small, medium, large")
            self.logger.info("💡 Accuracy: tiny < base < small < medium < large")
            self.logger.info("💡 Speed: large < medium < small < base < tiny")
            
            loop = asyncio.get_event_loop()
            start_time = time.time()
            self.whisper_model = await loop.run_in_executor(
                None,
                whisper.load_model,
                model_size
            )
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ Whisper model '{model_size}' loaded successfully in {load_time:.1f}s")
            
            # Show model info
            if hasattr(self.whisper_model, 'dims'):
                dims = self.whisper_model.dims
                self.logger.info(f"📊 Model info: {dims.n_audio_ctx} audio tokens, {dims.n_text_ctx} text tokens")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Whisper initialization failed: {e}")
            return False
    
    def detect_speech(self, audio_chunk: np.ndarray) -> tuple[bool, bool]:
        """Simple level-based speech detection."""
        try:
            # Calculate RMS (Root Mean Square) for volume detection
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            
            current_time = time.time()
            is_speech = rms > self.speech_threshold
            speech_ended = False
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.logger.info(f"🎤 Speech started (RMS: {rms:.4f})")
                self.last_speech_time = current_time
            elif self.is_speaking and self.last_speech_time:
                silence_duration = current_time - self.last_speech_time
                if silence_duration > self.max_silence_duration:
                    if self.speech_start_time:
                        speech_duration = current_time - self.speech_start_time
                        if speech_duration >= self.min_speech_duration:
                            speech_ended = True
                            self.logger.info(f"🎤 Speech ended (duration: {speech_duration:.2f}s)")
                    
                    self.is_speaking = False
                    self.speech_start_time = None
                    self.last_speech_time = None
            
            return is_speech, speech_ended
            
        except Exception as e:
            self.logger.error(f"❌ Speech detection error: {e}")
            return False, False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for sounddevice stream."""
        if status:
            self.logger.warning(f"⚠️  Audio callback status: {status}")
        
        try:
            # Convert stereo to mono if needed
            if len(indata.shape) > 1:
                audio_mono = np.mean(indata, axis=1)
            else:
                audio_mono = indata.copy()
            
            # Add audio to queue for processing
            self.audio_queue.put(audio_mono.flatten())
        except Exception as e:
            self.logger.error(f"❌ Audio callback error: {e}")
    
    def processing_worker(self):
        """Worker thread for processing audio chunks."""
        self.logger.info("🎯 Whisper audio processing worker started")
        
        while self.is_recording:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Detect speech using simple level-based detection
                is_speech, speech_ended = self.detect_speech(audio_chunk)
                
                # Always add to buffer when speaking
                if self.is_speaking:
                    self.audio_buffer.extend(audio_chunk)
                
                # Process complete speech segments
                if speech_ended and len(self.audio_buffer) > 0:
                    # Get accumulated audio
                    speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer.clear()
                    
                    # Process with Whisper directly in worker thread for testing
                    if self.callback_func:
                        try:
                            # For testing, run transcription directly in worker thread
                            self.logger.info("🎵 Processing audio with Whisper...")
                            
                            # Create a new event loop for this thread
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Run the callback
                                result = loop.run_until_complete(self.callback_func(speech_audio))
                                
                                loop.close()
                                self.logger.info("✅ Whisper audio processing completed")
                                
                            except Exception as e:
                                self.logger.error(f"❌ Callback processing error: {e}")
                            
                        except Exception as e:
                            self.logger.error(f"❌ Callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing worker error: {e}")
        
        self.logger.info("🎯 Whisper audio processing worker stopped")
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if not self.whisper_model:
            self.logger.error("❌ Whisper model not loaded")
            return None
            
        try:
            # Validate audio
            if len(audio_data) == 0:
                return None
            
            audio_duration = len(audio_data) / self.sample_rate
            audio_max_volume = np.max(np.abs(audio_data))
            audio_rms = np.sqrt(np.mean(np.square(audio_data)))
            
            self.logger.info(f"🎵 Transcribing audio - Duration: {audio_duration:.2f}s, Max: {audio_max_volume:.4f}, RMS: {audio_rms:.4f}")
            
            # Basic quality checks
            if audio_max_volume < 0.001:
                self.logger.warning(f"🔇 Audio volume too low (max: {audio_max_volume:.4f})")
                return None
            
            if audio_duration < 0.5:
                self.logger.warning(f"🔇 Audio too short ({audio_duration:.2f}s)")
                return None
            
            # Normalize audio
            if audio_max_volume > 0:
                normalized_audio = audio_data * (0.8 / audio_max_volume)
            else:
                normalized_audio = audio_data
            
            # Save to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert to 16-bit PCM for Whisper
                try:
                    import scipy.io.wavfile as wavfile
                    audio_int16 = np.clip(normalized_audio * 32767, -32767, 32767).astype(np.int16)
                    wavfile.write(temp_file.name, self.sample_rate, audio_int16)
                except ImportError:
                    # Fallback without scipy
                    self.logger.warning("⚠️ scipy not available, using basic audio processing")
                    # Simple WAV file creation
                    import wave
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        audio_int16 = np.clip(normalized_audio * 32767, -32767, 32767).astype(np.int16)
                        wav_file.writeframes(audio_int16.tobytes())
                
                temp_file_path = temp_file.name
            
            self.logger.info(f"🎵 Saved audio to {temp_file_path}, starting Whisper transcription...")
            
            # Run Whisper transcription
            loop = asyncio.get_event_loop()
            start_time = time.time()
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    temp_file_path,
                    language='en',
                    task='transcribe',
                    temperature=0.0,
                    best_of=1,
                    beam_size=1,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    initial_prompt="",
                    suppress_tokens=[-1],
                    logprob_threshold=-0.5,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4
                )
            )
            
            processing_time = time.time() - start_time
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if result and "text" in result:
                text = str(result["text"]).strip()
                
                # Log transcription results
                avg_logprob = result.get("avg_logprob", "unknown")
                no_speech_prob = result.get("no_speech_prob", "unknown")
                
                self.logger.info(f"📝 Whisper result: '{text}' (processed in {processing_time:.2f}s)")
                self.logger.info(f"📊 Whisper stats - Avg LogProb: {avg_logprob}, No Speech Prob: {no_speech_prob}")
                
                # Quality checks
                if isinstance(avg_logprob, (int, float)) and avg_logprob < -1.0:
                    self.logger.info(f"🚫 Rejected low confidence transcription (logprob: {avg_logprob})")
                    return None
                
                if isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.8:
                    self.logger.info(f"🚫 Rejected high no-speech probability ({no_speech_prob})")
                    return None
                
                # Basic hallucination filtering
                if text and len(text.strip()) > 0:
                    text_lower = text.lower().strip()
                    
                    # Filter out obvious hallucinations
                    hallucinations = [
                        "thank you for watching",
                        "thanks for watching", 
                        "subscribe",
                        "like and subscribe"
                    ]
                    
                    if text_lower in hallucinations:
                        self.logger.info(f"🚫 Filtered out hallucination: '{text}'")
                        return None
                    
                    if len(text_lower) <= 2:
                        self.logger.info(f"🚫 Filtered out very short text: '{text}'")
                        return None
                    
                    self.logger.info(f"✅ Accepted transcription: '{text}'")
                    return text
                else:
                    self.logger.info("🔇 Whisper returned empty text")
                    return None
            else:
                self.logger.warning("🔇 Whisper returned no result")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Whisper transcription error: {e}")
            return None
    
    def start_recording(self):
        """Start recording from audio device."""
        if not SOUNDDEVICE_AVAILABLE or sd is None:
            self.logger.error("❌ sounddevice not available")
            return False
            
        if self.is_recording:
            self.logger.warning("⚠️  Already recording")
            return True
        
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
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info("✅ Whisper audio recording started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.logger.info("🛑 Whisper audio recording stopped")

async def process_transcription(audio_data: np.ndarray):
    """Process transcribed audio."""
    global whisper_capture
    
    try:
        # Transcribe the audio
        transcription = await whisper_capture.transcribe_audio(audio_data)
        
        if transcription:
            print(f"\n🎤 TRANSCRIPTION: {transcription}")
            print("=" * 50)
        else:
            print("🔇 No clear speech detected")
            
    except Exception as e:
        logger.error(f"❌ Error processing transcription: {e}")

async def main():
    """Main test function."""
    global whisper_capture
    
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not available - install with: pip install openai-whisper")
        return
    
    if not SOUNDDEVICE_AVAILABLE:
        print("❌ sounddevice not available - install with: pip install sounddevice")
        return
    
    print("🎤 Whisper Audio Capture Test")
    print("=" * 50)
    
    # Create app context and audio capture
    app_context = SimpleAppContext()
    whisper_capture = WhisperAudioCapture(app_context, process_transcription)
    
    # Initialize Whisper
    print("🔧 Initializing Whisper...")
    if not await whisper_capture.initialize_whisper("base"):
        print("❌ Failed to initialize Whisper")
        return
    
    # Select audio device
    print("🎯 Selecting audio device...")
    if not whisper_capture.select_input_device():
        print("❌ Failed to select audio device")
        return
    
    # Start recording
    print("🎙️ Starting audio recording...")
    if not whisper_capture.start_recording():
        print("❌ Failed to start recording")
        return
    
    print("\n✅ Whisper audio capture is running!")
    print("🎤 Speak into your microphone or play audio through virtual audio cable")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping audio capture...")
        whisper_capture.stop_recording()
        print("✅ Audio capture stopped")

if __name__ == "__main__":
    whisper_capture = None
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"�� Fatal error: {e}") 