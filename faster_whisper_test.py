#!/usr/bin/env python3
"""
Faster-Whisper GPU Test - Compare performance and accuracy
Tests Faster-Whisper with GPU acceleration against regular OpenAI Whisper
"""

import logging
import time
import numpy as np
import queue
import threading
import json
import tempfile
import os
import argparse

# Audio capture
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

# STT with OpenAI Whisper (offline)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# STT with Faster-Whisper (optimized)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# Audio file processing
try:
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    wavfile = None

# CUDA support
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count()
except ImportError:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FasterWhisperTest")

class FasterWhisperTest:
    """Test Faster-Whisper performance with GPU acceleration."""
    
    def __init__(self, whisper_model="base", device="auto"):
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 4096
        self.dtype = np.float32
        
        # Model configuration
        self.whisper_model_name = whisper_model
        self.device = device
        
        # Model instances
        self.openai_whisper_model = None
        self.faster_whisper_model = None
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.processing_thread = None
        self.stream = None
        
        # Speech detection
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_threshold = 50
        self.min_speech_duration = 0.5
        
        # Performance tracking
        self.transcription_count = 0
        self.openai_time = 0
        self.faster_time = 0
        
    def initialize_models(self):
        """Initialize both OpenAI Whisper and Faster-Whisper models."""
        success = True
        
        # Determine device
        if self.device == "auto":
            if CUDA_AVAILABLE:
                device = "cuda"
                compute_type = "float16"  # Use FP16 for speed
                print(f"🚀 Auto-detected CUDA with {GPU_COUNT} GPU(s)")
            else:
                device = "cpu"
                compute_type = "int8"  # Use INT8 for CPU efficiency
                print("🖥️  Using CPU (CUDA not available)")
        else:
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"
        
        # Initialize OpenAI Whisper
        if WHISPER_AVAILABLE and whisper is not None:
            try:
                print(f"🔧 Loading OpenAI Whisper '{self.whisper_model_name}' model...")
                start_time = time.time()
                self.openai_whisper_model = whisper.load_model(self.whisper_model_name)
                load_time = time.time() - start_time
                print(f"✅ OpenAI Whisper ready (loaded in {load_time:.1f}s)")
            except Exception as e:
                print(f"❌ Failed to load OpenAI Whisper: {e}")
                success = False
        else:
            print("❌ OpenAI Whisper not available")
            success = False
        
        # Initialize Faster-Whisper
        if FASTER_WHISPER_AVAILABLE and WhisperModel is not None:
            try:
                print(f"🔧 Loading Faster-Whisper '{self.whisper_model_name}' model...")
                print(f"🎯 Device: {device}, Compute type: {compute_type}")
                
                start_time = time.time()
                self.faster_whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device=device,
                    compute_type=compute_type
                )
                load_time = time.time() - start_time
                print(f"✅ Faster-Whisper ready (loaded in {load_time:.1f}s)")
                
                # Show GPU info if using CUDA
                if device == "cuda" and CUDA_AVAILABLE:
                    for i in range(GPU_COUNT):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        print(f"   🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
            except Exception as e:
                print(f"❌ Failed to load Faster-Whisper: {e}")
                success = False
        else:
            print("❌ Faster-Whisper not available")
            success = False
        
        return success
    
    def find_audio_device(self):
        """Find the best audio input device."""
        if not SOUNDDEVICE_AVAILABLE or sd is None:
            print("❌ sounddevice not available")
            return None
            
        try:
            devices = sd.query_devices()
            print("🎵 Available audio devices:")
            
            virtual_devices = []
            input_devices = []
            
            for i, device in enumerate(devices):
                if isinstance(device, dict):
                    max_input = device.get('max_input_channels', 0)
                    name = device.get('name', f'Device {i}')
                else:
                    max_input = getattr(device, 'max_input_channels', 0)
                    name = getattr(device, 'name', f'Device {i}')
                
                if max_input > 0:
                    input_devices.append((i, name))
                    print(f"  {i}: {name}")
                    
                    # Prefer VB-Cable or Stereo Mix devices
                    if any(keyword in name.lower() for keyword in 
                          ['cable', 'stereo mix', 'what u hear', 'loopback', 'vb-audio']):
                        print(f"      ⭐ SPEAKER CAPTURE DEVICE")
                        virtual_devices.append((i, name))
            
            # Use virtual device if available
            if virtual_devices:
                device_id = virtual_devices[0][0]
                print(f"🎯 Auto-selected virtual device {device_id}: {virtual_devices[0][1]}")
                return device_id
            
            # Use first input device if no speaker capture found
            if input_devices:
                device_id = input_devices[0][0]
                print(f"🎯 Using device {device_id}: {input_devices[0][1]}")
                return device_id
            else:
                print("❌ No input devices found")
                return None
                
        except Exception as e:
            print(f"❌ Error finding audio device: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback."""
        if status:
            print(f"\n⚠️  Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio_chunk(self, audio_chunk):
        """Process audio for speech detection."""
        try:
            # Convert to mono
            if len(audio_chunk.shape) > 1:
                audio_mono = np.mean(audio_chunk, axis=1)
            else:
                audio_mono = audio_chunk.flatten()
            
            # Resample to 16kHz for STT
            target_length = int(len(audio_mono) * 16000 / self.sample_rate)
            if target_length > 0:
                audio_16k = np.interp(
                    np.linspace(0, len(audio_mono), target_length),
                    np.arange(len(audio_mono)),
                    audio_mono
                )
            else:
                return
            
            # Enhanced speech detection
            audio_level = np.sqrt(np.mean(np.square(audio_16k)))
            audio_max = np.max(np.abs(audio_16k))
            
            speech_detected = audio_level > 0.015 or audio_max > 0.1
            
            if speech_detected:
                self.speech_buffer.extend(audio_16k)
                self.silence_counter = 0
                print("🎤", end="", flush=True)
            else:
                if self.speech_buffer:
                    self.silence_counter += 1
                    if self.silence_counter > self.speech_threshold:
                        speech_duration = len(self.speech_buffer) / 16000
                        if speech_duration >= self.min_speech_duration:
                            self.process_speech()
                        else:
                            print(f"\n🔇 (speech too short: {speech_duration:.1f}s)")
                        
                        self.speech_buffer = []
                        self.silence_counter = 0
            
        except Exception as e:
            print(f"\n❌ Error processing audio: {e}")
    
    def transcribe_with_openai_whisper(self, speech_audio):
        """Transcribe with OpenAI Whisper."""
        if not self.openai_whisper_model:
            return None, 0
            
        try:
            start_time = time.time()
            
            # Normalize audio
            if np.max(np.abs(speech_audio)) > 0:
                speech_audio = speech_audio / np.max(np.abs(speech_audio)) * 0.8
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                if SCIPY_AVAILABLE:
                    audio_int16 = (speech_audio * 32767).astype(np.int16)
                    wavfile.write(temp_file.name, 16000, audio_int16)
                else:
                    temp_file.write((speech_audio * 32767).astype(np.int16).tobytes())
                
                temp_file_path = temp_file.name
            
            # Transcribe with OpenAI Whisper
            result = self.openai_whisper_model.transcribe(
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
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            processing_time = time.time() - start_time
            
            if result and "text" in result:
                text = result["text"].strip()
                avg_logprob = result.get("avg_logprob", "N/A")
                no_speech_prob = result.get("no_speech_prob", "N/A")
                return text, processing_time, avg_logprob, no_speech_prob
            else:
                return None, processing_time, None, None
                
        except Exception as e:
            print(f"\n❌ OpenAI Whisper error: {e}")
            return None, 0, None, None
    
    def transcribe_with_faster_whisper(self, speech_audio):
        """Transcribe with Faster-Whisper."""
        if not self.faster_whisper_model:
            return None, 0
            
        try:
            start_time = time.time()
            
            # Normalize audio
            if np.max(np.abs(speech_audio)) > 0:
                speech_audio = speech_audio / np.max(np.abs(speech_audio)) * 0.8
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                if SCIPY_AVAILABLE:
                    audio_int16 = (speech_audio * 32767).astype(np.int16)
                    wavfile.write(temp_file.name, 16000, audio_int16)
                else:
                    temp_file.write((speech_audio * 32767).astype(np.int16).tobytes())
                
                temp_file_path = temp_file.name
            
            # Transcribe with Faster-Whisper
            segments, info = self.faster_whisper_model.transcribe(
                temp_file_path,
                language='en',
                task='transcribe',
                temperature=0.0,
                beam_size=1,
                word_timestamps=False,
                condition_on_previous_text=False,
                initial_prompt="",
                suppress_tokens=[-1],
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            
            # Collect segments
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)
            
            text = " ".join(text_segments).strip()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            processing_time = time.time() - start_time
            
            # Get info
            language = getattr(info, 'language', 'unknown')
            language_probability = getattr(info, 'language_probability', 0.0)
            
            return text, processing_time, language, language_probability
                
        except Exception as e:
            print(f"\n❌ Faster-Whisper error: {e}")
            return None, 0, None, None
    
    def process_speech(self):
        """Process accumulated speech with both engines."""
        if not self.speech_buffer:
            return
            
        try:
            speech_audio = np.array(self.speech_buffer, dtype=np.float32)
            speech_duration = len(speech_audio) / 16000
            
            print(f"\n🎵 Processing {speech_duration:.1f}s of speech...")
            
            self.transcription_count += 1
            
            # Test OpenAI Whisper
            if self.openai_whisper_model:
                openai_result = self.transcribe_with_openai_whisper(speech_audio.copy())
                
                if len(openai_result) == 4:
                    openai_text, openai_time, avg_logprob, no_speech_prob = openai_result
                    self.openai_time += openai_time
                    
                    if openai_text:
                        confidence_info = ""
                        if avg_logprob != "N/A" and avg_logprob is not None:
                            confidence_info += f", confidence: {avg_logprob:.2f}"
                        if no_speech_prob != "N/A" and no_speech_prob is not None:
                            confidence_info += f", no_speech: {no_speech_prob:.2f}"
                        
                        print(f"🟦 OPENAI WHISPER ({openai_time:.2f}s{confidence_info}): '{openai_text}'")
                    else:
                        print(f"🟦 OPENAI WHISPER ({openai_time:.2f}s): (no speech detected)")
            
            # Test Faster-Whisper
            if self.faster_whisper_model:
                faster_result = self.transcribe_with_faster_whisper(speech_audio.copy())
                
                if len(faster_result) == 4:
                    faster_text, faster_time, language, lang_prob = faster_result
                    self.faster_time += faster_time
                    
                    if faster_text:
                        lang_info = ""
                        if language and language != "unknown":
                            lang_info += f", lang: {language}"
                        if lang_prob and lang_prob > 0:
                            lang_info += f" ({lang_prob:.2f})"
                        
                        print(f"🟩 FASTER-WHISPER ({faster_time:.2f}s{lang_info}): '{faster_text}'")
                    else:
                        print(f"🟩 FASTER-WHISPER ({faster_time:.2f}s): (no speech detected)")
            
            # Show performance comparison every 3 transcriptions
            if self.transcription_count % 3 == 0:
                self.show_performance_stats()
            
        except Exception as e:
            print(f"\n❌ STT processing error: {e}")
    
    def show_performance_stats(self):
        """Show performance statistics."""
        print(f"\n📊 Performance Stats (after {self.transcription_count} transcriptions):")
        
        if self.openai_time > 0:
            avg_openai = self.openai_time / self.transcription_count
            print(f"   🟦 OpenAI Whisper avg: {avg_openai:.2f}s per transcription")
        
        if self.faster_time > 0:
            avg_faster = self.faster_time / self.transcription_count
            print(f"   🟩 Faster-Whisper avg: {avg_faster:.2f}s per transcription")
        
        if self.openai_time > 0 and self.faster_time > 0:
            speed_ratio = self.openai_time / self.faster_time
            print(f"   ⚡ Faster-Whisper is {speed_ratio:.1f}x faster than OpenAI Whisper")
            
            # Calculate efficiency
            total_audio_time = self.transcription_count * 1.0  # Assume ~1s per transcription
            openai_realtime_factor = self.openai_time / total_audio_time
            faster_realtime_factor = self.faster_time / total_audio_time
            
            print(f"   🎯 OpenAI Whisper: {openai_realtime_factor:.2f}x realtime")
            print(f"   🎯 Faster-Whisper: {faster_realtime_factor:.2f}x realtime")
        
        print()
    
    def processing_worker(self):
        """Audio processing worker thread."""
        while self.is_recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                self.process_audio_chunk(audio_chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ Processing error: {e}")
    
    def start_capture(self, device_id):
        """Start audio capture."""
        try:
            if sd is None:
                print("❌ sounddevice not available")
                return False
                
            self.stream = sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("✅ Audio capture started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop audio capture."""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Process any remaining speech
        if self.speech_buffer:
            speech_duration = len(self.speech_buffer) / 16000
            if speech_duration >= self.min_speech_duration:
                self.process_speech()
            self.speech_buffer = []
        
        # Show final stats
        if self.transcription_count > 0:
            self.show_performance_stats()
        
        print("\n🛑 Capture stopped")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Faster-Whisper GPU Performance Test")
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--audio-device",
        type=int,
        help="Audio device ID to use (will auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Faster-Whisper GPU Performance Test")
    print("=" * 50)
    print(f"🤖 Model: {args.model}")
    print(f"🎯 Device: {args.device}")
    
    if CUDA_AVAILABLE:
        print(f"🎮 CUDA Available: {GPU_COUNT} GPU(s)")
        for i in range(GPU_COUNT):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
    else:
        print("🖥️  CUDA: Not available (CPU only)")
    
    print()
    
    # Check dependencies
    if not SOUNDDEVICE_AVAILABLE:
        print("❌ Install sounddevice: pip install sounddevice")
        return
    
    if not WHISPER_AVAILABLE:
        print("❌ Install OpenAI Whisper: pip install openai-whisper")
        return
    
    if not FASTER_WHISPER_AVAILABLE:
        print("❌ Install Faster-Whisper: pip install faster-whisper")
        return
    
    # Create test instance
    test = FasterWhisperTest(whisper_model=args.model, device=args.device)
    
    # Initialize models
    if not test.initialize_models():
        return
    
    # Find audio device
    if args.audio_device is not None:
        device_id = args.audio_device
        print(f"🎯 Using specified device {device_id}")
    else:
        device_id = test.find_audio_device()
        if device_id is None:
            print("💡 Try enabling 'Stereo Mix' in Windows Sound settings")
            print("💡 Or install VB-Cable for virtual audio routing")
            return
    
    # Start capture
    print(f"\n🎤 Starting capture on device {device_id}...")
    if not test.start_capture(device_id):
        return
    
    print("✅ Listening for speaker audio...")
    print("🎵 Play some audio with speech or talk near your speakers")
    print("🎤 = audio detected, transcriptions will appear below")
    print("🟦 = OpenAI Whisper results, 🟩 = Faster-Whisper results")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        test.stop_capture()
        print("✅ Done!")

if __name__ == "__main__":
    main() 