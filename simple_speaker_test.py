#!/usr/bin/env python3
"""
Enhanced Speaker STT Test - Compare Vosk vs OpenAI Whisper
Automatically detects audio devices and transcribes speaker audio with multiple STT engines.
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

# STT with Vosk (offline)
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    vosk = None

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

# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EnhancedSpeakerSTT")

class EnhancedSpeakerSTT:
    """Enhanced speaker audio capture and STT with multiple engines."""
    
    def __init__(self, stt_engine="both", whisper_model="base", use_faster_whisper=True):
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 4096
        self.dtype = np.float32
        
        # STT configuration
        self.stt_engine = stt_engine  # "vosk", "whisper", "faster-whisper", or "both"
        self.whisper_model_name = whisper_model
        self.use_faster_whisper = use_faster_whisper
        
        # Vosk components
        self.vosk_model = None
        self.vosk_recognizer = None
        
        # Whisper components
        self.whisper_model = None
        self.faster_whisper_model = None
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.processing_thread = None
        self.stream = None
        
        # Enhanced speech detection
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_threshold = 50  # Frames of silence before processing
        self.min_speech_duration = 0.5  # Minimum seconds of speech
        
        # Performance tracking
        self.transcription_count = 0
        self.total_processing_time = 0
        self.vosk_time = 0
        self.whisper_time = 0
        
    def initialize_stt(self):
        """Initialize selected STT engines."""
        success = True
        
        # Initialize Vosk if requested
        if self.stt_engine in ["vosk", "both"] and VOSK_AVAILABLE and vosk is not None:
            try:
                print("🔧 Loading Vosk model...")
                self.vosk_model = vosk.Model("models/vosk-model-small-en-us-0.15")
                self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
                print("✅ Vosk STT ready")
            except Exception as e:
                print(f"❌ Failed to load Vosk: {e}")
                if self.stt_engine == "vosk":
                    success = False
        elif self.stt_engine in ["vosk", "both"]:
            print("❌ Vosk not available - install with: pip install vosk")
            if self.stt_engine == "vosk":
                success = False
        
        # Initialize Whisper if requested
        if self.stt_engine in ["whisper", "both"] and WHISPER_AVAILABLE and whisper is not None:
            try:
                print(f"🔧 Loading OpenAI Whisper model '{self.whisper_model_name}'...")
                print("💡 Available models: tiny, base, small, medium, large")
                print("💡 Accuracy: tiny < base < small < medium < large")
                print("💡 Speed: large < medium < small < base < tiny")
                
                start_time = time.time()
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                load_time = time.time() - start_time
                
                print(f"✅ OpenAI Whisper '{self.whisper_model_name}' ready (loaded in {load_time:.1f}s)")
                
                # Show model info
                if hasattr(self.whisper_model, 'dims'):
                    dims = self.whisper_model.dims
                    print(f"📊 Model info: {dims.n_audio_ctx} audio tokens, {dims.n_text_ctx} text tokens")
                
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                if self.stt_engine == "whisper":
                    success = False
        elif self.stt_engine in ["whisper", "both"]:
            print("❌ Whisper not available - install with: pip install openai-whisper")
            if self.stt_engine == "whisper":
                success = False
        
        if not SCIPY_AVAILABLE and self.stt_engine in ["whisper", "both"]:
            print("⚠️  scipy not available - install for better audio processing: pip install scipy")
        
        return success
    
    def find_audio_device(self):
        """Find the best audio input device."""
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ sounddevice not available")
            return None
            
        try:
            if sd is None:
                return None
            devices = sd.query_devices()
            print("🎵 Available audio devices:")
            
            # Look for input devices
            input_devices = []
            virtual_devices = []
            
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
                print("💡 For speaker capture, install VB-Cable or enable Windows Stereo Mix")
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
            
            # Enhanced level-based speech detection
            audio_level = np.sqrt(np.mean(np.square(audio_16k)))
            audio_max = np.max(np.abs(audio_16k))
            
            # Dynamic threshold based on recent audio levels
            speech_detected = audio_level > 0.015 or audio_max > 0.1
            
            if speech_detected:
                self.speech_buffer.extend(audio_16k)
                self.silence_counter = 0
                print("🎤", end="", flush=True)  # Visual feedback
            else:
                if self.speech_buffer:  # We have speech data
                    self.silence_counter += 1
                    if self.silence_counter > self.speech_threshold:
                        # Check if we have enough speech
                        speech_duration = len(self.speech_buffer) / 16000
                        if speech_duration >= self.min_speech_duration:
                            self.process_speech()
                        else:
                            print(f"\n🔇 (speech too short: {speech_duration:.1f}s)")
                        
                        self.speech_buffer = []
                        self.silence_counter = 0
            
        except Exception as e:
            print(f"\n❌ Error processing audio: {e}")
    
    def transcribe_with_vosk(self, speech_audio):
        """Transcribe with Vosk."""
        if not self.vosk_recognizer:
            return None, 0
            
        try:
            start_time = time.time()
            
            # Normalize audio
            if np.max(np.abs(speech_audio)) > 0:
                speech_audio = speech_audio / np.max(np.abs(speech_audio)) * 0.8
            
            # Convert to 16-bit PCM
            speech_pcm = (speech_audio * 32767).astype(np.int16).tobytes()
            
            # Process with Vosk
            if self.vosk_recognizer.AcceptWaveform(speech_pcm):
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get("text", "").strip()
            else:
                result = json.loads(self.vosk_recognizer.FinalResult())
                text = result.get("text", "").strip()
            
            processing_time = time.time() - start_time
            return text, processing_time
            
        except Exception as e:
            print(f"\n❌ Vosk error: {e}")
            return None, 0
    
    def transcribe_with_whisper(self, speech_audio):
        """Transcribe with OpenAI Whisper."""
        if not self.whisper_model:
            return None, 0
            
        try:
            start_time = time.time()
            
            # Normalize audio for Whisper
            if np.max(np.abs(speech_audio)) > 0:
                speech_audio = speech_audio / np.max(np.abs(speech_audio)) * 0.8
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                if SCIPY_AVAILABLE:
                    # Use scipy for better quality
                    audio_int16 = (speech_audio * 32767).astype(np.int16)
                    wavfile.write(temp_file.name, 16000, audio_int16)
                else:
                    # Fallback: write raw audio (less reliable)
                    temp_file.write((speech_audio * 32767).astype(np.int16).tobytes())
                
                temp_file_path = temp_file.name
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_file_path,
                language='en',
                task='transcribe',
                temperature=0.0,  # Deterministic
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
                
                # Get confidence metrics
                avg_logprob = result.get("avg_logprob", "N/A")
                no_speech_prob = result.get("no_speech_prob", "N/A")
                
                return text, processing_time, avg_logprob, no_speech_prob
            else:
                return None, processing_time, None, None
                
        except Exception as e:
            print(f"\n❌ Whisper error: {e}")
            return None, 0, None, None
    
    def process_speech(self):
        """Process accumulated speech with selected STT engines."""
        if not self.speech_buffer:
            return
            
        try:
            speech_audio = np.array(self.speech_buffer, dtype=np.float32)
            speech_duration = len(speech_audio) / 16000
            
            print(f"\n🎵 Processing {speech_duration:.1f}s of speech...")
            
            self.transcription_count += 1
            
            # Process with Vosk
            if self.stt_engine in ["vosk", "both"] and self.vosk_recognizer:
                vosk_text, vosk_time = self.transcribe_with_vosk(speech_audio.copy())
                self.vosk_time += vosk_time
                
                if vosk_text:
                    print(f"🟦 VOSK ({vosk_time:.2f}s): '{vosk_text}'")
                else:
                    print(f"🟦 VOSK ({vosk_time:.2f}s): (no speech detected)")
            
            # Process with Whisper
            if self.stt_engine in ["whisper", "both"] and self.whisper_model:
                whisper_result = self.transcribe_with_whisper(speech_audio.copy())
                
                if len(whisper_result) == 4:  # Full result with confidence
                    whisper_text, whisper_time, avg_logprob, no_speech_prob = whisper_result
                    self.whisper_time += whisper_time
                    
                    if whisper_text:
                        confidence_info = ""
                        if avg_logprob != "N/A" and avg_logprob is not None:
                            confidence_info += f", confidence: {avg_logprob:.2f}"
                        if no_speech_prob != "N/A" and no_speech_prob is not None:
                            confidence_info += f", no_speech: {no_speech_prob:.2f}"
                        
                        print(f"🟩 WHISPER ({whisper_time:.2f}s{confidence_info}): '{whisper_text}'")
                    else:
                        print(f"🟩 WHISPER ({whisper_time:.2f}s): (no speech detected)")
                else:
                    whisper_text, whisper_time = whisper_result[:2]
                    self.whisper_time += whisper_time
                    
                    if whisper_text:
                        print(f"🟩 WHISPER ({whisper_time:.2f}s): '{whisper_text}'")
                    else:
                        print(f"🟩 WHISPER ({whisper_time:.2f}s): (no speech detected)")
            
            # Show performance comparison
            if self.stt_engine == "both" and self.transcription_count % 5 == 0:
                self.show_performance_stats()
            
        except Exception as e:
            print(f"\n❌ STT processing error: {e}")
    
    def show_performance_stats(self):
        """Show performance statistics."""
        print(f"\n📊 Performance Stats (after {self.transcription_count} transcriptions):")
        if self.vosk_time > 0:
            avg_vosk = self.vosk_time / self.transcription_count
            print(f"   🟦 Vosk avg: {avg_vosk:.2f}s per transcription")
        if self.whisper_time > 0:
            avg_whisper = self.whisper_time / self.transcription_count
            print(f"   🟩 Whisper avg: {avg_whisper:.2f}s per transcription")
        if self.vosk_time > 0 and self.whisper_time > 0:
            speed_ratio = self.whisper_time / self.vosk_time
            print(f"   ⚡ Whisper is {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'} than Vosk")
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
    parser = argparse.ArgumentParser(description="Enhanced Speaker STT Test - Compare Vosk vs OpenAI Whisper")
    parser.add_argument(
        "--engine", 
        choices=["vosk", "whisper", "both"], 
        default="both",
        help="STT engine to use (default: both)"
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base). Larger = more accurate but slower"
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Audio device ID to use (will auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    print("🔊 Enhanced Speaker STT Test - Vosk vs OpenAI Whisper")
    print("=" * 60)
    print(f"🎯 STT Engine: {args.engine}")
    if args.engine in ["whisper", "both"]:
        print(f"🤖 Whisper Model: {args.whisper_model}")
    print()
    
    # Check dependencies
    if not SOUNDDEVICE_AVAILABLE:
        print("❌ Install sounddevice: pip install sounddevice")
        return
    
    if args.engine in ["vosk", "both"] and not VOSK_AVAILABLE:
        print("❌ Install Vosk: pip install vosk")
        if args.engine == "vosk":
            return
    
    if args.engine in ["whisper", "both"] and not WHISPER_AVAILABLE:
        print("❌ Install OpenAI Whisper: pip install openai-whisper")
        if args.engine == "whisper":
            return
    
    # Create STT instance
    stt = EnhancedSpeakerSTT(stt_engine=args.engine, whisper_model=args.whisper_model)
    
    # Initialize STT
    if not stt.initialize_stt():
        return
    
    # Find audio device
    if args.device is not None:
        device_id = args.device
        print(f"🎯 Using specified device {device_id}")
    else:
        device_id = stt.find_audio_device()
        if device_id is None:
            print("💡 Try enabling 'Stereo Mix' in Windows Sound settings")
            print("💡 Or install VB-Cable for virtual audio routing")
            return
    
    # Start capture
    print(f"\n🎤 Starting capture on device {device_id}...")
    if not stt.start_capture(device_id):
        return
    
    print("✅ Listening for speaker audio...")
    print("🎵 Play some audio with speech or talk near your speakers")
    print("🎤 = audio detected, transcriptions will appear below")
    
    if args.engine == "both":
        print("🟦 = Vosk results, 🟩 = Whisper results")
    
    print("🛑 Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        stt.stop_capture()
        print("✅ Done!")

if __name__ == "__main__":
    main() 