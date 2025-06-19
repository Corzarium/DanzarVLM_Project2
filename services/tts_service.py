import os
import logging
import requests
import tempfile
import threading
import time
from typing import Optional, Callable, List
from queue import Queue

class TTSService:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.TTSService")
        self.config = self.ctx.global_settings.get('TTS_SERVER', {})
        
        # Chatterbox TTS endpoint configuration
        self.endpoint = self.config.get('endpoint') or os.getenv("TTS_ENDPOINT", "http://localhost:8055/tts")
        self.timeout = self.config.get('timeout', 60)
        
        # Ensure minimum timeout for Chatterbox (it can be slow)
        if self.timeout < 30:
            self.timeout = 30
            self.logger.info(f"[TTSService] Increased timeout to {self.timeout}s for Chatterbox compatibility")
        
        # Chatterbox TTS parameters with sensible defaults
        self.default_voice = self.config.get('default_voice', 'Emily.wav')
        self.temperature = self.config.get('temperature', 0.7)  # Controls randomness
        self.repetition_penalty = self.config.get('repetition_penalty', 1.1)  # Reduces repetition
        self.top_k = self.config.get('top_k', 50)  # Top-k sampling
        self.top_p = self.config.get('top_p', 0.8)  # Top-p sampling
        self.speed = self.config.get('speed', 1.0)  # Speech speed multiplier
        
        # Character limits for chunking
        self.max_chars = self.config.get('max_characters', 500)  # Conservative limit for Chatterbox
        
        # Streaming TTS configuration
        self.streaming_enabled = self.config.get('enable_streaming', True)
        self.streaming_chunk_size = self.config.get('streaming_chunk_size', 2)  # Play 2 sentences at a time
        
        self.logger.info(f"[TTSService] Initialized Chatterbox TTS - Endpoint: {self.endpoint}")
        self.logger.info(f"[TTSService] Voice: {self.default_voice}, Speed: {self.speed}, Timeout: {self.timeout}s")
        self.logger.info(f"[TTSService] Streaming: {'Enabled' if self.streaming_enabled else 'Disabled'} (chunk size: {self.streaming_chunk_size})")

    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio using Chatterbox TTS with proper API format"""
        if not text or len(text.strip()) == 0:
            self.logger.warning("[TTSService] Empty text provided")
            return None
        
        # Clean text for TTS
        cleaned_text = self._clean_text_for_tts(text)
        
        # Check if text is too long and needs chunking
        if len(cleaned_text) > self.max_chars:
            self.logger.info(f"[TTSService] Text too long ({len(cleaned_text)} chars), chunking into smaller pieces")
            return self._generate_chunked_audio(cleaned_text)
        
        return self._generate_single_audio(cleaned_text)

    def _generate_single_audio(self, text: str) -> Optional[bytes]:
        """Generate audio for a single text chunk using Chatterbox TTS API"""
        try:
            # Chatterbox TTS API payload format
            payload = {
                "text": text,
                "predefined_voice_id": self.default_voice,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "speed": self.speed,
                "format": "wav"  # Request WAV format
            }
            
            self.logger.debug(f"[TTSService] Generating audio for {len(text)} characters")
            self.logger.debug(f"[TTSService] Using voice: {self.default_voice}, speed: {self.speed}")
            
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'audio/wav'
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                audio_data = response.content
                if len(audio_data) > 0:
                    self.logger.info(f"[TTSService] Successfully generated {len(audio_data)} bytes of audio")
                    return audio_data
                else:
                    self.logger.error("[TTSService] Received empty audio response")
                    return None
            else:
                self.logger.error(f"[TTSService] HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"[TTSService] Request timed out after {self.timeout}s for {len(text)} characters")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"[TTSService] Connection failed to {self.endpoint}")
            return None
        except Exception as e:
            self.logger.error(f"[TTSService] Generation failed: {e}")
            return None

    def _generate_chunked_audio(self, text: str) -> Optional[bytes]:
        """Break long text into chunks and concatenate the audio"""
        chunks = self._smart_chunk_text(text)
        self.logger.info(f"[TTSService] Split text into {len(chunks)} chunks")
        
        audio_parts = []
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"[TTSService] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            audio_data = self._generate_single_audio(chunk)
            if audio_data:
                audio_parts.append(audio_data)
            else:
                self.logger.warning(f"[TTSService] Failed to generate audio for chunk {i+1}, continuing...")
        
        if not audio_parts:
            self.logger.error("[TTSService] All chunks failed to generate audio")
            return None
        
        if len(audio_parts) == 1:
            return audio_parts[0]
            
        # Concatenate WAV files properly
        return self._concatenate_wav_files(audio_parts)

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS output"""
        import re
        
        # Remove Discord formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Underline
        text = re.sub(r'~~([^~]+)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Inline code
        
        # Remove code blocks
        text = re.sub(r'```[a-zA-Z]*\n.*?\n```', '', text, flags=re.DOTALL)
        
        # Remove URLs in markdown format
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove emojis and special characters that might confuse TTS
        text = re.sub(r'[🎤🤖🎯🔊🎵✅❌⚠️💡🧠🎮🔥🛑📝📤📥🔄🔇]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _smart_chunk_text(self, text: str) -> list[str]:
        """Split text intelligently at sentence boundaries while respecting character limits"""
        import re
        
        # Split into sentences using common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split it further
                    word_chunks = self._split_long_sentence(sentence)
                    chunks.extend(word_chunks)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Split a very long sentence at word boundaries"""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    # Single word is too long, just truncate it
                    chunks.append(word[:self.max_chars])
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _concatenate_wav_files(self, audio_parts: list[bytes]) -> bytes:
        """Properly concatenate WAV files using temporary files with better error handling"""
        if len(audio_parts) == 1:
            return audio_parts[0]
            
        try:
            # Use pydub for proper WAV concatenation if available
            try:
                from pydub import AudioSegment
                import time
                
                # Create AudioSegment objects from bytes with better temp file handling
                audio_segments = []
                temp_files = []
                
                for i, audio_data in enumerate(audio_parts):
                    # Create temp file with unique name and explicit cleanup
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=f'_chunk_{i}_{int(time.time())}.wav', 
                        delete=False
                    )
                    temp_files.append(temp_file.name)
                    
                    try:
                        temp_file.write(audio_data)
                        temp_file.flush()
                        temp_file.close()  # Explicitly close before reading
                        
                        # Small delay to ensure file is written
                        time.sleep(0.01)
                        
                        segment = AudioSegment.from_wav(temp_file.name)
                        audio_segments.append(segment)
                        
                    except Exception as e:
                        self.logger.warning(f"[TTSService] Failed to process chunk {i}: {e}")
                        # Clean up this temp file and continue
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                        continue
                
                # Clean up all temp files
                for temp_file_path in temp_files:
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                    except Exception as e:
                        self.logger.debug(f"[TTSService] Could not delete temp file {temp_file_path}: {e}")
                
                if not audio_segments:
                    self.logger.error("[TTSService] No audio segments were successfully created")
                    return audio_parts[0] if audio_parts else b''
                
                # Concatenate all segments
                combined = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined += segment
                
                # Export to bytes with explicit temp file handling
                output_temp = tempfile.NamedTemporaryFile(
                    suffix=f'_output_{int(time.time())}.wav', 
                    delete=False
                )
                output_path = output_temp.name
                output_temp.close()
                
                try:
                    combined.export(output_path, format="wav")
                    
                    # Small delay to ensure file is written
                    time.sleep(0.01)
                    
                    with open(output_path, 'rb') as f:
                        result = f.read()
                    
                    # Clean up output temp file
                    os.unlink(output_path)
                    
                    self.logger.info(f"[TTSService] Successfully concatenated {len(audio_parts)} audio chunks using pydub")
                    return result
                    
                except Exception as e:
                    self.logger.error(f"[TTSService] Failed to export concatenated audio: {e}")
                    # Clean up output temp file if it exists
                    try:
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except:
                        pass
                    # Fall through to simple concatenation
                    
            except ImportError:
                self.logger.info("[TTSService] pydub not available, using simple concatenation")
            
            # Fallback to simple concatenation (more reliable for basic use cases)
            self.logger.info(f"[TTSService] Using simple WAV concatenation for {len(audio_parts)} chunks")
            result = audio_parts[0]  # Keep first file with header
            
            for i, audio_part in enumerate(audio_parts[1:], 1):
                # Skip WAV header (44 bytes) for subsequent files
                if len(audio_part) > 44:
                    result += audio_part[44:]
                    self.logger.debug(f"[TTSService] Added chunk {i+1} ({len(audio_part)} bytes)")
                    
            self.logger.info(f"[TTSService] Simple concatenation completed: {len(result)} total bytes")
            return result
                
        except Exception as e:
            self.logger.error(f"[TTSService] All concatenation methods failed: {e}")
            # Final fallback: return first chunk only
            self.logger.warning("[TTSService] Returning first audio chunk only as fallback")
            return audio_parts[0] if audio_parts else b''

    def test_connection(self) -> bool:
        """Test connection to Chatterbox TTS server"""
        try:
            # Try a simple health check or small test
            test_payload = {
                "text": "Test",
                "predefined_voice_id": self.default_voice,
                "format": "wav"
            }
            
            response = requests.post(
                self.endpoint,
                json=test_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("[TTSService] Connection test successful")
                return True
            else:
                self.logger.error(f"[TTSService] Connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"[TTSService] Connection test failed: {e}")
            return False

    def generate_audio_streaming(self, text: str, audio_callback: Callable[[bytes], None]) -> bool:
        """
        Generate audio with streaming playback - plays chunks as they're generated.
        
        Args:
            text: Text to convert to speech
            audio_callback: Function called with each audio chunk as it's ready
            
        Returns:
            True if streaming completed successfully, False otherwise
        """
        if not text or len(text.strip()) == 0:
            self.logger.warning("[TTSService] Empty text provided for streaming")
            return False
        
        if not self.streaming_enabled:
            # Fall back to regular generation
            audio_data = self.generate_audio(text)
            if audio_data:
                audio_callback(audio_data)
                return True
            return False
        
        # Clean text for TTS
        cleaned_text = self._clean_text_for_tts(text)
        
        # Split into streaming chunks (groups of sentences)
        streaming_chunks = self._create_streaming_chunks(cleaned_text)
        self.logger.info(f"[TTSService] Created {len(streaming_chunks)} streaming chunks for progressive playback")
        
        # Generate and play chunks progressively
        success_count = 0
        for i, chunk in enumerate(streaming_chunks):
            self.logger.info(f"[TTSService] Generating streaming chunk {i+1}/{len(streaming_chunks)} ({len(chunk)} chars)")
            
            # Generate audio for this chunk
            audio_data = self._generate_single_audio(chunk)
            if audio_data:
                # Immediately send to callback for playback
                try:
                    audio_callback(audio_data)
                    success_count += 1
                    self.logger.info(f"[TTSService] ✅ Streaming chunk {i+1} sent for playback ({len(audio_data)} bytes)")
                except Exception as e:
                    self.logger.error(f"[TTSService] Failed to send chunk {i+1} to callback: {e}")
            else:
                self.logger.warning(f"[TTSService] Failed to generate audio for streaming chunk {i+1}")
        
        success = success_count > 0
        if success:
            self.logger.info(f"[TTSService] ✅ Streaming TTS completed: {success_count}/{len(streaming_chunks)} chunks successful")
        else:
            self.logger.error("[TTSService] ❌ All streaming chunks failed to generate")
        
        return success

    def _create_streaming_chunks(self, text: str) -> List[str]:
        """
        Create chunks optimized for streaming playback.
        Each chunk contains a few sentences for natural speech flow.
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed character limit
            if len(current_chunk) + len(sentence) + 1 > self.max_chars:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                sentence_count = 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                sentence_count += 1
                
                # If we've reached the streaming chunk size, finalize this chunk
                if sentence_count >= self.streaming_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    sentence_count = 0
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Ensure we have reasonable chunk sizes
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chars:
                # Split large chunks further
                sub_chunks = self._smart_chunk_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
