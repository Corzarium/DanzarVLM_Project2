#!/usr/bin/env python3
"""
Better fix script for DanzarVLM.py issues:
1. Event loop error in processing worker
2. Whisper STT server not available
3. Type annotation issues
"""

import re

def fix_danzar_issues():
    """Apply fixes to DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace the entire problematic processing worker section
    # Find the start of the processing worker method
    start_pattern = r'def processing_worker\(self\):'
    end_pattern = r'async def transcribe_audio\(self, audio_data: np\.ndarray\) -> Optional\[str\]:'
    
    # Create the new processing worker content
    new_processing_worker = '''    def processing_worker(self):
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
                    
                    # Process with local Whisper model directly (no async needed)
                    try:
                        self.logger.info("🎯 Processing speech audio with local Whisper...")
                        
                        # Use local Whisper model directly in this thread
                        if hasattr(self, 'whisper_model') and self.whisper_model:
                            try:
                                # Run Whisper transcription directly in this thread
                                result = self.whisper_model.transcribe(speech_audio)
                                
                                if result and isinstance(result, dict) and 'text' in result:
                                    transcription = result['text'].strip()
                                    if transcription:
                                        self.logger.info(f"✅ Local Whisper transcription: '{transcription}'")
                                        
                                        # Put transcription in queue for Discord bot to process
                                        self.transcription_queue.put({
                                            'transcription': transcription,
                                            'timestamp': time.time(),
                                            'user': 'VirtualAudio'
                                        })
                                        self.logger.info("📤 Added local Whisper transcription to queue for Discord processing")
                                    else:
                                        self.logger.info("🔇 Local Whisper returned empty transcription")
                                else:
                                    self.logger.info("🔇 Local Whisper returned no result")
                                    
                            except Exception as e:
                                self.logger.error(f"❌ Local Whisper transcription error: {e}")
                        else:
                            self.logger.warning("⚠️ No local Whisper model available - audio detected but cannot transcribe")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Processing error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing worker error: {e}")
        
        self.logger.info("🎯 Whisper audio processing worker stopped")
    
    '''
    
    # Replace the processing worker method
    start_match = re.search(start_pattern, content)
    end_match = re.search(end_pattern, content)
    
    if start_match and end_match:
        start_pos = start_match.start()
        end_pos = end_match.start()
        
        # Replace the entire method
        content = content[:start_pos] + new_processing_worker + content[end_pos:]
    
    # Fix 2: Disable the callback function initialization
    content = re.sub(
        r'self\.virtual_audio = WhisperAudioCapture\(self\.app_context, self\.process_virtual_audio_sync\)',
        'self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors',
        content
    )
    
    # Fix 3: Add missing loop attribute initialization in __init__
    init_pattern = r'def __init__\(self, settings: dict, app_context: AppContext\):\s*# Initialize loop attribute for async operations\s*self\.loop = None'
    if not re.search(init_pattern, content, re.DOTALL):
        content = re.sub(
            r'def __init__\(self, settings: dict, app_context: AppContext\):',
            'def __init__(self, settings: dict, app_context: AppContext):\n        # Initialize loop attribute for async operations\n        self.loop = None',
            content
        )
    
    # Fix 4: Set the loop attribute in setup_hook
    setup_hook_pattern = r'async def setup_hook\(self\):\s*# Set the event loop for async operations\s*self\.loop = asyncio\.get_event_loop\(\)'
    if not re.search(setup_hook_pattern, content, re.DOTALL):
        content = re.sub(
            r'async def setup_hook\(self\):',
            'async def setup_hook(self):\n        # Set the event loop for async operations\n        self.loop = asyncio.get_event_loop()',
            content
        )
    
    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Applied fixes to DanzarVLM.py:")
    print("  1. Replaced problematic processing worker with direct local Whisper transcription")
    print("  2. Disabled callback function initialization")
    print("  3. Added missing loop attribute initialization")
    print("  4. Set loop attribute in setup_hook")

if __name__ == "__main__":
    fix_danzar_issues() 