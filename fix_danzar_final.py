#!/usr/bin/env python3
"""
Final comprehensive fix script for DanzarVLM.py issues:
1. Fix indentation error in processing_worker method
2. Fix event loop error by removing async callback logic
3. Fix type annotation issues with Whisper results
4. Fix missing service attributes
"""

import re

def fix_danzar_issues():
    """Apply comprehensive fixes to DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace the entire processing_worker method with proper indentation
    processing_worker_pattern = r'def processing_worker\(self\):\s*"""Worker thread for processing audio chunks\."""\s*self\.logger\.info\("🎯 Whisper audio processing worker started"\)\s*while self\.is_recording:\s*try:\s*# Get audio chunk with timeout\s*audio_chunk = self\.audio_queue\.get\(timeout=1\.0\)\s*# Detect speech using simple level-based detection\s*is_speech, speech_ended = self\.detect_speech\(audio_chunk\)\s*# Always add to buffer when speaking\s*if self\.is_speaking:\s*self\.audio_buffer\.extend\(audio_chunk\)\s*# Process complete speech segments\s*if speech_ended and len\(self\.audio_buffer\) > 0:\s*# Convert buffer to numpy array\s*speech_audio = np\.array\(self\.audio_buffer, dtype=np\.float32\)\s*self\.audio_buffer\.clear\(\)\s*self\.is_speaking = False\s*self\.logger\.info\(f"🎤 Speech ended \(duration: {len\(speech_audio\) / self\.sample_rate:.2f}s\)"\)\s*# Process with Whisper using direct call instead of async callback\s*# Disable callback to prevent event loop errors\s*self\.callback_func = None\s*if self\.callback_func:\s*try:\s*self\.logger\.info\("🎯 Processing speech audio directly\.\.\."\)\s*# Call the transcription directly in this thread\s*# This avoids the event loop threading issues\s*import asyncio\s*try:\s*loop = asyncio\.new_event_loop\(\)\s*asyncio\.set_event_loop\(loop\)\s*future = asyncio\.run_coroutine_threadsafe\(\s*self\.callback_func\(speech_audio\),\s*self\.loop\s*\)\s*result = future\.result\(timeout=30\)\s*if result:\s*self\.logger\.info\(f"✅ Transcription: '{result}'"\)\s*except Exception as e:\s*self\.logger\.error\(f"❌ Event loop error: {e}"\)\s*except Exception as e:\s*self\.logger\.error\(f"❌ Callback error: {e}"\)\s*# Fallback to local Whisper model\s*if not result:\s*self\.logger\.info\("🔄 Processing with STT service\.\.\."\)\s*result = self\.process_with_stt_service\(speech_audio\)\s*if result:\s*self\.logger\.info\(f"✅ STT Service transcription: '{result}'"\)\s*else:\s*self\.logger\.info\("🔇 No transcription from STT service"\)\s*except queue\.Empty:\s*continue\s*except Exception as e:\s*self\.logger\.error\(f"❌ Processing worker error: {e}"\)\s*continue'
    
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
                    # Convert buffer to numpy array
                    speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer.clear()
                    self.is_speaking = False
                    self.logger.info(f"🎤 Speech ended (duration: {len(speech_audio) / self.sample_rate:.2f}s)")
                    
                    # Process with local Whisper model directly
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
                                        # Process the transcription
                                        self.process_transcription(transcription)
                                    else:
                                        self.logger.info("🔇 Empty transcription from local Whisper")
                                else:
                                    self.logger.info("🔇 No transcription from local Whisper")
                            except Exception as e:
                                self.logger.error(f"❌ Local Whisper error: {e}")
                        else:
                            self.logger.warning("⚠️ Local Whisper model not available")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Processing error: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing worker error: {e}")
                continue'''
    
    # Apply the replacement
    content = re.sub(processing_worker_pattern, new_processing_worker, content, flags=re.DOTALL)
    
    # Fix 2: Fix the transcribe_audio method type annotations
    transcribe_pattern = r'if result and isinstance\(result, dict\) and \'text\' in result:\s*transcription = result\[\'text\'\]\.strip\(\)'
    transcribe_replacement = '''if result and isinstance(result, dict) and 'text' in result:
                        transcription = str(result['text']).strip()'''
    
    content = re.sub(transcribe_pattern, transcribe_replacement, content)
    
    # Fix 3: Fix the async transcribe method type annotations
    async_transcribe_pattern = r'if result and isinstance\(result, dict\) and \'text\' in result:\s*transcription = result\[\'text\'\]\.strip\(\)'
    async_transcribe_replacement = '''if result and isinstance(result, dict) and 'text' in result:
                        transcription = str(result['text']).strip()'''
    
    content = re.sub(async_transcribe_pattern, async_transcribe_replacement, content)
    
    # Fix 4: Remove the problematic qwen_omni_service calls
    qwen_pattern = r'response = await self\.app_context\.qwen_omni_service\.generate_response\([^)]+\)'
    qwen_replacement = '''# Qwen2.5-Omni service not available
                        self.logger.warning("⚠️ Qwen2.5-Omni service not available")
                        response = None'''
    
    content = re.sub(qwen_pattern, qwen_replacement, content)
    
    # Fix 5: Remove the problematic qwen_omni_service calls in the sync method
    qwen_sync_pattern = r'response = loop\.run_until_complete\(\s*self\.app_context\.qwen_omni_service\.generate_response\([^)]+\)\s*\)'
    qwen_sync_replacement = '''# Qwen2.5-Omni service not available
                        self.logger.warning("⚠️ Qwen2.5-Omni service not available")
                        response = None'''
    
    content = re.sub(qwen_sync_pattern, qwen_sync_replacement, content)
    
    # Fix 6: Fix the loop attribute initialization
    loop_pattern = r'self\.loop = None'
    loop_replacement = '''# Initialize loop attribute for async operations
        self.loop = None  # Will be set when needed'''
    
    content = re.sub(loop_pattern, loop_replacement, content)
    
    # Fix 7: Remove duplicate vad_voice_receiver declaration
    vad_duplicate_pattern = r'self\.vad_voice_receiver: Optional\[Any\] = None\s*self\.simple_voice_receiver: Optional\[SimpleVoiceReceiver\] = None\s*self\.vad_voice_receiver: Optional\[Any\] = None'
    vad_duplicate_replacement = '''self.vad_voice_receiver: Optional[Any] = None
        self.simple_voice_receiver: Optional[SimpleVoiceReceiver] = None'''
    
    content = re.sub(vad_duplicate_pattern, vad_duplicate_replacement, content)
    
    # Fix 8: Fix the VADVoiceReceiver type annotation
    vad_type_pattern = r'self\.vad_voice_receiver: Optional\[VADVoiceReceiver\] = None'
    vad_type_replacement = '''self.vad_voice_receiver: Optional[Any] = None  # VADVoiceReceiver type'''
    
    content = re.sub(vad_type_pattern, vad_type_replacement, content)
    
    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Applied comprehensive fixes to DanzarVLM.py")
    print("   - Fixed processing_worker indentation")
    print("   - Removed problematic async callback logic")
    print("   - Fixed type annotations for Whisper results")
    print("   - Removed unavailable service calls")
    print("   - Fixed duplicate variable declarations")

if __name__ == "__main__":
    fix_danzar_issues() 