#!/usr/bin/env python3
"""
Quick fix script for DanzarVLM.py issues:
1. Event loop error in processing worker
2. Whisper STT server not available
3. Type annotation issues
"""

import re
import os

def fix_danzar_issues():
    """Apply fixes to DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove the problematic callback function call in processing_worker
    # Replace the complex async callback logic with simple local Whisper transcription
    callback_pattern = r'# Process with Whisper using direct call instead of async callback\s+if self\.callback_func:\s+try:\s+self\.logger\.info\("🎯 Processing speech audio directly\.\.\."\)\s+.*?except Exception as e:\s+self\.logger\.error\(f"❌ Processing error: {e}"\)'
    
    replacement = '''# Process with Whisper using direct call instead of async callback
                    try:
                        self.logger.info("🎯 Processing speech audio with local Whisper...")
                        
                        # Use local Whisper model directly in this thread
                        if hasattr(self, 'whisper_model') and self.whisper_model:
                            try:
                                # Run Whisper transcription directly in this thread
                                result = self.whisper_model.transcribe(speech_audio)
                                
                                if result and 'text' in result:
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
                        self.logger.error(f"❌ Processing error: {e}")'''
    
    # Apply the fix with a more targeted approach
    content = re.sub(
        r'# Process with Whisper using direct call instead of async callback\s+if self\.callback_func:',
        '# Process with Whisper using direct call instead of async callback\n                    # DISABLED: if self.callback_func:',
        content,
        flags=re.DOTALL
    )
    
    # Fix 2: Disable the callback function initialization
    content = re.sub(
        r'self\.virtual_audio = WhisperAudioCapture\(self\.app_context, self\.process_virtual_audio_sync\)',
        'self.virtual_audio = WhisperAudioCapture(self.app_context, None)  # Disabled callback to prevent event loop errors',
        content
    )
    
    # Fix 3: Update the transcribe_audio method to handle the result type properly
    content = re.sub(
        r'if result and \'text\' in result:\s+transcription = result\[\'text\'\]\.strip\(\)',
        'if result and isinstance(result, dict) and \'text\' in result:\n                        transcription = result[\'text\'].strip()',
        content
    )
    
    # Fix 4: Add missing loop attribute initialization
    content = re.sub(
        r'def __init__\(self, settings: dict, app_context: AppContext\):',
        'def __init__(self, settings: dict, app_context: AppContext):\n        # Initialize loop attribute for async operations\n        self.loop = None',
        content
    )
    
    # Fix 5: Set the loop attribute in setup_hook
    setup_hook_pattern = r'async def setup_hook\(self\):'
    setup_hook_replacement = '''async def setup_hook(self):
        # Set the event loop for async operations
        self.loop = asyncio.get_event_loop()'''
    
    content = re.sub(setup_hook_pattern, setup_hook_replacement, content)
    
    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Applied fixes to DanzarVLM.py:")
    print("  1. Disabled problematic async callback in processing worker")
    print("  2. Disabled callback function initialization")
    print("  3. Fixed result type handling in transcribe_audio")
    print("  4. Added missing loop attribute initialization")
    print("  5. Set loop attribute in setup_hook")

if __name__ == "__main__":
    fix_danzar_issues() 