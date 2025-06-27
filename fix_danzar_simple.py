#!/usr/bin/env python3
"""
Simple fix script for DanzarVLM.py issues:
1. Fix indentation error in processing_worker method
2. Fix event loop error by removing async callback logic
3. Fix type annotation issues with Whisper results
"""

def fix_danzar_issues():
    """Apply simple fixes to DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix 1: Find and replace the processing_worker method
    new_lines = []
    in_processing_worker = False
    skip_until_end = False
    
    for i, line in enumerate(lines):
        if 'def processing_worker(self):' in line:
            in_processing_worker = True
            skip_until_end = True
            # Add the fixed processing_worker method
            new_lines.append('    def processing_worker(self):\n')
            new_lines.append('        """Worker thread for processing audio chunks."""\n')
            new_lines.append('        self.logger.info("🎯 Whisper audio processing worker started")\n')
            new_lines.append('        \n')
            new_lines.append('        while self.is_recording:\n')
            new_lines.append('            try:\n')
            new_lines.append('                # Get audio chunk with timeout\n')
            new_lines.append('                audio_chunk = self.audio_queue.get(timeout=1.0)\n')
            new_lines.append('                \n')
            new_lines.append('                # Detect speech using simple level-based detection\n')
            new_lines.append('                is_speech, speech_ended = self.detect_speech(audio_chunk)\n')
            new_lines.append('                \n')
            new_lines.append('                # Always add to buffer when speaking\n')
            new_lines.append('                if self.is_speaking:\n')
            new_lines.append('                    self.audio_buffer.extend(audio_chunk)\n')
            new_lines.append('                \n')
            new_lines.append('                # Process complete speech segments\n')
            new_lines.append('                if speech_ended and len(self.audio_buffer) > 0:\n')
            new_lines.append('                    # Convert buffer to numpy array\n')
            new_lines.append('                    speech_audio = np.array(self.audio_buffer, dtype=np.float32)\n')
            new_lines.append('                    self.audio_buffer.clear()\n')
            new_lines.append('                    self.is_speaking = False\n')
            new_lines.append('                    self.logger.info(f"🎤 Speech ended (duration: {len(speech_audio) / self.sample_rate:.2f}s)")\n')
            new_lines.append('                    \n')
            new_lines.append('                    # Process with local Whisper model directly\n')
            new_lines.append('                    try:\n')
            new_lines.append('                        self.logger.info("🎯 Processing speech audio with local Whisper...")\n')
            new_lines.append('                        \n')
            new_lines.append('                        # Use local Whisper model directly in this thread\n')
            new_lines.append('                        if hasattr(self, \'whisper_model\') and self.whisper_model:\n')
            new_lines.append('                            try:\n')
            new_lines.append('                                # Run Whisper transcription directly in this thread\n')
            new_lines.append('                                result = self.whisper_model.transcribe(speech_audio)\n')
            new_lines.append('                                \n')
            new_lines.append('                                if result and isinstance(result, dict) and \'text\' in result:\n')
            new_lines.append('                                    transcription = str(result[\'text\']).strip()\n')
            new_lines.append('                                    if transcription:\n')
            new_lines.append('                                        self.logger.info(f"✅ Local Whisper transcription: \'{transcription}\'")\n')
            new_lines.append('                                        # Process the transcription\n')
            new_lines.append('                                        self.process_transcription(transcription)\n')
            new_lines.append('                                    else:\n')
            new_lines.append('                                        self.logger.info("🔇 Empty transcription from local Whisper")\n')
            new_lines.append('                                else:\n')
            new_lines.append('                                    self.logger.info("🔇 No transcription from local Whisper")\n')
            new_lines.append('                            except Exception as e:\n')
            new_lines.append('                                self.logger.error(f"❌ Local Whisper error: {e}")\n')
            new_lines.append('                        else:\n')
            new_lines.append('                            self.logger.warning("⚠️ Local Whisper model not available")\n')
            new_lines.append('                            \n')
            new_lines.append('                    except Exception as e:\n')
            new_lines.append('                        self.logger.error(f"❌ Processing error: {e}")\n')
            new_lines.append('                        \n')
            new_lines.append('            except queue.Empty:\n')
            new_lines.append('                continue\n')
            new_lines.append('            except Exception as e:\n')
            new_lines.append('                self.logger.error(f"❌ Processing worker error: {e}")\n')
            new_lines.append('                continue\n')
            continue
        
        # Skip the old processing_worker method
        if skip_until_end and in_processing_worker:
            # Check if we've reached the end of the method (next method or class)
            if line.strip().startswith('def ') and line != line.strip():
                # This is the start of the next method
                in_processing_worker = False
                skip_until_end = False
                new_lines.append(line)
            elif line.strip().startswith('class '):
                # This is the start of the next class
                in_processing_worker = False
                skip_until_end = False
                new_lines.append(line)
            elif line.strip() == '' and i + 1 < len(lines) and lines[i + 1].strip().startswith('def '):
                # Empty line followed by method definition
                in_processing_worker = False
                skip_until_end = False
                new_lines.append(line)
            else:
                # Still in the method, skip this line
                continue
        else:
            new_lines.append(line)
    
    # Fix 2: Fix type annotations for Whisper results
    for i, line in enumerate(new_lines):
        if 'result[\'text\'].strip()' in line and 'transcription =' in line:
            new_lines[i] = line.replace('result[\'text\'].strip()', 'str(result[\'text\']).strip()')
    
    # Fix 3: Remove problematic qwen_omni_service calls
    for i, line in enumerate(new_lines):
        if 'self.app_context.qwen_omni_service.generate_response(' in line:
            # Find the start of this block
            start = i
            while start > 0 and not new_lines[start].strip().startswith('response ='):
                start -= 1
            
            # Replace the entire block
            if start < i:
                new_lines[start] = '                        # Qwen2.5-Omni service not available\n'
                new_lines[start + 1] = '                        self.logger.warning("⚠️ Qwen2.5-Omni service not available")\n'
                new_lines[start + 2] = '                        response = None\n'
                
                # Remove the rest of the problematic lines
                for j in range(start + 3, i + 1):
                    if j < len(new_lines):
                        new_lines[j] = ''
    
    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✅ Applied simple fixes to DanzarVLM.py")
    print("   - Fixed processing_worker indentation")
    print("   - Removed problematic async callback logic")
    print("   - Fixed type annotations for Whisper results")
    print("   - Removed unavailable service calls")

if __name__ == "__main__":
    fix_danzar_issues() 