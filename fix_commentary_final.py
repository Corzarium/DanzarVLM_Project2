#!/usr/bin/env python3
"""
Final fix for commentary text callback issue
"""
import os
import shutil
from datetime import datetime

def apply_fix():
    """Apply the final fix for commentary text callback"""
    
    print("🔧 Applying Final Commentary Text Callback Fix")
    print("=" * 50)
    
    # Backup current file
    backup_file = f"services/vision_integration_service_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy("services/vision_integration_service.py", backup_file)
    print(f"📁 Backed up to: {backup_file}")
    
    # Read current file
    with open("services/vision_integration_service.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if fix is already applied
    if ">>> FINAL TEXT CALLBACK DEBUG" in content:
        print("✅ Fix already applied")
        return True
    
    # Find the commentary generation section and add enhanced debug logging
    if "Commentary generated:" in content:
        print("🔧 Applying enhanced text callback debug logging...")
        
        # Add enhanced debug logging after commentary generation
        old_section = '''if self.logger:
                    self.logger.info(f"[VisionIntegration] Commentary generated: {response[:100]}...")
                    if text_sent:
                        self.logger.info("[VisionIntegration] ✅ Text commentary sent successfully")
                    else:
                        self.logger.warning("[VisionIntegration] ❌ Text commentary failed to send")'''
        
        new_section = '''if self.logger:
                    self.logger.info(f"[VisionIntegration] Commentary generated: {response[:100]}...")
                    self.logger.info(f"[VisionIntegration] >>> FINAL TEXT CALLBACK DEBUG")
                    self.logger.info(f"[VisionIntegration] Text callback exists: {self.text_callback is not None}")
                    self.logger.info(f"[VisionIntegration] Text callback callable: {callable(self.text_callback) if self.text_callback else False}")
                    self.logger.info(f"[VisionIntegration] Text sent flag: {text_sent}")
                    if text_sent:
                        self.logger.info("[VisionIntegration] ✅ Text commentary sent successfully")
                    else:
                        self.logger.warning("[VisionIntegration] ❌ Text commentary failed to send")
                        self.logger.warning("[VisionIntegration] >>> MANUAL TEXT CALLBACK ATTEMPT")
                        # Manual text callback attempt
                        if self.text_callback and callable(self.text_callback):
                            try:
                                self.logger.info("[VisionIntegration] >>> MANUAL TEXT CALLBACK CALLING")
                                if asyncio.iscoroutinefunction(self.text_callback):
                                    asyncio.create_task(self.text_callback(response))
                                    self.logger.info("[VisionIntegration] >>> MANUAL ASYNC TASK CREATED")
                                else:
                                    self.text_callback(response)
                                    self.logger.info("[VisionIntegration] >>> MANUAL SYNC CALLBACK CALLED")
                            except Exception as e:
                                self.logger.error(f"[VisionIntegration] >>> MANUAL TEXT CALLBACK ERROR: {e}")'''
        
        content = content.replace(old_section, new_section)
        
        # Write the fixed file
        with open("services/vision_integration_service.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ Enhanced text callback debug logging applied")
        return True
    else:
        print("❌ Could not find commentary generation section")
        return False

def main():
    print("🔧 DanzarAI Commentary Text Callback - Final Fix")
    print("=" * 50)
    
    if apply_fix():
        print("\n✅ Fix applied successfully!")
        print("\n📋 Next steps:")
        print("1. Restart DanzarAI")
        print("2. Use !watch command")
        print("3. Check logs for '>>> FINAL TEXT CALLBACK DEBUG' messages")
        print("4. Look for '>>> MANUAL TEXT CALLBACK' attempts")
        print("5. Commentary should now appear in Discord text channel")
    else:
        print("\n❌ Fix failed - check the error messages above")

if __name__ == "__main__":
    main() 