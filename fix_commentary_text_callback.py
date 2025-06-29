#!/usr/bin/env python3
"""
Fix for commentary text callback issue
"""
import os
import shutil
from datetime import datetime

def backup_and_fix():
    """Backup current file and apply fix"""
    
    # Backup the current vision integration service
    backup_file = f"services/vision_integration_service_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    try:
        shutil.copy("services/vision_integration_service.py", backup_file)
        print(f"📁 Backed up to: {backup_file}")
        
        # Read the current file
        with open("services/vision_integration_service.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if the fix is already applied
        if ">>> ABOUT TO CALL TEXT CALLBACK" in content:
            print("✅ Text callback fix already applied")
            return True
        
        # Find the commentary generation section and add text callback
        if "Commentary generated:" in content:
            print("🔧 Applying text callback fix...")
            
            # Add text callback call after commentary generation
            old_line = 'self.logger.info(f"[VisionIntegration] Commentary generated: {response[:100]}...")'
            new_line = '''self.logger.info(f"[VisionIntegration] Commentary generated: {response[:100]}...")
            
            # Call text callback for commentary
            if self.text_callback and callable(self.text_callback):
                self.logger.info(f"[VisionIntegration] >>> ABOUT TO CALL TEXT CALLBACK with: '{response.strip()[:50]}...'")
                try:
                    if asyncio.iscoroutinefunction(self.text_callback):
                        asyncio.create_task(self.text_callback(response))
                        self.logger.info(f"[VisionIntegration] >>> ASYNC TEXT CALLBACK TASK CREATED")
                    else:
                        self.text_callback(response)
                        self.logger.info(f"[VisionIntegration] >>> SYNC TEXT CALLBACK CALLED")
                except Exception as e:
                    self.logger.error(f"[VisionIntegration] >>> TEXT CALLBACK ERROR: {e}")
            else:
                self.logger.warning(f"[VisionIntegration] >>> NO TEXT CALLBACK AVAILABLE")'''
            
            content = content.replace(old_line, new_line)
            
            # Write the fixed file
            with open("services/vision_integration_service.py", "w", encoding="utf-8") as f:
                f.write(content)
            
            print("✅ Text callback fix applied successfully")
            return True
        else:
            print("❌ Could not find commentary generation section")
            return False
            
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        return False

def main():
    print("🔧 DanzarAI Commentary Text Callback Fix")
    print("=" * 50)
    
    if backup_and_fix():
        print("\n✅ Fix applied successfully!")
        print("\n📋 Next steps:")
        print("1. Restart DanzarAI")
        print("2. Use !watch command")
        print("3. Check logs for '>>> ABOUT TO CALL TEXT CALLBACK' messages")
        print("4. Commentary should now appear in Discord text channel")
    else:
        print("\n❌ Fix failed - check the error messages above")

if __name__ == "__main__":
    main() 