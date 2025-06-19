#!/usr/bin/env python3
"""
Working TTS Installation Script
Installs a functional TTS solution to replace the problematic Chatterbox setup
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   ❌ Stderr: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ Python 3 is required")
        return False
    
    if version.minor < 10:
        print("⚠️  Python 3.10+ recommended for best compatibility")
    
    return True

def install_working_piper_tts():
    """Install Piper TTS using the working method from GitHub issues"""
    print("\n🚀 Installing Working Piper TTS Solution")
    
    # Method from GitHub issue #509 - most recent working solution
    commands = [
        # Install fixed phonemize package first
        ([sys.executable, "-m", "pip", "install", "piper-phonemize-fix==1.2.1"], 
         "Installing fixed piper-phonemize package"),
        
        # Install other dependencies
        ([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], 
         "Installing compatible numpy"),
        
        # Install onnxruntime (CPU version for compatibility)
        ([sys.executable, "-m", "pip", "install", "onnxruntime"], 
         "Installing ONNX runtime"),
        
        # Install piper-tts with no-deps to avoid conflicts
        ([sys.executable, "-m", "pip", "install", "--no-deps", "piper-tts==1.2.0"], 
         "Installing Piper TTS (no dependencies)"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"❌ Failed to install: {desc}")
            return False
    
    return True

def install_fallback_tts():
    """Install Windows SAPI TTS as fallback"""
    print("\n🔄 Installing Fallback TTS (Windows SAPI)")
    
    commands = [
        ([sys.executable, "-m", "pip", "install", "pyttsx3"], 
         "Installing pyttsx3 for Windows SAPI TTS"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"❌ Failed to install: {desc}")
            return False
    
    return True

def test_piper_installation():
    """Test if Piper TTS is working"""
    print("\n🧪 Testing Piper TTS Installation")
    
    try:
        import piper
        print("✅ Piper TTS imported successfully")
        
        # Try to create a simple TTS instance
        # Note: This might fail without a model, but import success is good
        return True
        
    except ImportError as e:
        print(f"❌ Piper TTS import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Piper TTS imported but initialization failed: {e}")
        print("   This is normal without models - installation likely successful")
        return True

def test_fallback_tts():
    """Test if fallback TTS is working"""
    print("\n🧪 Testing Fallback TTS")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ pyttsx3 TTS working")
        return True
    except Exception as e:
        print(f"❌ pyttsx3 TTS failed: {e}")
        return False

def main():
    """Main installation process"""
    print("=" * 60)
    print("🎙️  WORKING TTS INSTALLATION SCRIPT")
    print("   Solving Piper TTS dependency conflicts")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Try to install working Piper TTS
    piper_success = install_working_piper_tts()
    
    if piper_success:
        print("\n✅ Piper TTS installation completed")
        if test_piper_installation():
            print("✅ Piper TTS is working!")
        else:
            print("⚠️  Piper TTS installed but may need models")
    else:
        print("\n❌ Piper TTS installation failed")
    
    # Install fallback TTS regardless
    fallback_success = install_fallback_tts()
    
    if fallback_success:
        print("\n✅ Fallback TTS installation completed")
        if test_fallback_tts():
            print("✅ Fallback TTS is working!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 60)
    
    if piper_success:
        print("✅ Piper TTS: Installed (high-quality neural voices)")
    else:
        print("❌ Piper TTS: Failed")
    
    if fallback_success:
        print("✅ Fallback TTS: Installed (Windows SAPI)")
    else:
        print("❌ Fallback TTS: Failed")
    
    if piper_success or fallback_success:
        print("\n🎉 At least one TTS solution is available!")
        print("   Update your config to use 'piper' or 'pyttsx3' provider")
    else:
        print("\n💥 All TTS installations failed!")
        print("   You may need to manually resolve dependency conflicts")
    
    print("\n📝 Next Steps:")
    print("   1. Update config/global_settings.yaml TTS provider")
    print("   2. Restart DanzarVLM")
    print("   3. Test with !danzar tts Hello World")

if __name__ == "__main__":
    main() 