#!/usr/bin/env python3
"""
Piper TTS Installation Script
Installs Piper TTS with GPU acceleration for Windows 11 + NVIDIA GPUs
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
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  NVIDIA GPU not detected - will use CPU fallback")
    return False

def install_piper_tts():
    """Install Piper TTS and dependencies"""
    packages = [
        "piper-tts",
        "onnxruntime-gpu",  # GPU acceleration
        "soundfile",        # Audio processing
        "numpy",           # Array operations
        "requests",        # Model downloads
        "webrtcvad",       # Voice activity detection
        "sounddevice",     # Real-time audio I/O
    ]
    
    print("📦 Installing Piper TTS and dependencies...")
    
    for package in packages:
        success = run_command([
            sys.executable, "-m", "pip", "install", package, "--upgrade"
        ], f"Installing {package}")
        
        if not success:
            print(f"⚠️  Failed to install {package} - continuing anyway")
    
    return True

def install_ffmpeg():
    """Install FFmpeg for audio processing"""
    if platform.system() == "Windows":
        print("📦 FFmpeg installation on Windows:")
        print("   Please install FFmpeg manually:")
        print("   1. Download from: https://ffmpeg.org/download.html")
        print("   2. Add to PATH environment variable")
        print("   3. Or use: winget install FFmpeg")
        
        # Try winget installation
        try:
            result = subprocess.run(["winget", "install", "FFmpeg"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ FFmpeg installed via winget")
                return True
        except FileNotFoundError:
            pass
        
        print("   ⚠️  Please install FFmpeg manually")
        return False
    else:
        # Linux/WSL
        return run_command(["sudo", "apt", "update"]) and \
               run_command(["sudo", "apt", "install", "-y", "ffmpeg"])

def test_installation():
    """Test Piper TTS installation"""
    print("🧪 Testing Piper TTS installation...")
    
    try:
        # Test basic import
        import piper
        print("   ✅ Piper TTS imported successfully")
        
        # Test ONNX Runtime GPU
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print("   ✅ ONNX Runtime GPU support available")
            else:
                print("   ⚠️  ONNX Runtime GPU support not available")
        except ImportError:
            print("   ❌ ONNX Runtime not available")
        
        # Test other dependencies
        deps = ['soundfile', 'numpy', 'webrtcvad', 'sounddevice']
        for dep in deps:
            try:
                __import__(dep)
                print(f"   ✅ {dep} available")
            except ImportError:
                print(f"   ❌ {dep} not available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Piper TTS import failed: {e}")
        return False

def create_test_script():
    """Create test script for Piper TTS"""
    test_script = """#!/usr/bin/env python3
# Test Piper TTS installation
import sys
sys.path.append('.')

from services.piper_tts_service import PiperTTSService

def test_piper_tts():
    print("🧪 Testing Piper TTS Service...")
    
    # Initialize service
    tts = PiperTTSService()
    
    # Test generation
    test_text = "Hello, this is a test of the Piper TTS service."
    print(f"Generating: {test_text}")
    
    audio_bytes = tts.generate_audio(test_text)
    
    if audio_bytes:
        print(f"✅ Generated {len(audio_bytes)} bytes")
        
        # Save test file
        with open("piper_test.wav", "wb") as f:
            f.write(audio_bytes)
        print("💾 Saved to piper_test.wav")
        
        # Show stats
        stats = tts.get_stats()
        print(f"📊 Stats: {stats}")
        
        return True
    else:
        print("❌ TTS generation failed")
        return False

if __name__ == "__main__":
    success = test_piper_tts()
    sys.exit(0 if success else 1)
"""
    
    with open("test_piper_tts.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_piper_tts.py")

def main():
    """Main installation process"""
    print("🚀 Piper TTS Installation for DanzarAI")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    has_cuda = check_cuda()
    
    # Install packages
    if not install_piper_tts():
        print("❌ Package installation failed")
        return False
    
    # Install FFmpeg
    install_ffmpeg()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Piper TTS Installation Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Test with: python test_piper_tts.py")
    print("2. Update DanzarAI configuration")
    print("3. Restart DanzarAI bot")
    
    if has_cuda:
        print("\n🚀 GPU acceleration enabled!")
        print("   Piper will use CUDA for faster TTS generation")
    else:
        print("\n💻 CPU mode enabled")
        print("   Consider installing CUDA for better performance")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 